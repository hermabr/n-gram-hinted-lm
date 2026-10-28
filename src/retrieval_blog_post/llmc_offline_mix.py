"""The hinted llm.c model's output mixture redone offline with other validation count rows: a checkpoint's per-token
LM loss (lm_nll, computed with the run's exact hint rows) mixed with the interpolated Witten-Bell n-gram of the given
count rows, with the checkpoint's own NgramMix parameters. Exact for a change of count rows only (other hint rows
change lm_nll too: give their eval.sh dumps as `lms`). The checkpoints (CKPTS, saved with HINT_CKPT) are hinted llm.c
runs of 330 to 880 steps; these sweeps chose the CPU validation rows' per-level shard caps (cpu_val.py).

    uv run src/retrieval_blog_post/llmc_offline_mix.py [family,...] [steps ...]   # default: marginal,long, all of CKPTS
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import furu
import numpy as np

from retrieval_blog_post.config import DATA, SCRATCH

EDGES = (2, 3, 5, 9, 17, 33, 100, 1000, 10000)  # NgramMix.COUNT_EDGES
BINS = len(EDGES) + 1
VAL = DATA / "fineweb_val_000000.bin"
T_VAL = 10_485_760

Config = tuple[tuple[int, int], ...]


def params(ckpt: Path) -> dict[str, np.ndarray]:
    """The NgramMix parameters of a checkpoint (float64): base [V], gamma [L*BINS], gamma_level [L], gamma_bin [BINS],
    lam_logit [(L+1)*BINS], lam_level [L+1], lam_sparse [2(L+1)]."""
    cfg = json.loads((ckpt / "config.json").read_text())["hint"]
    p = np.load(ckpt / "hint_params.npy").astype(np.float64)
    L, V, off = cfg["levels"], cfg["V"], cfg["offsets"]
    sizes = {"base": V, "gamma": L * BINS, "gamma_level": L, "gamma_bin": BINS, "lam_logit": (L + 1) * BINS,
             "lam_level": L + 1, "lam_sparse": 2 * (L + 1)}
    return {n: p[off[n] : off[n] + s] for n, s in sizes.items()}


def mixed_loss(prm: dict, lm_nll, tot, dist, cnt, target) -> np.ndarray:
    """Per position -log(lam p_LM + (1 - lam) q) (float64; tot/dist/cnt [n, L]), as NgramMix.forward."""
    n, L = tot.shape
    base = prm["base"]
    lse = base.max() + np.log(np.exp(base - base.max()).sum())
    b = (tot[..., None] > np.asarray(EDGES, np.float64)).sum(-1)
    g = np.exp(np.clip(prm["gamma"][np.arange(L) * BINS + b] + prm["gamma_level"] + prm["gamma_bin"][b], -20, 20))
    p = np.exp(base[target] - lse)
    for j in range(L):
        m = tot[:, j] > 0
        gd = g[:, j] * dist[:, j]
        p = np.where(m, (cnt[:, j] + gd * p) / np.where(m, tot[:, j] + gd, 1.0), p)
    logq = np.log(np.maximum(p, 1e-30))
    seen = tot > 0
    deepest = np.where(seen.any(1), L - 1 - np.argmax(seen[:, ::-1], 1), -1)
    at = np.maximum(deepest, 0)
    rows = np.arange(n)
    hit = deepest >= 0
    d = deepest + 1
    gate = np.where(hit, d * BINS + b[rows, at], 0)
    sp = np.where(hit, d * 2 + (dist[rows, at] == 1), 0)
    a = prm["lam_logit"][gate] + prm["lam_level"][gate // BINS] + prm["lam_sparse"][sp]
    logsig = lambda x: -np.logaddexp(0.0, -x)  # noqa: E731
    return -np.logaddexp(logsig(a) - lm_nll, logsig(-a) + logq)


def mixed_loss_all(prm: dict, lm_nll, tot, dist, cnt, target, threads: int = 16) -> np.ndarray:
    """mixed_loss over all positions, in chunks on `threads` threads (numpy releases the GIL)."""
    from concurrent.futures import ThreadPoolExecutor

    chunk = 1 << 18
    out = np.empty(len(target))

    def one(i: int) -> None:
        s = slice(i, i + chunk)
        out[s] = mixed_loss(prm, lm_nll[s], tot[s].astype(np.float64), dist[s].astype(np.float64),
                            cnt[s].astype(np.float64), target[s])

    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(one, range(0, len(target), chunk)))
    return out


def subsets(loss: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    return {name: float(loss[m].mean()) for name, m in
            (("all", np.ones_like(labels, bool)), ("noexact", labels >= 1), ("nonear", labels == 2))}


@dataclass(frozen=True)
class Sweep:
    directory: Path
    results: tuple[dict, ...]


class OfflineMixSweep(furu.Spec[Sweep]):
    """The mixture's validation loss (all / noexact / nonear) of checkpoint `ckpt` per configuration of `configs`:
    per-level shard caps ((k, K) pairs; K = 0 drops the level, unlisted levels exact), the count rows assembled from
    CappedValCounts and ValCountRows. `lms` (parallel to configs; "": the checkpoint's own) are directories whose
    lm_nll.npy replaces the checkpoint's. results.json's "dump" is the mean of the checkpoint's loss.npy, which the
    exact rows' entry () must match."""

    ckpt: str
    configs: tuple[Config, ...]
    lms: tuple[str, ...] = ()

    migrations = (furu.Added("lms", default=()),)

    def metadata(self) -> furu.Metadata:
        return furu.Metadata(requires=furu.Requires(cpus=furu.at_least(16), memory=furu.GiB(96), gpus=0))

    def _caps(self) -> set[tuple[int, int]]:
        return {(k, c) for config in self.configs for k, c in config if 0 < c < 1028}

    @furu.dependency
    def inputs(self) -> list[furu.Spec]:
        from retrieval_blog_post.cpu_val import _capped_counts
        from retrieval_blog_post.old_records import ValCountRows, ValSubsetLabels

        return [ValCountRows(), ValSubsetLabels(), *sorted({_capped_counts(k, c) for k, c in self._caps()}, key=repr)]

    def create(self) -> Sweep:
        from retrieval_blog_post.cpu_val import _capped_counts
        from retrieval_blog_post.old_records import ValCountRows, ValSubsetLabels
        from retrieval_blog_post.val_hints import ALL_LEVELS

        ckpt = Path(self.ckpt)
        names = ("tot", "dist", "cnt")
        full = ValCountRows().create().directory
        exact = {n: np.load(full / f"{n}.npy") for n in names}
        labels = np.load(ValSubsetLabels().create().directory / "labels.npy")
        target = np.asarray(np.memmap(VAL, np.uint16, "r", offset=1024)[1 : T_VAL + 1]).astype(np.int64)
        own = np.load(ckpt / "lm_nll.npy").astype(np.float64)
        assert not self.lms or len(self.lms) == len(self.configs)
        prm = params(ckpt)
        cache: dict[tuple[int, int], dict[str, np.ndarray]] = {}

        def column(k: int, c: int) -> dict[str, np.ndarray]:
            if (k, c) not in cache:
                counts = _capped_counts(k, c).create()
                cache[(k, c)] = {n: np.asarray(counts.get(k, n)) for n in names}
            return cache[(k, c)]

        results = []
        for i, config in enumerate(self.configs):
            other = self.lms[i] if self.lms else ""
            lm = np.load(Path(other) / "lm_nll.npy").astype(np.float64) if other else own
            cols = {n: exact[n].copy() for n in names}
            for k, c in config:
                j = ALL_LEVELS.index(k)
                for n in names:
                    cols[n][:, j] = 0 if c == 0 else (exact[n][:, j] if c >= 1028 else column(k, c)[n])
            loss = mixed_loss_all(prm, lm, *(cols[n] for n in names), target, min(32, os.cpu_count() or 16))
            results.append({"config": [list(p) for p in config], "lm": other, **subsets(loss, labels)})
            self.logger.info(f"{ckpt.name} {config}: {results[-1]}")
        out = self.directory.data
        dump = float(np.load(ckpt / "loss.npy").astype(np.float64).mean())
        (out / "results.json").write_text(json.dumps({"ckpt": str(ckpt), "dump": dump, "results": results}, indent=1))
        return Sweep(out, tuple(results))


CKPTS = SCRATCH / "llmc-cpu/ckpt"
SHORT_K = (10, 20, 50, 100, 200, 400)


def marginal_configs() -> tuple[Config, ...]:
    """The exact rows, then one level changed at a time: each short level capped over SHORT_K, each long level and
    all of them over 100 shards, each level dropped; then uniform short caps (the earlier experiment's)."""
    from retrieval_blog_post.cpu_val import LONG, SHORT

    configs: list[Config] = [()]
    configs += [((k, c),) for k in SHORT for c in SHORT_K]
    configs += [((k, c),) for k in LONG for c in (100,)] + [tuple((k, c) for k in LONG) for c in (100,)]
    configs += [((k, 0),) for k in (*SHORT, *LONG)]
    configs += [tuple((k, c) for k in SHORT) for c in SHORT_K]
    return tuple(configs)


def long_configs() -> tuple[Config, ...]:
    """The long levels over 300 shards (each, and all), alone and with uniform short caps of 20 / 50 / 100."""
    from retrieval_blog_post.cpu_val import LONG, SHORT

    configs: list[Config] = [((k, 300),) for k in LONG] + [tuple((k, 300) for k in LONG)]
    for c in (20, 50, 100):
        short = tuple((k, c) for k in SHORT)
        configs += [short + tuple((k, l) for k in LONG) for l in (100, 300)]
    return tuple(configs)


JOINT_K = (10, 20, 50, 100, 200, 400, 1028)


def joint_configs() -> tuple[Config, ...]:
    """Every nondecreasing per-level schedule K_2 <= K_3 <= ... <= K_6 over JOINT_K (1028: exact), K_2, K_3 <= 400 (the
    CPU scan's cost model wants them nondecreasing), the long levels exact; the exact rows first."""
    from itertools import combinations_with_replacement

    from retrieval_blog_post.cpu_val import SHORT

    configs: list[Config] = [()]
    for ks in combinations_with_replacement(JOINT_K, len(SHORT)):
        if ks[1] <= 400:
            configs.append(tuple(zip(SHORT, ks)))
    return tuple(configs)


FAMILIES = {"marginal": marginal_configs, "long": long_configs, "joint": joint_configs}


if __name__ == "__main__":

    def _main() -> None:
        import sys

        from retrieval_blog_post.cpu_val import create
        from retrieval_blog_post.llmc_offline_mix import FAMILIES, OfflineMixSweep

        families = sys.argv[1].split(",") if sys.argv[1:] else ["marginal", "long"]
        steps = sys.argv[2:] or sorted(p.name for p in CKPTS.iterdir() if (p / "lm_nll.npy").exists())
        specs = [OfflineMixSweep(ckpt=str(CKPTS / s), configs=FAMILIES[f]()) for f in families for s in steps]
        for spec, result in zip(specs, create(specs, job_name="offline-mix")):
            print(spec.ckpt, json.dumps(result.results[:3]), flush=True)

    _main()
