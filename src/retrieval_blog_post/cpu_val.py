"""Validation hint rows built on spare CPUs, approximately, on two ideas: the short contexts (levels 2-6) need far
fewer corpus tokens than the long ones, and the validation rows need not be exact.

  CappedValCounts: exact counts of the validation contexts over the first `shards` training shards only;
  ApproxValCountRows, ApproxValHintRows: ValCountRows and ValHintRows with some levels counted over fewer shards;
  ProxyLoss: what such count rows cost, by cpu_val_proxy.py (the n-gram alone, or mixed with an LM);
  CpuValTiming: exact_match.val_scan, the CPU scan that builds the rows, timed and checked against the above.

val_scan is exact. The long levels over all the shards are cheap on CPUs (8-20 in ~8 s at 96 threads), the short ones
are not (a random probe per corpus position and level); capping the short levels at 10-20 shards costs +0.03 (levels
<= 3) to +0.12 (<= 6) in the LM-mixture proxy.

    uv run src/retrieval_blog_post/cpu_val.py [quality | rows FILE.json]   # keep it running (e.g. in tmux)
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import furu
import numpy as np
from furu.resources import ResourceFloor
from furu.worker.backends.slurm.backend import SlurmWorkerBackend

from retrieval_blog_post.config import DATA, SCRATCH, TORCH_VENV
from retrieval_blog_post.fineweb import TRAIN_SHARDS, FinewebShards
from retrieval_blog_post.kgram import VAL_TOKENS
from retrieval_blog_post.old_records import CountRowFiles, HintRowFiles
from retrieval_blog_post.config import cpu
from retrieval_blog_post.val_hints import (
    ALL_LEVELS,
    ARRAYS,
    LEVELS,
    TOP,
    ContextCounts,
    Hints,
    check_variant,
    deepest,
)

SHORT = (2, 3, 4, 5, 6)  # the levels `short` caps
LONG = tuple(k for k in ALL_LEVELS if k not in SHORT)  # (8, 10, 12, 16, 20): level_shards may cap these too
HINT_ENTRIES = 8  # old_records.ENTRIES: the hint rows' candidates (the capped counts keep that many per context)
CAPS = (5, 10, 20, 50, 100, 200)
# Per level, the growth exponent of distinct (context, next token) pairs with the corpus size: gpu_val_stats.PAIRS'
# fits (validation contexts' pairs at 4 and 1028 shards); ApproxValCountRows(scale=True) grows dist with it
DIST_GROWTH = {2: 0.620, 3: 0.667, 4: 0.694, 5: 0.706, 6: 0.705}
PROXY = Path(__file__).resolve().parent / "cpu_val_proxy.py"
# ProxyLoss(mix=True)'s LM: token_losses.NoHintsRecordTokenLosses' per-token losses (the record trained without hints,
# 3.736), by validation position (its rows sorted by position, targets checked against the validation shard)
LM_LOSSES = SCRATCH / "oldrec-cpuval/lm_loss_nohints.npy"


class _Arrays:
    """A ContextCounts look-alike over in-memory arrays: get(k, name)."""

    def __init__(self, arrays: dict[int, dict[str, np.ndarray]]):
        self.arrays = arrays

    def get(self, k: int, name: str) -> np.ndarray:
        return self.arrays[k][name]


class _Composite:
    """ContextCounts whose levels <= short are `capped`'s, the others `full`'s."""

    def __init__(self, capped: ContextCounts, full: ContextCounts, short: int):
        self.capped, self.full, self.short = capped, full, short

    def get(self, k: int, name: str) -> np.ndarray:
        return (self.capped if k <= self.short else self.full).get(k, name)


class _PerLevel:
    """ContextCounts whose level k is per_level[k]'s if listed, else `full`'s."""

    def __init__(self, full: ContextCounts, per_level: dict):
        self.full, self.per_level = full, per_level

    def get(self, k: int, name: str) -> np.ndarray:
        return self.per_level.get(k, self.full).get(k, name)


def _capped_counts(k: int, shards: int) -> "CappedValCounts":
    """The CappedValCounts holding level k over the first `shards` shards (the short levels' or the long levels')."""
    return CappedValCounts(shards=shards) if k in SHORT else CappedValCounts(shards=shards, levels=LONG)


def _caps(short: int, shards: int, level_shards: tuple[tuple[int, int], ...]) -> dict[int, int]:
    """Per capped level its shard count: the levels <= short over `shards` unless level_shards lists them, and any
    other level level_shards lists over its own; levels at TRAIN_SHARDS or more are exact (left out)."""
    caps = {k: shards for k in _short_levels(short)} | dict(level_shards)
    return {k: c for k, c in sorted(caps.items()) if c < TRAIN_SHARDS}


def _hybrid(capped, full, levels: tuple[int, ...], f: float, below: int) -> "_Arrays":
    """ContextCounts arrays at `levels`: per context the capped counts with totals times f, or where the capped total
    is below `below`, the full counts (top HINT_ENTRIES next tokens)."""
    arrays = {}
    for k in levels:
        ct, ft = np.asarray(capped.get(k, "total")), np.asarray(full.get(k, "total"))
        rare = ct < below
        total = np.where(rare, ft, np.round(ct * f).astype(np.uint64))
        co = np.asarray(capped.get(k, "offsets")).astype(np.int64)
        fo = np.asarray(full.get(k, "offsets")).astype(np.int64)
        lo = np.where(rare, fo[:-1], co[:-1])
        n = np.where(rare, np.minimum(np.diff(fo), HINT_ENTRIES), np.diff(co))
        offsets = np.zeros(len(n) + 1, np.uint64)
        offsets[1:] = np.cumsum(n)
        index = np.repeat(lo, n) + np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n)
        source = np.repeat(rare, n)
        nxt, cnt = np.empty(len(index), np.uint16), np.empty(len(index), np.uint32)
        for name, out in (("next", nxt), ("next_counts", cnt)):
            out[source] = np.asarray(full.get(k, name))[index[source]] if source.any() else 0
            out[~source] = np.asarray(capped.get(k, name))[index[~source]]
        arrays[k] = {"context": np.asarray(capped.get(k, "context")), "total": total, "offsets": offsets, "next": nxt,
                     "next_counts": cnt, "distinct": np.asarray(capped.get(k, "distinct"))}
    return _Arrays(arrays)


def _truncate(total, distinct, offsets, nxt, ncount, top: int):
    """The ContextCounts arrays with each context's `top` most frequent next tokens only."""
    lens = np.minimum(np.diff(offsets.astype(np.int64)), top)
    new = np.zeros(len(offsets), np.uint64)
    new[1:] = np.cumsum(lens)
    first = np.repeat(offsets[:-1].astype(np.int64), lens)
    index = first + np.arange(int(new[-1])) - np.repeat(new[:-1].astype(np.int64), lens)
    return total, distinct, new, np.asarray(nxt[index]), np.asarray(ncount[index])


class CappedValCounts(furu.Spec[ContextCounts]):
    """Per level k of `levels` (ascending), the validation contexts' exact counts over the first `shards` training
    shards (exact_match.kgram_counts, as val_hints.ValContextCounts): in k{k}/ the ContextCounts arrays with each
    context's top HINT_ENTRIES next tokens only (for the hint rows), and the count rows' columns tot.npy (uint32),
    dist.npy (uint16) and cnt.npy (uint32) per validation position (old_records.val_count_features: the target's count
    if among its context's TOP most frequent)."""

    shards: int
    levels: tuple[int, ...] = SHORT

    def metadata(self) -> furu.Metadata:
        return furu.Metadata(requires=furu.Requires(cpus=furu.at_least(48), memory=furu.GiB(500), gpus=0))

    def create(self) -> ContextCounts:
        from exact_match import kgram_counts
        from retrieval_blog_post.old_records import val_count_features

        train = FinewebShards(shards=self.shards).create()
        (val,) = FinewebShards(shards="val").create()
        files = [str(s.path) for s in train]
        tokens = np.ascontiguousarray(val.tokens[: VAL_TOKENS + 1])
        targets = np.asarray(val.tokens[1 : VAL_TOKENS + 1])
        hits = None
        for k in self.levels:
            start = time.time()
            *arrays, hits = kgram_counts(files, tokens, k, TOP, 128 << 30, None, hits)
            context, total, distinct, offsets, nxt, ncount = arrays
            tot, dist, cnt = val_count_features(
                _Arrays({k: dict(zip(ARRAYS, arrays))}), targets, np.arange(VAL_TOKENS), (k,)
            )
            out = self.directory.data / f"k{k}"
            out.mkdir(exist_ok=True)
            for name, array in zip(ARRAYS, (context, *_truncate(total, distinct, offsets, nxt, ncount, HINT_ENTRIES))):
                np.save(out / f"{name}.npy", array)
            for name, array in (("tot", tot[:, 0]), ("dist", dist[:, 0]), ("cnt", cnt[:, 0])):
                np.save(out / f"{name}.npy", array)
            self.logger.info(
                f"k={k} over {self.shards} shards: {(tot > 0).mean():.4f} matched, {(cnt > 0).mean():.4f} with the "
                f"target, {time.time() - start:.0f} s"
            )
            del arrays, context, total, distinct, offsets, nxt, ncount
        return ContextCounts(self.directory.data, self.levels)


def _short_levels(short: int) -> tuple[int, ...]:
    return tuple(k for k in SHORT if k <= short)


class ApproxValCountRows(furu.Spec[CountRowFiles]):
    """old_records.ValCountRows (all of ALL_LEVELS) with each level <= `short` counted over the first `shards`
    training shards only (CappedValCounts), and the levels of `drop` left out (tot, dist and cnt all 0: as if never
    matched, so the n-gram skips them). short = 0 caps nothing. With `scale`, the capped levels' counts are
    extrapolated to all TRAIN_SHARDS shards (iid): tot and cnt times f = TRAIN_SHARDS / shards, dist times
    f ** DIST_GROWTH[k] (at most the scaled tot and 65535); contexts unseen in the capped shards stay unmatched. With
    `exact_below` T > 0, the contexts that occurred fewer than T times in the capped shards (unseen ones too) get their
    exact counts over all the shards instead (the hybrid: a full-corpus pass counts only these rare contexts).
    `level_shards` ((k, K_k), ...) caps level k over its own first K_k shards instead (a per-level schedule; any
    level of ALL_LEVELS, the long ones too; K_k >= TRAIN_SHARDS: exact). grid() builds the canonical specs."""

    short: int
    shards: int
    drop: tuple[int, ...] = ()
    scale: bool = False
    exact_below: int = 0
    level_shards: tuple[tuple[int, int], ...] = ()

    migrations = (furu.Added("scale", default=False), furu.Added("exact_below", default=0),
                  furu.Added("level_shards", default=()))

    def metadata(self) -> furu.Metadata:
        return furu.Metadata(requires=furu.Requires(cpus=furu.at_least(4), memory=furu.GiB(32), gpus=0))

    def _shards(self, k: int) -> int:
        """The capped shards of level k: level_shards' if it lists k, else shards."""
        return dict(self.level_shards).get(k, self.shards)

    @furu.dependency
    def inputs(self) -> list[furu.Spec]:
        from retrieval_blog_post.old_records import ValCountRows

        caps = _caps(self.short, self.shards, self.level_shards)
        return [ValCountRows(), *sorted({_capped_counts(k, c) for k, c in caps.items()}, key=repr)]

    def create(self):
        from retrieval_blog_post.old_records import COUNT_NAMES, ValCountRows

        full = ValCountRows().create()
        arrays = {n: np.load(full.directory / f"{n}.npy") for n in COUNT_NAMES}
        caps = _caps(self.short, self.shards, self.level_shards)
        if self.scale or self.exact_below:
            assert set(caps) <= set(SHORT), "scale / exact_below cap the short levels only"
        if caps:
            for k, c in caps.items():
                capped = _capped_counts(k, c).create()
                j = ALL_LEVELS.index(k)
                exact = {n: arrays[n][:, j].copy() for n in COUNT_NAMES}
                for n in COUNT_NAMES:
                    arrays[n][:, j] = capped.get(k, n)
                rare = arrays["tot"][:, j] < self.exact_below  # contexts rare in the capped shards: exact counts
                if self.scale:  # the corpus is iid: the capped counts extrapolated to all the shards
                    f = TRAIN_SHARDS / self._shards(k)
                    tot, dist, cnt = (arrays[n][:, j].astype(np.float64) for n in COUNT_NAMES)
                    arrays["tot"][:, j] = np.round(tot * f).astype(np.uint32)
                    arrays["cnt"][:, j] = np.round(cnt * f).astype(np.uint32)
                    grown = np.minimum(np.round(dist * f ** DIST_GROWTH[k]), np.minimum(tot * f, 65535))
                    arrays["dist"][:, j] = grown.astype(np.uint16)
                for n in COUNT_NAMES:
                    arrays[n][rare, j] = exact[n][rare]
        for k in self.drop:
            for n in COUNT_NAMES:
                arrays[n][:, ALL_LEVELS.index(k)] = 0
        for n in COUNT_NAMES:
            np.save(self.directory.data / f"{n}.npy", arrays[n])
        tot = arrays["tot"]
        self.logger.info("matched " + " ".join(f"{k}:{(tot[:, j] > 0).mean():.3f}" for j, k in enumerate(ALL_LEVELS)))
        return CountRowFiles(self.directory.data, ALL_LEVELS)


class ApproxValHintRows(furu.Spec[HintRowFiles]):
    """old_records.ValHintRows of the variant `levels` (one slot, LEVELS' buckets) with each level <= `short`
    counted over the first `shards` training shards only (CappedValCounts): a position's hint is at the deepest level
    whose context occurred in those counts, with that level's top ENTRIES next tokens. `scale` and `exact_below` as
    ApproxValCountRows's (the totals scaled for the cells' count bins; the rare contexts' totals and top next tokens
    exact over all the shards). `level_shards` as ApproxValCountRows's (per-level caps, long levels too)."""

    levels: tuple[int, ...] = LEVELS
    short: int = 4
    shards: int = 20
    scale: bool = False
    exact_below: int = 0
    level_shards: tuple[tuple[int, int], ...] = ()

    migrations = (furu.Added("scale", default=False), furu.Added("exact_below", default=0),
                  furu.Added("level_shards", default=()))

    def metadata(self) -> furu.Metadata:
        return furu.Metadata(requires=furu.Requires(cpus=furu.at_least(4), memory=furu.GiB(64), gpus=0))

    @furu.dependency
    def inputs(self) -> list[furu.Spec]:
        from retrieval_blog_post.val_hints import ValContextCounts

        if not self.level_shards:
            return [ValContextCounts(), CappedValCounts(shards=self.shards)]
        caps = {k: c for k, c in _caps(self.short, self.shards, self.level_shards).items() if k in self.levels}
        return [ValContextCounts(), *sorted({_capped_counts(k, c) for k, c in caps.items()}, key=repr)]

    def create(self):
        from retrieval_blog_post.old_records import ENTRIES, _val_rows, _weights
        from retrieval_blog_post.val_hints import ValContextCounts

        check_variant(self.levels)
        full = ValContextCounts().create()
        if self.level_shards:
            assert not (self.scale or self.exact_below), "level_shards: plain caps only"
            caps = {k: c for k, c in _caps(self.short, self.shards, self.level_shards).items() if k in self.levels}
            counts = _PerLevel(full, {k: _capped_counts(k, c).create() for k, c in caps.items()})
        else:
            capped = CappedValCounts(shards=self.shards).create()
            if self.scale or self.exact_below:
                capped = _hybrid(capped, full, _short_levels(self.short),
                                 TRAIN_SHARDS / self.shards if self.scale else 1.0, self.exact_below)
            counts = _Composite(capped, full, self.short)
        level, context, cell = deepest(counts, self.levels, VAL_TOKENS, 1, LEVELS)
        out = self.directory.data
        for name, array in (("level", level), ("context", context), ("cell", cell)):
            np.save(out / f"{name}.npy", array)
        (out / "buckets.json").write_text(json.dumps(LEVELS))
        hints = Hints(out, self.levels, counts)  # type: ignore[arg-type]
        cell, tokens, cnts = _val_rows(hints, VAL_TOKENS)
        assert tokens.shape[1] == ENTRIES
        for name in ("level", "context"):
            (out / f"{name}.npy").unlink()
        np.save(out / "cell.npy", cell)
        np.save(out / "tokens.npy", tokens)
        np.save(out / "weights.npy", _weights(cnts))
        self.logger.info(
            "hint at " + ", ".join(f"{k}: {(level == LEVELS.index(k)).mean():.4f}" for k in self.levels)
            + f"; {(cell > 0).mean():.4f} hinted"
        )
        return HintRowFiles(out, self.levels)


@dataclass(frozen=True)
class Proxy:
    """cpu_val_proxy.py's results (result.json in `directory`)."""

    directory: Path
    result: dict


class ProxyLoss(furu.Spec[Proxy]):
    """cpu_val_proxy.py's losses of ApproxValCountRows(short, shards, ...) (short = 0 and no drop: of ValCountRows):
    g1 and fit_B, with `mix` mix_B, and with `train` > 0 NgramMix trained `train` steps on TrainCountRows(shards=4)."""

    short: int
    shards: int
    drop: tuple[int, ...] = ()
    train: int = 0
    mix: bool = False
    scale: bool = False
    exact_below: int = 0
    level_shards: tuple[tuple[int, int], ...] = ()

    migrations = (furu.Added("mix", default=False), furu.Added("scale", default=False),
                  furu.Added("exact_below", default=0), furu.Added("level_shards", default=()))

    def metadata(self) -> furu.Metadata:
        return furu.Metadata(requires=furu.Requires(cpus=furu.at_least(16), memory=furu.GiB(96), gpus=0))

    def _rows(self) -> furu.Spec[CountRowFiles]:
        from retrieval_blog_post.old_records import ValCountRows

        if self.short == 0 and not self.drop:
            return ValCountRows()
        return ApproxValCountRows(short=self.short, shards=self.shards, drop=self.drop, scale=self.scale,
                                  exact_below=self.exact_below, level_shards=self.level_shards)

    @furu.dependency
    def inputs(self) -> list[furu.Spec]:
        from retrieval_blog_post.old_records import TrainCountRows, ValSubsetLabels

        return [self._rows(), ValSubsetLabels(), *([TrainCountRows(shards=4)] if self.train else [])]

    def create(self) -> Proxy:
        from retrieval_blog_post.old_records import TrainCountRows, ValSubsetLabels

        rows = self._rows().create()
        labels = ValSubsetLabels().create()
        out = self.directory.data / "result.json"
        cmd = [str(TORCH_VENV / "bin/python"), str(PROXY), str(rows.directory), str(labels.directory / "labels.npy"),
               str(out), "--data", str(DATA), "--threads", str(min(32, os.cpu_count() or 16))]
        if self.train:
            cmd += ["--train", str(TrainCountRows(shards=4).create().directory), "--steps", str(self.train)]
        if self.mix:
            cmd += ["--lm", str(LM_LOSSES)]
        start = time.time()
        subprocess.run(cmd, check=True, env={**os.environ, "OMP_NUM_THREADS": str(min(32, os.cpu_count() or 16))})
        result = json.loads(out.read_text())
        self.logger.info(f"proxy {self.short} {self.shards} {self.drop}: {result.get('g1')} {result.get('fit_B')} "
                         f"{result.get('train')} ({time.time() - start:.0f} s)")
        return Proxy(self.directory.data, result)


def _hint_rows(counts, levels: tuple[int, ...], scratch: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ValHintRows' cell (int16), tokens (uint16 [n, ENTRIES]) and weights (uint16 [n, ENTRIES]) of the variant
    `levels` (one slot, LEVELS' buckets) from ContextCounts-like `counts` (get(k, name)) with each context's top
    ENTRIES next tokens at least."""
    from retrieval_blog_post.old_records import _val_rows, _weights

    level, context, cell = deepest(counts, levels, VAL_TOKENS, 1, LEVELS)
    scratch.mkdir(parents=True, exist_ok=True)
    for name, array in (("level", level), ("context", context), ("cell", cell)):
        np.save(scratch / f"{name}.npy", array)
    (scratch / "buckets.json").write_text(json.dumps(LEVELS))
    cell, tokens, cnts = _val_rows(Hints(scratch, levels, counts), VAL_TOKENS)  # type: ignore[arg-type]
    shutil.rmtree(scratch)
    return cell, tokens, _weights(cnts)


def _hogs(n: int) -> list[subprocess.Popen]:
    """`n` processes copying 1 GB arrays in a loop (memory-bandwidth load, standing in for the training-row worker
    and the training processes beside the scan), started once they have allocated."""
    code = ("import numpy as np\na = np.ones(1 << 27); b = np.empty_like(a); b[:] = a\nprint('ready', flush=True)\n"
            "while True:\n    np.copyto(b, a)\n")
    hogs = [subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True) for _ in range(n)]
    for hog in hogs:
        assert hog.stdout is not None and hog.stdout.readline().strip() == "ready"
    return hogs


@dataclass(frozen=True)
class Timings:
    """CpuValTiming's results: per run its configuration, val_scan's timings and counters, and its checks."""

    directory: Path
    runs: tuple[dict, ...]


# CpuValTiming's runs: (short, short shards, threads, filter bits, scale, rare_below, anchor w, second filter bits,
# hogs); the first of each (short, shards, scale, rare_below) is checked. The earlier versions' runs are in their
# specs' metadata. Version 6's: the long scan's filter tested at anchors only (every w-gram of the shallowest long
# level's contexts in the filter, tested at shard offsets that are multiples of k0 - w + 1), and an anchor hit's
# candidates tested against a second filter of the k0-contexts before probing (0 bits: none).
TIMING_RUNS_V6 = (
    (6, 20, 96, 28, 0, 0, 6, 28),
    (6, 10, 96, 28, 0, 0, 6, 28),
    (6, 20, 96, 28, 0, 0, 6, 0),
    (5, 20, 96, 28, 0, 0, 5, 28),
    (6, 10, 64, 28, 0, 0, 6, 28),
    (6, 10, 48, 28, 0, 0, 6, 28),
    (6, 5, 96, 28, 0, 0, 6, 28),
    (4, 20, 96, 29, 0, 0, 4, 28),
    (5, 10, 96, 28, 0, 0, 5, 28),
    (6, 10, 96, 28, 0, 0, 6, 28),
)
# Version 7: version 6's runs, then memory-bandwidth hogs beside the scan (_hogs, on the same 96 CPUs)
TIMING_RUNS = (
    *TIMING_RUNS_V6,
    (6, 10, 80, 28, 0, 0, 6, 28, 0),
    (6, 10, 80, 28, 0, 0, 6, 28, 16),
    (6, 10, 64, 28, 0, 0, 6, 28, 32),
    (5, 20, 80, 28, 0, 0, 5, 28, 16),
)


class CpuValTiming(furu.Spec[Timings]):
    """exact_match.val_scan (short levels over the first shards, the rest over all of them) timed on a 96-CPU worker
    with the shards staged in /dev/shm, once per run of `runs`. The checked runs' count rows must equal
    ApproxValCountRows' and their L7 hint rows (also timed) ApproxValHintRows'."""

    runs: tuple[tuple[int, ...], ...] = TIMING_RUNS
    version: int = 7

    def metadata(self) -> furu.Metadata:
        return furu.Metadata(requires=furu.Requires(cpus=furu.at_least(96), memory=furu.GiB(1000), gpus=0))

    @furu.dependency
    def inputs(self) -> list[furu.Spec]:
        return [
            FinewebShards(shards=TRAIN_SHARDS),
            FinewebShards(shards="val"),
            *(spec for c in sorted({self._config(r) for r in self.runs}) for spec in self._references(*c)),
        ]

    @staticmethod
    def _config(run: tuple[int, ...]) -> tuple[int, int, int, int]:
        """A run's (short, shards, scale, rare_below)."""
        short, shards, _, _, *rest = run
        scale, below = rest[:2] if rest else (0, 0)
        return short, shards, scale, below

    @staticmethod
    def _references(short: int, shards: int, scale: int, below: int) -> tuple[furu.Spec, furu.Spec]:
        extra = {"scale": True, "exact_below": below} if scale or below else {}
        return (ApproxValCountRows(short=short, shards=shards, **extra),
                ApproxValHintRows(levels=LEVELS, short=short, shards=shards, **extra))

    def create(self) -> Timings:
        from exact_match import val_scan
        from retrieval_blog_post.old_records import COUNT_NAMES

        out = self.directory.data
        train = FinewebShards(shards=TRAIN_SHARDS).create()
        (val,) = FinewebShards(shards="val").create()
        tokens = np.ascontiguousarray(val.tokens[: VAL_TOKENS + 1])
        need = sum(s.path.stat().st_size for s in train)
        cpu = subprocess.run(["lscpu"], capture_output=True, text=True, check=False).stdout
        notes: dict = {"cpus": os.cpu_count(), "affinity": len(os.sched_getaffinity(0)),
                       "lscpu": [l for l in cpu.splitlines() if l.split(":")[0] in
                                 ("Model name", "Socket(s)", "Core(s) per socket", "L3 cache", "NUMA node(s)")]}
        if shutil.disk_usage("/dev/shm").free < 1.2 * need:
            raise RuntimeError(f"/dev/shm lacks {need / 1e9:.0f} GB")
        results = []
        with tempfile.TemporaryDirectory(dir="/dev/shm", prefix="cpuval-") as local:
            staged = [Path(local) / s.path.name for s in train]
            start = time.time()
            with ThreadPoolExecutor(32) as pool:
                list(pool.map(lambda c: shutil.copyfile(*c), zip((s.path for s in train), staged)))
            notes["stage_copy_s"] = time.time() - start
            self.logger.info(f"staged {len(staged)} shards ({need / 1e9:.0f} GB) in {local}: {notes}")
            checked = set()
            for spec_run in self.runs:
                short, shards, threads, bits = spec_run[:4]
                config = self._config(spec_run)
                _, _, scale, below = config
                f = TRAIN_SHARDS / shards if scale else 0.0
                growth = [DIST_GROWTH[k] for k in ALL_LEVELS if k <= short] if scale else []
                anchor = spec_run[6] if len(spec_run) > 6 else 0
                bits2 = spec_run[7] if len(spec_run) > 7 else 0
                hogs = _hogs(spec_run[8] if len(spec_run) > 8 else 0)
                start = time.time()
                try:
                    tot, dist, cnt, per, times = val_scan([str(p) for p in staged], tokens, list(ALL_LEVELS), short,
                                                          shards, threads, TOP, HINT_ENTRIES, bits, f, growth, below,
                                                          16, anchor, bits2)
                finally:
                    for hog in hogs:
                        hog.kill()
                wall = time.time() - start
                run = {"short": short, "shards": shards, "threads": threads, "filter_bits": bits, "scale": scale,
                       "rare_below": below, "anchor": anchor, "filter2_bits": bits2, "hogs": len(hogs), "wall_s": wall,
                       **dict(times)}
                start = time.time()
                if self.version >= 2:
                    from exact_match import val_hint_rows

                    sel = [per[ALL_LEVELS.index(k)] for k in LEVELS]
                    cell, toks, weights = val_hint_rows(list(LEVELS), [(a[0], a[1], a[3], a[4], a[5]) for a in sel],
                                                        list(LEVELS), HINT_ENTRIES, threads)
                    toks, weights = toks.reshape(VAL_TOKENS, -1), weights.reshape(VAL_TOKENS, -1)
                else:  # version 1: in Python
                    counts = _Arrays({k: dict(zip(ARRAYS, arrays)) for k, arrays in zip(ALL_LEVELS, per)})
                    cell, toks, weights = _hint_rows(counts, LEVELS, out / "scratch")
                run["hint_rows_s"] = time.time() - start
                if config not in checked:
                    checked.add(config)
                    ref, href = (spec.create() for spec in self._references(*config))
                    mine = dict(zip(COUNT_NAMES, (a.reshape(VAL_TOKENS, -1) for a in (tot, dist, cnt))))
                    for n in COUNT_NAMES:
                        r = np.load(ref.directory / f"{n}.npy")
                        run[f"exact_{n}"] = [bool(np.array_equal(m, r[:, j])) for j, m in enumerate(mine[n].T)]
                    for n, a in (("cell", cell), ("tokens", toks), ("weights", weights)):
                        run[f"exact_hint_{n}"] = bool(np.array_equal(a, np.load(href.directory / f"{n}.npy")))
                self.logger.info(f"run {json.dumps(run)}")
                results.append(run)
                del tot, dist, cnt, per
        (out / "timings.json").write_text(json.dumps({"notes": notes, "runs": results}, indent=1))
        return Timings(out, tuple(results))


def create[T](specs, job_name: str = "cpuval") -> list[T]:
    """furu.create on CPU workers only (config.cpu): big workers for the corpus passes and timings, small ones for the
    rest."""
    return furu.create(
        specs,
        on=(
            SlurmWorkerBackend(
                max_workers=3,
                resources=cpu(n_cpus=96, mem_gib=1000, time_limit="4:00:00"),
                job_name=f"{job_name}-1000g",
                reserve_for=ResourceFloor(memory_gib=400),
            ),
            SlurmWorkerBackend(
                max_workers=6,
                resources=cpu(n_cpus=32, mem_gib=256, time_limit="4:00:00"),
                job_name=f"{job_name}-256g",
            ),
        ),
    )


def hybrid_rows() -> list[furu.Spec]:
    """The hybrid rows (scaled capped counts, rare contexts exact) of the model-run candidates."""
    return [
        spec
        for s, k, t in ((5, 20, 16), (6, 20, 16), (4, 20, 16), (5, 50, 16), (5, 20, 4), (6, 20, 256), (6, 10, 256))
        for spec in (ApproxValHintRows(levels=LEVELS, short=s, shards=k, scale=True, exact_below=t),
                     ApproxValCountRows(short=s, shards=k, scale=True, exact_below=t))
    ]


def model_rows() -> list[furu.Spec]:
    """The capped rows of the model-run candidates (OldRecordRun.approx_val), precomputed here."""
    return [spec for s, k in ((3, 20), (4, 20), (5, 20), (6, 20), (6, 10))
            for spec in (ApproxValHintRows(levels=LEVELS, short=s, shards=k), ApproxValCountRows(short=s, shards=k))]


def grid(level_shards: dict[int, int], drop: tuple[int, ...] = ()) -> tuple[ApproxValCountRows, ApproxValHintRows]:
    """The canonical row specs (count rows, L7 hint rows) of the configuration `level_shards` {k: K_k} (every short
    level 2-6 listed, level 2 capped; K >= TRAIN_SHARDS: exact; a long level not listed: exact), with the levels of
    `drop` left out. Canonical: short = 6, shards = K_2, level_shards = the other non-dropped levels whose cap differs
    from K_2 (long ones only if capped), ascending; so a uniform cap is the plain (short=6, shards=K) whose rows are
    already cached."""
    assert all(k in level_shards for k in SHORT) and level_shards[2] < TRAIN_SHARDS, level_shards
    caps = {k: min(c, TRAIN_SHARDS) for k, c in level_shards.items()}
    base = caps[2]
    rest = tuple(sorted((k, c) for k, c in caps.items()
                        if k != 2 and k not in drop and c != base and not (k in LONG and c >= TRAIN_SHARDS)))
    hint_levels = tuple(k for k in LEVELS if k not in drop)
    return (ApproxValCountRows(short=6, shards=base, drop=tuple(sorted(drop)), level_shards=rest),
            ApproxValHintRows(levels=hint_levels, short=6, shards=base, level_shards=rest))


def config_name(level_shards: dict[int, int], drop: tuple[int, ...] = ()) -> str:
    """A short name of a configuration: e.g. "2-3:20 4-6:50 8-20:1028" (+ " -6" for dropped levels)."""
    caps = {k: min(level_shards.get(k, TRAIN_SHARDS), TRAIN_SHARDS) for k in ALL_LEVELS}
    parts, run = [], []
    for k in ALL_LEVELS:
        if run and caps[k] != caps[run[0]]:
            parts.append(run)
            run = []
        run.append(k)
    parts.append(run)
    name = " ".join(f"{r[0]}{'-' + str(r[-1]) if len(r) > 1 else ''}:{caps[r[0]]}" for r in parts)
    return name + "".join(f" -{k}" for k in drop)


# The quality grid, for llm.c's validation rows on CPUs: levels 2-3 over K_a, 4-6 over K_b, the long levels over all
# the shards; then the long levels capped, and levels left out
GRID_K = (10, 20, 50, 100, 200, 400)


def quality_configs() -> list[tuple[dict[int, int], tuple[int, ...]]]:
    """The quality grid's configurations: (level_shards, drop) for grid()."""
    configs = [({2: a, 3: a, 4: b, 5: b, 6: b}, ()) for a in GRID_K for b in GRID_K]
    for a, b in ((10, 10), (20, 20), (20, 50), (50, 100)):
        for c in (100, 300):
            configs.append(({2: a, 3: a, 4: b, 5: b, 6: b, **{k: c for k in LONG}}, ()))
    for a, b in ((20, 20), (20, 50)):
        for drop in ((8, 12, 16), (12, 16), (6,), (6, 8, 12, 16)):
            configs.append(({2: a, 3: a, 4: b, 5: b, 6: b}, drop))
    return configs


def proxy_of(rows: ApproxValCountRows) -> "ProxyLoss":
    """The LM-mixture proxy (mix_B) of count rows."""
    return ProxyLoss(short=rows.short, shards=rows.shards, drop=rows.drop, mix=True, level_shards=rows.level_shards)


def quality_specs() -> list[furu.Spec]:
    specs: list[furu.Spec] = []
    for level_shards, drop in quality_configs():
        rows, hints = grid(level_shards, drop)
        specs += [rows, hints, proxy_of(rows)]
    return list(dict.fromkeys(specs))  # dropping long levels leaves the hint rows as they are


def hint_only(level_shards: dict[int, int]) -> ApproxValHintRows:
    """L7 hint rows with only the levels of `level_shards` capped (the others exact): one level's hint-row effect."""
    return ApproxValHintRows(levels=LEVELS, short=0, shards=0, level_shards=tuple(sorted(level_shards.items())))


def file_specs(path: str) -> list[furu.Spec]:
    """The row specs of a JSON list of {"caps": {k: K}, "drop": [...], "hint_only": bool, "hints": bool}: grid()'s
    pair (with hints, its hint rows only), or with hint_only the hint rows of hint_only()."""
    specs: list[furu.Spec] = []
    for entry in json.loads(Path(path).read_text()):
        caps = {int(k): int(c) for k, c in entry["caps"].items()}
        if entry.get("hint_only"):
            specs.append(hint_only(caps))
        else:
            pair = grid(caps, tuple(entry.get("drop", ())))
            specs += pair[1:] if entry.get("hints") else pair
    return list(dict.fromkeys(specs))


def accuracy_specs() -> list[furu.Spec]:
    specs: list[furu.Spec] = [ProxyLoss(short=0, shards=0, train=250)]
    for k in CAPS:
        for s in (3, 4, 5, 6):
            specs.append(ProxyLoss(short=s, shards=k))
    specs += [ProxyLoss(short=0, shards=0, drop=(8, 12, 16)), ProxyLoss(short=0, shards=0, drop=(8, 12))]
    # the LM mixture (what the rows add to an LM): the exact rows and the grid
    specs.append(ProxyLoss(short=0, shards=0, mix=True))
    for k in CAPS:
        for s in (3, 4, 5, 6):
            specs.append(ProxyLoss(short=s, shards=k, mix=True))
    # the capped counts extrapolated to the whole corpus (scale)
    for k in CAPS[:4]:
        for s in (3, 4, 5, 6):
            specs.append(ProxyLoss(short=s, shards=k, mix=True, scale=True))
    # the hybrid: rare contexts (capped count < T) exact over all the shards, the others scaled
    for k in (10, 20, 50):
        for s in (4, 5, 6):
            for t in (4, 16):
                specs.append(ProxyLoss(short=s, shards=k, mix=True, scale=True, exact_below=t))
    for k in (20, 50):
        for s in (3, 4, 5, 6):
            for t in (64, 256):
                specs.append(ProxyLoss(short=s, shards=k, mix=True, scale=True, exact_below=t))
    # NgramMix trained on the training count rows (as a run learns its gammas), evaluated on the approximate rows
    for k in (10, 20, 50):
        for s in (3, 4, 5, 6):
            specs.append(ProxyLoss(short=s, shards=k, train=250))
    specs += [ProxyLoss(short=s, shards=20, train=250, scale=True, exact_below=16) for s in (4, 5, 6)]
    # per-level schedules (levels 2-3 over `shards`, the deeper short levels over more): CPU-feasible, a filter test
    # per position of the extra shards per level
    for sched in (((4, 50), (5, 100), (6, 200)), ((4, 20), (5, 50), (6, 200)), ((5, 100), (6, 200)), ((6, 200),),
                  ((4, 100), (5, 200), (6, 200))):
        for k in (10, 20):
            specs.append(ProxyLoss(short=6, shards=k, mix=True, level_shards=sched))
    return specs


if __name__ == "__main__":

    def _main() -> None:
        # Imported, not this file's own definitions: run as a file, they would be __main__'s, other specs.
        from retrieval_blog_post.cpu_val import (
            CpuValTiming,
            accuracy_specs,
            create,
            hybrid_rows,
            model_rows,
        )

        if sys.argv[1:] == ["quality"]:
            from retrieval_blog_post.cpu_val import quality_specs

            specs = quality_specs()
            results = create(specs, job_name="cpuval-quality")
        elif sys.argv[1:2] == ["rows"]:  # rows <file.json>: file_specs (the quality grid's candidates)
            from retrieval_blog_post.cpu_val import file_specs

            specs = file_specs(sys.argv[2])
            results = create(specs, job_name="cpuval-rows")
        else:
            specs = [*accuracy_specs(), CpuValTiming(), *hybrid_rows(), *model_rows()]
            results = create(specs)
        for spec, result in zip(specs, results):
            print(spec, json.dumps(getattr(result, "result", None))[:400], flush=True)

    _main()
