"""The n-gram-alone proxy of validation count rows (cpu_val.py's ProxyLoss runs it with the torch venv): the
interpolated Witten-Bell n-gram (hint_runtime.NgramMix's q, no LM) over the rows' levels, its loss on all validation
positions and on ValSubsetLabels' subsets (noexact: label >= 1, nonear: label 2), in up to four ways:
  g1:    gamma 1 everywhere, over the unigram of training shard 1 (+0.5);
  fit:   log gamma per level and total-count bin fitted by L-BFGS on 2M positions of half A (even 64K chunks),
         reported on half B: the best a learned gamma can do with these rows;
  mix:   (--lm: per-token losses of an LM without hints, e.g. token_losses.NoHintsRecordTokenLosses) the mixture
         lam * p_LM + (1 - lam) * q, gamma and lam fitted as in fit: what the rows add to an LM (lm_B: the LM alone);
  train: (--train: training count rows, NNNN/{tot,dist,cnt}.npy) NgramMix's gammas and unigram trained as in a run
         (Adam, record #1's hint lr and cosine schedule), evaluated on all validation positions.

    python cpu_val_proxy.py ROWS_DIR LABELS.npy OUT.json --data config.DATA [--train DIR --steps N] [--lm LOSSES.npy]
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

T_VAL = 10_485_760
EDGES = (2, 3, 5, 9, 17, 33, 100, 1000, 10000)  # NgramMix.COUNT_EDGES
CHUNK = 65_536
STEP_TOK = 524_288
PER_SHARD = 190


def tokens(path):
    return np.memmap(path, dtype=np.uint16, mode="r", offset=1024)


def bins(tot):
    return sum((tot > e).long() for e in EDGES)


def wb_logp(p0, tot, dist, cnt, g):
    """log q(target) with per-position-level gammas g [T, L] (float64 math)."""
    p = p0
    for j in range(tot.shape[1]):
        m = tot[:, j] > 0
        gd = g[:, j] * dist[:, j]
        den = torch.where(m, tot[:, j] + gd, torch.ones_like(gd))
        p = torch.where(m, (cnt[:, j] + gd * p) / den, p)
    return torch.log(p.clamp_min(1e-30))


def subsets(lp, labels, sel=None):
    out = {}
    for name, m in (("all", np.ones_like(labels, bool)), ("noexact", labels >= 1), ("nonear", labels == 2)):
        if sel is not None:
            m = m & sel
        out[name] = float(-lp[torch.from_numpy(m)].mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows")
    ap.add_argument("labels")
    ap.add_argument("out")
    ap.add_argument("--data", required=True)
    ap.add_argument("--train", default="")
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--lr", type=float, default=0.008)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--lm", default="")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    t0 = time.time()
    rows = Path(args.rows)
    tot = torch.from_numpy(np.load(rows / "tot.npy").astype(np.float64))
    dist = torch.from_numpy(np.load(rows / "dist.npy").astype(np.float64))
    cnt = torch.from_numpy(np.load(rows / "cnt.npy").astype(np.float64))
    labels = np.load(args.labels)
    target = np.asarray(tokens(Path(args.data) / "fineweb_val_000000.bin")[1 : T_VAL + 1]).astype(np.int64)
    uni = np.bincount(tokens(Path(args.data) / "fineweb_train_000001.bin"), minlength=50257)[:50257] + 0.5
    uni /= uni.sum()
    p0 = torch.from_numpy(uni[target])
    L = tot.shape[1]
    res = {"rows": str(rows), "levels": L}
    res["matched"] = [float((tot[:, j] > 0).double().mean()) for j in range(L)]
    res["target_seen"] = [float((cnt[:, j] > 0).double().mean()) for j in range(L)]
    # g1
    lp = wb_logp(p0, tot, dist, cnt, torch.ones_like(tot))
    res["g1"] = subsets(lp, labels)
    print("g1", res["g1"], f"{time.time() - t0:.0f} s", flush=True)
    # fit: log gamma per (level, bin), on 2M positions of half A, reported on half B
    A = (np.arange(T_VAL) // CHUNK) % 2 == 0
    rng = np.random.default_rng(0)
    fit_at = torch.from_numpy(rng.choice(np.flatnonzero(A), 2_000_000, replace=False))
    b = bins(tot)
    nb = len(EDGES) + 1
    idx = torch.arange(L)[None, :] * nb + b
    g = torch.zeros(L * nb, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([g], lr=1, max_iter=100, history_size=20, line_search_fn="strong_wolfe")
    sub = (p0[fit_at], tot[fit_at], dist[fit_at], cnt[fit_at], idx[fit_at])

    def closure():
        opt.zero_grad()
        l = -wb_logp(sub[0], sub[1], sub[2], sub[3], torch.exp(g.clamp(-20, 20))[sub[4]]).mean()
        l.backward()
        return l

    opt.step(closure)
    with torch.no_grad():
        lp = wb_logp(p0, tot, dist, cnt, torch.exp(g.clamp(-20, 20))[idx])
    res["fit_B"] = subsets(lp, labels, ~A)
    res["fit_gamma"] = g.detach().view(L, nb).tolist()
    print("fit_B", res["fit_B"], f"{time.time() - t0:.0f} s", flush=True)
    # mix: the rows' n-gram mixed with an LM's target probabilities
    if args.lm:
        lm = torch.from_numpy(np.load(args.lm).astype(np.float64))
        matched = tot > 0
        zero = torch.zeros(T_VAL, dtype=torch.long)
        deep = torch.where(matched.any(1), L - torch.flip(matched, [1]).double().argmax(1), zero)
        dtot = torch.where(deep > 0, tot.gather(1, (deep - 1).clamp_min(0)[:, None])[:, 0], zero.double())
        lidx = deep * nb + bins(dtot)
        g2 = torch.zeros(L * nb, dtype=torch.float64, requires_grad=True)
        lam = torch.zeros((L + 1) * nb, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([g2, lam], lr=1, max_iter=150, history_size=20, line_search_fn="strong_wolfe")
        sub = (*sub, lm[fit_at], lidx[fit_at])

        def mix_logp(p0_, tot_, dist_, cnt_, gidx, lm_, li):
            q = wb_logp(p0_, tot_, dist_, cnt_, torch.exp(g2.clamp(-20, 20))[gidx])
            a = torch.nn.functional.logsigmoid(lam[li])
            b = torch.nn.functional.logsigmoid(-lam[li])
            return torch.logaddexp(a - lm_, b + q)

        def closure2():
            opt.zero_grad()
            l = -mix_logp(*sub).mean()
            l.backward()
            return l

        opt.step(closure2)
        with torch.no_grad():
            lp = mix_logp(p0, tot, dist, cnt, idx, lm, lidx)
        res["mix_B"] = subsets(lp, labels, ~A)
        res["lm_B"] = subsets(-lm, labels, ~A)
        res["mix_lam"] = torch.sigmoid(lam.detach()).view(L + 1, nb).tolist()
        print("mix_B", res["mix_B"], "lm_B", res["lm_B"], f"{time.time() - t0:.0f} s", flush=True)
    # train: NgramMix trained on the training count rows, evaluated on all validation positions
    if args.train:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "old_records"))
        import hint_runtime

        tr = Path(args.train)
        unigram = np.load(tr / "unigram.npy").astype(np.float32)
        mod = hint_runtime.NgramMix(L, unigram)
        gammas = [mod.gamma, mod.gamma_level, mod.gamma_bin]
        groups = [{"params": gammas, "lr": 10 * args.lr, "initial_lr": 10 * args.lr},
                  {"params": [mod.base], "lr": args.lr, "initial_lr": args.lr}]
        opt2 = torch.optim.Adam(groups, betas=(0.9, 0.99), eps=1e-10, weight_decay=0.0)
        train_tok = {}
        warm = max(1, round(700 * args.steps / 19560))
        for step in range(args.steps):
            shard, k = step // PER_SHARD, step % PER_SHARD
            if shard not in train_tok:
                train_tok[shard] = tokens(Path(args.data) / f"fineweb_train_{shard + 1:06d}.bin")
            sl = slice(k * STEP_TOK, (k + 1) * STEP_TOK)
            d = tr / f"{shard:04d}"
            batch = torch.from_numpy(np.stack([np.asarray(np.load(d / f"{n}.npy", mmap_mode="r")[sl]).astype(np.float32)
                                               for n in ("tot", "dist", "cnt")], 1))
            tgt = torch.from_numpy(np.asarray(train_tok[shard][sl.start + 1 : sl.stop + 1]).astype(np.int64))
            if step < warm:
                f = (step + 1) / warm
            else:
                f = 0.5 * (1 + math.cos(math.pi * (step - warm) / (args.steps - warm)))
            for grp in opt2.param_groups:
                grp["lr"] = grp["initial_lr"] * f
            loss = -mod.log_q(tgt, batch).mean()
            opt2.zero_grad()
            loss.backward()
            opt2.step()
        with torch.no_grad():
            out = torch.zeros(T_VAL)
            tt = torch.from_numpy(target)
            for i in range(0, T_VAL, STEP_TOK):
                sl = slice(i, i + STEP_TOK)
                batch = torch.stack([tot[sl], dist[sl], cnt[sl]], 1).float()
                out[sl] = mod.log_q(tt[sl], batch)
        res["train"] = subsets(out.double(), labels)
        res["train_steps"] = args.steps
        print("train", res["train"], f"{time.time() - t0:.0f} s", flush=True)
    Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
