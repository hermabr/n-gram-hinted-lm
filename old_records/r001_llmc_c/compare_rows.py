"""The rows train_gpt2.cu computes on the clock (HINT_DUMP=<dir>; only llmc_onclock.py's v1-v7 builds have it, not
the committed train_gpt2.cu) against the references:

- training rows: exact_match.OnlineRows (old_records/online_hints.py's, the PyTorch port's on-clock rows) over the same
  stream and plan (llm.c's sequential loader: every rank's micro-batches), per rank;
- validation rows: a directory with tot/dist/cnt.npy and the hint rows (cell/tokens/weights.npy, there or in
  rows_2_3_4_5_6_10_20/): old_records.ValCountRows' and ValHintRows' (all 1028 shards), or gpu_val_stats_ref.py's
  (a shard subset).

    python compare_rows.py <dump dir> --train "<shard glob>" --nproc 8 --B 64 --T 1024 --accum 1 --steps 210 \
        --val-counts <dir> --val-hints <dir>
"""

import argparse
import glob
import os
import sys

import numpy as np

LEVELS = (2, 3, 4, 5, 6, 10, 20)
COUNT_LEVELS = (2, 3, 4, 5, 6, 8, 10, 12, 16, 20)
ENTRIES = 8
VAL_TOKENS = 10_485_760


def ntokens(path: str) -> int:
    return int(np.fromfile(path, dtype=np.int32, count=3)[2])


def plan(files: list[str], nproc: int, BT: int, accum: int, steps: int) -> list[tuple[int, int, int, int, int]]:
    """llm.c's sequential DataLoader: (step, micro, rank, shard, offset) of every micro-batch."""
    sizes = [ntokens(f) for f in files]
    out = []
    for r in range(nproc):
        shard = sample = 0
        for b in range(steps * accum):
            if sample >= (sizes[shard] - 1) // (nproc * BT):
                shard, sample = shard + 1, 0
            out.append((b // accum, b % accum, r, shard, sample * nproc * BT + r * BT))
            sample += 1
    return out


def differ(name: str, mine: np.ndarray, ref: np.ndarray) -> int:
    bad = mine != ref
    if bad.ndim > 1:
        bad = bad.reshape(len(bad), -1).any(1)
    n = int(bad.sum())
    print(f"  {name}: {n} of {len(bad)} positions differ" + (f" (first at {int(np.flatnonzero(bad)[0])})" if n else ""))
    return n


def load(dump: str, prefix: str, n: int) -> dict[str, np.ndarray]:
    L = len(COUNT_LEVELS)
    spec = {"cell": (np.int16, (n,)), "tokens": (np.uint16, (n, ENTRIES)), "weights": (np.uint16, (n, ENTRIES)),
            "tot": (np.uint32, (n, L)), "dist": (np.uint16, (n, L)), "cnt": (np.uint32, (n, L))}
    return {k: np.fromfile(os.path.join(dump, f"{prefix}{k}.bin"), dtype=d).reshape(s) for k, (d, s) in spec.items()}


def check_train(args) -> int:
    import exact_match

    files = sorted(glob.glob(args.train))
    BT = args.B * args.T
    p = plan(files, args.nproc, BT, args.accum, args.steps)
    ends: dict[int, int] = {}
    for _, _, _, shard, offset in p:
        ends[shard] = max(ends.get(shard, 0), offset + BT + 1)
    base, bounds, parts = {}, [0], []
    for shard in range(max(ends) + 1):
        size = min(ends.get(shard, 0), ntokens(files[shard]))
        base[shard] = bounds[-1]
        bounds.append(bounds[-1] + size)
        parts.append(np.memmap(files[shard], dtype=np.uint16, mode="r", offset=1024)[:size])
    stream = np.concatenate(parts)
    spans = np.array([(base[sh] + off, base[sh] + off + BT, step) for step, _, _, sh, off in p], np.int64)
    rows = exact_match.OnlineRows(stream, bounds, spans, 32, [])
    n, L = len(stream), len(COUNT_LEVELS)
    ref = {"cell": np.zeros(n, np.int16), "tokens": np.zeros((n, ENTRIES), np.uint16),
           "weights": np.zeros((n, ENTRIES), np.uint16), "tot": np.zeros((n, L), np.uint32),
           "dist": np.zeros((n, L), np.uint16), "cnt": np.zeros((n, L), np.uint32)}
    rows.advance(args.steps, list(LEVELS), list(LEVELS), ENTRIES, list(COUNT_LEVELS), ref["cell"], ref["tokens"],
                 ref["weights"], ref["tot"], ref["dist"], ref["cnt"])
    bad = 0
    for r in range(args.nproc):
        if not os.path.exists(os.path.join(args.dump, f"train_rank{r}_cell.bin")):
            continue
        mine = load(args.dump, f"train_rank{r}_", args.steps * args.accum * BT)
        at = np.empty(args.steps * args.accum * BT, np.int64)  # each row's stream position
        for step, micro, rank, shard, offset in p:
            if rank == r:
                row = (step * args.accum + micro) * BT
                at[row : row + BT] = base[shard] + offset + np.arange(BT)
        print(f"training rows, rank {r}: {len(at)} rows, {(mine['cell'] > 0).mean():.4f} hinted "
              f"(reference {(ref['cell'][at] > 0).mean():.4f})")
        for k in mine:
            bad += differ(k, mine[k], ref[k][at])
    return bad


def check_val(args) -> int:
    mine = load(args.dump, "val_", VAL_TOKENS)
    hints = args.val_hints
    if os.path.isdir(os.path.join(hints, "rows_" + "_".join(map(str, LEVELS)))):
        hints = os.path.join(hints, "rows_" + "_".join(map(str, LEVELS)))
    print(f"validation rows: {(mine['cell'] > 0).mean():.4f} hinted")
    bad = 0
    for k in ("tot", "dist", "cnt"):
        bad += differ(k, mine[k], np.load(os.path.join(args.val_counts, f"{k}.npy"), mmap_mode="r"))
    for k in ("cell", "tokens", "weights"):
        bad += differ(k, mine[k], np.load(os.path.join(hints, f"{k}.npy"), mmap_mode="r"))
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--train", help="the training shards' glob (as train_gpt2cu -i)")
    ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--val-counts", help="the reference count rows' directory")
    ap.add_argument("--val-hints", help="the reference hint rows' directory")
    args = ap.parse_args()
    bad = 0
    if args.train:
        bad += check_train(args)
    if args.val_counts:
        bad += check_val(args)
    print("all rows equal" if bad == 0 else f"{bad} differing (positions x arrays)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
