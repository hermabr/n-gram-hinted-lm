"""The CPU reference of gpu_val_stats.py on a shard subset, with the existing definitions: exact_match.kgram_counts per
level of ALL_LEVELS (as val_hints.ValContextCounts counts), then old_records.val_count_features (the count rows) and
val_hints.deepest + old_records._val_rows/_weights (each variant's hint rows). Run with the repository's venv:

    .venv/bin/python old_records/gpu_val_stats_ref.py <shards file> <val shard> <out dir> [memory GiB]

Writes <out>/tot.npy, dist.npy, cnt.npy and <out>/rows_<levels>/{cell,tokens,weights}.npy (gpu_val_stats.py's layout)."""

import json
import sys
import time
from pathlib import Path

import numpy as np

from exact_match import kgram_counts
from retrieval_blog_post.kgram import VAL_TOKENS
from retrieval_blog_post.old_records import _val_rows, _weights, val_count_features
from retrieval_blog_post.val_hints import (
    ALL_LEVELS,
    ARRAYS,
    LEVELS,
    TOP,
    VARIANTS,
    ContextCounts,
    Hints,
    deepest,
)


def main() -> None:
    shards, val_path, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    memory = int(sys.argv[4]) << 30 if len(sys.argv) > 4 else 16 << 30
    files = [p for p in shards.read_text().split() if p]
    val = np.memmap(val_path, dtype="<u2", mode="r", offset=1024)
    tokens = np.ascontiguousarray(val[: VAL_TOKENS + 1])
    counts_dir = out / "counts"
    counts_dir.mkdir(parents=True, exist_ok=True)
    hits = None
    log = {}
    for k in ALL_LEVELS:
        done = counts_dir / f"k{k}"
        if done.exists():
            hits = None
            continue
        t0 = time.time()
        *arrays, hits = kgram_counts(files, tokens, k, TOP, memory, None, hits)
        part = counts_dir / f"k{k}.partial"
        part.mkdir(exist_ok=True)
        for name, a in zip(ARRAYS, arrays):
            np.save(part / f"{name}.npy", a)
        part.rename(done)
        log[k] = time.time() - t0
        print(f"k={k}: {len(arrays[1]):,} contexts, {int(arrays[1].sum()):,} occurrences, {log[k]:.0f} s", flush=True)
    counts = ContextCounts(counts_dir, ALL_LEVELS)
    t0 = time.time()
    tot, dist, cnt = val_count_features(counts, tokens[1:], np.arange(VAL_TOKENS), ALL_LEVELS)
    for name, a in (("tot", tot), ("dist", dist), ("cnt", cnt)):
        np.save(out / f"{name}.npy", a)
    for levels in VARIANTS:
        hdir = out / f"hints_{'_'.join(map(str, levels))}"
        hdir.mkdir(exist_ok=True)
        level, context, cell = deepest(counts, levels, VAL_TOKENS, 1, LEVELS)
        for name, a in (("level", level), ("context", context), ("cell", cell)):
            np.save(hdir / f"{name}.npy", a)
        (hdir / "buckets.json").write_text(json.dumps(LEVELS))
        cell, toks, cnts = _val_rows(Hints(hdir, levels, counts), VAL_TOKENS)
        rdir = out / f"rows_{'_'.join(map(str, levels))}"
        rdir.mkdir(exist_ok=True)
        np.save(rdir / "cell.npy", cell)
        np.save(rdir / "tokens.npy", toks)
        np.save(rdir / "weights.npy", _weights(cnts))
    print(f"rows: {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
