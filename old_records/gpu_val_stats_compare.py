"""Compare gpu_val_stats.py's outputs with a reference: per level of the count rows (tot, dist, cnt) and per variant of
the hint rows (cell, tokens, weights), the positions that differ.

    python old_records/gpu_val_stats_compare.py <gpu out dir> <ref counts dir> [<variant levels>=<ref rows dir> ...]

The reference count dir holds tot.npy, dist.npy and cnt.npy (gpu_val_stats_ref.py's, or ValCountRows' data); the rows
dirs cell.npy, tokens.npy and weights.npy (ValHintRows' data). Without explicit rows dirs, <ref counts dir>/rows_* are
used. Prints and returns a JSON report."""

import json
import sys
from pathlib import Path

import numpy as np

ALL_LEVELS = (2, 3, 4, 5, 6, 8, 10, 12, 16, 20)


def compare(gpu: Path, ref_counts: Path, ref_rows: dict[str, Path]) -> dict:
    report: dict = {"counts": {}, "rows": {}}
    for name in ("tot", "dist", "cnt"):
        a = np.load(gpu / f"{name}.npy", mmap_mode="r")
        b = np.load(ref_counts / f"{name}.npy", mmap_mode="r")
        if a.shape != b.shape or a.dtype != b.dtype:
            report["counts"][name] = f"shape/dtype {a.shape} {a.dtype} vs {b.shape} {b.dtype}"
            continue
        per = {}
        for j, k in enumerate(ALL_LEVELS):
            x, y = np.asarray(a[:, j]), np.asarray(b[:, j])
            per[k] = {"differ": int((x != y).sum()), "nonzero": int((y > 0).sum())}
        report["counts"][name] = per
    for variant, rdir in ref_rows.items():
        r = {}
        for name in ("cell", "tokens", "weights"):
            a = np.load(gpu / f"rows_{variant}" / f"{name}.npy")
            b = np.load(rdir / f"{name}.npy")
            if a.shape != b.shape or a.dtype != b.dtype:
                r[name] = f"shape/dtype {a.shape} {a.dtype} vs {b.shape} {b.dtype}"
                continue
            bad = (a != b).reshape(len(a), -1).any(1)
            r[name] = {"rows_differ": int(bad.sum())}
        report["rows"][variant] = r
    return report


def main() -> None:
    gpu, ref = Path(sys.argv[1]), Path(sys.argv[2])
    rows = {}
    for arg in sys.argv[3:]:
        v, d = arg.split("=")
        rows[v] = Path(d)
    if not rows:
        rows = {d.name[len("rows_") :]: d for d in sorted(ref.glob("rows_*"))}
    report = compare(gpu, ref, rows)
    print(json.dumps(report, indent=1))
    (gpu / "compare.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
