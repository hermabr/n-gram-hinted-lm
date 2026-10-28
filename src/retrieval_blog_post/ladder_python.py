"""The ladder's first rung (ladder.py): the naive cache, Python dicts. For k = 6, 10 and 20, a dict from the hash of a
k-token context to a counter of the tokens that followed it, at most CAP different ones per context.

One dict cannot be filled from 48 processes, so each process owns the keys that are its number modulo the process
count: in phase 1 every process hashes its slice of the positions and writes each (key, next token) to the owner's file
in /dev/shm, and in phase 2 every owner reads its files and fills its three dicts. cache_ladder's HashMapIndex (Rust) is
the same algorithm."""

import multiprocessing as mp
import os
import resource
import time
from array import array
from pathlib import Path

import numpy as np

from retrieval_blog_post.fineweb import HEADER_BYTES

KS = (6, 10, 20)
CAP = 288  # next tokens kept per context


class PythonDicts:
    """The cache_ladder interface (create untimed, build timed) for the dicts: create starts the processes, pinned one
    per CPU of `cpus`, and build has them fill the dicts, which they keep until close."""

    def __init__(self, files: list[str], tokens: int, processes: int, cpus: list[int]):
        self.scratch = Path(f"/dev/shm/cache-ladder-dicts-{os.getpid()}")
        self.scratch.mkdir()
        lengths = []
        for path in files:
            n = int(np.fromfile(path, dtype="<i4", count=3)[2])
            lengths.append(min(n, tokens - sum(lengths)))
            if sum(lengths) == tokens:
                break
        if sum(lengths) != tokens:
            raise ValueError(
                f"the files hold {sum(lengths)} tokens, fewer than {tokens}"
            )
        shards = list(zip(files, lengths))
        ctx = mp.get_context("fork")
        self.go, self.stop = ctx.Event(), ctx.Event()
        self.results = ctx.Queue()
        barrier = ctx.Barrier(processes)
        self.processes = [
            ctx.Process(
                target=_worker,
                args=(
                    w,
                    processes,
                    cpus[w % len(cpus)] if cpus else None,
                    shards,
                    tokens,
                    self.scratch,
                    barrier,
                    self.go,
                    self.stop,
                    self.results,
                ),
            )
            for w in range(processes)
        ]
        for p in self.processes:
            p.start()

    def build(self) -> dict[str, float]:
        self.go.set()
        stats: dict[str, float] = {}
        for _ in self.processes:
            for name, value in self.results.get().items():
                stats[name] = stats.get(name, 0) + value
        return stats

    def close(self) -> None:
        self.stop.set()
        for p in self.processes:
            p.join()
        for f in self.scratch.iterdir():
            f.unlink()
        self.scratch.rmdir()


def _worker(w, workers, cpu, shards, tokens, scratch, barrier, go, stop, results):
    if cpu is not None:
        os.sched_setaffinity(0, {cpu})
    go.wait()
    # Phase 1: the positions [lo, hi) of the concatenated shards; position i's contexts end just before it.
    lo, hi = w * tokens // workers, (w + 1) * tokens // workers
    keys = [[array("q") for _ in KS] for _ in range(workers)]
    nexts = [[array("H") for _ in KS] for _ in range(workers)]
    start = 0
    for path, n in shards:
        a, b = max(lo, start) - start, min(hi, start + n) - start
        start += n
        if a >= b:
            continue
        first = max(a - max(KS), 0)
        toks = np.fromfile(
            path, dtype="<u2", count=b - first, offset=HEADER_BYTES + 2 * first
        ).tolist()
        for i in range(a - first, b - first):
            nxt = toks[i]
            for j, k in enumerate(KS):
                if i + first >= k:
                    key = hash(tuple(toks[i - k : i]))
                    d = key % workers
                    keys[d][j].append(key)
                    nexts[d][j].append(nxt)
    for d in range(workers):
        for j in range(len(KS)):
            with open(scratch / f"{w}-{d}-{j}.keys", "wb") as f:
                keys[d][j].tofile(f)
            with open(scratch / f"{w}-{d}-{j}.next", "wb") as f:
                nexts[d][j].tofile(f)
    del keys, nexts
    barrier.wait()
    # Phase 2: this process's keys, from every process's files.
    dicts = [{} for _ in KS]
    dropped = 0
    for s in range(workers):
        for j in range(len(KS)):
            k_path, n_path = (
                scratch / f"{s}-{w}-{j}.keys",
                scratch / f"{s}-{w}-{j}.next",
            )
            ks, ns = array("q"), array("H")
            with open(k_path, "rb") as f:
                ks.fromfile(f, k_path.stat().st_size // 8)
            with open(n_path, "rb") as f:
                ns.fromfile(f, n_path.stat().st_size // 2)
            d = dicts[j]
            for key, nxt in zip(ks, ns):
                counter = d.get(key)
                if counter is None:
                    d[key] = {nxt: 1}
                elif nxt in counter:
                    counter[nxt] += 1
                elif len(counter) < CAP:
                    counter[nxt] = 1
                else:
                    dropped += 1
    # Peak memory of this process (build sums it over the processes).
    stats = {
        "dropped": dropped,
        "rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
    }
    for j, k in enumerate(KS):
        stats[f"keys{k}"] = len(dicts[j])
        stats[f"values{k}"] = sum(map(len, dicts[j].values()))
    results.put(stats)
    stop.wait()


if __name__ == "__main__":
    # A check against a single-process count: python -m retrieval_blog_post.ladder_python SHARD TOKENS PROCESSES
    import sys
    from collections import Counter

    path, tokens, processes = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    index = PythonDicts([path], tokens, processes, [])
    t = time.perf_counter()
    stats = index.build()
    print(f"{time.perf_counter() - t:.2f} s", stats)
    index.close()
    toks = np.fromfile(path, dtype="<u2", count=tokens, offset=HEADER_BYTES).tolist()
    for k in KS:
        pairs = Counter((tuple(toks[i - k : i]), toks[i]) for i in range(k, tokens))
        contexts = {c for c, _ in pairs}
        assert stats[f"keys{k}"] == len(contexts), (k, stats[f"keys{k}"], len(contexts))
        assert stats[f"values{k}"] == len(pairs) or stats["dropped"], (
            k,
            stats[f"values{k}"],
            len(pairs),
        )
    print("keys and values match a single-process count")
