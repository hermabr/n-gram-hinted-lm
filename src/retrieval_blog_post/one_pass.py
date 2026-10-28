"""Two stages or one pass: how much the record's two-stage cache build (scan into a partitioned arena, then place each
partition into its buckets) gains over a one-pass build that inserts every entry straight into its bucket
(cache_ladder's SlotIndexOnePass: the same hash and table, AVX-512, no branch on the data), and what hashing alone
(reading the tokens, hashing them and computing both buckets, no inserts) takes, at cache sizes from 37B tokens halved down to about 1B.
Timed as ladder.py times the ladder: on a miso-lo worker of its own (ladder.WORKER_GIB), the shards staged in /dev/shm, THREADS
threads pinned one per core, a fresh process per build, only the build on the clock.

    uv run src/retrieval_blog_post/one_pass.py      # measures every size SAMPLES times on Slurm (cached ones load), prints a table

The Overview shows them (its "Two stages or one pass?" section)."""

import json
import os
import shutil
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

import furu

from retrieval_blog_post.fineweb import FinewebShards
from retrieval_blog_post.ladder import (
    SHARD_TOKENS,
    THREADS,
    WORKER_CPUS,
    WORKER_GIB,
    core_cpus,
    staged,
)

# The builds, in the order they run: (SlotIndexOnePass mode, prefetch distance), None for the record's SlotIndex.
BUILDS: dict[str, tuple[str, int] | None] = {
    "hash only": ("hash", 0),
    "two pass": None,
    # plain stores, the fill counts and table lines prefetched 32 entries ahead
    "one pass": ("racy", 32),
}
# 37B tokens, the record's cache, halved down to about 1B, in whole shards (the record indexes whole shards)
SIZES = tuple(round(37e9 / 2**i / SHARD_TOKENS) * SHARD_TOKENS for i in range(6))
# Independent measurements per size: jobs of their own, at other times and maybe on other nodes. The repeats within a
# job agree to a few percent, but a job's node and its neighbours' load shift all of them, by more than that.
SAMPLES = 5


class OnePassTime(furu.Spec[dict]):
    """`repeats` timed builds by each of BUILDS (in turn, so drift on the node hits them alike) over the first `tokens`
    training tokens, each in a fresh process; `sample` tells apart independent measurements of the same."""

    tokens: int
    repeats: int = 3
    sample: int = 0

    migrations = (
        furu.Added("sample", default=0),
    )  # the first measurements, made before there were samples

    def metadata(self) -> furu.Metadata:
        return furu.Metadata(
            requires=furu.Requires(
                cpus=furu.at_least(WORKER_CPUS), memory=furu.GiB(WORKER_GIB)
            ),
            reuse="never",  # a fresh process per measurement: no memory left from the last one
        )

    def create(self) -> dict:
        if self.tokens % SHARD_TOKENS:
            raise ValueError("the record's SlotIndex indexes whole shards")
        shards = FinewebShards(shards=self.tokens // SHARD_TOKENS).create()
        stage = Path(f"/dev/shm/one-pass-shards-{os.getpid()}")
        try:
            files = staged([str(shard.path) for shard in shards], stage)
            cpus = core_cpus(THREADS)
            runs = {build: [] for build in BUILDS}
            for _ in range(self.repeats):
                for build in BUILDS:
                    out = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "retrieval_blog_post.one_pass",
                            "measure",
                            build,
                            str(self.tokens),
                            json.dumps(cpus),
                            *files,
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    runs[build].append(json.loads(out.stdout.strip().splitlines()[-1]))
                    self.logger.info(
                        f"{build} {self.tokens / 1e9:g}B: {runs[build][-1]['seconds']:.3f} s"
                    )
        finally:
            shutil.rmtree(stage, ignore_errors=True)
        return {
            "tokens": self.tokens,
            "seconds": {b: [r["seconds"] for r in rs] for b, rs in runs.items()},
            "median": {
                b: statistics.median(r["seconds"] for r in rs) for b, rs in runs.items()
            },
            "runs": runs,
            "node": socket.gethostname(),
            "cpus": cpus,
        }


def measure(build: str, tokens: int, cpus: list[int], files: list[str]) -> dict:
    """One timed build, in this process: create untimed, build timed. The one-pass builds also report what they kept
    (counted after the clock): positions, entries, lost (slots two threads wrote at once), dropped (both buckets full)."""
    if (how := BUILDS[build]) is None:
        from exact_match import SlotIndex

        path = f"/dev/shm/one-pass-record-{os.getpid()}"
        try:
            index = SlotIndex.create(path, files, len(cpus), cpus, [])
            t = time.perf_counter()
            index.build()
            seconds = time.perf_counter() - t
            index.release()
        finally:
            if os.path.exists(path):
                os.unlink(path)
        return {"seconds": seconds}
    import cache_ladder

    mode, dist = how
    index = cache_ladder.SlotIndexOnePass.create(
        files, tokens, len(cpus), cpus, mode, dist
    )
    t = time.perf_counter()
    stats = index.build()
    seconds = time.perf_counter() - t
    return {"seconds": seconds, "stats": stats, "counts": index.counts()}


def specs() -> list[OnePassTime]:
    return [OnePassTime(tokens=n, sample=k) for k in range(SAMPLES) for n in SIZES]


def seconds(result: dict) -> dict[str, list[float]]:
    """An OnePassTime result's seconds per build; the results made before "hash only" was renamed call it "scan
    only"."""
    s = result["seconds"]
    return {("hash only" if b == "scan only" else b): v for b, v in s.items()}


def combined(results: list[dict]) -> list[dict]:
    """Per size, in order: the median over every build of every sample of each of BUILDS, how many builds (of each)
    and jobs, the nodes."""
    sizes = sorted({r["tokens"] for r in results})
    out = []
    for n in sizes:
        rs = [r for r in results if r["tokens"] == n]
        out.append(
            {
                "tokens": n,
                "median": {
                    b: statistics.median(s for r in rs for s in seconds(r)[b])
                    for b in BUILDS
                },
                "builds": sum(len(r["seconds"]["two pass"]) for r in rs),
                "jobs": len(rs),
                "nodes": sorted({r["node"].split(".")[0] for r in rs}),
            }
        )
    return out


if __name__ == "__main__":

    def _main() -> None:
        # A measurement process (OnePassTime's): measure BUILD TOKENS CPUS FILE...
        if sys.argv[1:2] == ["measure"]:
            build, tokens, cpus, *files = sys.argv[2:]
            print(json.dumps(measure(build, int(tokens), json.loads(cpus), files)))
            return
        # Imported, not this file's own definitions: run as a file, they would be __main__.OnePassTime, another spec.
        from retrieval_blog_post.one_pass import BUILDS, combined, specs
        from retrieval_blog_post.config import create

        print(f"{'tokens':>8s}" + "".join(f"{b:>18s}" for b in BUILDS))
        for r in combined(create(specs(), job_name="one-pass")):
            m = r["median"]
            print(
                f"{r['tokens'] / 1e9:7.3g}B"
                + "".join(f"{m[b]:16.3f} s" for b in BUILDS)
                + f"   ({r['builds']} builds each, on {', '.join(r['nodes'])})"
            )

    _main()
