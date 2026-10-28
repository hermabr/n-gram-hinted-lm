"""The cache-build ladder: how many training tokens each version of the validation cache indexes in BUDGET seconds of
build on THREADS threads, from the naive Python dicts to the record's SlotIndex. Every version builds about the same
data, a 6-gram-keyed record per position with the next token and the 10- and 20-token contexts, from FineWeb100B's
first training shards, warm in the page cache; only the build is timed, not what each design did before its clock.

    uv run src/retrieval_blog_post/ladder.py [VERSION...]   # measures on miso-lo workers until each is within TOL

Every measurement runs on a WORKER_GIB worker of its own (config.create's ladder pool, which runs nothing else), in a
fresh process per repeat, with the THREADS threads (or processes) pinned one per physical core."""

import itertools
import json
import os
import shutil
import socket
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import furu

from retrieval_blog_post.fineweb import FinewebShards

# seconds: the target of every version's search (next_tokens), the record's build on Nebius as of 2026-10-07 (35B tokens on
# 48 threads). measured() replays the searches with it, so it stays.
BUDGET = 6.35
THREADS = 48
WORKER_CPUS = 2 * THREADS  # whole cores, with their hyperthreads idle
# 800 until 2026-10-20; the record's build of 37B tokens, with the shards in /dev/shm, needs about 790 (its arena blocks
# grew fourfold). The memory a worker reserves does not change its speed.
WORKER_GIB = 1000
SHARD_TOKENS = 100_000_000
TOL = 0.04  # a version is done once a measurement is within this of BUDGET

Version = Literal[
    "python-dicts",
    "rust-hashmap",
    "per-length-tables",
    "fast-checksum",
    "slot-v1",
    "slot-branch-free",
    "slot-ifma",
    "slot-balanced",
]
# The record's build: exact_match's SlotIndex (exact_match/src/slot.rs, modded-nanogpt e390cb25), whole shards only.
RECORD: Version = "slot-balanced"


@dataclass(frozen=True)
class Rung:
    name: str
    what: str
    start: int  # tokens of the first measurement
    whole_shards: bool = False  # its builder takes whole shards only
    about: str = ""  # what changed against the rung before it


RUNGS: dict[Version, Rung] = {
    "python-dicts": Rung(
        "Python dicts",
        "a dict per n-gram length, a counter of next tokens per context",
        100_000_000,
        about="The naive index. For each of the 6-, 10- and 20-token contexts, a Python dict maps the hash of the context to a counter of the tokens that followed it (at most 288 different ones). 48 processes share the work by key: each hashes its slice of the corpus and hands every (key, next token) pair to the process that owns the key, through files in /dev/shm; each owner then fills its three dicts.",
    ),
    "rust-hashmap": Rung(
        "Rust hash maps",
        "the same maps and counters in Rust (hashbrown)",
        3_600_000_000,
        about="The same work as the Python dicts (three maps from a context's hash to a counter of its next tokens, "
        "at most 288, every count kept, the pairs handed to the key's owner thread), written in Rust with hashbrown "
        "hash maps. The first, literal version (std HashMap of HashMaps) was no faster than Python: 48 threads of one "
        "process allocating hundreds of millions of small maps contend for one allocator and address space. Here "
        "each owner's map is split into hashbrown maps small enough to stay in a core's cache, filled one at a "
        "time, with a context's single next token stored inline in the value, allocating from memory mapped once "
        "before the threads fill it.",
    ),
    "per-length-tables": Rung(
        "Per-length tables",
        "three slot tables, one per n-gram length",
        6_000_000_000,
        about="The first SlotIndex's build (2026-09-29) applied to the per-length idea of 2026-09-16: three "
        "separate tables, for the 6-, 10- and 20-token contexts, each keyed by the hash of its own context. Each "
        "position becomes three 8-byte entries (a slot, a tag and the next token). Table by table, one vectorized "
        "scan writes the entries into partitions through write-combining cache lines and non-temporal stores, with "
        "no counting pass, and each partition is then counting-sorted into fixed slots while it fits in cache, every "
        "slot keeping its first 288 entries. Three times the entries of the one table that follows it.",
    ),
    "fast-checksum": Rung(
        "Fast checksum index",
        "6-gram entries with 10/20 fingerprints, counted and sorted",
        12_000_000_000,
        about="One index instead of three. Every position becomes one small entry keyed by its 6-gram, holding the next token and 12-bit fingerprints of its 10- and 20-token contexts. The entries are counted, scattered into partitions and sorted, with non-temporal stores and a cheap fingerprint hash (2026-09-29, ckfast).",
    ),
    "slot-v1": Rung(
        "One table",
        "one table for all three lengths, keyed by the 6-gram",
        14_000_000_000,
        about="One table instead of three, built the same way (2026-09-29, the first SlotIndex). Every position "
        "becomes a single 8-byte entry keyed by its 6-token context, holding the next token and fingerprints of its "
        "10- and 20-token contexts (12 and 6 bits): a third of the entries, written by one scan and sorted into "
        "slots in one pass.",
    ),
    "slot-branch-free": Rung(
        "Branch-free + AVX-512",
        "the two-step SlotIndex with branch-free AVX-512 loops",
        20_000_000_000,
        about="Today's two-step layout: the scan fills an arena through one cache line per partition held in L2, and the finish places each partition into 2^14 buckets of the final table through staging lines, both with non-temporal stores. The inner loops are written with AVX-512 and have no branches on the data (2026-10-05).",
    ),
    "slot-ifma": Rung(
        "Branch-free & more SIMD",
        "plus IFMA hash, no NUMA balancing, a padded block counter",
        33_000_000_000,
        about="The record code until 2026-10-20 (9025a886). The scan fills an arena through one cache line per "
        "partition held in L2, and the finish places each partition into 2^14 buckets of the final table through "
        "staging lines, both with non-temporal stores; the inner loops are written with AVX-512 and have no branches "
        "on the data. On top of that: the hash computed with IFMA multiplies on byte-permuted token words, the "
        "mappings kept out of automatic NUMA balancing, and the arena's shared block counter on a cache line of its "
        "own.",
    ),
    "slot-balanced": Rung(
        "Balanced, fused scan",
        "plus scan load balancing, larger arena blocks, fused hash and scatter, 8 B staging inserts",
        40_000_000_000,
        True,
        about="The record code (2026-10-20, e390cb25). A scan thread done with its own pieces takes the back half of "
        "the busiest remaining run of pieces, so no thread waits on the slowest; the scan hashes 16 positions at a "
        "time, one group ahead of their inserts, instead of 4096 into a buffer first; the arena's blocks are four "
        "times larger (8192 entries), and the finish prefetches them 256 entries ahead and places each entry with "
        "one 8-byte store into its bucket's staging line. The table is the same as before.",
    ),
}


class BuildTime(furu.Spec[dict]):
    """`repeats` timed builds of `version` over the first `tokens` training tokens, each in a fresh process."""

    version: Version
    tokens: int
    repeats: int = 3

    # The versions until 2026-10-20, when "slot-record" (the previous record's build, through exact_match) gave way to
    # "slot-ifma" (that build in cache_ladder) and "slot-balanced" (the record's): its measurements stay behind.
    migrations = (
        furu.Retyped(
            "version",
            was=Literal[
                "python-dicts",
                "rust-hashmap",
                "per-length-tables",
                "fast-checksum",
                "slot-v1",
                "slot-branch-free",
                "slot-record",
            ],
        ),
    )

    def metadata(self) -> furu.Metadata:
        return furu.Metadata(
            requires=furu.Requires(
                cpus=furu.at_least(WORKER_CPUS), memory=furu.GiB(WORKER_GIB)
            ),
            reuse="never",  # a fresh process per measurement: no memory left from the last one
        )

    def create(self) -> dict:
        shards = FinewebShards(shards=-(-self.tokens // SHARD_TOKENS)).create()
        stage = Path(f"/dev/shm/cache-ladder-shards-{os.getpid()}")
        try:
            files = staged([str(shard.path) for shard in shards], stage)
            cpus = core_cpus(THREADS)
            runs = self._runs(files, cpus)
        finally:
            shutil.rmtree(stage, ignore_errors=True)
        return {
            "version": self.version,
            "tokens": self.tokens,
            "seconds": [r["seconds"] for r in runs],
            "median": statistics.median(r["seconds"] for r in runs),
            "runs": runs,
            "node": socket.gethostname(),
            "cpus": cpus,
        }

    def _runs(self, files: list[str], cpus: list[int]) -> list[dict]:
        runs = []
        for _ in range(self.repeats):
            out = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "retrieval_blog_post.ladder",
                    "measure",
                    self.version,
                    str(self.tokens),
                    json.dumps(cpus),
                    *files,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            runs.append(
                json.loads(out.stdout.strip().splitlines()[-1])
                | {"stderr": out.stderr[-4000:]}
            )
            self.logger.info(
                f"{self.version} {self.tokens / 1e9:.3f}B: {runs[-1]['seconds']:.3f} s"
            )
        return runs


def staged(files: list[str], stage: Path) -> list[str]:
    """Copies of `files` in `stage`, a new directory in /dev/shm, 16 at a time. The record keeps the shards in the page
    cache; on a shared node other jobs' reads can evict them from it (and the build then reads NFS), but not from
    shared memory."""
    stage.mkdir()
    copies = [str(stage / Path(f).name) for f in files]
    with ThreadPoolExecutor(16) as pool:
        list(pool.map(shutil.copyfile, files, copies))
    return copies


def core_cpus(n: int) -> list[int]:
    """`n` CPUs of this process's affinity, one per physical core while there are enough cores (in CPU order)."""
    cores: dict[tuple[str, str], list[int]] = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        core = (
            (topology / "physical_package_id").read_text(),
            (topology / "core_id").read_text(),
        )
        cores.setdefault(core, []).append(cpu)
    firsts = sorted(cpus[0] for cpus in cores.values())
    rest = sorted(cpu for cpus in cores.values() for cpu in cpus[1:])
    return (firsts + rest)[:n]


def measure(version: Version, tokens: int, cpus: list[int], files: list[str]) -> dict:
    """One timed build, in this process: create untimed, build timed."""
    import resource

    if version == RECORD:
        from exact_match import SlotIndex

        path = f"/dev/shm/cache-ladder-record-{os.getpid()}"
        if len(files) * SHARD_TOKENS != tokens:
            raise ValueError("the record's SlotIndex indexes whole shards")
        try:
            index = SlotIndex.create(path, files, len(cpus), cpus, [])
            t = time.perf_counter()
            index.build()
            seconds = time.perf_counter() - t
            stats = {}
            index.release()
        finally:
            if os.path.exists(path):
                os.unlink(path)
    elif version == "python-dicts":
        from retrieval_blog_post.ladder_python import PythonDicts

        index = PythonDicts(files, tokens, len(cpus), cpus)
        try:
            t = time.perf_counter()
            stats = index.build()
            seconds = time.perf_counter() - t
        finally:
            index.close()
    else:
        import cache_ladder

        cls = {
            "rust-hashmap": cache_ladder.HashMapIndex,
            "per-length-tables": cache_ladder.PerLengthTables,
            "fast-checksum": cache_ladder.FastChecksumIndex,
            "slot-v1": cache_ladder.SlotIndexV1,
            "slot-branch-free": cache_ladder.SlotIndexBranchFree,
            "slot-ifma": cache_ladder.SlotIndexIfma,
        }[version]
        index = cls.create(files, tokens, len(cpus), cpus)
        t = time.perf_counter()
        stats = index.build()
        seconds = time.perf_counter() - t
        del index
    # The dicts' processes report their own peaks, which add up; shard copies a build maps count too.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20 + stats.get(
        "rss_gib", 0
    )
    return {"seconds": seconds, "stats": stats, "peak_rss_gib": rss}


def next_tokens(version: Version, trials: list[dict]) -> int | None:
    """The next measurement's tokens for `version` given its `trials` (BuildTime results), None when one is within
    TOL of BUDGET: the tokens where the two trials closest to BUDGET put it (log-log), or scaled in proportion from
    the only one."""
    if not trials:
        return RUNGS[version].start
    if any(abs(t["median"] / BUDGET - 1) <= TOL for t in trials):
        return None
    trials = sorted(trials, key=lambda t: abs(t["median"] - BUDGET))
    if len(trials) == 1 or trials[0]["median"] == trials[1]["median"]:
        tokens = trials[0]["tokens"] * BUDGET / trials[0]["median"]
    else:
        import math

        (n0, t0), (n1, t1) = [
            (math.log(t["tokens"]), math.log(t["median"])) for t in trials[:2]
        ]
        slope = (t1 - t0) / (n1 - n0) if n1 != n0 else 1.0
        tokens = math.exp(n0 + (math.log(BUDGET) - t0) / max(slope, 0.2))
    # Two significant digits, or three where two land on a measured count; None where three do too.
    seen = {t["tokens"] for t in trials}
    for digits in (2, 3):
        if (n := rounded(version, tokens, digits)) not in seen:
            return n
    return None


def rounded(version: Version, tokens: float, digits: int) -> int:
    """`tokens` to `digits` significant digits, or to whole shards for a builder that takes only those."""
    if RUNGS[version].whole_shards:
        return max(1, round(tokens / SHARD_TOKENS)) * SHARD_TOKENS
    step = 10 ** max(0, len(str(int(tokens))) - digits)
    return max(1, round(tokens / step)) * step


def tokens_in_budget(trials: list[dict]) -> float:
    """The tokens built in BUDGET seconds: the trial closest to BUDGET, scaled in proportion."""
    best = min(trials, key=lambda t: abs(t["median"] - BUDGET))
    return best["tokens"] * BUDGET / best["median"]


# The run time a version would give the record, if the record's validation cache were as large as that version builds in
# time: the time to the target loss (3.28) against the cache size (B tokens), Nebius seconds, piecewise linear between
# these runs (run_estimate goes no further than the last).
TRAIN_ANCHORS = (
    (0.0, 40.0),  # no cache: the record before retrieval (ANVIL2 f9b6266, 40.00 s)
    (10.26, 21.55),  # our record pool of 2026-10-01: 538 steps, 18 runs, 21.555 s
    (25.0, 9.4),  # about 765 steps (miso step curve, 2026-10-05) x 12.1 ms + 0.15 s
    (27.5, 8.7),  # about 707 steps (miso, 2026-10-05)
    (30.0, 8.15),  # Nebius 300 shards at 660 steps, 3.2747 at 8.17 s
    (32.5, 7.4),  # Nebius 325 shards at 600 steps, 7.40 s
    # the record pools of 2026-10-07 (350 shards, 570 steps, 16 runs) and 2026-10-20 (370 shards, 540 steps, 16 runs)
    (35.0, 7.065),
    (37.0, 6.702),
)
T_START = 0.11  # s: the build starts once the training rows are built on the same pool
# The record is conservative: its 37B build is done 0.11 + 6.15 s (medians) into its 6.702 s of training, at 93% of it,
# not at the end. Every version gets the same margin.
MARGIN = (T_START + 6.15) / 6.702


def train_time(billions: float) -> float:
    """Seconds to the target with a validation cache of `billions` B tokens (TRAIN_ANCHORS, up to the last)."""
    for (n0, t0), (n1, t1) in itertools.pairwise(TRAIN_ANCHORS):
        if billions <= n1:
            return t0 + (t1 - t0) * (billions - n0) / (n1 - n0)
    raise ValueError(f"no record runs past {TRAIN_ANCHORS[-1][0]}B tokens")


def build_time(trials: list[dict], tokens: float) -> float:
    """Seconds a version's build would take for `tokens`, from its measurements (medians): at least as long as any
    measurement of at most that many tokens, scaled in proportion (or the smallest one, if all are larger). Builds that
    slow down per token as they grow (the hash maps) keep their slowest measured rate from then on."""
    below = [t for t in trials if t["tokens"] <= tokens] or [
        min(trials, key=lambda t: t["tokens"])
    ]
    return max(t["median"] * tokens / t["tokens"] for t in below)


def run_estimate(trials: list[dict]) -> dict:
    """The record redone with this build: the largest whole-shard cache (up to the record's, TRAIN_ANCHORS' last, as
    nothing past it was run) whose build ends by MARGIN of the training time it gives, and that training time."""
    shards = 0
    for n in range(1, round(TRAIN_ANCHORS[-1][0] * 1e9 / SHARD_TOKENS) + 1):
        if T_START + build_time(trials, n * SHARD_TOKENS) > MARGIN * train_time(
            n * SHARD_TOKENS / 1e9
        ):
            break
        shards = n
    tokens = shards * SHARD_TOKENS
    return {
        "shards": shards,
        "tokens": tokens,
        "build_s": T_START + build_time(trials, tokens) if shards else 0.0,
        "run_s": train_time(tokens / 1e9),
    }


def search(versions: list[Version], max_rounds: int = 6) -> dict[Version, list[dict]]:
    """Measure every version until one of its measurements is within TOL of BUDGET, a round of measurements at a time
    (one per unfinished version, in parallel on the ladder pool)."""
    from retrieval_blog_post.config import create

    trials: dict[Version, list[dict]] = {v: [] for v in versions}
    for _ in range(max_rounds):
        # The measurements follow from the earlier ones, so a rerun loads the same ones from the cache and resumes.
        todo = {
            v: BuildTime(version=v, tokens=n)
            for v in versions
            if (n := next_tokens(v, trials[v])) is not None
        }
        if not todo:
            break
        for v, result in zip(todo, create(list(todo.values()), job_name="ladder")):
            trials[v].append(result)
    return trials


# Measurements beyond the search's (BuildTime specs like the search's): every version the Overview shows at the cache
# size run_estimate picks for it, or as close as the 800 GB worker allowed, so its estimates rest on measurements at
# those sizes rather than on extrapolation.
EXTRA: dict[Version, tuple[int, ...]] = {
    "python-dicts": (200_000_000, 300_000_000, 400_000_000),
    "rust-hashmap": (1_000_000_000, 8_000_000_000),  # 84 B/token: 8B is about 630 GiB
    "per-length-tables": (
        14_000_000_000,
        22_000_000_000,
    ),  # 34.5 B/token: about 24B fit
    "slot-v1": (
        22_500_000_000,
    ),  # it cannot index 35B: its arena addresses 8-entry lines with u32 indexes
    # the previous record's corpus and the record's
    "slot-ifma": (35_000_000_000, 37_000_000_000),
    "slot-balanced": (37_000_000_000,),  # the record's corpus: overview.record_build_s
}


def measured(versions: list[Version]) -> dict[Version, list[dict]]:
    """The measurements `search` has made so far, without making any, and those of EXTRA."""
    trials: dict[Version, list[dict]] = {v: [] for v in versions}
    for v in versions:
        while (n := next_tokens(v, trials[v])) is not None:
            spec = BuildTime(version=v, tokens=n)
            if spec.status != "done":
                break
            trials[v].append(spec.load_existing())
        for n in EXTRA.get(v, ()):
            if (spec := BuildTime(version=v, tokens=n)).status == "done":
                trials[v].append(spec.load_existing())
    return trials


def export(path: str) -> None:
    """The measurements so far as JSON at `path`, for the blog post's figure (artifacts/cache-build-measured-build.py)."""
    out = []
    for v, ts in measured(list(RUNGS)).items():
        rung = RUNGS[v]
        out.append(
            {
                "version": v,
                "name": rung.name,
                "what": rung.what,
                "done": next_tokens(v, ts) is None,
                "tokens_in_budget": tokens_in_budget(ts) if ts else None,
                "trials": [
                    {k: t[k] for k in ("tokens", "seconds", "median", "node")}
                    | {
                        "stats": t["runs"][0]["stats"],
                        "peak_rss_gib": max(r["peak_rss_gib"] for r in t["runs"]),
                    }
                    for t in ts
                ],
            }
        )
    with open(path, "w") as f:
        json.dump({"budget": BUDGET, "threads": THREADS, "rungs": out}, f, indent=1)


if __name__ == "__main__":

    def _main() -> None:
        # A measurement process (BuildTime's): measure VERSION TOKENS CPUS FILE...
        if sys.argv[1:2] == ["measure"]:
            version, tokens, cpus, *files = sys.argv[2:]
            print(
                json.dumps(
                    measure(
                        cast(Version, version), int(tokens), json.loads(cpus), files
                    )
                )
            )
            return
        # Imported, not this file's own definitions: run as a file, they would be __main__.BuildTime, another spec.
        from retrieval_blog_post.ladder import RUNGS, export, search, tokens_in_budget

        if sys.argv[1:2] == ["export"]:  # export PATH
            export(sys.argv[2])
            return

        for v, ts in search(cast(list[Version], sys.argv[1:]) or list(RUNGS)).items():
            print(
                f"{v:18s} {tokens_in_budget(ts) / 1e9:8.3f}B tokens in {BUDGET} s  ({len(ts)} measurements)"
            )

    _main()
