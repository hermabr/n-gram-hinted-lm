"""The hints of the validation positions, shared by every experiment that adds exact-match retrieval to an older
record: per position, the next tokens that followed its context in the training shards, at the deepest matched level of
a hint variant, counted exactly by exact_match.kgram_counts (exact_match/src/kgram.rs has the definitions).

ValContextCounts counts every level of its `levels`, a pass over the shards per level. ValHints reads one variant's
hints off those counts. The cells are the record's (exact_match/src/lib.rs's row) with one length bucket per level of
`buckets`, so variants with the same buckets share cells, and the model's tables do not depend on the variant. The
record's SlotIndex differs from these exact counts in its drops (~3% of the entries at 35B), its hash collisions, and
in starting a segment at every validation batch.

    uv run src/retrieval_blog_post/val_hints.py   # VARIANTS' hints on Slurm workers
"""

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import furu
import numpy as np

from exact_match import COUNTS, PURITIES, kgram_counts
from retrieval_blog_post.fineweb import BOS, TRAIN_SHARDS, FinewebShard, FinewebShards
from retrieval_blog_post.kgram import VAL_TOKENS
from retrieval_blog_post.ladder import WORKER_GIB

# The levels (context lengths in tokens) of the first variants, and the default length buckets of the cells.
LEVELS = (2, 3, 4, 5, 6, 10, 20)
ALL_LEVELS = (2, 3, 4, 5, 6, 8, 10, 12, 16, 20)
VARIANTS = ((6, 10, 20), (2, 3, 6, 10, 20), LEVELS)


def cells(buckets: tuple[int, ...] = LEVELS) -> int:
    """The record's cells with the length buckets `buckets`: no hint, then a length bucket per level, by bins of the
    total count and of the top token's share (a cell is ((1 + bucket) * 7 + count bin) * 6 + share bin)."""
    return (len(buckets) + 1) * (len(COUNTS) + 1) * (len(PURITIES) + 1)


# Next tokens kept per context, the most frequent: far more than a record's rows hold (8 entries per position), and
# all of them for nearly every context of 6 or more tokens. A 2-context has up to ~50K.
TOP = 1024
NO_CONTEXT = np.iinfo(np.uint32).max  # kgram_counts's context id where a position's k-context does not fit its segment
ARRAYS = ("context", "total", "distinct", "offsets", "next", "next_counts")


def _ascending(levels: tuple[int, ...]) -> bool:
    return bool(levels) and levels[0] >= 1 and list(levels) == sorted(set(levels))


def check_variant(levels: tuple[int, ...], buckets: tuple[int, ...] = LEVELS, slots: int = 1) -> None:
    """Raise unless `levels` ascend and are of `buckets`, which ascend, and there is a slot at least."""
    if not _ascending(buckets):
        raise ValueError(f"buckets must ascend: {buckets}")
    if not _ascending(levels) or not set(levels) <= set(buckets):
        raise ValueError(f"levels must ascend and be of {buckets}: {levels}")
    if slots < 1:
        raise ValueError(f"slots must be at least 1: {slots}")


@dataclass(frozen=True)
class ContextCounts:
    """Per level k of `levels`, in directory/k{k}/: context.npy, each validation position's context id (uint32,
    NO_CONTEXT where its k-context does not fit its segment); per context id total.npy (uint64, its occurrences),
    distinct.npy (uint32, its distinct next tokens) and offsets.npy (uint64) into next.npy (uint16) and
    next_counts.npy (uint32): its TOP most frequent next tokens and their counts, count descending, then token
    ascending."""

    directory: Path
    levels: tuple[int, ...]

    def get(self, k: int, name: str) -> np.ndarray:
        """Array `name` of level `k`, mapped read-only."""
        return np.load(self.directory / f"k{k}" / f"{name}.npy", mmap_mode="r")


class ValContextCounts(furu.Spec[ContextCounts]):
    """ContextCounts of the validation shard's first VAL_TOKENS positions at every level of `levels` (ascending), in
    the first `shards` training shards. With a superset of LEVELS, the levels of LEVELS are symlinks to
    ValContextCounts(shards=shards)'s, and only the others are counted."""

    shards: int = TRAIN_SHARDS
    levels: tuple[int, ...] = LEVELS

    migrations = (furu.Added("levels", default=LEVELS),)

    def metadata(self) -> furu.Metadata:
        # The shards stay in the page cache across the levels' passes (205 GB at 1028 shards), next to the
        # occurrences (_memory(), and up to as much again in the per-thread buffers' slack) and two hit bitmaps
        # (26 GB at 1028 shards). Up to 370 shards fit kgram.py's 256 GB workers, as KGramMatchStats does.
        if self.shards <= 370:
            return furu.Metadata(requires=furu.Requires(cpus=furu.at_least(32), memory=furu.GiB(256), gpus=0))
        return furu.Metadata(requires=furu.Requires(cpus=furu.at_least(48), memory=furu.GiB(WORKER_GIB), gpus=0))

    def _memory(self) -> int:
        """Bytes of occurrences kgram_counts holds at once: more rounds over the corpus when smaller."""
        return (64 << 30) if self.shards <= 370 else (256 << 30)

    def _reused(self) -> bool:
        """Whether the levels of LEVELS are ValContextCounts(shards=shards)'s."""
        return self.levels != LEVELS and set(LEVELS) <= set(self.levels)

    def create(self) -> ContextCounts:
        if not _ascending(self.levels):
            raise ValueError(f"levels must ascend: {self.levels}")
        base = ValContextCounts(shards=self.shards).create() if self._reused() else None
        counted = [k for k in self.levels if base is None or k not in base.levels]
        train = FinewebShards(shards=self.shards).create()
        (val,) = FinewebShards(shards="val").create()
        files = [str(shard.path) for shard in train]
        tokens = np.ascontiguousarray(val.tokens[: VAL_TOKENS + 1])
        # Each level is saved in scratch, which a retry keeps, so a preempted run skips its finished levels. The
        # previous level's hits bitmap is not saved (13 GB at 1028 shards): without it a level probes every corpus
        # position, slower but counting the same.
        progress, hits = self.directory.scratch, None
        for k in counted:
            done = progress / f"k{k}"
            if done.exists():
                hits = None
                continue
            start = time.time()
            *arrays, hits = kgram_counts(files, tokens, k, TOP, self._memory(), None, hits)
            partial = progress / f"k{k}.partial"
            shutil.rmtree(partial, ignore_errors=True)
            partial.mkdir()
            for name, array in zip(ARRAYS, arrays):
                np.save(partial / f"{name}.npy", array)
            partial.rename(done)
            context, total, _, _, kept, _ = arrays
            valid = context != NO_CONTEXT
            matched = (total[context[valid]] > 0).sum()
            self.logger.info(
                f"k={k}: {len(total):,} contexts, {matched / valid.sum():.4f} of {valid.sum():,} valid positions "
                f"matched, {len(kept):,} next tokens kept, {time.time() - start:.0f} s"
            )
        for k in self.levels:
            target = self.directory.data / f"k{k}"
            if k in counted:
                (progress / f"k{k}").rename(target)
            else:
                assert base is not None
                if target.is_symlink():  # an earlier attempt's
                    target.unlink()
                target.symlink_to((base.directory / f"k{k}").resolve())
        return ContextCounts(self.directory.data, self.levels)


class QueryContextCounts(furu.Spec[ContextCounts]):
    """ContextCounts of the positions of the training shards `query` (1-based ids, ascending; each shard starts a
    segment) at every level of `levels`, in the first `corpus_shards` training shards, which must precede the query
    shards: a query position only counts occurrences earlier in the corpus. Each context keeps its `top` most frequent
    next tokens, and each level also has cnt.npy (uint32, per query position): the count of the position's own next
    token among its context's TOP most frequent, as the validation count rows count it (0 outside them, and at each
    query shard's last position)."""

    query: tuple[int, ...]
    corpus_shards: int
    levels: tuple[int, ...] = ALL_LEVELS
    top: int = 8

    def metadata(self) -> furu.Metadata:
        # the corpus in the page cache (205 GB at 1023 shards), _memory() of occurrences and up to as much again in
        # the threads' slack, and a query of up to 500M positions: its context table (13 GB), its positions by context
        # and the level's per-context arrays (~30 GB); less for a small corpus (tests)
        if self.corpus_shards <= 32:
            return furu.Metadata(requires=furu.Requires(cpus=furu.at_least(16), memory=furu.GiB(64), gpus=0))
        return furu.Metadata(requires=furu.Requires(cpus=furu.at_least(48), memory=furu.GiB(600), gpus=0))

    def _memory(self) -> int:
        """Bytes of occurrences kgram_query_counts holds at once (ValContextCounts._memory)."""
        return (128 << 30) if self.corpus_shards > 32 else (8 << 30)

    def create(self) -> ContextCounts:
        from exact_match import kgram_query_counts

        if not _ascending(self.levels) or not _ascending(self.query):
            raise ValueError(f"levels and query must ascend: {self.levels}, {self.query}")
        if not 1 <= self.corpus_shards < self.query[0] or self.query[-1] > TRAIN_SHARDS:
            raise ValueError(f"the corpus (shards 1-{self.corpus_shards}) must precede the query shards {self.query}")
        corpus = FinewebShards(shards=self.corpus_shards).create()
        query = furu.create([FinewebShard(shard=i) for i in self.query])
        files = [str(shard.path) for shard in corpus]
        lengths = [len(shard.tokens) for shard in query]
        ends = np.cumsum(lengths)
        tokens = np.concatenate([np.asarray(shard.tokens) for shard in query] + [np.array([BOS], np.uint16)])
        progress, hits = self.directory.scratch, None
        for k in self.levels:
            done = progress / f"k{k}"
            if done.exists():  # an earlier attempt's (ValContextCounts.create)
                hits = None
                continue
            start = time.time()
            *arrays, hits, cnt = kgram_query_counts(
                files, tokens, k, self.top, TOP, self._memory(), [int(e) for e in ends[:-1]], hits
            )
            cnt[ends - 1] = 0  # a query shard's last position: its next token is the next shard's
            partial = progress / f"k{k}.partial"
            shutil.rmtree(partial, ignore_errors=True)
            partial.mkdir()
            for name, array in zip(ARRAYS, arrays):
                np.save(partial / f"{name}.npy", array)
            np.save(partial / "cnt.npy", cnt)
            partial.rename(done)
            context, total, _, _, kept, _ = arrays
            valid = context != NO_CONTEXT
            matched = (total[context[valid]] > 0).sum()
            self.logger.info(
                f"k={k}: {len(total):,} contexts, {matched / valid.sum():.4f} of {valid.sum():,} valid positions "
                f"matched, {(cnt > 0).mean():.4f} of positions with the target, {len(kept):,} next tokens kept, "
                f"{time.time() - start:.0f} s"
            )
            del arrays, context, total, kept, valid, cnt
        for k in self.levels:
            (progress / f"k{k}").rename(self.directory.data / f"k{k}")
        (self.directory.data / "query.json").write_text(json.dumps({"query": list(self.query), "lengths": lengths}))
        return ContextCounts(self.directory.data, self.levels)


@dataclass(frozen=True)
class Hints:
    """Per validation position, in directory/: level.npy (int8, the index in `buckets` of its hint's level, the
    deepest of the variant's `levels` whose context occurred; -1 without a hint), context.npy (int32, the context id
    at that level in `counts`; -1 without a hint) and cell.npy (int32, its cell of cells(buckets); 0 without a hint).
    With slots, each is [n, slots]: per slot, the slot-th deepest level whose context occurred (empty slots as without
    a hint); else [n]."""

    directory: Path
    levels: tuple[int, ...]
    counts: ContextCounts

    @property
    def buckets(self) -> tuple[int, ...]:
        """The levels whose index is a cell's length bucket (LEVELS in the hints made before buckets.json)."""
        path = self.directory / "buckets.json"
        return tuple(json.loads(path.read_text())) if path.exists() else LEVELS

    @property
    def slots(self) -> int:
        level = self.get("level")
        return 1 if level.ndim == 1 else level.shape[1]

    def get(self, name: str) -> np.ndarray:
        """Array `name`, mapped read-only."""
        return np.load(self.directory / f"{name}.npy", mmap_mode="r")

    def candidates(
        self, start: int, stop: int, top: int = TOP, slot: int = 0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The candidates of positions [start, stop) in slot `slot`, each position's `top` most frequent next tokens
        at its hint's level: per candidate its position (from start), next token and count, by position, then count
        descending."""
        level, context = self.get("level")[start:stop], self.get("context")[start:stop]
        if level.ndim == 2:
            level, context = level[:, slot], context[:, slot]
        elif slot != 0:
            raise ValueError(f"slot {slot} of hints with one slot")
        parts = []
        for i, k in enumerate(self.buckets):
            at = np.flatnonzero(level == i)
            if len(at) == 0:
                continue
            offsets = self.counts.get(k, "offsets")
            lo, hi = offsets[context[at]], offsets[context[at] + 1]
            n = np.minimum(hi - lo, top).astype(np.int64)
            # entry j of position a is lo[a] + j
            first = np.repeat(np.cumsum(n) - n, n)
            index = np.repeat(lo.astype(np.int64), n) + np.arange(n.sum()) - first
            next_tokens, next_counts = self.counts.get(k, "next")[index], self.counts.get(k, "next_counts")[index]
            parts.append((np.repeat(at, n), next_tokens, next_counts))
        if not parts:
            return np.zeros(0, np.int64), np.zeros(0, np.uint16), np.zeros(0, np.uint32)
        at, tokens, counts = map(np.concatenate, zip(*parts))
        order = np.argsort(at, kind="stable")  # a position's candidates are of one level, already by count
        return at[order], tokens[order], counts[order]


def deepest(
    counts: ContextCounts,
    levels: tuple[int, ...],
    positions: int,
    slots: int = 1,
    buckets: tuple[int, ...] = LEVELS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hints.level, Hints.context and Hints.cell of the first `positions` positions in `counts`, for the variant
    `levels` with the length buckets `buckets`: per position its `slots` deepest levels whose context occurred, deepest
    first ([positions, slots]; [positions] with one slot)."""
    level = np.full((positions, slots), -1, np.int8)
    context = np.full((positions, slots), -1, np.int32)
    total = np.zeros((positions, slots), np.uint64)
    top = np.zeros((positions, slots), np.uint64)
    filled = np.zeros(positions, np.int64)  # per position, its slots filled so far
    for k in reversed(levels):  # deepest first: a level that matched fills its position's first empty slot
        ids = np.asarray(counts.get(k, "context")[:positions])
        totals = counts.get(k, "total")
        matched = ids != NO_CONTEXT
        matched[matched] = totals[ids[matched]] > 0
        at = np.flatnonzero(matched & (filled < slots))
        found, slot = ids[at], filled[at]
        level[at, slot] = buckets.index(k)
        context[at, slot] = found
        total[at, slot] = totals[found]
        top[at, slot] = counts.get(k, "next_counts")[counts.get(k, "offsets")[found]]
        filled[at] += 1
    # lib.rs's bins: the edges at most the total count, and at most the top token's share of it
    count = np.searchsorted(COUNTS, total, side="right")
    purity = np.searchsorted(PURITIES, top / np.maximum(total, 1), side="right")
    cell = ((1 + level.astype(np.int64)) * (len(COUNTS) + 1) + count) * (len(PURITIES) + 1) + purity
    cell = np.where(level >= 0, cell, 0).astype(np.int32)
    if slots == 1:
        return level[:, 0], context[:, 0], cell[:, 0]
    return level, context, cell


def counted_levels(levels: tuple[int, ...]) -> tuple[int, ...]:
    """The levels of the ValContextCounts a variant's hints read: LEVELS if they cover the variant, else
    ALL_LEVELS."""
    if set(levels) <= set(LEVELS):
        return LEVELS
    if set(levels) <= set(ALL_LEVELS):
        return ALL_LEVELS
    raise ValueError(f"levels must be of {ALL_LEVELS}: {levels}")


class ValHints(furu.Spec[Hints]):
    """The Hints of the variant `levels` (ascending, of `buckets` and of ALL_LEVELS) in `slots` slots, with the length
    buckets `buckets`, from the counts in the first `shards` training shards."""

    levels: tuple[int, ...]
    shards: int = TRAIN_SHARDS
    slots: int = 1
    buckets: tuple[int, ...] = LEVELS

    migrations = (furu.Added("slots", default=1), furu.Added("buckets", default=LEVELS))

    def metadata(self) -> furu.Metadata:
        return furu.Metadata(requires=furu.Requires(memory=furu.GiB(32), gpus=0))

    def create(self) -> Hints:
        check_variant(self.levels, self.buckets, self.slots)
        counts = ValContextCounts(shards=self.shards, levels=counted_levels(self.levels)).create()
        level, context, cell = deepest(counts, self.levels, VAL_TOKENS, self.slots, self.buckets)
        first = level if level.ndim == 1 else level[:, 0]
        for k in self.levels:
            share = (first == self.buckets.index(k)).mean()
            self.logger.info(f"hint at k={k}: {share:.4f} of positions")
        if level.ndim == 2:
            filled = np.bincount((level >= 0).sum(1), minlength=self.slots + 1)
            self.logger.info(f"slots filled: {', '.join(f'{s}: {f / len(level):.4f}' for s, f in enumerate(filled))}")
        for name, array in [("level", level), ("context", context), ("cell", cell)]:
            np.save(self.directory.data / f"{name}.npy", array)
        (self.directory.data / "buckets.json").write_text(json.dumps(self.buckets))
        return Hints(self.directory.data, self.levels, counts)


if __name__ == "__main__":

    def _main() -> None:
        # Imported, not this file's own definitions: run as a file, they would be __main__.ValHints, another spec.
        from retrieval_blog_post.config import create
        from retrieval_blog_post.val_hints import VARIANTS, ValHints

        for hints in create([ValHints(levels=v) for v in VARIANTS], job_name="val-hints"):
            level = hints.get("level")
            level = level if level.ndim == 1 else level[:, 0]
            shares = ", ".join(f"{k}: {(level == hints.buckets.index(k)).mean():.3f}" for k in hints.levels)
            print(f"{hints.levels}: {(level >= 0).mean():.4f} of positions hinted ({shares})")

    _main()
