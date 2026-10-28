"""The oldest speedrun record that reaches 3.28 with exact-match hints in #360's time (old_records/README.md): the
hint rows the ported records read, and their runs on a miso node's 8 GPUs.

ValHintRows and TrainHintRows are a variant's validation rows (val_hints.py, counted over all 1028 shards) and its
training rows (causal in the corpus order): per position a cell and the top ENTRIES candidates with weights.
OldRecordRun runs one port (old_records/rNNN_*/, or record_360/ for #360) once. Run BATCH with

    uv run src/retrieval_blog_post/old_records.py      # keep it running (e.g. in tmux): it coordinates the workers

To add runs to a running batch, edit BATCH and relaunch with FURU_TAKEOVER=<the running execution's id prefix>: the
new coordinator inherits the workers and cancels the running runs BATCH no longer lists."""

import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import furu
import numpy as np
from furu.resources import ResourceFloor
from furu.worker.backends.slurm.backend import SlurmWorkerBackend

from retrieval_blog_post.config import LOCAL_DISK, SCRATCH, TORCH_VENV
from retrieval_blog_post.fineweb import TRAIN_SHARDS, FinewebShard, FinewebShards, Shard
from retrieval_blog_post.kgram import RECORD_360, SHARD_TOKENS, VAL_TOKENS
from retrieval_blog_post.ladder import WORKER_GIB
from retrieval_blog_post.config import cpu, gpu_node, shared_cpu
from retrieval_blog_post.val_cache import REPO
from retrieval_blog_post.val_hints import (
    ALL_LEVELS,
    LEVELS,
    VARIANTS,
    ContextCounts,
    Hints,
    ValHints,
    cells,
    check_variant,
)

OLD = REPO / "old_records"
ENTRIES = 8  # candidates per position's row: the record's validation rows average 4.9 at 35B
WEIGHT_ONE = 65535  # hint_runtime.py's
VOCAB = 50257  # GPT-2's tokens
# Training shards with hint rows: #360 trains on 329M tokens in its 39.9 s, the fastest of the records; 8 shards cover
# that with any loader's skipped document tails.
TRAIN_HINT_SHARDS = 8
# The least training shards with hint rows for variants with slots or other buckets: the early records read up to 3.
SLOT_HINT_SHARDS = 4
RECORD_360_SHARDS = math.ceil(1.35 * RECORD_360[1] / SHARD_TOKENS) + 1


@dataclass(frozen=True)
class HintRowFiles:
    """Rows in `directory`: cell.npy, tokens.npy and weights.npy (hint_runtime.py), directly (validation) or in a
    subdirectory NNNN/ per training shard."""

    directory: Path
    levels: tuple[int, ...]


def _weights(counts: np.ndarray) -> np.ndarray:
    """Counts [..., ENTRIES] as weights summing to WEIGHT_ONE per row (all 0 for empty rows)."""
    total = counts.sum(-1, keepdims=True, dtype=np.float64)
    return np.round(counts / np.maximum(total, 1) * WEIGHT_ONE).astype(np.uint16)


def _train_hint_bytes(shards: int, slots: int) -> int:
    """Bytes exact_match.train_hints holds over `shards` shards (exact_match/src/train_hints.rs: 39 + slots *
    (3 + 4 * ENTRIES) per position): 55 GiB at 8 shards with one slot, 54 GiB at 4 with 3."""
    return shards * SHARD_TOKENS * (39 + slots * (3 + 4 * ENTRIES))


def _shares(
    level: np.ndarray, levels: tuple[int, ...], buckets: tuple[int, ...]
) -> str:
    """The share of positions whose (first slot's) hint is at each level of `levels`, and the slots filled."""
    first = level if level.ndim == 1 else level[:, 0]
    shares = ", ".join(f"{k}: {(first == buckets.index(k)).mean():.4f}" for k in levels)
    if level.ndim == 1:
        return shares
    filled = np.bincount((level >= 0).sum(1), minlength=level.shape[1] + 1) / len(level)
    return f"{shares}; slots filled {', '.join(f'{s}: {f:.4f}' for s, f in enumerate(filled))}"


def _val_rows(
    hints: Hints, positions: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The first `positions` positions' rows: cell (int16), the top ENTRIES candidates' tokens (uint16) and counts
    (uint64, 0 for padding); [n] and [n, ENTRIES] with one slot, else [n, slots] and [n, slots, ENTRIES]."""
    return _val_rows_range(hints, 0, positions)


def _val_rows_range(
    hints: Hints, start: int, stop: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """_val_rows of positions [start, stop) of `hints`."""
    positions = stop - start
    slots = hints.slots
    tokens = np.zeros((positions, slots, ENTRIES), np.uint16)
    counts = np.zeros((positions, slots, ENTRIES), np.uint64)
    for s in range(slots):
        at, next_tokens, next_counts = hints.candidates(
            start, stop, top=ENTRIES, slot=s
        )
        rank = np.arange(len(at)) - np.searchsorted(
            at, at, side="left"
        )  # rank within its position's candidates
        tokens[at, s, rank] = next_tokens
        counts[at, s, rank] = next_counts
    cell = np.asarray(hints.get("cell")[start:stop]).astype(np.int16)
    if not np.array_equal(cell.reshape(positions, slots) > 0, counts[..., 0] > 0):
        raise RuntimeError("positions with a cell and positions with candidates differ")
    if slots == 1:  # no slot axis
        return cell, tokens[:, 0], counts[:, 0]
    return cell, tokens, counts


class ValHintRows(furu.Spec[HintRowFiles]):
    """The validation rows of the variant `levels` in `slots` slots, with the length buckets `buckets`: per position
    (and slot) its ValHints cell and top ENTRIES candidates, in _val_rows' shapes."""

    levels: tuple[int, ...]
    slots: int = 1
    buckets: tuple[int, ...] = LEVELS

    migrations = (
        furu.Added("slots", default=1),
        furu.Added("buckets", default=LEVELS),
    )  # the rows made before there were slots and other buckets

    def metadata(self) -> furu.Metadata:
        return furu.Metadata(requires=furu.Requires(memory=furu.GiB(48), gpus=0))

    def create(self) -> HintRowFiles:
        check_variant(self.levels, self.buckets, self.slots)
        hints = ValHints(
            levels=self.levels, slots=self.slots, buckets=self.buckets
        ).create()
        cell, tokens, counts = _val_rows(hints, VAL_TOKENS)
        out = self.directory.data
        np.save(out / "cell.npy", cell)
        np.save(out / "tokens.npy", tokens)
        np.save(out / "weights.npy", _weights(counts))
        hinted = (cell.reshape(len(cell), -1)[:, 0] > 0).mean()
        self.logger.info(
            f"{hinted:.4f} of positions hinted, {(counts > 0).sum(-1).mean():.2f} candidates per row"
        )
        return HintRowFiles(out, self.levels)


class TrainHintRows(furu.Spec[HintRowFiles]):
    """ValHintRows' rows for the first `shards` training shards, counted only over the occurrences that end before
    the position's document starts (exact_match.train_hints)."""

    levels: tuple[int, ...]
    shards: int = TRAIN_HINT_SHARDS
    slots: int = 1
    buckets: tuple[int, ...] = LEVELS

    migrations = (
        furu.Added("slots", default=1),
        furu.Added("buckets", default=LEVELS),
    )  # the rows made before there were slots and other buckets

    def metadata(self) -> furu.Metadata:
        # _train_hint_bytes() with room to spare: 160 GiB up to 64 GiB of it (8 shards with one slot, 4 with 3)
        need = _train_hint_bytes(self.shards, self.slots) / 2**30
        return furu.Metadata(
            requires=furu.Requires(
                cpus=furu.at_least(32),
                memory=furu.GiB(max(160, math.ceil(2.5 * need))),
                gpus=0,
            )
        )

    def create(self) -> HintRowFiles:
        from exact_match import train_hints

        check_variant(self.levels, self.buckets, self.slots)
        shards = FinewebShards(shards=self.shards).create()
        start = time.time()
        cell, tokens, weights, level = train_hints(
            [str(s.path) for s in shards],
            list(self.levels),
            ENTRIES,
            buckets=list(self.buckets),
            slots=self.slots,
        )
        self.logger.info(f"train_hints: {time.time() - start:.0f} s")
        hinted = (level >= 0).mean() if level.ndim == 1 else (level[:, 0] >= 0).mean()
        self.logger.info(
            f"{hinted:.4f} of positions hinted ({_shares(level, self.levels, self.buckets)})"
        )
        offset = 0
        for j, shard in enumerate(shards):
            n = len(shard.tokens)
            out = self.directory.data / f"{j:04d}"
            out.mkdir()
            np.save(out / "cell.npy", cell[offset : offset + n])
            np.save(out / "tokens.npy", tokens[offset : offset + n])
            np.save(out / "weights.npy", weights[offset : offset + n])
            offset += n
        if offset != len(cell):
            raise RuntimeError(
                f"train_hints returned {len(cell)} rows for {offset} positions"
            )
        return HintRowFiles(self.directory.data, self.levels)


# The n-gram count rows (hint_runtime.py's NgramMix): per position and level its context's total count, distinct next
# tokens and the count of its own next token. #1 at 250 steps reads 131M tokens: 2 shards.
COUNT_LEVELS = ALL_LEVELS
COUNT_SHARDS = 2
COUNT_NAMES = ("tot", "dist", "cnt")


@dataclass(frozen=True)
class CountRowFiles:
    """Count rows in `directory`: tot.npy (uint32 [n, L]), dist.npy (uint16 [n, L]) and cnt.npy (uint32 [n, L]) per
    level of `levels` (ascending), directly (validation) or in a subdirectory NNNN/ per training shard; the training
    rows also have unigram.npy (float32 [VOCAB], the first training shard's token frequencies, + 0.5, normalized)."""

    directory: Path
    levels: tuple[int, ...]


def val_count_features(
    counts, targets: np.ndarray, positions: np.ndarray, levels: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """tot, dist and cnt ([len(positions), L]: uint32, uint16, uint32) of the validation `positions` at each level of
    `levels`, from val_hints.ContextCounts `counts`: the position's k-context's total count, its distinct next tokens,
    and the count of the position's target (`targets`, indexed by position) among its context's TOP most frequent next
    tokens (0 if outside them: then the n-gram is a sub-normalized distribution, never an over-normalized one). All 0
    where the k-context does not fit its segment or never occurred."""
    from retrieval_blog_post.val_hints import NO_CONTEXT

    n, nl = len(positions), len(levels)
    tot = np.zeros((n, nl), np.uint32)
    dist = np.zeros((n, nl), np.uint16)
    cnt = np.zeros((n, nl), np.uint32)
    target = np.asarray(targets[positions]).astype(np.int64)
    for j, k in enumerate(levels):
        ctx = np.asarray(counts.get(k, "context")[positions])
        total, distinct = counts.get(k, "total"), counts.get(k, "distinct")
        offsets = counts.get(k, "offsets")
        nxt, ncount = counts.get(k, "next"), counts.get(k, "next_counts")
        at = np.flatnonzero(ctx != NO_CONTEXT)
        c = ctx[at].astype(np.int64)
        t = np.asarray(total[c])
        if t.max(initial=0) > np.iinfo(np.uint32).max:
            raise ValueError(f"k={k}: a total of {t.max()} does not fit uint32")
        tot[at, j] = t
        dist[at, j] = np.minimum(np.asarray(distinct[c]), np.iinfo(np.uint16).max)
        # the target's count: the entries of the contexts used, keyed (context rank, token) and sorted, in batches
        # of about 200M entries
        used = np.unique(c[t > 0])
        lo = np.asarray(offsets[used]).astype(np.int64)
        lens = np.asarray(offsets[used + 1]).astype(np.int64) - lo
        ends = np.cumsum(lens)
        rank = np.searchsorted(used, c)  # each query's context's rank in used (where its total is > 0)
        a = 0
        while a < len(used):
            b = int(np.searchsorted(ends, (ends[a - 1] if a else 0) + 200_000_000, side="right"))
            b = max(b, a + 1)
            ln, first = lens[a:b], lo[a:b]
            m = int(ln.sum())
            if b - a == 1 or (np.diff(first) == ln[:-1]).all():  # contiguous: one slice
                tokens = np.asarray(nxt[int(first[0]) : int(first[0]) + m]).astype(np.int64)
                values = np.asarray(ncount[int(first[0]) : int(first[0]) + m])
            else:
                index = np.repeat(first - (np.cumsum(ln) - ln), ln) + np.arange(m, dtype=np.int64)
                tokens = np.asarray(nxt[index]).astype(np.int64)
                values = np.asarray(ncount[index])
                del index
            keys = np.repeat(np.arange(a, b, dtype=np.int64), ln) * 65536 + tokens
            del tokens
            order = np.argsort(keys)
            keys = keys[order]
            sel = np.flatnonzero((rank >= a) & (rank < b) & (t > 0))
            qk = rank[sel] * 65536 + target[at[sel]]
            ix = np.minimum(np.searchsorted(keys, qk), len(keys) - 1)
            hit = keys[ix] == qk
            cnt[at[sel[hit]], j] = values[order[ix[hit]]]
            a = b
    return tot, dist, cnt


class ValCountRows(furu.Spec[CountRowFiles]):
    """The validation count rows (val_count_features) per level of `levels`, counted over all 1028 training shards."""

    levels: tuple[int, ...] = COUNT_LEVELS

    def metadata(self) -> furu.Metadata:
        # 200M-entry batches of int64 keys, their order and values (~7 GB), and the [n, L] outputs (1 GB)
        return furu.Metadata(requires=furu.Requires(cpus=furu.at_least(8), memory=furu.GiB(48), gpus=0))

    def create(self) -> CountRowFiles:
        from retrieval_blog_post.val_hints import ValContextCounts, counted_levels

        check_variant(self.levels, ALL_LEVELS)
        counts = ValContextCounts(levels=counted_levels(self.levels)).create()
        (val,) = FinewebShards(shards="val").create()
        targets = np.asarray(val.tokens[1 : VAL_TOKENS + 1])
        start = time.time()
        arrays = val_count_features(counts, targets, np.arange(VAL_TOKENS), self.levels)
        self.logger.info(f"val_count_features: {time.time() - start:.0f} s")
        tot, _, cnt = arrays
        for j, k in enumerate(self.levels):
            self.logger.info(
                f"k={k}: {(tot[:, j] > 0).mean():.4f} of positions matched, {(cnt[:, j] > 0).mean():.4f} with the target"
            )
        for name, array in zip(COUNT_NAMES, arrays):
            np.save(self.directory.data / f"{name}.npy", array)
        return CountRowFiles(self.directory.data, self.levels)


def _train_count_bytes(shards: int, levels: int) -> int:
    """Bytes exact_match.train_counts holds over `shards` shards: 34 per position and 10 per position and level."""
    return shards * SHARD_TOKENS * (34 + 10 * levels)


class TrainCountRows(furu.Spec[CountRowFiles]):
    """The count rows of the first `shards` training shards, causal as TrainHintRows' (exact_match.train_counts), and
    the unigram base."""

    levels: tuple[int, ...] = COUNT_LEVELS
    shards: int = COUNT_SHARDS

    def metadata(self) -> furu.Metadata:
        need = _train_count_bytes(self.shards, len(self.levels)) / 2**30
        return furu.Metadata(
            requires=furu.Requires(
                cpus=furu.at_least(32), memory=furu.GiB(max(128, math.ceil(2.5 * need))), gpus=0
            )
        )

    def create(self) -> CountRowFiles:
        from exact_match import train_counts

        check_variant(self.levels, ALL_LEVELS)
        shards = FinewebShards(shards=self.shards).create()
        start = time.time()
        tot, dist, cnt = train_counts([str(s.path) for s in shards], list(self.levels))
        self.logger.info(f"train_counts: {time.time() - start:.0f} s")
        for j, k in enumerate(self.levels):
            self.logger.info(
                f"k={k}: {(tot[:, j] > 0).mean():.4f} of positions matched, {(cnt[:, j] > 0).mean():.4f} with the target"
            )
        offset = 0
        for j, shard in enumerate(shards):
            n = len(shard.tokens)
            out = self.directory.data / f"{j:04d}"
            out.mkdir()
            for name, array in zip(COUNT_NAMES, (tot, dist, cnt)):
                np.save(out / f"{name}.npy", array[offset : offset + n])
            offset += n
        if offset != len(tot):
            raise RuntimeError(f"train_counts returned {len(tot)} rows for {offset} positions")
        frequency = np.bincount(np.asarray(shards[0].tokens), minlength=VOCAB)[:VOCAB] + 0.5
        np.save(self.directory.data / "unigram.npy", (frequency / frequency.sum()).astype(np.float32))
        return CountRowFiles(self.directory.data, self.levels)


# Training rows as strong as the validation rows, still causal: the run trains on the last shards, LATE_FIRST to
# TRAIN_SHARDS (the port's shard j is LATE_FIRST + j), with rows counted over shards 1 to LATE_FIRST - 1. Occurrences
# earlier in the late shards themselves are left out (0.3% of the corpus). 5 shards, 474M tokens: 901 of #1's steps.
LATE_FIRST = 1024
LATE_SHARDS = TRAIN_SHARDS - LATE_FIRST + 1


def _late_counts(first: int, last: int = TRAIN_SHARDS) -> ContextCounts:
    """The context counts of training shards first..last over shards 1..first - 1."""
    from retrieval_blog_post.val_hints import QueryContextCounts

    return QueryContextCounts(
        query=tuple(range(first, last + 1)), corpus_shards=first - 1, levels=ALL_LEVELS
    ).create()


def _late_shards(first: int, last: int = TRAIN_SHARDS) -> list[Shard]:
    return furu.create([FinewebShard(shard=i) for i in range(first, last + 1)])


def _no_huge_pages() -> None:
    """No transparent huge pages: on a worker with fragmented memory every fault on numpy's madvised allocations
    compacts memory, which made the rows' array work ~100x slower."""
    import ctypes

    from numpy._core.multiarray import _set_madvise_hugepage

    _set_madvise_hugepage(False)
    ctypes.CDLL(None, use_errno=True).prctl(41, 1, 0, 0, 0)  # PR_SET_THP_DISABLE


@dataclass(frozen=True)
class _LoadedCounts(ContextCounts):
    """ContextCounts whose arrays are read whole, not mapped: page faults through mappings of these GB-sized arrays
    cost ~20x the reads. The arrays `cached` are kept once read."""

    cached: tuple[str, ...] = ()
    cache: dict = field(default_factory=dict, compare=False)

    def get(self, k: int, name: str) -> np.ndarray:
        if (k, name) in self.cache:
            return self.cache[k, name]
        array = np.load(self.directory / f"k{k}" / f"{name}.npy")
        if name in self.cached:
            self.cache[k, name] = array
        return array


@dataclass(frozen=True)
class _LoadedHints(Hints):
    """Hints whose level, context and cell arrays are in memory (`arrays`), not mapped from files."""

    arrays: dict = field(default_factory=dict, compare=False)

    def get(self, name: str) -> np.ndarray:
        return self.arrays[name]


class LateCountRows(furu.Spec[CountRowFiles]):
    """The count rows of training shards `first`..`last` (in NNNN/, j = shard - first), counted over shards
    1..first - 1 as val_count_features counts, and the first late shard's unigram base."""

    levels: tuple[int, ...] = COUNT_LEVELS
    first: int = LATE_FIRST
    last: int = TRAIN_SHARDS  # a smaller one for tests

    def metadata(self) -> furu.Metadata:
        # the [n, L] outputs (27 GB at 274M positions) and a level's per-position temporaries
        return furu.Metadata(requires=furu.Requires(cpus=furu.at_least(16), memory=furu.GiB(128), gpus=0))

    def create(self) -> CountRowFiles:
        check_variant(self.levels, ALL_LEVELS)
        _no_huge_pages()
        late = _late_counts(self.first, self.last)
        counts = _LoadedCounts(late.directory, late.levels)
        shards = _late_shards(self.first, self.last)
        lengths = [len(s.tokens) for s in shards]
        for j, n in enumerate(lengths):
            (self.directory.data / f"{j:04d}").mkdir(exist_ok=True)  # exist_ok: an earlier attempt's
        nl = len(self.levels)
        outs = [  # per shard, the [n, L] arrays, filled level by level
            {name: np.zeros((n, nl), dtype) for name, dtype in zip(COUNT_NAMES, (np.uint32, np.uint16, np.uint32))}
            for n in lengths
        ]
        from retrieval_blog_post.val_hints import NO_CONTEXT

        for j, k in enumerate(self.levels):
            ctx = np.asarray(counts.get(k, "context"))
            valid = ctx != NO_CONTEXT
            c = np.where(valid, ctx, 0).astype(np.int64)
            total = np.asarray(counts.get(k, "total"))
            if total.max(initial=0) > np.iinfo(np.uint32).max:
                raise ValueError(f"k={k}: a total of {total.max()} does not fit uint32")
            tot = np.where(valid, total[c], 0).astype(np.uint32)
            dist = np.where(valid, np.minimum(np.asarray(counts.get(k, "distinct"))[c], 65535), 0).astype(np.uint16)
            cnt = np.asarray(counts.get(k, "cnt"))
            self.logger.info(
                f"k={k}: {(tot > 0).mean():.4f} of positions matched, {(cnt > 0).mean():.4f} with the target"
            )
            offset = 0
            for out, n in zip(outs, lengths):
                out["tot"][:, j] = tot[offset : offset + n]
                out["dist"][:, j] = dist[offset : offset + n]
                out["cnt"][:, j] = cnt[offset : offset + n]
                offset += n
            if offset != len(ctx):
                raise RuntimeError(f"{len(ctx)} positions counted for {offset} in the late shards")
            del ctx, valid, c, tot, dist, cnt, total
        for j, out in enumerate(outs):
            for name in COUNT_NAMES:
                np.save(self.directory.data / f"{j:04d}/{name}.npy", out[name])
        frequency = np.bincount(np.asarray(shards[0].tokens), minlength=VOCAB)[:VOCAB] + 0.5
        np.save(self.directory.data / "unigram.npy", (frequency / frequency.sum()).astype(np.float32))
        return CountRowFiles(self.directory.data, self.levels)


class LateHintRows(furu.Spec[HintRowFiles]):
    """ValHintRows' rows of training shards `first`..`last`, counted over shards 1..first - 1, in TrainHintRows'
    layout."""

    levels: tuple[int, ...]
    slots: int = 1
    buckets: tuple[int, ...] = LEVELS
    first: int = LATE_FIRST
    last: int = TRAIN_SHARDS  # a smaller one for tests

    def metadata(self) -> furu.Metadata:
        return furu.Metadata(requires=furu.Requires(cpus=furu.at_least(16), memory=furu.GiB(128), gpus=0))

    def create(self) -> HintRowFiles:
        check_variant(self.levels, self.buckets, self.slots)
        if not set(self.levels) <= set(ALL_LEVELS):
            raise ValueError(f"levels must be of {ALL_LEVELS}: {self.levels}")
        _no_huge_pages()
        LOCAL_DISK.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=LOCAL_DISK) as local:
            late = _late_counts(self.first, self.last)
            # the candidates' arrays are read per chunk of positions: kept (~5 GB per level)
            counts = _LoadedCounts(late.directory, late.levels, cached=("offsets", "next", "next_counts"))
            return self._rows(counts, Path(local) / "hints")

    def _rows(self, counts: ContextCounts, hints_dir: Path) -> HintRowFiles:
        from retrieval_blog_post.val_hints import deepest

        lengths = [len(s.tokens) for s in _late_shards(self.first, self.last)]
        positions = sum(lengths)
        level, context, cell = deepest(counts, self.levels, positions, self.slots, self.buckets)
        self.logger.info(
            f"{(level.reshape(positions, -1)[:, 0] >= 0).mean():.4f} of positions hinted "
            f"({_shares(level, self.levels, self.buckets)})"
        )
        hints_dir.mkdir()
        (hints_dir / "buckets.json").write_text(json.dumps(self.buckets))
        hints = _LoadedHints(
            hints_dir, self.levels, counts, arrays={"level": level, "context": context, "cell": cell}
        )
        del level, context, cell
        offset = 0
        for j, n in enumerate(lengths):
            out = self.directory.data / f"{j:04d}"
            out.mkdir(exist_ok=True)  # an earlier attempt's
            cell = np.zeros((n, self.slots) if self.slots > 1 else (n,), np.int16)
            tokens = np.zeros((n, self.slots, ENTRIES) if self.slots > 1 else (n, ENTRIES), np.uint16)
            weights = np.zeros_like(tokens)
            chunk = 1 << 24
            for a in range(0, n, chunk):
                b = min(n, a + chunk)
                c, t, w = _val_rows_range(hints, offset + a, offset + b)
                cell[a:b], tokens[a:b], weights[a:b] = c, t, _weights(w)
            np.save(out / "cell.npy", cell)
            np.save(out / "tokens.npy", tokens)
            np.save(out / "weights.npy", weights)
            offset += n
        return HintRowFiles(self.directory.data, self.levels)


SUBSET_SHARES = {"noexact": 0.9248712539672852, "nonear": 0.8804489135742187}  # the dedup analysis' token shares


@dataclass(frozen=True)
class SubsetLabelFiles:
    """labels.npy in `directory` (ValSubsetLabels), and each subset's share of the validation targets."""

    directory: Path
    shares: dict[str, float]


class ValSubsetLabels(furu.Spec[SubsetLabelFiles]):
    """Each validation target's subset label (uint8 [VAL_TOKENS], hint_runtime.ValSubsets), by its document
    (ValOverlap): 2 without an exact copy in the training shards and a best 13-word n-gram Jaccard < `jaccard_max`
    with every training document, 1 with a near-copy only, 0 with an exact copy or outside every document's body. The
    subsets "noexact" (label >= 1) and "nonear" (label 2) are 92.49% and 88.04% of the targets."""

    jaccard_max: float = 0.8

    def create(self) -> SubsetLabelFiles:
        from retrieval_blog_post.fineweb import ValOverlap

        overlap = ValOverlap().create()
        starts, lengths = np.asarray(overlap.starts, np.int64), np.asarray(overlap.lengths, np.int64)
        exact, jaccard = np.asarray(overlap.exact), np.asarray(overlap.jaccard)
        target = np.arange(1, VAL_TOKENS + 1)  # row i predicts val token i + 1
        doc = np.searchsorted(starts, target, side="right") - 1
        d = np.maximum(doc, 0)
        body = (doc >= 0) & (target < starts[d] + lengths[d])
        noexact = body & (exact[d] == 0)
        labels = noexact.astype(np.uint8) + (noexact & (jaccard[d] < self.jaccard_max))
        out = self.directory.data
        np.save(out / "labels.npy", labels)
        shares = {"noexact": float((labels >= 1).mean()), "nonear": float((labels == 2).mean())}
        self.logger.info(f"subset shares {shares} (the dedup analysis': {SUBSET_SHARES})")
        return SubsetLabelFiles(out, shares)


@dataclass(frozen=True)
class RunResult:
    """A run's final validation loss, its timed training and its steps."""

    val_loss: float
    train_time_ms: int
    steps: int


def port(record: int) -> Path:
    """The directory of record `record`'s port."""
    if record == 92:
        return REPO / "record_360"
    (directory,) = (d for d in OLD.glob(f"r{record:03d}_*/") if (d / "port.json").exists())  # not r001_llmc_c/
    return directory


def port_shards(record: int) -> int:
    """The training shards record `record`'s port reads at its default steps (its port.json)."""
    if record == 92:
        return RECORD_360_SHARDS
    return json.loads((port(record) / "port.json").read_text())["train_shards"]


class OldRecordRun(furu.Spec[RunResult]):
    """Record `record`'s port, seed `seed`, with the hint rows of the variant `levels` (None: no hints; "fake": random
    rows, a smoke test) and `env` added to its environment (TRAIN_STEPS, LR_SCALE, REC_* knobs).

    - `slots` S > 1: each position gets its S deepest matched levels' rows, deepest first. `buckets`: the levels
      whose index is a cell's length bucket (None: LEVELS).
    - `counts` C > 0: also the n-gram count rows (ValCountRows, and TrainCountRows of the first C training shards),
      and HINT_NGRAM=1 (unless `env` sets it): the output is mixed with the full-count n-gram (NgramMix).
    - `late`: train on the last LATE_SHARDS training shards, with LateHintRows (and LateCountRows, counts =
      LATE_SHARDS).
    - `online` S / `online_counts` C > 0: the rows are computed in the run (old_records/online_hints.py): the training
      rows causal in the steps, and the validation index built on the clock, a SlotIndex over the first S training
      shards and an exact CountTable over the first C.
    - `gpu_val` (#1 only): the training rows as `online`'s, and the exact validation rows built from all training
      shards on the run's 8 GPUs, on the clock (old_records/gpu_val_stats.py). The shards (206 GB) are staged in
      /dev/shm before the clock.
    - `approx_val` (short, shards[, scale[, exact_below]]): cpu_val.py's approximate validation rows instead, levels
      <= short counted over the first `shards` training shards only."""

    record: int
    levels: tuple[int, ...] | None | Literal["fake"] = None
    env: tuple[tuple[str, str], ...] = ()
    seed: int = 0
    slots: int = 1
    buckets: tuple[int, ...] | None = None
    counts: int = 0
    late: bool = False
    online: int = 0
    online_counts: int = 0
    gpu_val: bool = False
    approx_val: tuple[int, ...] = ()

    migrations = (
        furu.Added("slots", default=1),
        furu.Added("buckets", default=None),
        furu.Added("counts", default=0),
        furu.Added("late", default=False),
        furu.Added("online", default=0),
        furu.Added("online_counts", default=0),
        furu.Added("gpu_val", default=False),
        furu.Added("approx_val", default=()),
    )  # the runs made before there were slots, other buckets, count rows, late shards, online rows and GPU val rows

    def _online(self) -> bool:
        return self.online > 0 or self.online_counts > 0 or self.gpu_val

    def metadata(self) -> furu.Metadata:
        return furu.Metadata(requires=furu.Requires(gpus=8))

    def _train_shards(self) -> int:
        steps = dict(self.env).get("TRAIN_STEPS")
        shards = port_shards(self.record)
        if steps is not None and self.record != 92:
            default = json.loads((port(self.record) / "port.json").read_text()).get(
                "default_steps"
            )
            if default and int(steps) > default:
                shards = math.ceil(shards * int(steps) / default)
        return shards

    @furu.dependency
    def shards(self) -> list[furu.Spec]:
        if self.late:
            return [*(FinewebShard(shard=i) for i in range(LATE_FIRST, TRAIN_SHARDS + 1)), FinewebShards(shards="val")]
        if self._online():  # the validation index's shards too
            index = TRAIN_SHARDS if self.gpu_val else max(self.online, self.online_counts)
            return [FinewebShards(shards=self._train_shards()), FinewebShards(shards=index), FinewebShards(shards="val")]
        return [FinewebShards(shards=self._train_shards()), FinewebShards(shards="val")]

    def _buckets(self) -> tuple[int, ...]:
        return LEVELS if self.buckets is None else self.buckets

    def _hint_shards(self) -> int:
        """The training shards with hint rows: TRAIN_HINT_SHARDS for the first variants (one slot, LEVELS), as they
        were made; else the run's, at least SLOT_HINT_SHARDS."""
        if self.late:
            return LATE_SHARDS
        if self.slots == 1 and self._buckets() == LEVELS:
            return TRAIN_HINT_SHARDS
        return max(SLOT_HINT_SHARDS, self._train_shards())

    def _row_specs(self) -> tuple[ValHintRows, TrainHintRows | LateHintRows]:
        assert isinstance(self.levels, tuple)
        variant = {
            "levels": self.levels,
            "slots": self.slots,
            "buckets": self._buckets(),
        }
        if self.late:
            return ValHintRows(**variant), LateHintRows(**variant)
        if self.approx_val:
            from retrieval_blog_post.cpu_val import ApproxValHintRows

            short, shards, *rest = self.approx_val
            extra = {"scale": True, "exact_below": rest[1]} if len(rest) > 1 and rest[1] else {}
            return ApproxValHintRows(levels=self.levels, short=short, shards=shards, **extra), TrainHintRows(  # type: ignore[return-value]
                **variant, shards=self._hint_shards()
            )
        return ValHintRows(**variant), TrainHintRows(
            **variant, shards=self._hint_shards()
        )

    def _count_specs(self) -> tuple[ValCountRows, TrainCountRows | LateCountRows]:
        if self.late:
            return ValCountRows(), LateCountRows()
        if self.approx_val:
            from retrieval_blog_post.cpu_val import ApproxValCountRows

            short, shards, *rest = self.approx_val
            extra = {"scale": bool(rest[0]), "exact_below": rest[1] if len(rest) > 1 else 0} if rest else {}
            return ApproxValCountRows(short=short, shards=shards, **extra), TrainCountRows(  # type: ignore[return-value]
                shards=self.counts
            )
        return ValCountRows(), TrainCountRows(shards=self.counts)

    @furu.dependency
    def rows(self) -> list[furu.Spec[HintRowFiles]]:
        if not isinstance(self.levels, tuple) or self._online():
            return []
        return list(self._row_specs())

    @furu.dependency
    def count_rows(self) -> list[furu.Spec[CountRowFiles]]:
        if not (isinstance(self.levels, tuple) and self.counts > 0) or self._online():
            return []
        return list(self._count_specs())

    def create(self) -> RunResult:
        train_shards = min(LATE_SHARDS, self._train_shards()) if self.late else self._train_shards()
        if self._online() and (not isinstance(self.levels, tuple) or self.late or self.slots != 1 or self.counts not in (0, 1)):
            raise ValueError("online runs need a hint variant's levels, one slot, not late, and counts 0 or 1")
        if self.approx_val and (len(self.approx_val) not in (2, 3, 4) or self._online() or self.late or self.slots != 1
                                or self._buckets() != LEVELS):
            raise ValueError("approx_val (short, shards) needs precomputed rows, one slot, LEVELS' buckets, not late")
        if self.gpu_val and (self.online or self.online_counts or self._buckets() != LEVELS or self.record != 1):
            raise ValueError("gpu_val runs (#1 only) build the validation rows on the GPUs: online = online_counts = 0, "
                             "buckets LEVELS")
        if isinstance(self.levels, tuple) and not self._online() and train_shards > self._hint_shards():
            raise ValueError(
                f"the run reads {train_shards} shards, past the {self._hint_shards()} with hint rows"
            )
        if self.counts > 0 and not isinstance(self.levels, tuple):
            raise ValueError("count rows need a hint variant's levels")
        if self.late and (not isinstance(self.levels, tuple) or self.counts not in (0, LATE_SHARDS)):
            raise ValueError(f"late runs need a hint variant's levels, and counts 0 or {LATE_SHARDS}")
        out = self.directory.data
        shutil.rmtree(out / "logs", ignore_errors=True)  # an earlier attempt's
        for stale in Path("/dev/shm").glob("oldrec-gpuval-*"):  # killed runs' (one worker per node)
            shutil.rmtree(stale, ignore_errors=True)
        LOCAL_DISK.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=LOCAL_DISK) as local:
            local = Path(local)
            data = local / "data/fineweb100B"
            data.mkdir(parents=True)
            train_files = _late_shards(LATE_FIRST)[:train_shards] if self.late else FinewebShards(shards=train_shards).create()
            copies = [
                (s.path, data / s.path.name)
                for s in [*train_files, *FinewebShards(shards="val").create()]
            ]
            run_env = {
                "SEED": str(self.seed),
                "TRAIN_SEED": str(self.seed),
                **dict(self.env),
            }
            gpu_stage, staged = None, []  # gpu_val: the staged shards' directory (removed after the run) and paths
            if self.levels is None:
                run_env["HINTS"] = "0"
            elif self.levels == "fake":
                run_env |= {"HINTS": "1", "HINT_DIR": "fake"}
                if self.slots > 1:
                    run_env["HINT_FAKE_SLOTS"] = str(self.slots)
            elif self._online():
                # the index shards on the local disk too (from NFS, the SlotIndex build took 26 s instead of 7), and
                # this snapshot's exact_match for the port
                import exact_match

                if not self.gpu_val:
                    index = local / "index.txt"
                    shards = FinewebShards(shards=max(self.online, self.online_counts)).create()
                    (local / "index").mkdir()
                    copies += [(s.path, local / "index" / s.path.name) for s in shards]
                    index.write_text("\n".join(str(local / "index" / s.path.name) for s in shards) + "\n")
                    run_env["HINT_ONLINE_INDEX"] = str(index)
                shutil.copytree(Path(exact_match.__file__).parent, local / "pylib/exact_match")
                run_env |= {
                    "HINTS": run_env.get("HINTS", "1"),
                    "HINT_ONLINE": "1",
                    "HINT_LEVELS": ",".join(map(str, self.levels)),
                    "HINT_BUCKETS": ",".join(map(str, self._buckets())),
                    "HINT_ONLINE_SLOT_SHARDS": str(self.online),
                    "HINT_ONLINE_COUNT_SHARDS": str(self.online_counts),
                }
                if self.counts > 0:
                    run_env.setdefault("HINT_NGRAM", "1")
                if self.gpu_val:
                    # all training shards in /dev/shm (else the local disk), read once before the clock; the kernels
                    # built now; NCCL as gpu_val_stats.py was timed (with cuMem, the first all-to-all's peer setup
                    # took ~6 s)
                    every = FinewebShards(shards=TRAIN_SHARDS).create()
                    need = sum(s.path.stat().st_size for s in every)
                    stage_root = Path("/dev/shm") if shutil.disk_usage("/dev/shm").free > 1.2 * need else local
                    gpu_stage = tempfile.TemporaryDirectory(dir=stage_root, prefix="oldrec-gpuval-")
                    staged = [Path(gpu_stage.name) / s.path.name for s in every]
                    copies += list(zip((s.path for s in every), staged))
                    (local / "gpu_val_shards.txt").write_text("".join(f"{p}\n" for p in staged))
                    build = local / "build"
                    subprocess.run([TORCH_VENV / "bin/python", OLD / "gpu_val_stats.py"], check=True,
                                   env={**os.environ, "GVS_BUILD_ONLY": "1", "GVS_BUILD_DIR": str(build)})
                    run_env |= {"HINT_GPU_VAL": "1", "HINT_GPU_VAL_SHARDS": str(local / "gpu_val_shards.txt"),
                                "GVS_BUILD_DIR": str(build)}
                    run_env.setdefault("NCCL_CUMEM_ENABLE", "0")
                    run_env.setdefault("NCCL_NVLS_ENABLE", "0")
                    self.logger.info(f"gpu_val: {len(every)} shards ({need / 1e9:.0f} GB) staged in {gpu_stage.name}")
            else:
                hints = local / "hints"
                val, train = (spec.create() for spec in self._row_specs())
                for name in ("cell", "tokens", "weights"):
                    copies.append(
                        (val.directory / f"{name}.npy", hints / "val" / f"{name}.npy")
                    )
                    for j in range(train_shards):
                        copies.append(
                            (
                                train.directory / f"{j:04d}/{name}.npy",
                                hints / f"train/{j:04d}/{name}.npy",
                            )
                        )
                for _, target in copies:
                    target.parent.mkdir(parents=True, exist_ok=True)
                meta = {
                    "levels": list(self.levels),
                    "cells": cells(self._buckets()),
                    "entries": ENTRIES,
                    "train_shards": train_shards,
                    "val_tokens": VAL_TOKENS,
                    "slots": self.slots,
                    "buckets": list(self._buckets()),
                }
                if self.counts > 0:
                    val_counts, train_counts = (spec.create() for spec in self._count_specs())
                    for name in COUNT_NAMES:
                        copies.append(
                            (val_counts.directory / f"{name}.npy", hints / "val" / f"{name}.npy")
                        )
                        for j in range(train_shards if self.late else self.counts):
                            copies.append(
                                (
                                    train_counts.directory / f"{j:04d}/{name}.npy",
                                    hints / f"train/{j:04d}/{name}.npy",
                                )
                            )
                    copies.append((train_counts.directory / "unigram.npy", hints / "unigram.npy"))
                    for _, target in copies:
                        target.parent.mkdir(parents=True, exist_ok=True)
                    meta |= {
                        "count_levels": list(COUNT_LEVELS),
                        "count_shards": train_shards if self.late else self.counts,
                    }
                    run_env.setdefault("HINT_NGRAM", "1")
                if self.late:
                    meta["late_first"] = LATE_FIRST  # training shard j is FineWeb-100B's LATE_FIRST + j
                (hints / "meta.json").write_text(json.dumps(meta))
                # HINTS=0 in env (a late run's no-hint control: its shards staged, the rows unused) is kept
                run_env |= {"HINTS": run_env.get("HINTS", "1"), "HINT_DIR": str(hints)}
            # VAL_SUBSETS: the ports that support it print the final loss on the deduplicated subsets after the timer
            try:
                labels = ValSubsetLabels().create()
                copies.append((labels.directory / "labels.npy", local / "val_subsets/labels.npy"))
                (local / "val_subsets").mkdir(exist_ok=True)
                run_env.setdefault("VAL_SUBSETS", str(local / "val_subsets/labels.npy"))
            except Exception as error:  # never fails the run
                self.logger.warning(f"no VAL_SUBSETS: {error!r}")
            try:
                start = time.time()
                with ThreadPoolExecutor(32) as pool:
                    list(pool.map(lambda c: shutil.copyfile(*c), copies))
                self.logger.info(f"copied {len(copies)} files to {local} in {time.time() - start:.0f} s")
                if staged:  # gpu_val: the staged shards read once (ValStatsTiming's prefault)
                    def prefault(path: Path) -> int:
                        with open(path, "rb", buffering=0) as f:
                            n = 0
                            while chunk := f.read(64 << 20):
                                n += len(chunk)
                        return n

                    start = time.time()
                    with ThreadPoolExecutor(32) as pool:
                        read = sum(pool.map(prefault, staged))
                    self.logger.info(f"gpu_val: read {read / 1e9:.0f} GB of staged shards in {time.time() - start:.0f} s")
            except BaseException:
                if gpu_stage is not None:
                    gpu_stage.cleanup()
                raise
            env = {
                **os.environ,
                "VIRTUAL_ENV": str(TORCH_VENV),
                "PATH": f"{TORCH_VENV}/bin:{os.environ['PATH']}",
                "PYTHONPATH": f"{OLD}:{local / 'pylib'}" if self._online() else str(OLD),
                "HF_HUB_OFFLINE": "1",  # skips kernels' publisher check: the FA3 kernel is in the HF cache already
                "DATA_PATH": str(local),
                "TMPDIR": str(local),
                "OMP_NUM_THREADS": "1",
                "PYTHONUNBUFFERED": "1",
                "TORCHINDUCTOR_CACHE_DIR": str(local / "inductor"),
                "TRITON_CACHE_DIR": str(local / "triton"),
                **run_env,
            }
            self.logger.info(f"record {self.record}: {run_env}")
            try:
                with open(out / "console.txt", "w") as console:
                    subprocess.run(
                        [
                            TORCH_VENV / "bin/python",
                            "-m",
                            "torch.distributed.run",
                            "--standalone",
                            "--nproc_per_node=8",
                            port(self.record) / "train_gpt.py",
                        ],
                        cwd=out,  # the run's logs go to out/
                        env=env,
                        stdout=console,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
            finally:
                if gpu_stage is not None:  # gpu_val: the staged shards removed (206 GB of /dev/shm)
                    gpu_stage.cleanup()
        text = (out / "console.txt").read_text()
        final = re.findall(
            r"FINAL val_loss:([\d.]+) train_time_ms:(\d+) steps:(\d+)", text
        )
        if not final:  # record_360's own last line
            final = [
                (loss, ms, step)
                for step, loss, ms in re.findall(
                    r"step:(\d+)/\1 val_loss:([\d.]+) train_time:(\d+)ms", text
                )
            ]
        if not final:
            raise RuntimeError("the run's console lacks its final line")
        loss, ms, steps = final[-1]
        result = RunResult(
            val_loss=float(loss), train_time_ms=int(ms), steps=int(steps)
        )
        self.logger.info(
            f"record {self.record} {self.levels} {dict(self.env)}: {result}"
        )
        for line in re.findall(r"^VAL_SUBSETS .*$", text, re.MULTILINE)[-1:]:  # extra output only (not in RunResult)
            self.logger.info(f"record {self.record}: {line}")
        return result


def create[T](specs: Sequence[furu.Spec[T]], job_name: str = "old-records") -> list[T]:
    """config.create's pools, with one GPU node for the runs: it stays up for 20 idle minutes, for a relaunched
    batch."""
    return furu.create(
        specs,
        on=(
            SlurmWorkerBackend(
                max_workers=1,
                resources=gpu_node(),
                job_name=f"{job_name}-8gpu",
                reserve_for=ResourceFloor(gpus=8),
                worker_idle_timeout=20 * 60,
            ),
            SlurmWorkerBackend(
                max_workers=2,
                resources=cpu(n_cpus=96, mem_gib=WORKER_GIB, time_limit="3:00:00"),
                job_name=f"{job_name}-{WORKER_GIB}g",
                reserve_for=ResourceFloor(memory_gib=WORKER_GIB),
            ),
            SlurmWorkerBackend(
                max_workers=4,
                resources=cpu(n_cpus=32, mem_gib=256),
                job_name=f"{job_name}-256g",
                reserve_for=ResourceFloor(memory_gib=101),
            ),
            SlurmWorkerBackend(
                max_workers=32, resources=shared_cpu(), job_name=f"{job_name}-sc"
            ),
        ),
    )


def probe(record: int, smoke_steps: int, **kw) -> list[OldRecordRun]:
    """A bisection probe's first runs: a smoke test with random rows, then without hints and with each variant, at the
    port's default steps."""
    return [
        OldRecordRun(
            record=record, levels="fake", env=(("TRAIN_STEPS", str(smoke_steps)),)
        ),
        OldRecordRun(record=record, **kw),
        *(OldRecordRun(record=record, levels=v, **kw) for v in VARIANTS),
    ]


from retrieval_blog_post.val_stats_timing import ValStatsTiming  # noqa: E402

# #1's best recipe (4x512, 650 steps, aux, wsd cooldown 0.1, LR 3, the hint lr held) and its pass at its own 12x768
BEST_1 = (
    ("TRAIN_STEPS", "650"), ("REC_N_LAYER", "4"), ("REC_N_EMBD", "512"), ("REC_N_HEAD", "8"), ("REC_LM_AUX", "1"),
    ("REC_LR_SCHEDULE", "wsd"), ("REC_COOLDOWN_FRAC", "0.1"), ("LR_SCALE", "3"), ("REC_HINT_LR", "0.002667"),
)


def on_clock_1(steps: str, seed: int = 0, extra: tuple[tuple[str, str], ...] = ()) -> OldRecordRun:
    """#1's PyTorch port with BEST_1's recipe (warmup 50) for `steps` steps, every hint row computed on the clock.
    At 210 steps it reaches 3.27 in about 16.5 s: the recipe llm.c's own trainer runs (llmc_onclock.py)."""
    return OldRecordRun(
        record=1, levels=LEVELS, env=(*BEST_1[1:], ("TRAIN_STEPS", steps), ("REC_WARMUP_STEPS", "50"), *extra),
        counts=1, gpu_val=True, seed=seed,
    )


# BEST_1 at 325 steps (env entries later in a tuple override earlier ones: the port reads dict(env))
BEST_1_325 = (*BEST_1[1:], ("TRAIN_STEPS", "325"))
PASS_1_12X768 = (
    ("TRAIN_STEPS", "245"), ("REC_LM_AUX", "1"), ("REC_LR_SCHEDULE", "wsd"), ("REC_COOLDOWN_FRAC", "0.3"),
    ("REC_WARMUP_STEPS", "30"), ("LR_SCALE", "1.5"),
)


def _llmc_specs() -> list[furu.Spec]:
    """llmc_runs.py's runs, or none while llmc_runs.py is the module importing this one (partly initialized)."""
    import sys

    partial = sys.modules.get("retrieval_blog_post.llmc_runs")
    if partial is not None and not hasattr(partial, "LlmcRun"):
        return []
    from retrieval_blog_post.llmc_runs import LLMC_V1, LLMC_V1_COSINE, LLMC_V2, PRISTINE, LlmcRun

    # A 40-step smoke, v2 (BEST_1) and v1 (12x768) with their no-hint baselines and seeds, v1 with llm.c's cosine
    return [
        LlmcRun(version="smoke", levels=LEVELS, env=(*LLMC_V2, ("TRAIN_STEPS", "40")), counts=4),
        LlmcRun(version="v2", levels=LEVELS, env=LLMC_V2, counts=4),
        LlmcRun(version="baseline", env=LLMC_V2),
        LlmcRun(version="v1", levels=LEVELS, env=LLMC_V1, counts=2),
        LlmcRun(version="baseline", env=LLMC_V1),
        LlmcRun(version="v2", levels=LEVELS, env=LLMC_V2, counts=4, seed=1),
        LlmcRun(version="v2", levels=LEVELS, env=LLMC_V2, counts=4, seed=2),
        LlmcRun(version="v1-cosine", levels=LEVELS, env=LLMC_V1_COSINE, counts=2),
        LlmcRun(version="baseline", env=LLMC_V1_COSINE),
        # v2 3.0745 in 29.35 s, 0.09 behind the port: v2 from the port's initial weights, and at 760 steps
        LlmcRun(
            version="v2-portinit",
            levels=LEVELS,
            counts=4,
            env=(*LLMC_V2, ("LLMC_INIT", str(SCRATCH / "llmc-hints/init/port_4x512_seed0.bin")),
                 ("REC_LOG_LM", "1"), ("REC_LOG_EVERY", "25")),
        ),
        LlmcRun(version="v2", levels=LEVELS, env=(*LLMC_V2, ("TRAIN_STEPS", "760")), counts=4),
        # v1 3.2657 in 34.19 s: 280 steps fit 39.9 s
        *(
            LlmcRun(version="v1", levels=LEVELS, env=(*LLMC_V1, ("TRAIN_STEPS", "280")), counts=2, seed=seed)
            for seed in (0, 1)
        ),
        # v2 at 760 steps 2.9406 in 34.42 s: seeds
        *(
            LlmcRun(version="v2", levels=LEVELS, env=(*LLMC_V2, ("TRAIN_STEPS", "760")), counts=4, seed=seed)
            for seed in (1, 2)
        ),
        # unmodified llm.c at d12: the record's own per-step time on this node
        LlmcRun(version="pristine", env=LLMC_V1_COSINE, binary=PRISTINE),
    ]


# Every run of the search; the comments give the results that motivated the runs after them.
BATCH: list[furu.Spec] = [
    OldRecordRun(record=92),  # #360 as is: the time budget on this node
    *(ValHintRows(levels=v) for v in VARIANTS),
    *(TrainHintRows(levels=v) for v in VARIANTS),
    *probe(1, 30),
    # #1: 250 steps fit 39.9 s (265 took 41.7-42.1 s); llm.c's lr is set for 19,560 steps
    *(
        OldRecordRun(record=1, levels=v, env=(("TRAIN_STEPS", "250"), ("LR_SCALE", lr)))
        for v in VARIANTS[1:]
        for lr in ("0.35", "0.7", "1", "2", "4")
    ),
    *probe(46, 40),
    # #46 passed (3.0070 at 38.6 s with 6/10/20): the next probe is #23, then #12 or #34
    *probe(23, 30),
    # #23 at 3.2749 with 6/10/20: the next probe is #12, then #6 or #17
    *probe(12, 40),
    # #12 at 3.9557 without hints (#23: 3.7110, -0.76 with hints): likely a pass, so #6
    *probe(6, 40),
    # #6's 245 steps took 35.6 s: 270 fit 39.9 s
    *(
        OldRecordRun(record=6, levels=v, env=(("TRAIN_STEPS", "270"),))
        for v in VARIANTS[1:]
    ),
    # #1 with the full-count n-gram mixed into the output (NgramMix: interpolated Witten-Bell over ALL_LEVELS' counts):
    # without an LM it alone gives 3.571 on val. counts=2 covers 380 steps; the 6x512 model's ~2.3x faster steps need 3
    *(
        OldRecordRun(record=1, levels=LEVELS, env=(("TRAIN_STEPS", "250"), *extra), counts=2)
        for extra in ((), (("HINT_INJECT", "0"),), (("LR_SCALE", "2"),))
    ),
    OldRecordRun(
        record=1,
        levels=LEVELS,
        env=(
            ("TRAIN_STEPS", "500"),
            ("REC_N_LAYER", "6"),
            ("REC_N_EMBD", "512"),
            ("REC_N_HEAD", "8"),
        ),
        counts=3,
    ),
    # 3.4353 at 250 steps but 39.95 s (HINT_INJECT=0 3.4446, LR_SCALE=2 3.4497): 248 steps, and the lambda init (the
    # LM's mixture logit; 0 = 1/2), which sets the deep gates the causal training counts seldom reach
    *(
        OldRecordRun(
            record=1, levels=LEVELS, env=(("TRAIN_STEPS", "248"), *extra), counts=2
        )
        for extra in ((), (("HINT_NGRAM_LAM_INIT", "-1"),), (("HINT_NGRAM_LAM_INIT", "-2"),))
    ),
    # #360 here gives 3.3192 / 3.3495 against 3.279-3.284 for #367's baselines on H100s. The console logs BOS_RACE
    # (the loader's partial BOS index running out before its async scan, ending the shard early); REC360_BOS_FIX=1
    # waits for the full index
    *(OldRecordRun(record=92, env=env, seed=seed) for env, seed in (((), 1), ((("REC360_BOS_FIX", "1"),), 1), ((), 2))),
    # Seeds 1 and 2 are at 3.281; both bad runs were seed 0. Seed 0 again (REC360_RERUN only changes the hash), seed 3
    *(OldRecordRun(record=92, env=env, seed=seed) for env, seed in (((("REC360_RERUN", "1"),), 0), ((), 3))),
    # seed 0 again 3.3535: seed 0 with the fix separates the seed from the race
    OldRecordRun(record=92, env=(("REC360_BOS_FIX", "1"),), seed=0),
    # The fewest seconds under 3.28, every row on the clock (~4.4 s + 59.6 ms x steps): steps, then warmup, LR, seeds
    *(
        on_clock_1(steps, seed, extra)
        for steps, extra, seed in (
            ("250", (), 0),
            ("225", (), 0),
            ("275", (), 0),
            ("200", (), 0),
            # 250 3.2350 (18.33 s), 225 3.2572 (17.08 s): the target is ~205-210
            ("210", (), 0),
            ("225", (("REC_WARMUP_STEPS", "30"),), 0),
            ("225", (("REC_WARMUP_STEPS", "80"),), 0),
            ("225", (("LR_SCALE", "4"),), 0),
            ("225", (("LR_SCALE", "2"),), 0),
            ("210", (), 1),
            ("210", (), 2),
            ("200", (), 1),
            ("200", (), 2),
            # 200 3.2736 (15.50 s): below it
            ("190", (), 0),
            ("180", (), 0),
            ("225", (), 1),
            ("225", (), 2),
        )
    ),
    # cpu_val.py's validation rows, the short levels capped to the first shards (vs 3.1846 exact). (3, 20) bounds the
    # capping, (6, 20, 1, 256) is the hybrid (rare contexts exact)
    *(
        OldRecordRun(record=1, levels=LEVELS, env=(*BEST_1_325, ("REC_WARMUP_STEPS", "50")), counts=4, approx_val=av)
        for av in ((6, 10), (5, 20), (3, 20), (6, 20, 1, 256))
    ),
    ValStatsTiming(),
    *_llmc_specs(),
    # gpu_val: the build after the last step (325, 450 steps), or overlapping the training (650, 600)
    OldRecordRun(record=1, levels=LEVELS, env=BEST_1_325, counts=1, gpu_val=True),
    OldRecordRun(record=1, levels=LEVELS, env=(*BEST_1_325, ("LR_SCALE", "2")), counts=1, gpu_val=True),
    OldRecordRun(record=1, levels=LEVELS, env=(*BEST_1[1:], ("TRAIN_STEPS", "450")), counts=1, gpu_val=True),
    OldRecordRun(record=1, levels=LEVELS, env=(*BEST_1, ("HINT_GPU_VAL_OVERLAP", "2"), ("HINT_GPU_VAL_THREADS", "4")), counts=1, gpu_val=True),
    OldRecordRun(record=1, levels=LEVELS, env=(*BEST_1[1:], ("TRAIN_STEPS", "600"), ("HINT_GPU_VAL_OVERLAP", "2"), ("HINT_GPU_VAL_THREADS", "4")), counts=1, gpu_val=True),
    # The overlap slows the steps by ~4.4 s, more than the 3.7 s build: the build after the last step, at the most
    # steps that fit (570)
    *(
        OldRecordRun(record=1, levels=LEVELS, env=(*BEST_1[1:], ("TRAIN_STEPS", "570"), *extra), counts=1, gpu_val=True)
        for extra in ((), (("LR_SCALE", "2"),))
    ),
    # 570: 3.1179 in 38.35 s. Overlap 1 (only the host-to-GPU loads beside the training) at 600; 16 load threads
    OldRecordRun(record=1, levels=LEVELS, env=(*BEST_1[1:], ("TRAIN_STEPS", "600"), ("HINT_GPU_VAL_OVERLAP", "1"), ("HINT_GPU_VAL_THREADS", "4"), ("HINT_GPU_VAL_CHUNK_MB", "64")), counts=1, gpu_val=True),
    OldRecordRun(record=1, levels=LEVELS, env=(*BEST_1[1:], ("TRAIN_STEPS", "570"), ("HINT_GPU_VAL_THREADS", "16"), ("HINT_GPU_VAL_CHUNK_MB", "64")), counts=1, gpu_val=True),
    # neither helps: seeds of 570 steps LR 2 (3.0846 in 38.62 s)
    *(
        OldRecordRun(
            record=1, levels=LEVELS, env=(*BEST_1[1:], ("TRAIN_STEPS", "570"), ("LR_SCALE", "2")), counts=1, gpu_val=True,
            seed=seed,
        )
        for seed in (1, 2)
    ),
    # at 325 steps warmup 50 beat LR 2: LR 3 with warmup 50 and 90
    *(
        OldRecordRun(
            record=1, levels=LEVELS, env=(*BEST_1[1:], ("TRAIN_STEPS", "570"), ("REC_WARMUP_STEPS", w)), counts=1,
            gpu_val=True,
        )
        for w in ("50", "90")
    ),
    # warmup 50 3.0458 (38.31 s), 90 3.0548: warmup 50's seeds
    *(
        OldRecordRun(
            record=1, levels=LEVELS, env=(*BEST_1[1:], ("TRAIN_STEPS", "570"), ("REC_WARMUP_STEPS", "50")), counts=1,
            gpu_val=True, seed=seed,
        )
        for seed in (1, 2)
    ),
    # Online hints (#1 only): SlotIndex 6/10/20 over 370 shards (#367's 37B), an exact CountTable over 10
    *(
        OldRecordRun(record=1, levels=levels, env=env, counts=counts, online=370, online_counts=online_counts)
        for levels, env, counts, online_counts in (
            # not BEST_1 with LEVELS: its 60 ms steps outrun the online worker's training index
            (LEVELS, PASS_1_12X768, 1, 10),
            ((6, 10, 20), BEST_1, 0, 0),
        )
    ),
    # The 1-GPU searches' best #1 (BEST_1: the hint lr held at its LR-1 value), at several steps; and #1's own 12x768
    # with the hint lr held and cooldown 0.1
    *(
        OldRecordRun(
            record=1,
            levels=LEVELS,
            env=(("TRAIN_STEPS", steps), ("REC_N_LAYER", "4"), ("REC_N_EMBD", "512"), ("REC_N_HEAD", "8"),
                 ("REC_LM_AUX", "1"), ("REC_LR_SCHEDULE", "wsd"), ("REC_COOLDOWN_FRAC", "0.1"), ("LR_SCALE", "3"),
                 ("REC_HINT_LR", "0.002667")),
            counts=4,
            seed=seed,
        )
        # 450 3.1738 (27.2 s), 325 3.2596 (19.7 s): seeds at both, which leave room for the GPU val build
        for steps, seed in (("650", 0), ("650", 1), ("650", 2), ("450", 0), ("325", 0), ("250", 0), ("325", 1),
                            ("325", 2), ("450", 1), ("450", 2), ("300", 0))
    ),
    *(
        OldRecordRun(
            record=1,
            levels=LEVELS,
            env=(("TRAIN_STEPS", "245"), ("REC_LM_AUX", "1"), ("REC_LR_SCHEDULE", "wsd"), ("REC_COOLDOWN_FRAC", cd),
                 ("REC_WARMUP_STEPS", "30"), ("LR_SCALE", "1.5"), ("REC_HINT_LR", "0.005333")),
            counts=2,
        )
        for cd in ("0.3", "0.1")
    ),
    # online: 12x768 4.1115 in 46.0 s, BEST_1 6/10/20 4.1338: the 10-shard val counts are far too weak. The index
    # shards on local disk: 12x768 again, over 530 shards, and BEST_1 + counts
    OldRecordRun(record=1, levels=LEVELS, env=PASS_1_12X768, counts=1, online=370, online_counts=10, seed=1),
    OldRecordRun(record=1, levels=LEVELS, env=PASS_1_12X768, counts=1, online=530, online_counts=10),
    OldRecordRun(record=1, levels=LEVELS, env=(*BEST_1[1:], ("TRAIN_STEPS", "620")), counts=1, online=370, online_counts=10),
    # Sweeps at 325 steps (~19.7 s): LR, cooldown, warmup, aux weight, and time-matched shapes
    *(
        OldRecordRun(record=1, levels=LEVELS, env=(*BEST_1_325, *extra), counts=4)
        for extra in (
            (("LR_SCALE", "2"),),
            (("LR_SCALE", "4"),),
            (("REC_COOLDOWN_FRAC", "0.2"),),
            (("REC_COOLDOWN_FRAC", "0.3"),),
            (("REC_WARMUP_STEPS", "50"),),
            (("REC_LM_AUX", "0.5"),),
            (("REC_LM_AUX", "2"),),
        )
    ),
    *(
        OldRecordRun(
            record=1,
            levels=LEVELS,
            env=(*BEST_1_325, ("TRAIN_STEPS", steps), ("REC_N_LAYER", layers), ("REC_N_EMBD", embd), ("REC_N_HEAD", heads)),
            counts=4,
        )
        for steps, layers, embd, heads in (("270", "4", "640", "10"), ("293", "2", "768", "12"), ("260", "3", "768", "12"))
    ),
    # LR 2 3.2152, LR 3 3.2548, LR 4 3.2577: LR 2's seeds, LR 1.5, and LR 2 at fewer steps
    *(
        OldRecordRun(record=1, levels=LEVELS, env=(*BEST_1_325, *extra), counts=4, seed=seed)
        for extra, seed in (
            ((("LR_SCALE", "2"),), 1),
            ((("LR_SCALE", "2"),), 2),
            ((("LR_SCALE", "1.5"),), 0),
            ((("LR_SCALE", "2"), ("TRAIN_STEPS", "275")), 0),
            ((("LR_SCALE", "2"), ("TRAIN_STEPS", "250")), 0),
        )
    ),
    # warmup 50 (default 12) 3.1859: its seeds, longer warmups, with LR 2, and at 650 steps
    *(
        OldRecordRun(record=1, levels=LEVELS, env=(*BEST_1_325, *extra), counts=4, seed=seed)
        for extra, seed in (
            ((("REC_WARMUP_STEPS", "50"),), 1),
            ((("REC_WARMUP_STEPS", "50"),), 2),
            ((("REC_WARMUP_STEPS", "80"),), 0),
            ((("REC_WARMUP_STEPS", "120"),), 0),
            ((("REC_WARMUP_STEPS", "50"), ("LR_SCALE", "2")), 0),
            ((("REC_WARMUP_STEPS", "80"), ("LR_SCALE", "2")), 0),
        )
    ),
    *(
        OldRecordRun(record=1, levels=LEVELS, env=(*BEST_1, ("REC_WARMUP_STEPS", w)), counts=4)
        for w in ("50", "100")
    ),
    # The 1-GPU search's best at 12x768: ALL_LEVELS rows in 3 slots, cooldown 0.1, aux 3
    *(
        OldRecordRun(
            record=1,
            levels=ALL_LEVELS,
            slots=3,
            buckets=ALL_LEVELS,
            env=(("TRAIN_STEPS", "245"), ("REC_LM_AUX", "3"), ("REC_LR_SCHEDULE", "wsd"), ("REC_COOLDOWN_FRAC", "0.1"),
                 ("REC_WARMUP_STEPS", "30"), ("LR_SCALE", "1.5"), ("REC_HINT_LR", "0.005333")),
            counts=2,
            seed=seed,
        )
        for seed in (0, 1)
    ),
    # #1's passing configs again, for their subset losses
    *(
        OldRecordRun(record=record, levels=LEVELS, env=env, counts=counts, seed=3)
        for record, env, counts in (
            (1, (("TRAIN_STEPS", "245"), ("REC_LM_AUX", "1"), ("REC_LR_SCHEDULE", "wsd"), ("REC_COOLDOWN_FRAC", "0.3"),
                 ("REC_WARMUP_STEPS", "30"), ("LR_SCALE", "1.5")), 2),
            (1, (("TRAIN_STEPS", "635"), ("REC_N_LAYER", "4"), ("REC_N_EMBD", "512"), ("REC_N_HEAD", "8"),
                 ("REC_LM_AUX", "1")), 4),
        )
    ),
    # late: #1 at its own 12x768 (cosine and wsd), then #2 and 4x512
    *(
        OldRecordRun(record=record, levels=LEVELS, env=env, counts=5, late=True)
        for record, env in (
            (1, (("TRAIN_STEPS", "245"), ("REC_LM_AUX", "1"))),
            (1, (("TRAIN_STEPS", "248"), ("REC_LM_AUX", "1"), ("REC_LR_SCHEDULE", "wsd"), ("REC_COOLDOWN_FRAC", "0.3"))),
            (1, (("TRAIN_STEPS", "248"),)),
            (2, (("TRAIN_STEPS", "238"), ("REC_LM_AUX", "1"))),
            (1, (("TRAIN_STEPS", "650"), ("REC_N_LAYER", "4"), ("REC_N_EMBD", "512"), ("REC_N_HEAD", "8"),
                 ("REC_LM_AUX", "1"), ("REC_LR_SCHEDULE", "wsd"), ("REC_COOLDOWN_FRAC", "0.3"), ("LR_SCALE", "3"))),
        )
    ),
    # 12x768 aux1 wsd 0.3 warmup 30 LR 1.5: 3.2716 (38.9 s), the first 12x768 pass: seeds
    *(
        OldRecordRun(
            record=1,
            levels=LEVELS,
            env=(("TRAIN_STEPS", "245"), ("REC_LM_AUX", "1"), ("REC_LR_SCHEDULE", "wsd"),
                 ("REC_COOLDOWN_FRAC", "0.3"), ("REC_WARMUP_STEPS", "30"), ("LR_SCALE", "1.5")),
            counts=2,
            seed=s,
        )
        for s in (1, 2)
    ),
    # warmup 30 LR 2: 3.2626 (39.0 s): seeds; LR 3 with warmup 50
    *(
        OldRecordRun(
            record=1,
            levels=LEVELS,
            env=(("TRAIN_STEPS", "245"), ("REC_LM_AUX", "1"), ("REC_LR_SCHEDULE", "wsd"),
                 ("REC_COOLDOWN_FRAC", "0.3"), ("REC_WARMUP_STEPS", warmup), ("LR_SCALE", lr)),
            counts=2,
            seed=s,
        )
        for warmup, lr, s in (("30", "2", 1), ("30", "2", 2), ("50", "3", 0))
    ),
    # No-hint controls on the late shards (HINTS=0; data order only) and #1's first shards at 245 steps (#2's: 5.9134)
    OldRecordRun(record=2, levels=LEVELS, env=(("TRAIN_STEPS", "245"), ("HINTS", "0")), late=True),
    OldRecordRun(record=1, levels=LEVELS, env=(("TRAIN_STEPS", "245"), ("HINTS", "0")), late=True),
    OldRecordRun(record=1, env=(("TRAIN_STEPS", "245"),)),
    # #1 no hints at 245: late 6.2514, first shards 6.2479; #2 late 5.8157: #2's first shards at the same 245 steps
    OldRecordRun(record=2, env=(("TRAIN_STEPS", "245"),)),
    # #2 12x768 first shards, aux1 LR_SCALE=0.7, 238 steps: 3.2721 (38.5 s): seeds
    *(
        OldRecordRun(
            record=2, levels=LEVELS, env=(("TRAIN_STEPS", "238"), ("REC_LM_AUX", "1"), ("LR_SCALE", "0.7")), counts=3,
            seed=s,
        )
        for s in (1, 2)
    ),
    # LR 1.5 seeds 3.2643-3.2716, LR 2 seeds up to 3.2947: around the pass
    *(
        OldRecordRun(
            record=1,
            levels=LEVELS,
            env=(("TRAIN_STEPS", steps), ("REC_LM_AUX", "1"), ("REC_LR_SCHEDULE", "wsd"),
                 ("REC_COOLDOWN_FRAC", "0.3"), ("REC_WARMUP_STEPS", warmup), ("LR_SCALE", lr)),
            counts=2,
        )
        for steps, warmup, lr in (("245", "50", "1.5"), ("245", "30", "1.75"), ("248", "30", "1.5"))
    ),
    # late #2 12x768 aux 238: 3.2764 (38.3 s), a pass at the record's own size: 245 steps, 3 seeds
    *(
        OldRecordRun(
            record=2, levels=LEVELS, env=(("TRAIN_STEPS", "245"), ("REC_LM_AUX", "1")), counts=5, late=True, seed=s
        )
        for s in (0, 1, 2)
    ),
    # The 1-GPU searches' wsd schedule (cooldown 0.3, LR 3): 3.0324 vs llm.c's cosine 3.2114
    *(
        OldRecordRun(
            record=1,
            levels=LEVELS,
            env=(("TRAIN_STEPS", steps), *size, ("REC_LM_AUX", "1"), ("REC_LR_SCHEDULE", "wsd"),
                 ("REC_COOLDOWN_FRAC", "0.3"), ("LR_SCALE", "3")),
            counts=counts,
            seed=seed,
        )
        for steps, size, counts, seed in (
            ("650", (("REC_N_LAYER", "4"), ("REC_N_EMBD", "512"), ("REC_N_HEAD", "8")), 4, 0),
            ("245", (), 2, 0),
            ("650", (("REC_N_LAYER", "4"), ("REC_N_EMBD", "512"), ("REC_N_HEAD", "8")), 4, 1),
            ("650", (("REC_N_LAYER", "4"), ("REC_N_EMBD", "512"), ("REC_N_HEAD", "8")), 4, 2),
        )
    ),
    # 4x512 650 3.0187 (38.9 s); 12x768 3.3741, worse than cosine: lower lr for 12x768
    *(
        OldRecordRun(
            record=1,
            levels=LEVELS,
            env=(("TRAIN_STEPS", "245"), ("REC_LM_AUX", "1"), ("REC_LR_SCHEDULE", "wsd"),
                 ("REC_COOLDOWN_FRAC", "0.3"), ("LR_SCALE", lr)),
            counts=2,
        )
        for lr in ("1", "1.5")
    ),
    # LR 1 3.2902, LR 1.5 3.3216: lower lr, and a longer cooldown
    *(
        OldRecordRun(
            record=1,
            levels=LEVELS,
            env=(("TRAIN_STEPS", "248"), ("REC_LM_AUX", "1"), ("REC_LR_SCHEDULE", "wsd"),
                 ("REC_COOLDOWN_FRAC", cd), ("LR_SCALE", lr)),
            counts=2,
        )
        for lr, cd in (("0.7", "0.3"), ("0.5", "0.3"), ("1", "0.5"))
    ),
    # The 4-GPU search: a longer warmup (50 of 640) fixes high-lr seeds. 12x768: warmup 30 (default 9) at LR 1.5 / 2
    *(
        OldRecordRun(
            record=1,
            levels=LEVELS,
            env=(("TRAIN_STEPS", "245"), ("REC_LM_AUX", "1"), ("REC_LR_SCHEDULE", "wsd"),
                 ("REC_COOLDOWN_FRAC", "0.3"), ("REC_WARMUP_STEPS", "30"), ("LR_SCALE", lr)),
            counts=2,
        )
        for lr in ("1.5", "2")
    ),
    # #1 4x512 650 seeds mean 3.2797: borderline. REC_LM_AUX=1 (the LM's own CE in the loss) gained 0.095 on 1 GPU
    *(
        OldRecordRun(
            record=1,
            levels=LEVELS,
            env=(("TRAIN_STEPS", steps), ("REC_N_LAYER", layers), ("REC_N_EMBD", "512"), ("REC_N_HEAD", "8"),
                 ("REC_LM_AUX", "1")),
            counts=4,
            seed=seed,
        )
        for steps, layers, seed in (
            ("635", "4", 0), ("635", "4", 1), ("635", "4", 2), ("545", "6", 0),
        )
    ),
    # #1 passes: 4x512 635 aux1 3.2133-3.2150 in 38.3-38.7 s. Does aux alone pass #1 and #2 at 12x768?
    OldRecordRun(
        record=1, levels=LEVELS, env=(("TRAIN_STEPS", "245"), ("REC_LM_AUX", "1")), counts=2
    ),
    OldRecordRun(record=2, levels=LEVELS, env=(("TRAIN_STEPS", "238"), ("REC_LM_AUX", "1")), counts=3),
    # both fail (#1 3.3451, #2 3.3241). The smallest size change that passes: depth only (6x768) or width only
    *(
        OldRecordRun(
            record=record,
            levels=LEVELS,
            env=(("TRAIN_STEPS", steps), ("REC_N_LAYER", layers), ("REC_N_EMBD", embd), ("REC_N_HEAD", heads),
                 ("REC_LM_AUX", "1")),
            counts=3,
        )
        for record, steps, layers, embd, heads in (
            (1, "365", "6", "768", "12"),
            (1, "375", "12", "512", "8"),
            (2, "355", "6", "768", "12"),
            (2, "365", "12", "512", "8"),
        )
    ),
    # 6x768 3.2678 passes, 12x512 3.3010 fails: depth at the full budget, seeds; 8 and 10 layers; 6x768 without aux
    *(
        OldRecordRun(
            record=1,
            levels=LEVELS,
            env=(("TRAIN_STEPS", steps), ("REC_N_LAYER", layers), *aux),
            counts=3,
            seed=seed,
        )
        for steps, layers, aux, seed in (
            ("380", "6", (("REC_LM_AUX", "1"),), 0),
            ("380", "6", (("REC_LM_AUX", "1"),), 1),
            ("380", "6", (("REC_LM_AUX", "1"),), 2),
            ("300", "8", (("REC_LM_AUX", "1"),), 0),
            ("385", "6", (), 0),
            ("265", "10", (("REC_LM_AUX", "1"),), 0),
        )
    ),
    # 8x768 3.2993 and 6x768 without aux 3.3922 fail: depth and aux both needed. The best #1 without the injected hint
    # (n-gram mixing only), and at 650 steps
    *(
        OldRecordRun(
            record=1,
            levels=LEVELS,
            env=(("TRAIN_STEPS", steps), ("REC_N_LAYER", "4"), ("REC_N_EMBD", "512"), ("REC_N_HEAD", "8"),
                 ("REC_LM_AUX", "1"), *extra),
            counts=4,
        )
        for steps, extra in (("635", (("HINT_INJECT", "0"),)), ("650", ()))
    ),
    # On 1 GPU wider/shallower is better per step (4x640 3.1787, 2x768 3.1852 vs 4x512 3.2114): time-matched here
    *(
        OldRecordRun(
            record=1,
            levels=LEVELS,
            env=(("TRAIN_STEPS", steps), ("REC_N_LAYER", layers), ("REC_N_EMBD", embd), ("REC_N_HEAD", heads),
                 ("REC_LM_AUX", "1")),
            counts=4,
        )
        for steps, layers, embd, heads in (
            ("530", "4", "640", "10"),
            ("570", "2", "768", "12"),
            ("500", "3", "768", "12"),
        )
    ),
    # 4x512 aux 650 3.2092 (39.1 s): seeds; #2 at 12x768 with more aux or a lower lr
    *(
        OldRecordRun(
            record=1,
            levels=LEVELS,
            env=(("TRAIN_STEPS", "650"), ("REC_N_LAYER", "4"), ("REC_N_EMBD", "512"), ("REC_N_HEAD", "8"),
                 ("REC_LM_AUX", "1")),
            counts=4,
            seed=seed,
        )
        for seed in (1, 2)
    ),
    *(
        OldRecordRun(record=2, levels=LEVELS, env=(("TRAIN_STEPS", "238"), *extra), counts=3)
        for extra in ((("REC_LM_AUX", "3"),), (("REC_LM_AUX", "1"), ("LR_SCALE", "0.7")))
    ),
    # #1 4x512 650 + n-gram: 3.2787 in 39.66 s, the first #1 below 3.28: seeds and LR 2
    *(
        OldRecordRun(
            record=1,
            levels=LEVELS,
            env=(("TRAIN_STEPS", steps), ("REC_N_LAYER", layers), ("REC_N_EMBD", "512"), ("REC_N_HEAD", "8"), *extra),
            counts=4,
            seed=seed,
        )
        for steps, layers, seed, extra in (
            ("650", "4", 1, ()),
            ("650", "4", 2, ()),
            ("650", "4", 0, (("LR_SCALE", "2"),)),
        )
    ),
    # #3 with the n-gram mixing
    *(
        OldRecordRun(record=3, levels=LEVELS, env=(("TRAIN_STEPS", "230"), *extra), counts=2)
        for extra in ((), (("LR_SCALE", "2"),))
    ),
    # #2 6x512 530 steps + n-gram: 3.0698 in 38.4 s: seeds
    *(
        OldRecordRun(
            record=2,
            levels=LEVELS,
            env=(("TRAIN_STEPS", "530"), ("REC_N_LAYER", "6"), ("REC_N_EMBD", "512"), ("REC_N_HEAD", "8")),
            counts=4,
            seed=s,
        )
        for s in (1, 2)
    ),
    # #3 + n-gram at 230 steps: LR 2 3.1268 but 40.8 s: 220 steps, 3 seeds; then the next probe, #2
    *(
        OldRecordRun(
            record=3, levels=LEVELS, env=(("TRAIN_STEPS", "220"), ("LR_SCALE", "2")), counts=2, seed=s
        )
        for s in (0, 1, 2)
    ),
    # #3 + n-gram passes (3.1364-3.1429 in 38.8-39.3 s). #2: smoke test and no hints only
    *probe(2, 30)[:2],
    # #2 + n-gram at its default 242 steps
    *(
        OldRecordRun(record=2, levels=LEVELS, env=extra, counts=3)
        for extra in ((), (("LR_SCALE", "2"),))
    ),
    # #2 + n-gram 3.3719 (39.5 s): a 6x512 model at 530 steps
    *(
        OldRecordRun(
            record=2,
            levels=LEVELS,
            env=(("TRAIN_STEPS", "530"), ("REC_N_LAYER", "6"), ("REC_N_EMBD", "512"), ("REC_N_HEAD", "8"), *extra),
            counts=4,
        )
        for extra in ((), (("LR_SCALE", "2"),))
    ),
    # #1 6x512 at 500 steps 3.3477 in 35.5 s (12x768: 3.4368): smaller models, more steps; 190 steps per counts
    # shard
    *(
        OldRecordRun(
            record=1,
            levels=LEVELS,
            env=(
                ("TRAIN_STEPS", steps),
                ("REC_N_LAYER", layers),
                ("REC_N_EMBD", "512"),
                ("REC_N_HEAD", "8"),
                *extra,
            ),
            counts=shards,
        )
        for steps, layers, shards, extra in (
            ("555", "6", 3, ()),
            ("650", "4", 4, ()),
            ("480", "8", 3, ()),
            ("560", "6", 3, (("LR_SCALE", "2"),)),
        )
    ),
    # #6 passed (3.0499 at 270 steps, 39.3 s): the next probe is #3, then #2 or #4
    *probe(3, 30),
    # #3's 176 steps take 30.1 s (none 5.0384, 6/10/20 3.7458, 2/3/6/10/20 3.4339): 230 steps fit 39.9 s
    *(
        OldRecordRun(record=3, levels=LEVELS, env=(("TRAIN_STEPS", "230"), *extra))
        for extra in ((), (("LR_SCALE", "2"),))
    ),
    # seeds to confirm #6
    *(
        OldRecordRun(record=6, levels=LEVELS, env=(("TRAIN_STEPS", "270"),), seed=s)
        for s in (1, 2)
    ),
]


if __name__ == "__main__":

    def _main() -> None:
        # Imported, not this file's own definitions: run as a file, they would be __main__.OldRecordRun, another spec.
        from retrieval_blog_post.old_records import BATCH, OldRecordRun, _llmc_specs, create

        if not any(type(spec).__name__ == "LlmcRun" for spec in BATCH):
            # llmc_runs.py imported this module first: its runs go after ValStatsTiming
            at = next(i for i, spec in enumerate(BATCH) if type(spec).__name__ == "ValStatsTiming") + 1
            BATCH[at:at] = _llmc_specs()

        for spec, result in zip(BATCH, create(BATCH)):
            if isinstance(spec, OldRecordRun):
                print(
                    f"record {spec.record} {spec.levels} {dict(spec.env)} seed {spec.seed}: {result}"
                )

    _main()
