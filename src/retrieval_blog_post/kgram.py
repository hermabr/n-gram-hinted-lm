"""How often a validation position's k-context occurred in the first training tokens, and how often the most frequent
tokens after it there include the position's next token: the appendix figure of sep30-how-the-cache-works (made from
artifacts/sep30-fullcorpus-k23.py over all 103 FineWeb10B shards), for the start of FineWeb100B. The counting is
exact_match.kgram_stats (exact_match/src/kgram.rs), which has the definitions."""

import time
from dataclasses import dataclass
from typing import Literal

import furu
import numpy as np

from exact_match import kgram_stats
from retrieval_blog_post.fineweb import FinewebShards, Val

KS = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 20, 24, 32, 48, 64)
VAL_TOKENS = 10_485_760  # as track_1_short/config.py
# Bytes of occurrences kgram_stats holds at once: more rounds over the corpus when smaller (5 for k = 2 at 350 shards).
# The worker's memory (WORKER_GIB) also holds up to as much again in the per-thread buffers' slack, the bitmaps (9 GB
# at 350 shards), and the shards in the page cache (70 GB at 350), so that the rounds do not reread them over NFS.
MEMORY = 64 << 30
WORKER_GIB = 256
SHARD_TOKENS = 100_000_000  # per FineWeb100B training shard

# The training data: the first that many shards, or the first that many tokens of the shards (where the training loader
# starts; it reads on from there, though it skips the tails of documents cut at the stage's sequence length).
TrainData = tuple[Literal["shards"], int] | tuple[Literal["tokens"], int]

# The training tokens of record runs (every rank's batch of every step), with the n-gram statistics over as much data
# as the model trains on: #360 (ANVIL2), 1194 steps at batch 8/16/24/20/8 x 16,384 tokens (320/361/426/67/20 steps);
# #367 (the retrieval-augmented n-gram model), 530 + 10 steps at batch 8 x 16,384.
RECORD_360: TrainData = ("tokens", 328_663_040)
RECORD_367: TrainData = ("tokens", 70_778_880)


@dataclass(frozen=True)
class KGramStats:
    """Per k of `ks`: the validation positions whose k-context is valid, those whose k-context occurred in the
    `train_tokens` training tokens, and per n of `tops` those whose next token is among the n most frequent there."""

    train_tokens: int
    ks: tuple[int, ...]
    tops: tuple[int, ...]
    eligible: tuple[int, ...]
    matched: tuple[int, ...]
    correct: tuple[tuple[int, ...], ...]


class KGramMatchStats(furu.Spec[KGramStats]):
    """KGramStats of the first VAL_TOKENS positions of the validation shard `val` against the training data `train`."""

    train: TrainData = ("shards", 370)
    tops: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    val: Val = "val"

    migrations = (furu.Added("val", default="val"),)

    throttle = furu.Throttle(max_running=8)  # each reads its shards over NFS once

    def metadata(self) -> furu.Metadata:
        shards, _ = self._data()
        if shards <= 20:  # everything fits the general pools' workers
            return furu.Metadata(
                requires=furu.Requires(cpus=furu.at_least(16), memory=furu.GiB(64))
            )
        return furu.Metadata(
            requires=furu.Requires(cpus=furu.at_least(32), memory=furu.GiB(WORKER_GIB))
        )

    def _data(self) -> tuple[int, int | None]:
        """The training shards to read, and the tokens to use from them (None: all)."""
        match self.train:
            case ("shards", n):
                return n, None
            case ("tokens", n):
                return -(-n // SHARD_TOKENS), n

    def create(self) -> KGramStats:
        shards, train_tokens = self._data()
        train = FinewebShards(shards=shards).create()
        (val,) = FinewebShards(shards=self.val).create()
        files = [str(shard.path) for shard in train]
        tokens = np.ascontiguousarray(val.tokens[: VAL_TOKENS + 1])
        # Each k's counts are saved in the scratch directory, which a retry keeps: after a preemption, the finished
        # ks are read back. The hits bitmap of the previous k is not saved (4.4 GB at 350 shards); without it the
        # first k after a resume probes every corpus position, which is slower but counts the same.
        progress = self.directory.scratch
        eligible, matched, ranks, hits = [], [], [], None
        for k in KS:
            path = progress / f"k{k}.npz"
            if path.exists():
                done = np.load(path)
                hits = None
            else:
                start = time.time()
                e, m, r, hits = kgram_stats(
                    files, tokens, k, max(self.tops), MEMORY, train_tokens, hits
                )
                np.savez(progress / f"k{k}.partial.npz", eligible=e, matched=m, ranks=r)
                (progress / f"k{k}.partial.npz").rename(path)
                done = np.load(path)
                self.logger.info(
                    f"k={k}: {m / e:.4f} matched, top-1 {r[0] / max(m, 1):.3f}, "
                    f"{time.time() - start:.0f} s"
                )
            eligible.append(int(done["eligible"]))
            matched.append(int(done["matched"]))
            ranks.append(done["ranks"])
        correct = np.cumsum(ranks, axis=1)[:, [n - 1 for n in self.tops]]
        return KGramStats(
            train_tokens=train_tokens or sum(len(shard.tokens) for shard in train),
            ks=KS,
            tops=self.tops,
            eligible=tuple(eligible),
            matched=tuple(matched),
            correct=tuple(map(tuple, correct.tolist())),
        )
