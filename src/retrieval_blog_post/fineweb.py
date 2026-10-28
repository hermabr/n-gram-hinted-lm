"""The GPT-2 tokens of FineWeb100B (kjj0/fineweb100B-gpt2): 1028 training shards of 100M tokens and one validation
shard, linked from the local copy when it has them and downloaded otherwise, and the validation shard without the
documents that also occur in the training shards, exactly or fuzzily."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import furu
import numpy as np
import tiktoken
from huggingface_hub import hf_hub_download

from exact_match import val_overlap
from retrieval_blog_post.config import DATA

LOCAL = DATA
HEADER_BYTES = (
    1024  # 256 int32 (magic, version, token count, ...), then the tokens as u16
)
BOS = 50256
TRAIN_SHARDS = 1028

# The validation shard as is, without the documents with an exact copy in the training shards, or without Gopher's
# near-duplicates at the given amount: those with an exact copy or with a training document at a word 13-gram Jaccard
# similarity of at least the amount (Gopher removed at 0.8).
Val = Literal["val", "val-exact-dedup"] | tuple[Literal["val-dedup-fuzzy"], float]


@dataclass(frozen=True)
class Shard:
    """A shard file."""

    path: Path

    @property
    def tokens(self) -> np.memmap:
        """The tokens, as uint16, mapped read-only."""
        return np.memmap(self.path, dtype="<u2", mode="r", offset=HEADER_BYTES)


@dataclass(frozen=True)
class Overlap:
    """Per validation document: the start and length of its body (after its BOS), its exact copies in the training
    shards, and its best word n-gram Jaccard similarity with a training document (exact_match/src/overlap.rs)."""

    starts: np.ndarray
    lengths: np.ndarray
    exact: np.ndarray
    jaccard: np.ndarray


class ValOverlap(furu.Spec[Overlap]):
    """The overlap of the validation shard with all training shards, as Gopher (Rae et al., 2021) deduplicates:
    exact copies, and Jaccard similarities of the sets of `words_per_ngram`-word n-grams (whitespace normalized,
    punctuation ignored)."""

    words_per_ngram: int = 13

    migrations = (furu.Renamed("n", to="words_per_ngram"),)

    def metadata(self) -> furu.Metadata:
        # one pass over all training shards, read in parallel; peaked at 10.7 GB
        return furu.Metadata(
            requires=furu.Requires(cpus=furu.at_least(12), memory=furu.GiB(32), gpus=0)
        )

    def create(self) -> Overlap:
        train = FinewebShards(shards=TRAIN_SHARDS).create()
        val = FinewebShard(shard="val").create()
        files = [str(shard.path) for shard in train]
        vocab = tiktoken.get_encoding("gpt2")
        pieces = [vocab.decode_single_token_bytes(t) for t in range(vocab.n_vocab)]
        offsets = np.cumsum([0, *map(len, pieces)], dtype=np.uint64)
        bytes_ = np.frombuffer(b"".join(pieces), np.uint8)
        return Overlap(
            *val_overlap(str(val.path), files, self.words_per_ngram, offsets, bytes_)
        )


class FinewebShard(furu.Spec[Shard]):
    """Training shard `shard` (1-1028) or a validation shard (see Val)."""

    shard: int | Val
    repo: str = "kjj0/fineweb100B-gpt2"

    throttle = furu.Throttle(max_running=10)  # concurrent downloads

    def metadata(self) -> furu.Metadata:
        # hf_xet's full concurrency (it is throttled by default); HF_TOKEN, if set, raises the Hub's rate limits. No
        # GPUs: a data job never takes a GPU worker.
        return furu.Metadata(
            environment={"HF_XET_HIGH_PERFORMANCE": "1"},
            requires=furu.Requires(gpus=0),
        )

    def create(self) -> Shard:
        match self.shard:
            case "val-exact-dedup":
                overlap = ValOverlap().create()
                path = self._without(overlap, overlap.exact > 0)
            case ("val-dedup-fuzzy", amount):
                overlap = ValOverlap().create()
                path = self._without(
                    overlap, (overlap.exact > 0) | (overlap.jaccard >= amount)
                )
            case "val":
                path = self._fetch("fineweb_val_000000.bin")
            case int(i):
                path = self._fetch(f"fineweb_train_{i:06d}.bin")
        header = np.fromfile(path, dtype="<i4", count=256)
        if (
            header[0] != 20240520
            or header[1] != 1
            or path.stat().st_size != HEADER_BYTES + 2 * int(header[2])
        ):
            raise ValueError(f"invalid shard: {path}")
        return Shard(path)

    def _fetch(self, name: str) -> Path:
        """The shard file `name`: linked from the local copy, or downloaded."""
        path = self.directory.data / name
        if (LOCAL / name).exists():
            path.symlink_to(LOCAL / name)
        else:
            Path(
                hf_hub_download(
                    self.repo,
                    name,
                    repo_type="dataset",
                    local_dir=self.directory.scratch,
                )
            ).rename(path)
        return path

    def _without(self, overlap: Overlap, drop: np.ndarray) -> Path:
        """The validation shard without the documents `drop` (per document of `overlap`), each with its BOS."""
        val = FinewebShard(shard="val", repo=self.repo).create()
        tokens = val.tokens
        if (tokens[overlap.starts - 1] != BOS).any():
            raise ValueError("a validation document does not follow a BOS")
        keep = np.ones(len(tokens), bool)
        for start, length in zip(overlap.starts[drop], overlap.lengths[drop]):
            keep[start - 1 : start + length] = False
        header = np.fromfile(val.path, dtype="<i4", count=256)
        header[2] = keep.sum()
        path = self.directory.data / "fineweb_val_000000.bin"
        with open(path, "wb") as f:
            f.write(header.tobytes())
            f.write(tokens[keep].tobytes())
        self.logger.info(
            f"dropped {drop.sum()} of {len(drop)} documents, "
            f"{len(tokens) - keep.sum()} of {len(tokens)} tokens"
        )
        return path


class FinewebShards(furu.Spec[list[Shard]]):
    """The first `shards` training shards, or a validation shard alone (see Val)."""

    shards: int | Val

    def metadata(self) -> furu.Metadata:
        return furu.Metadata(requires=furu.Requires(gpus=0))

    def create(self) -> list[Shard]:
        if isinstance(self.shards, int):
            return furu.create(
                [FinewebShard(shard=i) for i in range(1, self.shards + 1)]
            )
        return [FinewebShard(shard=self.shards).create()]
