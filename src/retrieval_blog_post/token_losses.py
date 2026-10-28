"""The validation loss at every token (the 10,485,760 targets of the final validation), of the #367 record with its
hints and without them, of the record trained without hints, and of the previous record (#360, ANVIL2, 1194 steps):
one run each on the 8 GPUs of a miso node, with VAL_TOKEN_LOSSES set (the record's track_1_short/token_losses.py, off
the clock after the final validation). The record's loss without hints is the same trained model's, evaluated again
with every row blank; the record trained without hints (NO_HINTS) has blank rows in training and validation, as
data_scaling's runs. The previous record is record_360/: its code at 4ea6b937, trained and validated on FineWeb-100B
as the record's baseline runs are (the FineWeb-10B validation shard holds other tokens). Every frame has a row per
target, at its position in the validation shard, so they join on `position`:

    uv run src/retrieval_blog_post/token_losses.py      # keep it running (e.g. in tmux): it coordinates the workers
"""

import math
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar

import furu
import numpy as np
import polars as pl

from exact_match import CELLS
from retrieval_blog_post.data_scaling import SHARD_TOKENS_PER_TOKEN
from retrieval_blog_post.fineweb import FinewebShards
from retrieval_blog_post.kgram import RECORD_360, RECORD_367, SHARD_TOKENS
from retrieval_blog_post.val_cache import INDEX_SHARDS, LOCAL_DISK, REPO, train_record

# A hint's match length per length bucket of its cell (exact_match/src/lib.rs's row: 0 is no match, then MIN and
# LEVELS), each bucket CELLS / 4 cells.
HINT_LENGTHS = np.array([0, 6, 10, 20], dtype=np.uint8)
# The shards a run without the validation index reads: what it trains on, with data_scaling's margin, and the one its
# loader preloads.
RECORD_360_SHARDS = math.ceil(SHARD_TOKENS_PER_TOKEN * RECORD_360[1] / SHARD_TOKENS) + 1
RECORD_367_SHARDS = math.ceil(SHARD_TOKENS_PER_TOKEN * RECORD_367[1] / SHARD_TOKENS) + 1


class LazyParquet(furu.Codec[pl.LazyFrame]):
    """A LazyFrame, collected into data.parquet and scanned from it when loaded."""

    auto_register: ClassVar[bool] = False

    @classmethod
    def matches(cls, value: object) -> bool:
        return isinstance(value, pl.LazyFrame)

    @classmethod
    def save(
        cls, value: pl.LazyFrame, artifact_directory: Path
    ) -> Mapping[str, object]:
        value.sink_parquet(artifact_directory / "data.parquet")
        return {}

    @classmethod
    def load(
        cls, metadata: Mapping[str, object], artifact_directory: Path
    ) -> pl.LazyFrame:
        return pl.scan_parquet(artifact_directory / "data.parquet")


def _token_losses(
    out: Path, script: Path, train_shards: int, run_env: dict[str, str], logger
) -> dict[str, np.ndarray]:
    """Run `script` with `run_env` and VAL_TOKEN_LOSSES, check that its per-token losses average to the logged
    validation loss, and return its columns."""
    LOCAL_DISK.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=LOCAL_DISK) as dump:
        text = train_record(
            out,
            train_shards,
            {**run_env, "VAL_TOKEN_LOSSES": dump},
            logger,
            script=script,
        )
        with np.load(Path(dump) / "token_losses.npz") as npz:
            columns = dict(npz)
    final = re.search(r"val_loss:([\d.]+) train_time:", text)
    if final is None:
        raise RuntimeError("the run log lacks the final validation loss")
    mean = columns["loss"].mean(dtype=np.float64)
    if abs(mean - float(final[1])) > 1e-4:
        raise RuntimeError(
            f"the per-token losses average {mean:.6f}, the run logged {final[1]}"
        )
    return columns


def _positions(columns: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Each target's position in the validation shard, and the target."""
    return {
        "position": np.arange(1, len(columns["target"]) + 1, dtype=np.uint32),
        "target": columns["target"].astype(np.uint16),
    }


def _losses(columns: dict[str, np.ndarray]) -> pl.LazyFrame:
    """Each target's position, the target and its loss."""
    return pl.DataFrame({**_positions(columns), "loss": columns["loss"]}).lazy()


class RecordTokenLosses(furu.Spec[pl.LazyFrame]):
    """The #367 record trained with seed `seed`, per validation target: its position and token, the loss with hints and
    without (the same weights, every row blank), whether it was hinted, the hint's match length (0, 6, 10 or 20) and
    cell (the model's embedding of length, count and top share bins), and over the candidates the model read, their
    total count, the top one's count and the target's count (0: the target is not among them; the top candidate when
    equal to hint_top_count, ties included)."""

    seed: int = 0

    result_codecs = (LazyParquet,)

    def metadata(self) -> furu.Metadata:
        return furu.Metadata(requires=furu.Requires(gpus=8))

    @furu.dependency
    def shards(self) -> list[FinewebShards]:
        return [FinewebShards(shards=INDEX_SHARDS), FinewebShards(shards="val")]

    def create(self) -> pl.LazyFrame:
        columns = _token_losses(
            self.directory.data,
            REPO / "train_gpt.py",
            INDEX_SHARDS,
            {"TRAIN_SEED": str(self.seed)},
            self.logger,
        )
        cell = columns["hint_cell"]
        return pl.DataFrame(
            {
                **_positions(columns),
                "loss": columns["loss"],
                "loss_without_hints": columns["loss_without_hints"],
                "hinted": cell > 0,
                "hint_length": HINT_LENGTHS[cell // (CELLS // len(HINT_LENGTHS))],
                "hint_cell": cell.astype(np.int16),
                "hint_total": columns["hint_total"],
                "hint_top_count": columns["hint_top_count"],
                "hint_target_count": columns["hint_target_count"],
            }
        ).lazy()


class NoHintsRecordTokenLosses(furu.Spec[pl.LazyFrame]):
    """The #367 record trained and validated without hints (NO_HINTS), seed `seed`, per validation target: its position
    and token, and the loss."""

    seed: int = 0

    result_codecs = (LazyParquet,)

    def metadata(self) -> furu.Metadata:
        return furu.Metadata(requires=furu.Requires(gpus=8))

    @furu.dependency
    def shards(self) -> list[FinewebShards]:
        return [FinewebShards(shards=RECORD_367_SHARDS), FinewebShards(shards="val")]

    def create(self) -> pl.LazyFrame:
        columns = _token_losses(
            self.directory.data,
            REPO / "train_gpt.py",
            RECORD_367_SHARDS,
            {"TRAIN_SEED": str(self.seed), "NO_HINTS": "1"},
            self.logger,
        )
        return _losses(columns)


class PreviousRecordTokenLosses(furu.Spec[pl.LazyFrame]):
    """The previous record (#360) trained with seed `seed`, per validation target: its position and token, and the
    loss. With `bos_fix` (REC360_BOS_FIX), its loader waits for a shard's full BOS index where upstream's ends the shard
    early: on miso, runs that hit the race land at 3.32-3.35 (3.3495 with seed 0, 3.3266 with seed 1 here), fixed ones
    at 3.281, as on Nebius."""

    seed: int = 0
    bos_fix: bool = True

    migrations = (furu.Added("bos_fix", default=False),)  # the runs before the fix
    result_codecs = (LazyParquet,)

    def metadata(self) -> furu.Metadata:
        return furu.Metadata(requires=furu.Requires(gpus=8))

    @furu.dependency
    def shards(self) -> list[FinewebShards]:
        return [FinewebShards(shards=RECORD_360_SHARDS), FinewebShards(shards="val")]

    def create(self) -> pl.LazyFrame:
        columns = _token_losses(
            self.directory.data,
            REPO / "record_360/train_gpt.py",
            RECORD_360_SHARDS,
            {"TRAIN_SEED": str(self.seed), "REC360_BOS_FIX": str(int(self.bos_fix))},
            self.logger,
        )
        return _losses(columns)


if __name__ == "__main__":

    def _main() -> None:
        # Imported, not this file's own definitions: run as a file, they would be __main__.RecordTokenLosses, another
        # spec.
        from retrieval_blog_post.config import create
        from retrieval_blog_post.token_losses import (
            NoHintsRecordTokenLosses,
            PreviousRecordTokenLosses,
            RecordTokenLosses,
        )

        record, no_hints, previous = create(
            [
                RecordTokenLosses(),
                NoHintsRecordTokenLosses(),
                PreviousRecordTokenLosses(),
            ],
            job_name="token-losses",
        )
        every = record.join(
            no_hints.select("position", trained_without_hints_loss="loss"),
            on="position",
        ).join(previous.select("position", previous_loss="loss"), on="position")
        print(
            every.group_by("hint_length")
            .agg(
                tokens=pl.len(),
                loss=pl.col("loss").mean(),
                loss_without_hints=pl.col("loss_without_hints").mean(),
                trained_without_hints_loss=pl.col("trained_without_hints_loss").mean(),
                previous_loss=pl.col("previous_loss").mean(),
            )
            .sort("hint_length")
            .collect()
        )

    _main()
