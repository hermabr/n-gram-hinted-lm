"""Record #367 with exact n-gram counts built on the GPUs (old_records/gpu_val_stats.py's build mixed into the model's
output distribution: the blog post's 3.1 s variant), checked position by position against these CPU references: the
validation count rows over the first 103 (10.3B tokens) and 370 (37B) training shards. GpuCountsRun runs the variant
itself (record_367_gpu_counts/: 82 + 10 steps to ~3.25 in ~3.13 s).

    uv run src/retrieval_blog_post/r367_gpu_counts.py   # keep it running (e.g. in tmux): it coordinates the workers
"""

import re
import shutil
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import furu
import numpy as np

from retrieval_blog_post.config import LOCAL_DISK
from retrieval_blog_post.config import create as slurm_create
from retrieval_blog_post.fineweb import TRAIN_SHARDS, FinewebShards
from retrieval_blog_post.kgram import VAL_TOKENS
from retrieval_blog_post.old_records import (
    COUNT_NAMES,
    CountRowFiles,
    val_count_features,
)
from retrieval_blog_post.val_cache import REPO, train_record
from retrieval_blog_post.val_hints import ALL_LEVELS, ValContextCounts

# The 3.13 s variant: train_gpt.py and track_1_short/ of modded-nanogpt's r367-gpu-counts-clean (b8a31606), kept as
# the files that differ from the root's track_1_short/ (the rest are the same bytes; assemble() lays them over it). Its counts
# cover the first COUNT_SHARDS training shards (its config.py's count_shards); the next one is held out for the mixture's
# fit, so a run reads all TRAIN_SHARDS.
GPU_COUNTS = REPO / "record_367_gpu_counts"
COUNT_SHARDS = 1027
SHM_STAGE = Path("/dev/shm/r367gpu")  # the shards' staging, as the record's runs had them (one worker per node)


class ValCountRowsAt(furu.Spec[CountRowFiles]):
    """old_records.ValCountRows over the first `shards` training shards (rather than all of them)."""

    shards: int

    def metadata(self) -> furu.Metadata:
        return furu.Metadata(requires=furu.Requires(cpus=furu.at_least(8), memory=furu.GiB(48), gpus=0))

    def create(self) -> CountRowFiles:
        counts = ValContextCounts(shards=self.shards, levels=ALL_LEVELS).create()
        (val,) = FinewebShards(shards="val").create()
        targets = np.asarray(val.tokens[1 : VAL_TOKENS + 1])
        start = time.time()
        arrays = val_count_features(counts, targets, np.arange(VAL_TOKENS), ALL_LEVELS)
        self.logger.info(f"val_count_features: {time.time() - start:.0f} s")
        tot, _, cnt = arrays
        for j, k in enumerate(ALL_LEVELS):
            matched, with_target = (tot[:, j] > 0).mean(), (cnt[:, j] > 0).mean()
            self.logger.info(f"k={k}: {matched:.4f} of positions matched, {with_target:.4f} with the target")
        for name, array in zip(COUNT_NAMES, arrays):
            np.save(self.directory.data / f"{name}.npy", array)
        return CountRowFiles(self.directory.data, ALL_LEVELS)


@dataclass(frozen=True)
class GpuCountsResult:
    val_loss: float
    train_time_ms: int


class GpuCountsRun(furu.Spec[GpuCountsResult]):
    """record_367_gpu_counts/ trained and validated with seed `seed` on a node's 8 GPUs, its 1028 training shards
    staged in /dev/shm (else on the NVMe) before the clock, as the record's runs staged them."""

    seed: int = 0

    def metadata(self) -> furu.Metadata:
        return furu.Metadata(requires=furu.Requires(gpus=8))

    @furu.dependency
    def shards(self) -> list[furu.Spec]:
        return [FinewebShards(shards=TRAIN_SHARDS), FinewebShards(shards="val")]

    def create(self) -> GpuCountsResult:
        assert COUNT_SHARDS + 1 == TRAIN_SHARDS
        shutil.rmtree(SHM_STAGE, ignore_errors=True)  # a killed run's
        need = sum(s.path.stat().st_size for s in FinewebShards(shards=TRAIN_SHARDS).create())
        stage = SHM_STAGE if shutil.disk_usage("/dev/shm").free > 1.2 * need else None
        if stage is not None:
            stage.mkdir()
        self.logger.info(f"staging {TRAIN_SHARDS} shards ({need / 1e9:.0f} GB) in {stage or LOCAL_DISK}")
        LOCAL_DISK.mkdir(parents=True, exist_ok=True)
        try:
            # the code assembled on local disk, so the count kernels' nvcc build (track_1_short/counts/.build/) also
            # stays out of the snapshot
            with tempfile.TemporaryDirectory(dir=LOCAL_DISK) as code:
                script = assemble(Path(code) / "record")
                text = train_record(self.directory.data, TRAIN_SHARDS, {"TRAIN_SEED": str(self.seed)}, self.logger,
                                    script=script, stage=stage)
        finally:
            shutil.rmtree(SHM_STAGE, ignore_errors=True)
        final = re.findall(r"val_loss:([\d.]+) train_time:(\d+)ms", text)
        if not final:
            raise RuntimeError("the run log lacks the final validation")
        loss, ms = final[-1]
        return GpuCountsResult(val_loss=float(loss), train_time_ms=int(ms))


def assemble(to: Path) -> Path:
    """The variant's code in `to`: the root's track_1_short/ with record_367_gpu_counts/ laid over it. Returns its
    train_gpt.py."""
    ignore = shutil.ignore_patterns(".build", "__pycache__")
    shutil.copytree(REPO / "track_1_short", to / "track_1_short", ignore=ignore)
    shutil.copytree(GPU_COUNTS, to, ignore=ignore, dirs_exist_ok=True)
    return to / "train_gpt.py"


def create[T](specs: Sequence[furu.Spec[T]]) -> list[T]:
    """config.create's pools (the counts over 103 or 370 shards ask for 48 GiB)."""
    return slurm_create(specs, job_name="r367gpu")


BATCH: list[furu.Spec] = [ValCountRowsAt(shards=103), ValCountRowsAt(shards=370)]
# the record's four seeds (record_367_gpu_counts/README.md)
RUNS: list[furu.Spec] = [GpuCountsRun(seed=seed) for seed in range(4)]


if __name__ == "__main__":

    def _main() -> None:
        from retrieval_blog_post.r367_gpu_counts import BATCH, RUNS, create

        specs = [*BATCH, *RUNS]
        for spec, result in zip(specs, create(specs)):
            print(f"{spec}: {result}", flush=True)

    _main()
