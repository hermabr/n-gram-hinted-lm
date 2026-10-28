"""Record #367 with exact n-gram counts built on the GPUs (old_records/gpu_val_stats.py's build mixed into the model's
output distribution: the blog post's 3.1 s variant), checked position by position against these CPU references: the
validation count rows over the first 103 (10.3B tokens) and 370 (37B) training shards.

    uv run src/retrieval_blog_post/r367_gpu_counts.py   # keep it running (e.g. in tmux): it coordinates the workers
"""

import time
from collections.abc import Sequence

import furu
import numpy as np

from retrieval_blog_post.fineweb import FinewebShards
from retrieval_blog_post.kgram import VAL_TOKENS
from retrieval_blog_post.old_records import COUNT_NAMES, CountRowFiles, val_count_features
from retrieval_blog_post.config import create as slurm_create
from retrieval_blog_post.val_hints import ALL_LEVELS, ValContextCounts


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


def create[T](specs: Sequence[furu.Spec[T]]) -> list[T]:
    """config.create's pools (the counts over 103 or 370 shards ask for 48 GiB)."""
    return slurm_create(specs, job_name="r367gpu")


BATCH: list[furu.Spec] = [ValCountRowsAt(shards=103), ValCountRowsAt(shards=370)]


if __name__ == "__main__":

    def _main() -> None:
        from retrieval_blog_post.r367_gpu_counts import BATCH, create

        for spec, result in zip(BATCH, create(BATCH)):
            print(f"{spec}: {result}", flush=True)

    _main()
