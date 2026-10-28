"""How the validation loss scales with the training data alone: models trained as the #367 record trains them, but
without hints (train_gpt.py's NO_HINTS: blank retrieval rows in training and validation, no index built), each on its
own N training tokens, at the validation-cache sweep's sizes N (val_cache.py). The model and the batch (131,072 tokens)
stay the record's; the run is N / 131,072 steps, the record's 560 + 10 step recipe stretched to it (the record now
runs the same recipe at 530 + 10, its step counts these times 540 / 570): the stage lengths, the lr cooldown and the
loss-weight schedules are fractions of the run already, the step counts written for 570 steps (warmups, tail-average
windows, cadence switches) are stretched by STEP_SCALE, and the weight decays scaled by
WD_SCALE = STEP_SCALE ** -wd_exponent: 1 keeps the run's total decay the record's (Wang & Aitchison's timescale
proportional to the run), 0.5 decays more over longer runs (the repo's sweeps from 570 to 1450 steps put the best
decay falling slower than 1/steps, without telling 0.5 from 1).

DataScaling decides between them first: it creates the CHECK runs at 2.4B (two weight-decay exponents by two lr
floors), then every size with the best of them. The shards of all its runs are made first, on CPU workers (data jobs
ask for no GPUs); then DataScaling and the runs, which all ask for a miso node's 8 GPUs, go to the GPU workers (up to
config.create's five GPU nodes, one run each at a time):

    uv run src/retrieval_blog_post/data_scaling.py      # keep it running (e.g. in tmux): it coordinates the workers

N counts the tokens trained on. The loader skips the tails of documents past the stage's sequence length, so the run
reads further into the shards than N (35B trained tokens read the first ~446 shards; the sweep's 37B cache is the
first 370)."""

import math
import re
from dataclasses import dataclass

import furu

from retrieval_blog_post.fineweb import TRAIN_SHARDS, FinewebShards
from retrieval_blog_post.kgram import SHARD_TOKENS
from retrieval_blog_post.val_cache import SIZES, train_record

# every stage's batch (track_1_short/config.py's TRAINING_STAGES)
BATCH_TOKENS = 8 * 2048 * 8
RECORD_STEPS = 570  # 560 + the 10-step extension stage: the recipe the runs stretch (track_1_short's scaled_steps)
EXTENSION_STEPS = 10
# Shard tokens the loader reads per trained token, with a margin: measured on shard 1, 1.49 at sequence length 896,
# 1.22 at 2048 and 1.16 at 3072, 1.27 over the record's stages.
SHARD_TOKENS_PER_TOKEN = 1.35
# The check before the sizes: the weight-decay rule and the lr floor at 2.4B (18,312 steps), where the 1/steps and
# 1/sqrt(steps) decays are 5.7x apart. Seed noise is about +-0.003.
CHECK_TOKENS = 2_400_000_000
CHECK_WD_EXPONENTS = (0.5, 1.0)
CHECK_LR_FLOORS = (0.30, 0.10)


@dataclass(frozen=True)
class NoHintsLoss:
    """The validation loss of a model trained without hints on `tokens` tokens (`steps` steps)."""

    tokens: int
    steps: int
    val_loss: float
    train_time_ms: int


class NoHintsTraining(furu.Spec[NoHintsLoss]):
    """The record without hints, seed `seed`, trained on about `tokens` tokens: a whole number of 4-step blocks of
    BATCH_TOKENS (the tail-average windows and the n-gram table's period-4 cadence want an even, 4-aligned run). The
    weight decays scale by STEP_SCALE ** -`wd_exponent`; the lr cools down to `lr_floor` (the record's 0.30)."""

    tokens: int
    wd_exponent: float
    lr_floor: float
    seed: int = 0

    def metadata(self) -> furu.Metadata:
        return furu.Metadata(requires=furu.Requires(gpus=8))

    @property
    def steps(self) -> int:
        return 4 * round(self.tokens / (4 * BATCH_TOKENS))

    @property
    def shards(self) -> int:
        """The training shards the run stages: what it reads, with SHARD_TOKENS_PER_TOKEN's margin and one more."""
        shards = (
            math.ceil(SHARD_TOKENS_PER_TOKEN * self.steps * BATCH_TOKENS / SHARD_TOKENS)
            + 1
        )
        if shards > TRAIN_SHARDS:
            raise ValueError(
                f"{self.steps} steps read past the {TRAIN_SHARDS} training shards"
            )
        return shards

    def create(self) -> NoHintsLoss:
        steps = self.steps
        step_scale = steps / RECORD_STEPS
        text = train_record(
            self.directory.data,
            self.shards,
            {
                "TRAIN_SEED": str(self.seed),
                "NO_HINTS": "1",
                "NUM_SCHEDULED_ITERATIONS": str(steps - EXTENSION_STEPS),
                "STEP_SCALE": repr(step_scale),
                "WD_SCALE": repr(step_scale**-self.wd_exponent),
                "LR_FLOOR": repr(self.lr_floor),
            },
            self.logger,
        )
        final = re.search(
            rf"step:{steps}/{steps} val_loss:([\d.]+) train_time:(\d+)ms", text
        )
        if final is None:
            raise RuntimeError("the run log lacks the final validation loss")
        return NoHintsLoss(
            tokens=steps * BATCH_TOKENS,
            steps=steps,
            val_loss=float(final[1]),
            train_time_ms=int(final[2]),
        )


@dataclass(frozen=True)
class DataScalingLosses:
    """The validation loss per size `tokens` (the nominal N), at the weight-decay exponent and lr floor the check
    chose, and the check's own losses."""

    tokens: tuple[int, ...]
    losses: tuple[NoHintsLoss, ...]
    wd_exponent: float
    lr_floor: float
    check: tuple[
        tuple[float, float, float], ...
    ]  # (wd_exponent, lr_floor, val_loss) per check run


class DataScaling(furu.Spec[DataScalingLosses]):
    """The record without hints, seed `seed`, at each size of `tokens`, with the weight-decay exponent and lr floor of
    the lowest validation loss among the CHECK runs. The runs are found as it runs: the check runs first, then the
    sizes. Not a GPU job itself, but it asks for one, as they do, so that the GPU worker that ran the last check picks
    it up and goes on to the sizes (an idle worker exits after a minute)."""

    tokens: tuple[int, ...] = SIZES
    seed: int = 0

    def metadata(self) -> furu.Metadata:
        return furu.Metadata(requires=furu.Requires(gpus=8))

    @furu.dependency
    def shards(self) -> list[FinewebShards]:
        """The shards every run stages (train_record's FinewebShards), made on CPU workers before any GPU run, which
        would otherwise wait for them. The counts depend on the sizes alone, not on the check's choice."""
        counts = {run.shards for run in (*self._checks(), *self._runs(0.5, 0.30))}
        return [
            *(FinewebShards(shards=n) for n in sorted(counts)),
            FinewebShards(shards="val"),
        ]

    def _checks(self) -> list[NoHintsTraining]:
        return [
            NoHintsTraining(
                tokens=CHECK_TOKENS, wd_exponent=e, lr_floor=f, seed=self.seed
            )
            for e in CHECK_WD_EXPONENTS
            for f in CHECK_LR_FLOORS
        ]

    def _runs(self, wd_exponent: float, lr_floor: float) -> list[NoHintsTraining]:
        return [
            NoHintsTraining(
                tokens=n, wd_exponent=wd_exponent, lr_floor=lr_floor, seed=self.seed
            )
            for n in self.tokens
        ]

    def create(self) -> DataScalingLosses:
        checks = self._checks()
        check_losses = furu.create(checks)
        best = min(zip(checks, check_losses), key=lambda c: c[1].val_loss)[0]
        self.logger.info(
            f"check: wd exponent {best.wd_exponent} and lr floor {best.lr_floor}"
        )
        runs = self._runs(best.wd_exponent, best.lr_floor)
        return DataScalingLosses(
            tokens=self.tokens,
            losses=tuple(furu.create(runs)),
            wd_exponent=best.wd_exponent,
            lr_floor=best.lr_floor,
            check=tuple(
                (c.wd_exponent, c.lr_floor, loss.val_loss)
                for c, loss in zip(checks, check_losses)
            ),
        )


if __name__ == "__main__":

    def _main() -> None:
        # Imported, not this file's own definitions: run as a file, they would be __main__.DataScaling, another spec.
        from retrieval_blog_post.data_scaling import DataScaling
        from retrieval_blog_post.config import create

        (result,) = create([DataScaling()], job_name="data-scaling")
        for e, f, loss in result.check:
            print(f"check at 2.4B: wd exponent {e}, lr floor {f}: val loss {loss:.4f}")
        print(f"chosen: wd exponent {result.wd_exponent}, lr floor {result.lr_floor}")
        for n, r in zip(result.tokens, result.losses):
            print(f"{n:>14,} tokens: {r.steps:>7,} steps, val loss {r.val_loss:.4f}")

    _main()
