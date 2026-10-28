"""Run configuration: hyperparameters and the training-stage table.

The run's own environment variables (DATA_PATH, NUM_SCHEDULED_ITERATIONS, TRAIN_SEED, VAL_CACHE_SWEEP, NO_HINTS,
VAL_TOKEN_LOSSES, STEP_SCALE, WD_SCALE) are read here; LR_FLOOR in schedule.py.
The others: torchrun's RANK / LOCAL_RANK / WORLD_SIZE (distributed.py; train_gpt.py also reads
LOCAL_RANK), PYTORCH_ALLOC_CONF (set by train_gpt.py), and CUDA_HOME (perf/kernels/cross_entropy.py,
for the CE kernel's CUDA headers).

Provenance: the five-stage table (seq 896 / 2048 / 3072 under a 2560 attention cap, the terminal stage and the
extension) is record #360's; here every stage trains at batch 8.
"""
import os
import uuid
from dataclasses import dataclass

@dataclass(slots=True)
class Hyperparameters:
    # data
    data_path = os.environ.get("DATA_PATH", ".")
    train_files: str = os.path.join(data_path, "data/fineweb100B/fineweb_train_*.bin") # input .bin to train on
    val_files: str = os.path.join(data_path, "data/fineweb100B/fineweb_val_*.bin") # input .bin to eval validation loss on
    val_index_shards: int = 370 # the validation retrieval index covers the first this many training shards (37B tokens)
    # Off the clock, after the final validation: evaluate the same model again with a validation index over the first
    # this many training tokens each (0: no hints), comma-separated (track_1_short/val_cache_sweep.py). Unset in records.
    val_cache_sweep: tuple[int, ...] = tuple(int(n) for n in os.environ.get("VAL_CACHE_SWEEP", "").split(",") if n)
    # Train and validate without retrieval hints: every row blank, no index built (the data-scaling runs,
    # src/retrieval_blog_post/data_scaling.py). Unset in records.
    no_hints: bool = os.environ.get("NO_HINTS") == "1"
    # Off the clock, after the final validation: write its per-token losses and hints, and the same model's losses
    # without hints, to this directory (track_1_short/token_losses.py). Unset in records.
    val_token_losses: str = os.environ.get("VAL_TOKEN_LOSSES", "")
    val_tokens: int = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    # batch sizes
    val_batch_size: int = 4 * 64 * 1024 * 8
    # schedule: the step count of the main stages, then the extension stage. The override is for
    # step-count sweeps (record #360's KX_STEPS); the step constants elsewhere are tuned for 530 + 10.
    num_scheduled_iterations: int = int(os.environ.get("NUM_SCHEDULED_ITERATIONS", "530"))
    num_extension_iterations: int = 10
    # evaluation and logging
    run_id: str = f"{uuid.uuid4()}"
    # Every how many steps to evaluate val loss; 0 = only at the end (record #360: each intermediate
    # validation is on the clock -- the val reads and n-gram pulls, the value_embeds gather, the drained
    # CUDA-graph run-ahead and the cycle re-land after it cost ~0.2-0.5 s over a run at 250).
    val_loss_every: int = 0
    save_checkpoint: bool = False
    run_evals: bool = False  # run additional evaluations after training is completed
    # reproducibility: unset means a random init; any integer (0 included) seeds it
    train_seed: int | None = int(os.environ["TRAIN_SEED"]) if "TRAIN_SEED" in os.environ else None

@dataclass(slots=True)
class TrainingStage:
    lr_mul: float
    batch_size: int
    window_sizes: tuple[int, int]  # (short, long) in block units
    mtp_weights_start: list[float]
    mtp_weights_end: list[float]
    train_max_seq_len: int  # documents are cut to this many tokens
    # Prefix-token-prediction loss weight, stepped: the stage's steps split into equal parts, one weight each.
    prefix_weights: tuple[float, ...]
    duration: float | None = None  # fraction of num_scheduled_iterations; None for the extension stage


# The record's recipe stretched to a longer run (the data-scaling runs): the step counts written for its 560 + 10 steps
# (warmups, tail-average windows, cadence switches; the record now runs 530 + 10, with its own counts) are multiplied
# by STEP_SCALE, and the weight decays by WD_SCALE. Unset in records (None and 1).
STEP_SCALE = float(os.environ["STEP_SCALE"]) if "STEP_SCALE" in os.environ else None
WD_SCALE = float(os.environ.get("WD_SCALE", "1"))


def scaled_steps(record: int, at_570: int, multiple: int = 1) -> int:
    """The record's step count `record`; in a stretched run, the 560 + 10 step recipe's `at_570` times STEP_SCALE,
    rounded to a multiple of `multiple`."""
    if STEP_SCALE is None:
        return record
    return max(multiple, round(at_570 * STEP_SCALE / multiple) * multiple)


# Stage lengths as fractions of num_scheduled_iterations (normalized to sum to 1). The last entry is
# the seq-3072 stage plus record #360's terminal (batch taper) stage, which takes TAPER_DURATION of it.
STAGE_DURATIONS = [0.2854, 0.3209, 0.3931]
STAGE_DURATIONS = [d / sum(STAGE_DURATIONS) for d in STAGE_DURATIONS]
TAPER_DURATION = 0.06
# Model width (record #360: 768); heads are 128 wide.
MODEL_DIM = 512
# Attention window sizes are counted in blocks of this many tokens (record #360).
BLOCK_SIZE = 128
# Documents may run to 3072 tokens in the last stages, but attention segments are cut at this many
# tokens (the loader adds segment boundaries; the token stream is untouched). A whole number of blocks.
VIRTUAL_SEQ_CAP = 2560
# The lr decays linearly to its floor over the last LR_COOLDOWN_FRAC of the main stages (record #360: 0.80).
LR_COOLDOWN_FRAC = 0.5
# embed unties from lm_head at the start of this stage, the extension stage (record #360).
SPLIT_EMBED_STAGE = 4
# The final validation extends the long attention window to this many blocks (record #360).
WS_POST_YARN_EXT = 20

# window_sizes are in units of BLOCK_SIZE tokens. Every stage trains at batch 8 (units of 2048 * 8 tokens), with
# lr_mul 1 (record #360 grew the batch 8 -> 16 -> 24, tapered to 20, with lr_mul 1.52, 1.73, 1.58).
TRAINING_STAGES = [
    TrainingStage(duration=STAGE_DURATIONS[0], train_max_seq_len=896, batch_size=8 * 2048 * 8, window_sizes=(1, 3), lr_mul=1.0,
                  mtp_weights_start=[1.0, 0.5, 0.25], mtp_weights_end=[1.0, 0.5, 0.0], prefix_weights=(0.25,)),
    TrainingStage(duration=STAGE_DURATIONS[1], train_max_seq_len=2048, batch_size=8 * 2048 * 8, window_sizes=(3, 7), lr_mul=1.0,
                  mtp_weights_start=[1.0, 0.5], mtp_weights_end=[1.0, 0.0], prefix_weights=(0.20, 0.15, 0.10, 0.05)),
    TrainingStage(duration=STAGE_DURATIONS[2] - TAPER_DURATION, train_max_seq_len=3072, batch_size=8 * 2048 * 8, window_sizes=(5, 11), lr_mul=1.0,
                  mtp_weights_start=[1.0], mtp_weights_end=[1.0], prefix_weights=(0.0,)),
    # record #360's terminal batch taper: the same batch and windows as the stage before it here, but the full
    # softmax (sampled_softmax.py samples candidates in stages 0-2 only)
    TrainingStage(duration=TAPER_DURATION, train_max_seq_len=3072, batch_size=8 * 2048 * 8, window_sizes=(5, 11), lr_mul=1.0,
                  mtp_weights_start=[1.0], mtp_weights_end=[1.0], prefix_weights=(0.0,)),
    # extension stage at batch 8, the lr at its floor
    TrainingStage(train_max_seq_len=3072, batch_size=8 * 2048 * 8, window_sizes=(6, 13), lr_mul=1.0,  # lr_mul is not used (lr sits at the floor)
                  mtp_weights_start=[1.0], mtp_weights_end=[1.0], prefix_weights=(0.0,)),
]
assert VIRTUAL_SEQ_CAP % BLOCK_SIZE == 0
assert any(s.train_max_seq_len > VIRTUAL_SEQ_CAP for s in TRAINING_STAGES), "VIRTUAL_SEQ_CAP would be inert"
