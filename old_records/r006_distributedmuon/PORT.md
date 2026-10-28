# Record #6: Distributed the overhead of Muon (2024-10-18, 13.1 min)

## Source

The script at the top of modded-nanogpt `records/track_1_short/2024-10-17_DistributedMuon/22d24867-eb5a-4fcc-ae2c-263d0277dfd1.txt`
(= modded-nanogpt-public `train_gpt2.py` at 7e1f20b). Its log: 5100 steps, val 3.2747, 153.8 ms/step on 8 H100,
torch 2.4.1. Record #7 is the same script on torch 2.5.0: 141.8 ms/step.

The record: 12 layers, 768 dim, 6 heads of 128, vocab padded to 50304, no biases, weightless RMSNorm, QK norm, rotary,
ReLU², zero-inited projections, wte tied to lm_head (nn.Linear's init). fp32 parameters under bf16 autocast, SDPA,
524,288 tokens per step, torch.compile then DDP. AdamW on the head (lr 0.0036, wd 0); Muon on the 72 block matrices
(lr 0.00036, momentum 0.95, each rank orthogonalizing every 8th matrix, one all_reduce sharing the updates). Constant
lr, then linear decay over the last 1450 steps.

## Changes (each marked `port:` in train_gpt.py)

- **Stack**: the Rotary cos/sin caches are filled eagerly before compiling (in the record the first compiled forward
  filled them on the CPU and the next call recompiled). No `coordinate_descent_tuning`.
- **Loader**: the record's, quirks included (ranks 6-7 switch shards a step early); `next_batch` also returns the
  batch's hint rows (`train_span` / `val_span`), looked up while the GPU runs the step.
- **First batch**: the record trained on batch 0 twice (loaded before the clock, then the loader reset); the port
  loads it on the clock after the reset.
- **Hints**: `HintEmbedding(768, cells, embed_std=0.02083)` (ratio 4.167) is the GPT's `hint_embedding` submodule,
  outside both record optimizers; `hint_optimizer` with a `LambdaLR(get_lr)` like theirs. Sites:
  `x = wte(idx) + s0 * hint`; after 7 of 12 blocks (`REC_HINT_MID`); before the final rms_norm.
- **Budget**: `TRAIN_STEPS` (245), warmdown `round(1450/5100 x TRAIN_STEPS)` (70). `LR_SCALE` scales every lr, the
  hint's included. `SEED` seeds torch / numpy.
- **Batch**: `REC_DEVICE_BATCH` smaller than 64 keeps 512 sequences per step by accumulation (smoke tests only).
- **Timing**: the record reset its clock at step 10. The port runs `REC_COMPILE_WARMUP` (10) steps before the clock
  instead, then restores the model, every optimizer and the loader; every step is timed. The per-step
  `train_loss.item()` stays on the clock as in the record.
- **Output**: the `FINAL` line. The final checkpoint save is off (`REC_SAVE_CHECKPOINT=1`).

## Default steps: 245

~154 ms/step for the record (its log; r001 ran at 158.8 ms/step on 8 H200 with hints, similar compute) + ~1-2 ms for
the hints: ~155 ms/step, 245 steps ~38 s. 128.5M tokens: 2 shards, `train_shards` 3 (up to ~378 steps).

## Knobs

| env | default | what |
| --- | --- | --- |
| `TRAIN_STEPS` | 245 | total steps; the warmdown scales with it |
| `LR_SCALE` | 1 | multiplies every lr (AdamW 0.0036, Muon 0.00036, hint 0.008) |
| `SEED` | 0 | torch/numpy seed |
| `REC_WARMDOWN_FRAC` | 0.2843 (1450/5100) | the lr decay's fraction of the run |
| `REC_HINT_EMBED_RATIO` | 4.167 | hint_runtime's embedding ratio (1 = #367's literal inits) |
| `REC_HINT_LR` | 0.008 | the hint's Adam lr, times LR_SCALE |
| `REC_HINT_MID` | 7 | site 1 is after this many of the 12 blocks |
| `REC_VAL_EVERY` | 125 | intermediate validations (off the clock; 0 = only the final one) |
| `REC_COMPILE_WARMUP` | 10 | training steps before the clock |
| `REC_DEVICE_BATCH` | 64 | sequences per GPU (smoke tests only) |
| `REC_SAVE_CHECKPOINT` | 0 | 1 = the record's final checkpoint save |

## Checks

GPU smoke test on 2 RTX 6000 Ada (`REC_DEVICE_BATCH=8`): fake rows and `HINTS=0`, exit 0, no recompiles. On the CPU:
123,568,128 parameters; with the site scales at 0 the loss equals the no-hint loss; the optimizers cover every
non-hint parameter; one dynamo graph; the loader on the real shards.
