# Record #3: Introduced the Muon optimizer (2024-10-04, 24.9 min)

## Source

No log in modded-nanogpt `records/`. The record is modded-nanogpt-public's `train_gpt2.py` + `run.sh` at d4bc211
(2024-10-05): c26c2a1's README describes the record ("7000 steps on 3.67B tokens ... ~3.280"), and the commits from
c26c2a1 to d4bc211 only rename `ProposedOptimizer` to `OrthogonalNesterov` and edit comments. (0ee2c25, on the
`nowarmup` branch, is the next recipe, not this record.) run.sh's arguments are the script's defaults now:
`--model d12 --batch_size 64 --sequence_length 1024 --val_loss_every 128 --num_iterations 7000 --weight_decay 0.1
--learning_rate 0.0018 --warmup_iters 250 --warmdown_iters 2000`.

The record: GPT-2 124M (12 layers, 12 heads of 64, 768 dims, vocab 50257), no biases, weightless rmsnorm (eps 1e-6,
fp32), rotary, GELU, attention output x 1/sqrt(24), wte tied to lm_head, nn.Linear's default init. fp32 parameters
under bf16 autocast, SDPA, torch.compile then DDP; 524,288 tokens per step. AdamW on the head (lr 0.0018, wd 0) and
OrthogonalNesterov ("Muon": lr 0.018, momentum 0.95, Nesterov, 5 bf16 Newton-Schulz steps, every rank orthogonalizing
every matrix) on the 48 block matrices. run.sh's `--weight_decay 0.1` is never used (kept so). Linear warmup 250,
constant, linear decay over the last 2000. Validation in fp32 (no autocast).

## Changes (each marked `port:` in train_gpt.py)

- **Stack**: the Rotary cos/sin caches are filled eagerly before compiling; `CombinedOptimizer` gets a
  `load_state_dict` for the restore. No `coordinate_descent_tuning`.
- **Validation**: `val_tokens` = 10,485,760 replaces `--val_max_steps 20`.
- **Loader**: the record's, quirks included (ranks 6-7 switch shards a step early: 189 vs 190 steps per shard);
  `next_batch` also returns the batch's hint rows (`train_span` / `val_span`), looked up while the GPU runs the step.
- **Hints**: `HintEmbedding(768, cells, embed_std=0.02083)`, the tied weight's init std (ratio 4.167; site scales
  start at 0.417, 8, 8), is the GPT's `hint_embedding` submodule, outside both record optimizers, with
  `hint_optimizer` following `get_lr(step)`. Sites: `x = wte(idx) + s0 * hint`; after 7 of 12 blocks
  (`REC_HINT_MID`); before the final rmsnorm.
- **Budget**: `TRAIN_STEPS` (176); warmup and warmdown as fractions (6 and 50 steps at 176). `LR_SCALE` scales both
  record optimizers and the hint. `SEED` seeds torch / numpy (the record had no seed).
- **Batch**: 512 sequences per step; `REC_DEVICE_BATCH` smaller than 64 or fewer GPUs keep it through the record's
  accumulation (smoke tests only).
- **Timing**: the record timed each step on its own, the first including the compile. The port runs
  `REC_COMPILE_WARMUP` (10) steps before the clock, then restores every optimizer's state, the model and the loader.
  The record's per-step sync, all_reduce and print stay on the clock (~0.3 ms). The first batch is loaded on the clock.
- **Output**: the `FINAL` line. The final checkpoint save is off (`REC_SAVE_CHECKPOINT=1`).
- **Optional modes**: `HINT_NGRAM=1` as r001's (on the record's fp32 `logits.float()`), `HINT_MIX=1` through
  `mixed_nll`, and `HINT_ONLINE=1` (online_hints.py).

## Default steps: 176

No log: 24.9 min / 7000 steps = 213.4 ms/step on 8 H100, matching record #4's log (same model and optimizer, 216.3
ms/step). H200 and the missing coordinate-descent tuning taken to cancel out; + ~2 ms for the hints: ~215.5 ms/step,
38 s / 0.2155 s = 176. 92.3M tokens, all in shard 0; `train_shards` 3 covers up to ~378 steps.

## Knobs

| env | default | what |
| --- | --- | --- |
| `TRAIN_STEPS` | 176 | total steps; the warmup and warmdown scale with it |
| `LR_SCALE` | 1 | multiplies every lr (AdamW 0.0018, OrthogonalNesterov 0.018, hint 0.008) |
| `SEED` | 0 | torch/numpy seed |
| `REC_WARMUP_FRAC` | 0.0357 (250/7000) | the lr warmup's fraction of the run |
| `REC_WARMDOWN_FRAC` | 0.2857 (2000/7000) | the lr decay's fraction of the run |
| `REC_HINT_EMBED_RATIO` | 4.167 | hint_runtime's embedding ratio (1 = #367's literal inits) |
| `REC_HINT_LR` | 0.008 | the hint's Adam lr, times LR_SCALE |
| `REC_HINT_MID` | 7 | site 1 is after this many of the 12 blocks |
| `REC_VAL_EVERY` | 128 | intermediate validations (off the clock; 0 = only the final one) |
| `REC_COMPILE_WARMUP` | 10 | training steps before the clock |
| `REC_DEVICE_BATCH` | 64 | sequences per GPU (smoke tests only) |
| `REC_SAVE_CHECKPOINT` | 0 | 1 = the record's final checkpoint save |

## Checks

GPU smoke tests on 2 A40 / L40S (`REC_DEVICE_BATCH=8` or 16, accumulating): fake rows, `HINTS=0`, and real rows with
`HINT_NGRAM=1`, each exit 0 with no recompiles or graph breaks. On the CPU: 123,532,032 parameters; with the site
scales at 0 the loss equals the no-hint loss; the optimizers cover every non-hint parameter; the restore; one dynamo
graph; the loader on the real shards; `HINT_NGRAM` off is bit-identical to the port without it.
