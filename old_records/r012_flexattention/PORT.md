# Record #12: 1024-ctx dense causal attention -> 64K-ctx FlexAttention (2024-11-19, 5.03 min)

## Source

The script at the top of modded-nanogpt `records/track_1_short/2024-11-19_FlexAttention/8384493d-dba9-4991-b16b-8696953f5e6d.txt`
(= modded-nanogpt-public `train_gpt2.py` at ebce774, but for a comment). Its log: 1875 steps, val 3.2783, 301.8 s on
8 H100 (run-to-run std ~0.005).

The record: 12 layers, 768 dim, 6 heads of 128, U-net with learned skip weights, value residual, x0 mixing, QK norm,
rotary, ReLU², zero-inited projections and head, tanh softcap 30. FlexAttention over one 64K-token sequence per GPU,
document-causal with a fixed 1024-token window. bf16 model but for the CastedLinear weights. Adam on wte (0.6),
lm_head (0.008) and the scalars (0.04); Muon on the block matrices (0.04, momentum warmup 0.85 -> 0.95 over 500
steps). Constant lr, then linear decay over the last 562 steps.

## Changes (each marked `port:` in train_gpt.py)

- **Stack**: `create_block_mask(..., _compile=True)` inside the compiled forward is deprecated in torch 2.10, and its
  warning breaks the graph; the port drops `_compile=True` (as the warning recommends), so the mask compiles inside
  the model's graph (1 graph, 0 breaks). `REC_COORD_DESCENT=0` switches off `coordinate_descent_tuning` (smoke tests).
- **Loader**: the record's, quirks included (rank 7 switches shards a step early); `next_batch` also returns the
  batch's hint rows (`train_span` / `val_span`), looked up while the GPU runs the step.
- **Hints**: `HintEmbedding(768, cells, embed_std=1.0)`, nn.Embedding's init (ratio 200: site 0's scale starts at
  20), attached after the record's `.bfloat16()` so it stays fp32; outside the record's four optimizers;
  `hint_optimizer` with a `LambdaLR(get_lr)` like theirs. Sites: `x = wte(idx) + s0 * hint` before `x0 = rms_norm(x)`
  (so x0 and every block's `lambdas[1] * x0` see it, as in #367); after 7 of 12 blocks (`REC_HINT_MID`; after the
  first decoder block, before block 7's skip); before the final rms_norm. Cast to bf16.
- **Budget**: `TRAIN_STEPS` (240), warmdown `round(562/1875 x TRAIN_STEPS)` (72), momentum warmup
  `round(500 x STEP_SCALE)` (64). `LR_SCALE` scales every lr, the hint's included. `SEED` seeds torch / numpy.
- **Timing**: the record reset its clock at step 10. The port runs `REC_COMPILE_WARMUP` (10) steps before the clock
  instead, then restores the model, every optimizer and the loader.
- **Output**: the `FINAL` line. The final checkpoint save is off (`REC_SAVE_CHECKPOINT=1`).

## Default steps: 240

161.8 ms/step on H100 (flat over the run: the window is fixed), ~157.8 on H200, + ~1 ms for the hints: 240 steps
~38.1 s. 125.8M tokens: 2 shards, `train_shards` 3.

## Knobs

| env | default | what |
| --- | --- | --- |
| `TRAIN_STEPS` | 240 | total steps; warmdown and momentum warmup scale with it |
| `LR_SCALE` | 1 | multiplies every lr (Adam 0.6 / 0.008 / 0.04, Muon 0.04, hint 0.008) |
| `SEED` | 0 | torch/numpy seed |
| `REC_WARMDOWN_FRAC` | 0.2997 (562/1875) | the lr decay's fraction of the run |
| `REC_HINT_EMBED_RATIO` | 200 | hint_runtime's embedding ratio (1 = #367's literal inits) |
| `REC_HINT_LR` | 0.008 | the hint's Adam lr, times LR_SCALE |
| `REC_HINT_MID` | 7 | site 1 is after this many of the 12 blocks (6..11) |
| `REC_VAL_EVERY` | 125 | intermediate validations (off the clock; 0 = only the final one) |
| `REC_COMPILE_WARMUP` | 10 | training steps before the clock |
| `REC_SAVE_CHECKPOINT` | 0 | 1 = the record's final checkpoint save |
| `REC_COORD_DESCENT` | 1 | 0 = no coordinate-descent tuning (faster compile; smoke tests only) |

## Checks

No GPU smoke test finished (the compile outlasted the preemptible job); the GPU test is the smoke run of `probe` in old_records.py. On
the CPU (eager FlexAttention, 12 x 256 model): with the site scales at 0 the loss equals the no-hint loss; the record's
optimizers cover every non-hint parameter; one dynamo graph with and without hints; the loader on the real shards.
