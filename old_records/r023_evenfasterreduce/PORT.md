# Record #23: Overlap computation and gradient communication (2025-05-25, 2.979 min)

## Source

The script at the top of modded-nanogpt `records/track_1_short/2025-05-25_EvenFasterReduce/6ae86d05-5cb2-4e40-a512-63246fd08e45.txt`
(modded-nanogpt-public 16648e0). Its log: 1770 steps, val 3.2782, 178.7 s on 8 H100. 12 layers, 768 dim, U-net,
FlexAttention with a sliding window growing 128 -> 1792 tokens, FP8 lm_head, Muon + Adam, gradients averaged by
bucketed `all_reduce_coalesced` from post-accumulate-grad hooks.

## Changes (each marked `port:` in train_gpt.py)

- **Stack**: runs unchanged on torch 2.10. The `world_size == 8` assert allows 1-2 GPUs (smoke tests).
- **Loader**: also yields each rank's shard and first offset; the step gets `hints.train_span(shard, start, 49152)`,
  validation `hints.val_span(start, 262144)`. Rows are looked up on the host and copied from pinned memory, so the
  record's loop still never syncs.
- **Hints**: `HintEmbedding` is a GPT submodule, compiled with the model and averaged by the record's bucketed
  all-reduce; outside the record's Adam groups; `hint_optimizer` sits in the record's `optimizers` list, so it follows
  `get_lr` and is restored after the compile warmup. Sites: `embed(input) + s0 * hint` before `x0 = norm(x)` (as
  #367); after 7 of 12 blocks; before the final norm. Cast to bf16.
- **Hint inits** (`REC_HINT_EMBED_RATIO` k, default 200): #367's inits assume an embedding of std 0.005; this one has
  std 1, so the next scale starts at 4/k and site 0's scale at 0.1·k, giving the hint #367's size relative to the
  stream at every site.
- **Budget**: `TRAIN_STEPS` (380). The cooldown and window ramp are fractions already; the momentum warmup becomes
  `round(300 x STEP_SCALE)`. `LR_SCALE` scales every lr, the hint's included.
- **Timing**: as the record (10 compile steps on random tokens before the clock, states restored, the hint's too).
- **Output**: the `FINAL` line.

## Default steps: 380

100.7 ms/step on H100 averaged over the whole window ramp, ~98.2 on H200, + ~0.5-1 ms for the hints: 380 steps
~38.0 s. 149.4M tokens: 2 shards, `train_shards` 3.

## Knobs

| env | default | what |
| --- | --- | --- |
| `TRAIN_STEPS` | 380 | total steps |
| `LR_SCALE` | 1 | multiplies every lr (Adam 0.22/0.6/0.04, Muon 0.05, hint 0.008) |
| `SEED` | 0 | torch/numpy seed |
| `REC_HINT_EMBED_RATIO` | 200 | next scale 4/k, site-0 scale 0.1·k (1 = #367's literal inits) |
| `REC_HINT_LR` | 0.008 | the hint's Adam lr, times LR_SCALE |
| `REC_HINT_MID` | 7 | site 1 is after this many of the 12 blocks |
| `REC_COOLDOWN_FRAC` | 0.4 | the lr cooldown's fraction of the run |
| `REC_VAL_EVERY` | 125 | intermediate validations (off the clock; 0 = only the final one) |

## Checks

CPU only (needs sm89+ for the FP8 head); the GPU test is the smoke run of `probe` in old_records.py. On a 12 x 256 model: with the
site scales at 0 the loss equals the no-hint loss; the hint's parameters are outside the record's Adam groups; the
loader's (shard, start) on the real shards across the switch to shard 1; the torch 2.10 APIs the record uses exist.
