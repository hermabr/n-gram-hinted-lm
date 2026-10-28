# Record #46: Batch size schedule (2025-11-29, 2.203 min)

## Source

The script at the top of modded-nanogpt `records/track_1_short/2025-11-29_BatchSizeSchedule/10e8f7c6-7175-4467-bdb0-a5de25d771a6.txt`
(= modded-nanogpt-public `train_gpt.py` at 25500ab). Its log: 2160 steps, val 3.2780, 132.2 s on 8 H100 (torch
2.10.0.dev, triton 3.6.0).

The record: 12 blocks at 768 dim (block 0 empty, block 7 without attention), skip 4 -> 7, backout of block 8's output,
smear gate, value embeddings, FP8 lm_head. Varlen FA3 over BOS-aligned documents with windows (3, 7, 11, 13 blocks of
128) and YaRN. NorMuon on the matrices and gates, DistAdam on the embeddings, head and scalars (odd steps only). Batch
8 -> 16 -> 24 x 16384 tokens over the thirds of 2120 scheduled steps, then 40 extension steps.

## Changes (each marked `[port]` in train_gpt.py)

- **Stack**: the record's `get_kernel('varunneal/flash-attention-3')` is not in the HF cache; the port loads the stock
  `kernels-community/flash-attn3` (same `flash_attn_varlen_func`, the same exact attention) with
  `kernels.get_local_kernel` on its snapshot (`get_kernel` with `HF_HUB_OFFLINE=1` fails on the partial snapshot).
  The import-time `torch.empty(1, device="cuda").backward()` uses `cuda:$LOCAL_RANK`, as #360 does, so ranks don't
  all create a context on GPU 0.
- **Loader**: the BOS-aligned loader's logic is kept; it also yields each input token's offset in its shard, and looks
  the rows up itself (`hints.train(shard, offsets)` / `hints.val(offsets)`) before the batch's pageable `.to(cuda)`,
  so the host gather (~2-3 ms) overlaps the GPU's previous step instead of adding to it. The lookups are timed.
- **Hints** (`HintEmbedding` as the GPT's `hint` submodule, compiled with the model):
  - site 0: added to `embed(input)` before the smear and `x0 = norm(...)`, as #367;
  - site 1 (`REC_HINT_MID`, default 8): right before block 8. #367 is this architecture without the empty block 0, so
    its layer i is block i + 1 here and this is #367's place (after its layer 6, the attention-less one with the
    skip). The hint enters before block 8's output is saved for the backout, as in #367;
  - site 2: after the backout, before the final norm;
  - cast to bf16, the site scales too, as #367.
  - The hint's parameters are outside the record's groups (whose filters would have put them in DistAdam). The record
    is not DDP, so the hint's gradients get `allreduce_grads`; `hint_optimizer` steps with the record's Adam on odd
    steps, on the step pair's summed gradients, at `set_lr_factor(get_lr(step))`.
  - Embedding scale (`REC_HINT_EMBED_RATIO` k, default 200): the hint reads `embed.weight / k` and site 0's scale
    starts at 0.1·k (#367's inits assume embedding std 0.005; this one's is 1). r023 instead puts the 1/k into
    next_scale's init: the same forward at init, but next_scale's relative Adam steps differ.
- **Budget**: `TRAIN_STEPS` (620), `STEP_SCALE = TRAIN_STEPS / 2160`: `num_scheduled_iterations =
  round(2120 x STEP_SCALE)` (609, 203 per stage) and the rest extension (11); the lr, window and batch-size schedules
  are fractions of the scheduled steps; Muon's momentum warmup and cooldown scale (86, 14). `LR_SCALE` scales every
  lr, `REC_HINT_LR_SCALE` the hint's on top. `SEED` seeds torch / numpy.
- **Timing**: as the record (4 x 10 compile steps before the clock, the states restored, the hint optimizer's too, a
  fresh loader); every training step is timed.
- **Output**: the `FINAL` line; with hints each validation also logs a `[port]` line of the hint's scales.

## Default steps: 620

The log's ms/step per stage on H100: 33.74, 60.13, 88.08, and 88.41 for the extension, + ~66 ms once. At 620 steps:
37.97 s on H100, ~37.0 on H200, + ~1-1.5 ms/step for the hints: ~37.6-38.0 s. 164.0M tokens, x 1.35 = 3 shards, + 1:
`train_shards` 4.

## Knobs

| env | default | what |
| --- | --- | --- |
| `TRAIN_STEPS` | 620 | total steps |
| `LR_SCALE` | 1 | multiplies every lr (record Adam 0.008 x lr_mul, NorMuon 0.023, hint 0.008) |
| `SEED` | 0 | torch/numpy seed |
| `REC_COOLDOWN_FRAC` | 0.55 | the lr cooldown's fraction of the scheduled steps |
| `REC_BS_SCHEDULE` | 131072,262144,393216 | the three stages' tokens per step (divisible by 8 x 2048) |
| `REC_BS_EXTENSION` | last stage's | the extension steps' batch size |
| `REC_LR_MAX` | 1.0,1.51,1.93 | the stages' lr peaks (the record's (bs / bs_0)^0.6); change with `REC_BS_SCHEDULE` |
| `REC_WS_SCHEDULE` | 3,7,11 | long attention window per stage (blocks of 128; short = half) |
| `REC_WS_FINAL` / `REC_WS_VALIDATE` | 13 / 20 | the extension's window / the final validation's long window |
| `REC_HINT_EMBED_RATIO` | 200 | the hint reads embed / k, site 0's scale starts at 0.1·k (1 = #367's literal inits) |
| `REC_HINT_MID` | 8 | site 1 is added right before this block |
| `REC_HINT_LR_SCALE` | 1 | the hint's lr on top of `LR_SCALE` |
| `REC_VAL_EVERY` | 250 | intermediate validations (off the clock; 0 = only the final one) |

## Checks

CPU only (FP8 and FA3 need Hopper); the GPU test is the smoke run of `probe` in old_records.py. On the real shards: the training
loader's `inputs == tokens[shard][offsets]` across a shard boundary, and the validation offsets; real-format rows
through the loader's lookup; the hint path under `torch.compile(fullgraph=True)` (aot_eager) equals eager, with
gradients on all three hint parameters.
