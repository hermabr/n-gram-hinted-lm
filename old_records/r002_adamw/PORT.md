# Record #2: Tuned learning rate & rotary embeddings (2024-06-06, 31.4 min)

## Source

- Log: modded-nanogpt `records/track_1_short/2024-06-06_AdamW/f66d43d7-e449-4029-8adf-e8537bab49ea.log` (llm.c
  format, `tel` = val loss): 9536 steps, val 3.275959, 2^19 tokens per step.
- Code: modded-nanogpt-public `train_gpt2.py` at 3e48a54 ("upgrade with rotary embeddings") with `run.sh` at
  7751345, the commit that set 9536 steps (`train_gpt2.py` is unchanged up to b9d141e). run.sh's arguments are the
  script's defaults now: `--model d12 --batch_size 64 --sequence_length 1024 --total_batch_size 524288
  --val_loss_every 128 --num_iterations 9536 --weight_decay 0.1 --learning_rate 0.0018 --warmup_iters 256
  --warmdown_iters 2048` (the log README's "250 / 2000" is from memory).
- 31.4 min / 9536 = 197.6 ms/step on 8 H100.

The record: GPT-2 124M (12 layers, 12 heads of 64, 768 dims, vocab 50257), no biases, weightless rmsnorm (eps 1e-6,
fp32), rotary, exact GELU, attention output x 1/sqrt(24), no position embedding, wte tied to lm_head (so the tied
weight is lm_head's nn.Linear init, std 0.02083). fp32 parameters under bf16 autocast, SDPA, torch.compile then DDP.
AdamW lr 0.0018, betas (0.9, 0.95), wd 0.1 on every parameter, each gradient divided by its norm + 1e-6 first, no
clip. Linear warmup 256, constant, linear warmdown over the last 2048. Validation in fp32 (no autocast).

## Changes (each marked `port:` in train_gpt.py)

- **Stack**: the Rotary cos/sin caches are filled eagerly before compiling (under `fullgraph=True` they must exist
  before tracing); AdamW `fused=True`. No `coordinate_descent_tuning`.
- **Loader**: the record's, but `next_batch` also returns (shard, start); all ranks switch shards on the same step
  (as r001); shards are memory-mapped; running out of shards raises.
- **Validation**: `val_tokens` = 10,485,760 replaces `--val_max_steps 20`, averaged over ranks; still fp32. Only the
  final one by default (`REC_VAL_EVERY`).
- **Hints**: `HintEmbedding(768, cells, embed_std=0.02083)` (ratio 4.167: site scales start at 0.417, 8, 8) is the
  GPT's `hint` submodule. Rows are looked up when the record loads the next batch, overlapping the GPU. Sites: x0 =
  wte(idx); after 7 of 12 blocks (`REC_HINT_MID_LAYER`); before the final rmsnorm. The hint's parameters are outside
  the record's AdamW and gradient normalization (`record_parameters()`).
- **Budget**: `TRAIN_STEPS` (242), warmup `round(256 x STEP_SCALE)` = 6, warmdown `round(2048 x STEP_SCALE)` = 52.
  `SEED` seeds torch / numpy before the model is built (the record had no seed).
- **Timing**: `REC_COMPILE_WARMUP` (3) steps before the clock, then the restore (as r001). The record printed every
  step; the port every `REC_LOG_EVERY` (25). The first batch is loaded on the clock.
- **Output**: the `FINAL` line; the llm.c-format log still goes to `pylog124M/`. Checkpoint saves dropped.
- **Optional modes**: `HINT_NGRAM=1` as r001's (the record has no `logits.float()`; the per-position nll is fp32
  anyway), and `HINT_MIX=1` through `mixed_nll`; `REC_LM_AUX`; `REC_N_LAYER`, `REC_N_HEAD`, `REC_N_EMBD`.

## Default steps: 242

Same model size and batch as r001, whose measured 157 ms/step on 8 H200 with hints gives 38 s / 0.157 s = 242.
126.9M tokens: 2 shards, x 1.35 + 1 = `train_shards` 3 (enough up to ~380 steps).

## Knobs

| env | default | meaning |
| --- | --- | --- |
| `TRAIN_STEPS` | 242 | total steps; `STEP_SCALE = TRAIN_STEPS / 9536` |
| `LR_SCALE` | 1.0 | multiplies the AdamW lr 0.0018 and the hint's 0.008 |
| `SEED` | 0 | torch / numpy seed |
| `REC_WARMUP_STEPS` | round(256 x STEP_SCALE), at least 1 | linear warmup steps |
| `REC_WARMDOWN_STEPS` | round(2048 x STEP_SCALE) | linear warmdown steps |
| `REC_WEIGHT_DECAY` | 0.1 | AdamW weight decay (all record parameters) |
| `REC_HINT_EMBED_RATIO` | 4.167 | hint_runtime's embedding ratio |
| `REC_HINT_MID_LAYER` | 7 | site 1 is after this many blocks |
| `REC_BATCH` | 64 | sequences per GPU; smaller only for smoke tests |
| `REC_VAL_EVERY` | 0 | extra validations (off the clock) |
| `REC_LOG_EVERY` | 25 | steps between train-loss prints (on the clock) |
| `REC_COMPILE_WARMUP` | 3 | compile steps before the clock |

## Checks

GPU smoke tests on 2 RTX A6000 / L40S (`REC_BATCH=16`): fake rows, `HINTS=0`, and real rows with `HINT_NGRAM=1`, each
exit 0. On the CPU: 123,532,032 record parameters, all in the AdamW and none of the hint's; with the site scales at 0
the loss equals the no-hint loss; one dynamo graph with and without hints; the loader's (shard, start) on the real
shards; the restore; `HINT_NGRAM` off is bit-identical to the port without it, and with lam -> 1 the mixture's loss is
the record's.
