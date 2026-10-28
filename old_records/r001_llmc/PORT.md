# Record #1: llm.c GPT-2 124M baseline, ported to PyTorch

## Source

- modded-nanogpt `records/track_1_short/2024-10-13_llmc/`: llm.c's CUDA trainer, 8 H100, 140 ms/step, 524,288
  tokens/step, 19,560 steps (one epoch of FineWeb-10B), val loss 3.2722. The command
  (https://github.com/karpathy/llm.c/discussions/481):
  `train_gpt2cu -e d12 -b 64 -t 1024 -d 524288 -r 1 -z 1 -c 0.1 -l 0.0006 -q 0.0 -u 700 -n 5000 -v 250 -s 20000 -h 1`.
- The CUDA code cannot take hints, so `train_gpt.py` reimplements the model and recipe in PyTorch, after llm.c's
  reference `train_gpt2.py`, checked against `train_gpt2.cu` and `llmc/schedulers.h` (init, weight-decay set, AdamW
  constants, schedule). (modded-nanogpt-public's `train_gpt2.py` at 4a452a2 is not the record: it already has
  RMSNorm, no biases and lr 0.0015; only its loader's structure was used.)

## The port

- Model: GPT-2 124M (12 layers, 12 heads, 768 dims), LayerNorm with bias (eps 1e-5), tanh-GELU, learned position
  embeddings, tied wte / lm_head. Vocab padded to 50304 as llm.c, with the padding rows 0 and their logits sliced off
  before the loss. 124,475,904 parameters, llm.c's count.
- Init: N(0, 0.02), N(0, 0.02 / sqrt(24)) for the two residual projections per block, biases 0 (llm.c's scheme, from
  a CPU generator seeded with SEED).
- AdamW (fused), betas (0.9, 0.95), eps 1e-8, wd 0.1 on the 2D tensors only, grad-norm clip 1.0 over the record's
  parameters. fp32 master weights under bf16 autocast.
- Schedule: llm.c's linear warmup, then cosine from 6e-4 to 0.
- Batch: 64 x 1024 per GPU x 8 = 524,288 tokens. On 1, 2 or 4 GPUs a step accumulates `8 // world` micro-steps
  (DDP `no_sync` on all but the last), reading exactly the 8-GPU step's 8 chunks, so the math is the same.
- Data: llm.c's sequential `DistributedDataLoader`: rank r reads `tokens[p + r*B*T :][:B*T+1]`, p advances by 8*B*T,
  and the next shard starts when a step's chunks no longer fit (190 steps per 100M-token shard). Fixed: llm.c decided
  the shard switch per rank, so ranks 6-7 switched a step early; here all switch together. Shards are memory-mapped.
  **Unlike the record**, which shuffles shards and chunks, the data is read sequentially.
- Attention: SDPA with only the flash backend. `torch.compile` of the whole model (with `coordinate_descent_tuning`,
  as llm.c's train_gpt2.py), then DDP.
- Validation: the first 10,485,760 tokens, llm.c's 20 batches of 8 x 64 x 1024, only at the end by default.
- Timing: `REC_COMPILE_WARMUP` (3) steps compile before the clock; model, optimizers and loader are then restored.

## Hints

- The loader returns each chunk's (shard, start); the step gets `hints.train_span(shard, start, B*T)`, validation
  `hints.val_span(start, B*T)`.
- `HintEmbedding(768, hints.cells)` is a model submodule (DDP averages it, torch.compile compiles it), fed the tied
  `wte.weight`. Sites: x = wte + wpe; the residual stream after 7 of 12 blocks (0.58 of the depth, as #367's 7 of 11);
  right before ln_f.
- `hint_optimizer(model.hint)` with `set_lr_factor(LR_SCALE x the record's schedule factor)`; outside the record's
  AdamW, weight decay and grad clip.

## Optional modes (all off by default; without them the run is unchanged)

- `HINT_NGRAM=1` (HINT_DIR with count rows; `OldRecordRun(counts=...)` sets it): `Hints` also returns per-position
  counts (float32 [T, 3, 10]: tot, dist, cnt at k = 2, 3, 4, 5, 6, 8, 10, 12, 16, 20), and the loss is
  `hint.mixed_nll(nll, targets, rows)` = -log(lam * p_LM + (1 - lam) * q_WB): an interpolated Witten-Bell n-gram with
  learned gammas and lambdas over context-side counts (hint_runtime.NgramMix), in training and validation. Its 263
  scalars and 50,257 unigram logits are in the hint optimizer. Combinable with `HINT_INJECT=0/1`, not `HINT_MIX=1`.
  Knobs: `HINT_NGRAM_LAM_INIT`, `HINT_NGRAM_GAMMA_INIT`.
- `REC_LM_AUX=a`: with an output mixture, adds a x the LM's own cross-entropy to the training loss.
- `REC_N_LAYER`, `REC_N_HEAD`, `REC_N_EMBD`: smaller models (site 1 stays at 7/12 of the depth).
- `HINT_ONLINE=1`: the rows computed in the run (online_hints.py; `OldRecordRun(online=..., online_counts=...)`).
  The port passes its loader plan; only the final validation runs.
- `HINT_GPU_VAL=1`: the validation rows built on the GPUs after the last step (gpu_val_stats.InProcess);
  `HINT_GPU_VAL_CHECK=<dir>` compares them with a reference's.

## Default steps: 265

An estimate: ~144 ms/step on 8 H200 with hints (llm.c's 140 ms/step on H100; modded-nanogpt's PyTorch records of the
same size ran 142-154 ms/step; H200 ~2.5% faster; ~3 ms for the hint lookups). 38 s / 0.144 s = 264, so 265
(9 warmup steps). 139M tokens: 2 shards, `train_shards` = ceil(1.35 x 1.39) + 1 = 3.

## Knobs

| env | default | meaning |
| --- | --- | --- |
| `TRAIN_STEPS` | 265 | total steps; `STEP_SCALE = TRAIN_STEPS / 19560` |
| `LR_SCALE` | 1.0 | multiplies the record's lr 6e-4 and the hint's |
| `SEED` | 0 | torch / numpy seed, and the init generator's |
| `REC_WARMUP_STEPS` | round(700 * STEP_SCALE), at least 1 | linear warmup steps (llm.c `-u`) |
| `REC_FINAL_LR_FRAC` | 0.0 | the cosine's final lr fraction (llm.c `-q`) |
| `REC_WEIGHT_DECAY` | 0.1 | AdamW weight decay on the 2D tensors (llm.c `-c`) |
| `REC_HINT_MID_LAYER` | 7 | site 1 is after this many blocks |
| `REC_HINT_LR` | 0.008 | the hint optimizer's base lr |
| `REC_LR_SCHEDULE` | cosine | `cosine` (llm.c's) or `wsd`: warmup, constant, then linear to the final fraction |
| `REC_COOLDOWN_FRAC` | 0.5 | `wsd`: the share of the steps in the final decay |
| `REC_BETA2` | 0.95 | AdamW's beta2 |
| `REC_GRAD_CLIP` | 1.0 | the record's grad-norm clip |
| `REC_BATCH` | 64 | sequences per GPU; smaller only for smoke tests (it changes the recipe) |
| `REC_VAL_EVERY` | 0 | extra validations every that many steps (off the clock) |
| `REC_LOG_EVERY` | 25 | steps between train-loss prints (each syncs, on the clock) |
| `REC_COMPILE_WARMUP` | 3 | compile steps before the clock |

## Checks

On the CPU (torch 2.10): the parameter count, init stds and zero padding rows; gradients reach all hint parameters;
one graph under `torch.compile(fullgraph=True, backend="aot_eager")` with and without hints; `DDP(torch.compile(model))`
on 2 gloo ranks; the loader's (shard, offset) and tokens on the real shards across the shard switch, at world 8, 4
and 2; the restore after the compile steps; `HINT_NGRAM` off is bit-identical to the port without it. The GPU runs are
`OldRecordRun`s.
