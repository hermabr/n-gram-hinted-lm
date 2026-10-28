# Old records with exact-match hints

Which is the oldest speedrun record that, with record #367's exact-match hints added, reaches 3.28 validation loss in
no more training time than record #360 (39.9 s) on the same node? Each probed record is ported into `rNNN_<name>/`,
run on 8 H200 through furu (`OldRecordRun` in `src/retrieval_blog_post/old_records.py`, whose `BATCH` lists every run
with its result), and the search bisects over the record numbers.

## The hints

Per position, the hint is the next tokens that followed the position's context before, at the deepest matched level of
a variant (levels are context lengths: (6,10,20), (2,3,6,10,20) or (2,3,4,5,6,10,20)):

- validation: counted exactly over all 1028 FineWeb-100B training shards (`val_hints.py`);
- training: causal, counted over the occurrences that end before the position's document starts;
- training with `OldRecordRun(late=True)`: the run trains on the last shards (1024-1028) and their rows are counted
  over shards 1-1023, so they are about as strong as the validation rows while still causal.

A row is a cell (one of 336: level bucket x count bin x top-share bin; 0 = no hint) and the top 8 candidates with
weights. `hint_runtime.py` reads the rows and has #367's mechanism, `HintEmbedding`.

## A port: `rNNN_<name>/train_gpt.py`

The record's own code (from its log in modded-nanogpt `records/track_1_short/`), with only these changes, each marked
`port:` in the script and described in its `PORT.md`:

1. **Stack**: `$RETRIEVAL_BLOG_POST_TORCH_VENV/bin/python -m torch.distributed.run --standalone --nproc_per_node=8
   train_gpt.py` (torch 2.10, made as in val_cache.py; FA3 from the HF cache with `HF_HUB_OFFLINE=1`). Fixes for the
   newer torch don't change the math. Compile flags are #360's: `torch.compile(model, dynamic=False, fullgraph=True)`,
   default inductor config.
2. **Data**: `$DATA_PATH/data/fineweb100B/fineweb_train_*.bin` (shard j = position j in sorted order) and the first
   10,485,760 tokens of `fineweb_val_000000.bin`.
3. **Hints** (`HINTS=0` gives the plain record): the loader yields each input token's (shard, offset), and the step
   gets `hints.train(...)` / `train_span(...)` rows (validation: `hints.val(...)` / `val_span(...)`); row i is for
   predicting token i + 1. `hint_embedding(embed_weight, rows)` is added at three sites: the embedding output, the
   residual stream at ~0.6 of the depth, and right before the final norm. `embed_std` (the port's embedding init std)
   rescales the hint's inits to #367's sizes. `hint_optimizer` (Adam, lr 0.008, scales 10x, no wd) follows the
   record's lr schedule factor.
4. **Budget** (env): `TRAIN_STEPS` (every schedule length scales with `STEP_SCALE = TRAIN_STEPS / record_steps`),
   `LR_SCALE` (every lr, the hint's included), `SEED`, and record-specific `REC_*` knobs. The default `TRAIN_STEPS`
   aims at ~38 s of training on 8 H200.
5. **Timing**: every training step is timed, hint lookups included. Compile warmup steps run before the clock and the
   model, optimizer and loader states are restored after them. Validation is off the clock.
6. **Output**: rank 0 prints `FINAL val_loss:<4 decimals> train_time_ms:<int> steps:<int>`. With
   `VAL_SUBSETS=<labels.npy>` (r001, r002, r003, r006) it then prints
   `VAL_SUBSETS all:<loss> noexact:<loss> nonear:<loss> tokens:<n>/<n>/<n> time:<s>s`: the loss over all targets,
   over those in val documents without an exact copy in the training shards, and without a near-copy either.
7. **`port.json`**: `{"record": NNN, "default_steps": D, "train_shards": S}`, S being the shards read at D steps
   (x1.35 for loaders that skip document tails), + 1.

Ports were smoke-tested on 2 small GPUs with `HINT_DIR=fake` and `HINTS=0`; Hopper-only ones on the CPU, then through
an `OldRecordRun(levels="fake")`.

## Contents

- Ports: r001 (record #1, llm.c, in PyTorch), r002, r003, r006, r012, r023, r046; record #360 itself is
  `record_360/` in the repo root. `r001_llmc_c/` is llm.c's own CUDA trainer with the hints (see its README).
- `gpu_val_stats.py` (+ `.cu`): the validation rows built on a node's GPUs; `gpu_val_stats_ref.py` and
  `gpu_val_stats_compare.py` check it against a CPU reference.
- `online_hints.py`: every row computed in the run, on the clock (`HINT_ONLINE=1`).
