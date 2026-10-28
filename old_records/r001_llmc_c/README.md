# Record #1: llm.c's own trainer, with hints

Record #1 of the speedrun (`records/track_1_short/2024-10-13_llmc/`) is Karpathy's GPT-2 124M run with llm.c's
CUDA trainer. This is that trainer as llm.c's master had it at the record
([7ecd8906](https://github.com/karpathy/llm.c/commit/7ecd8906afe6ed7a2b2cdb731c042f26d525b820)), with two changes:
the training data is read in order (`permute_train_loader = 0`, as the PyTorch port old_records/r001_llmc reads it),
and the hints (`HINTS=1`, off by default), with what their recipe uses: a `custom:L:C:H` model, `SEED` and the
`wsdlin` schedule.

- `build.sh`: builds `build/train_gpt2cu`; llmc_cpu_val.py's `LlmcBuild` runs it with config.py's paths.
- `run.sh DATA`: the record's command on one node's 8 GPUs.
- `run_hints.py`: one run with the hints, or without them on the same clock (below).

## Hints

`HINTS=1` adds record #367's exact-match hints: the hint injected into the residual stream at three sites and the
n-gram output mixture (`llmc/hints.cuh`, ports of old_records/hint_runtime.py's HintEmbedding and NgramMix), with
hand-written backward passes and their own Adam. Every row is computed in the run (`llmc/hint_stats.cuh`):

- training rows: at the start of step 0, on each GPU for its own positions, strictly causal (exact_match's
  OnlineRows);
- validation rows, over up to all 1028 FineWeb-100B training shards: by default on the GPUs, which load the shards
  beside the training and count the validation contexts in them after the last step. With `HINT_CPU_VAL=1` on the CPUs
  instead (`cpuval/`, Rust, and `llmc/cpu_val.h`): the corpus is built beside the training from the training shards
  alone, and after the clock the validation tokens are only looked up in it. Levels 2-3 count over the first 110
  shards, 4-6 over 10 and 8-20 over all (`HINT_CPU_VAL_LEVEL_SHARDS`).

The clock is llm.c's per-step CUDA-event times, summed: every step, then the validation build's remaining work after
the last step (with `HINT_CPU_VAL=1`, the wait for it). The cuBLAS / cuDNN / NCCL setup (a forward and backward of the
first batch, no update) and the validation forward passes are off it. `CLOCK_EVERY_STEP=1` gives a run without the
hints the same clock.

`HINT_CPU_VAL_CHECK=1` also runs the GPU build and compares the rows at every position (set
`HINT_CPU_VAL_LEVEL_SHARDS` so every level counts over all shards); `cpuval_rows` (`cargo build --release` in
`cpuval/`) builds the CPU rows without GPUs.

## Results

8x H200, the PyTorch port's best on-clock recipe (4x512, 8 heads, aux 1, warmup 50, `wsdlin`, lr 3 x 6e-4, hint lr 3
x 0.002667), seeds 0 / 1 / 2:

| | steps | val loss | on the clock (s) |
|---|---|---|---|
| hints, validation rows on CPUs | 470 | 3.2609 / 3.2628 / 3.2662 | 21.78 / 21.38 / 21.45 |
| no hints | 470 | 4.8314 / 4.7692 / 4.8398 | 20.14 / 20.16 / 20.16 |
| hints, validation rows on GPUs | 210 | 3.2698 / 3.2740 / 3.2694 | 12.43 / 12.51 / 12.44 |

The first two are src/retrieval_blog_post/llmc_cpu_val.py's `LlmcCpuValRun`, the last llmc_onclock.py's `LlmcRun`.
The last row was measured on an earlier build of this trainer.
`uv run src/retrieval_blog_post/llmc_cpu_val.py` builds the trainer and runs the first two.
