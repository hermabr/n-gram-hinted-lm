The 3.13-second variant of record #367 (exact n-gram counts on the GPUs): `train_gpt.py` and `track_1_short/` of
modded-nanogpt's `r367-gpu-counts-clean` (b8a31606). Only the files that differ from the root's `track_1_short/` are
here; the trainer is the root's `track_1_short/` with this directory laid over it
(`src/retrieval_blog_post/r367_gpu_counts.py`'s `assemble`, which `GpuCountsRun` runs it from). It needs the first 1028 training shards: the counts cover the first 1027 (102.7B tokens), and
shard 1028 is held out for the mixture's fit. The count kernels (`track_1_short/counts/kernels.cu`) are built with
`$CUDA_HOME/bin/nvcc` (default `/usr/local/cuda`) on the first run.

On top of the 6.7 s record (`train_gpt.py` and `track_1_short/` at the repo root, but with its 530 + 10 steps and a
370-shard SlotIndex):

- **Exact validation rows from the GPUs.** After the last step, an exact count build on the GPUs over the first 1027
  training shards produces the validation rows, on the clock. It replaces the CPU SlotIndex. The build counts every
  validation context at levels 2, 3, 4, 5, 6, 10 and 20: level 2 over the first 103 shards, 3-5 over the first 370,
  and 6/10/20 over all 1027. The rows at 6/10/20 are exact, with the top 8 candidates.
- **Counts mixed into the output.** Witten-Bell backoff onto the model mixes the counts into the model's validation
  probabilities. The mixture's weights are fit on the clock on 8 x 65536 positions of the held-out shard 1028. No
  validation tokens are used for the fit.
- **92 trained steps** (82 + 10) instead of 540, with the run-length constants scaled to 92.

Before the clock, each rank pins its 1/8 of the count corpus (12.9B tokens, 25.8 GB) in host memory. During training,
the host-to-device copies run on a low-priority stream in equal shares per step.

Four seeds on one 8x H200 node (miso), interleaved with the base record: mean val loss 3.2507 (3.2404, 3.2441, 3.2379,
3.2805; one-sided t = 2.9 against 3.28), mean 3.131 s (sd 0.085). The base record ran 3.2733 in 6.389 s on the same
node. On the clock: 92 steps 1.470 s, validation reads and count queries 0.186 s, count build 1.422 s, fit forward and
mixture fit 0.053 s. Peak GPU memory is 60.6 GB allocated per rank, so it fits an H100.

Caveat: most of the gain comes from validation documents that have copies in the 102.7B-token count corpus. On
validation positions without a near-copy, the variant is at ~3.65, against ~3.41 for the base record.
