/*
Retrieval hints for llm.c (record #1 of modded-nanogpt): a C/CUDA port of old_records/hint_runtime.py's HintEmbedding
(record #367's hint, injected into the residual stream at 3 sites) and NgramMix (the output mixture
-log(lam * p_LM + (1 - lam) * q_WB) with an interpolated Witten-Bell n-gram over count rows), with hand-written
backward passes and the hint parameters' own Adam. The rows are computed in the run, on the GPUs
(llmc/hint_stats.cuh): the training rows at the start of the clock, the validation rows after the last step.

Parameters (fp32, one contiguous buffer, in this order; the PyTorch port's names):
  next_scale [cells], cell_embed [cells, C], site_scales [3]                          (HintEmbedding)
  base [V], gamma [L*BINS], gamma_level [L], gamma_bin [BINS],
  lam_logit [(L+1)*BINS], lam_level [L+1], lam_sparse [2*(L+1)]                      (NgramMix)

Env: HINTS=1 turns them on; HINT_VAL_SHARDS (a file listing the training shards the validation rows count over);
HINT_LR (the hint optimizer's peak lr, 0.008; the scales 10x), HINT_AUX (the LM's own cross-entropy's weight in the
training loss, 0), HINT_MID_LAYER (site 1 is the input of this block, round(7 L / 12)); HINT_VAL_THREADS (4) and
HINT_VAL_CHUNK_MB (8): the validation shards' loads (from the clock's start, beside the training); HINT_VAL_SUBSETS (a labels.npy: the losses on the validation subsets, after the clock).
HINT_CPU_VAL=1: the validation rows built on CPUs instead (llmc/cpu_val.h: the corpus built from the training shards
beside the training, waited for on the clock; the validation tokens looked up in it after the clock stops;
HINT_VAL_SHARDS lists the shards they count over), the GPU build not set up (with HINT_CPU_VAL_CHECK=1 it runs too,
after the CPU rows, and the two are compared).
*/
#ifndef LLMC_HINTS_CUH
#define LLMC_HINTS_CUH

#include <vector>
#include <string>
#include <math.h>
#include "cuda_common.h"
#include "cuda_utils.cuh"
#include "hint_stats.cuh"
#include "cpu_val.h"

#define HINT_WEIGHT_ONE 65535.0f
#define HINT_BINS 10          // len(NgramMix.COUNT_EDGES) + 1
#define HINT_MAX_LEVELS 10    // ALL_LEVELS
#define HINT_MAX_ENTRIES 8
#define HINT_CELLS 336        // (len(LEVELS) + 1) * 7 * 6
#define HINT_RECORD_EMBED_STD 0.005f

__constant__ float c_count_edges[9] = {2.f, 3.f, 5.f, 9.f, 17.f, 33.f, 100.f, 1000.f, 10000.f};

// one batch's rows (views into hint_stats::Rows)
struct HintBatch {
    const int16_t* cell; const uint16_t* tokens; const uint16_t* weights;
    const uint32_t* tot; const uint16_t* dist; const uint32_t* cnt;
};

struct HintState {
    int cells = HINT_CELLS, entries = HINT_MAX_ENTRIES, levels = HINT_MAX_LEVELS, C = 0, V = 50257, Vp = 50304, N = 0;
    int mid_layer = 0;
    float embed_ratio = 4.0f, aux = 0.0f, lr = 0.008f;
    // rows: every training micro-batch's of this rank, and the validation's (all positions)
    hint_stats::Rows train, val, blank;  // blank: one batch without hints (the warmup's)
    hint_stats::ValBuild val_build;
    hint_stats::CpuValBuild cpu_val;
    bool use_cpu_val = false, check_cpu_val = false;
    hint_stats::TrainPlan train_plan;
    HintBatch dev;
    int* order = nullptr; uint32_t* sort_out = nullptr;  // the batch's positions sorted by cell, and their cells
    uint32_t* sort_in = nullptr; int* iota = nullptr;
    // parameters: offsets into params / grads / m / v (floats)
    size_t n_params = 0;
    size_t o_next_scale, o_cell_embed, o_site_scales, o_base, o_gamma, o_gamma_level, o_gamma_bin, o_lam_logit, o_lam_level, o_lam_sparse;
    float* params = nullptr; float* grads = nullptr; float* m = nullptr; float* v = nullptr;
    int adam_t = 0;
    // activations
    float* hint = nullptr; float* mean = nullptr; float* dhint = nullptr;  // [N, C]
    float* ng_a = nullptr; float* ng_logq = nullptr; float* ng_r = nullptr; float* lm_nll = nullptr;  // [N]
    float* scalars = nullptr;  // [0] base logsumexp, [1] the sum of dL/dlog p0 (base gradient's dense part)
    // the validation subsets (HINT_VAL_SUBSETS): per position 0, 1 (no exact copy) or 2 (no near-copy either)
    std::vector<unsigned char> subsets; double subset_sum[3] = {0, 0, 0}, subset_n[3] = {0, 0, 0};
    // timings (ms)
    float train_build_ms = 0, val_build_ms = 0;
};

static HintState* g_hint = nullptr;

static void hint_layout(HintState* h) {
    size_t o = 0;
    const int L = h->levels, C = h->C;
    h->o_next_scale = o; o += h->cells;
    h->o_cell_embed = o; o += (size_t)h->cells * C;
    h->o_site_scales = o; o += 3;
    h->o_base = o; o += h->V;
    h->o_gamma = o; o += L * HINT_BINS;
    h->o_gamma_level = o; o += L;
    h->o_gamma_bin = o; o += HINT_BINS;
    h->o_lam_logit = o; o += (L + 1) * HINT_BINS;
    h->o_lam_level = o; o += L + 1;
    h->o_lam_sparse = o; o += 2 * (L + 1);
    h->n_params = o;
}

// the PyTorch port's init (HintEmbedding / NgramMix with lam_init = gamma_init = 0); the unigram logits (base) are
// set at the start of the clock, from shard 0's token frequencies (hint_stats::train_rows_build)
static void hint_init_params(HintState* h) {
    std::vector<float> p(h->n_params, 0.0f);
    for (int c = 0; c < h->cells; c++) p[h->o_next_scale + c] = 4.0f;
    p[h->o_site_scales + 0] = 0.1f * h->embed_ratio;
    p[h->o_site_scales + 1] = 8.0f;
    p[h->o_site_scales + 2] = 8.0f;
    for (int v = 0; v < h->V; v++) p[h->o_base + v] = logf(1.0f / h->V);
    cudaCheck(cudaMemcpy(h->params, p.data(), h->n_params * sizeof(float), cudaMemcpyHostToDevice));
}

static float env_float(const char* name, float fallback) { return getenv(name) ? (float)atof(getenv(name)) : fallback; }
static int env_int(const char* name, int fallback) { return getenv(name) ? atoi(getenv(name)) : fallback; }

static std::vector<std::string> read_lines(const char* path) {
    std::vector<std::string> out;
    FILE* f = fopenCheck(path, "r");
    char buf[4096];
    while (fgets(buf, sizeof(buf), f)) {
        std::string s(buf);
        while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back();
        if (!s.empty()) out.push_back(s);
    }
    fcloseCheck(f);
    return out;
}

// HINTS=1: the hint state for B x T batches, and before the clock the validation build's buffers (HINT_VAL_SHARDS)
static HintState* hint_setup(int C, int V, int Vp, int num_layers, int B, int T, const char* train_pattern, int accum,
                             int steps, const char* val_path) {
    if (getenv("HINTS") == NULL || strcmp(getenv("HINTS"), "1") != 0) { return nullptr; }
    HintState* h = new HintState();
    h->C = C; h->V = V; h->Vp = Vp; h->N = B * T;
    h->mid_layer = env_int("HINT_MID_LAYER", (int)nearbyint(7.0 * num_layers / 12.0));
    // the candidates' mean embedding is divided by embed_std / RECORD_EMBED_STD (0.02 / 0.005: llm.c's init std)
    h->embed_ratio = 0.02f / HINT_RECORD_EMBED_STD;
    h->aux = env_float("HINT_AUX", 0.0f);
    h->lr = env_float("HINT_LR", 0.008f);
    hint_layout(h);
    size_t np = h->n_params * sizeof(float);
    cudaCheck(cudaMalloc(&h->params, np)); cudaCheck(cudaMalloc(&h->grads, np));
    cudaCheck(cudaMalloc(&h->m, np)); cudaCheck(cudaMalloc(&h->v, np));
    cudaCheck(cudaMemset(h->grads, 0, np)); cudaCheck(cudaMemset(h->m, 0, np)); cudaCheck(cudaMemset(h->v, 0, np));
    size_t nc = (size_t)h->N * h->C * sizeof(float);
    cudaCheck(cudaMalloc(&h->hint, nc)); cudaCheck(cudaMalloc(&h->mean, nc)); cudaCheck(cudaMalloc(&h->dhint, nc));
    cudaCheck(cudaMalloc(&h->ng_a, h->N * sizeof(float))); cudaCheck(cudaMalloc(&h->ng_logq, h->N * sizeof(float)));
    cudaCheck(cudaMalloc(&h->ng_r, h->N * sizeof(float))); cudaCheck(cudaMalloc(&h->lm_nll, h->N * sizeof(float)));
    cudaCheck(cudaMalloc(&h->scalars, 2 * sizeof(float))); cudaCheck(cudaMemset(h->scalars, 0, 2 * sizeof(float)));
    cudaCheck(cudaMemset(h->lm_nll, 0, h->N * sizeof(float)));
    cudaCheck(cudaMalloc(&h->order, h->N * sizeof(int)));
    cudaCheck(cudaMalloc(&h->sort_in, h->N * sizeof(uint32_t))); cudaCheck(cudaMalloc(&h->sort_out, h->N * sizeof(uint32_t)));
    std::vector<int> iota(h->N);
    for (int i = 0; i < h->N; i++) iota[i] = i;
    cudaCheck(cudaMalloc(&h->iota, h->N * sizeof(int)));
    cudaCheck(cudaMemcpy(h->iota, iota.data(), h->N * sizeof(int), cudaMemcpyHostToDevice));
    hint_init_params(h);
    hint_stats::rows_alloc(h->blank, h->N, 0);
    glob_t g;
    if (glob(train_pattern, 0, NULL, &g) != 0) { fprintf(stderr, "no shards match %s\n", train_pattern); exit(EXIT_FAILURE); }
    hint_stats::train_plan(h->train_plan, std::vector<std::string>(g.gl_pathv, g.gl_pathv + g.gl_pathc),
                           multi_gpu_config.num_processes, (hint_stats::i64)B * T, accum, steps);
    globfree(&g);
    if (getenv("HINT_VAL_SHARDS") == NULL) { fprintf(stderr, "HINTS=1 needs HINT_VAL_SHARDS\n"); exit(EXIT_FAILURE); }
    h->use_cpu_val = env_int("HINT_CPU_VAL", 0) == 1;
    h->check_cpu_val = h->use_cpu_val && env_int("HINT_CPU_VAL_CHECK", 0) == 1;
    if (h->use_cpu_val) {  // the rows' device buffers and the shared host mapping, before the clock
        hint_stats::rows_alloc(h->val, hint_stats::VAL_TOKENS, 0);
        cudaCheck(cudaDeviceSynchronize());
        h->cpu_val.setup(getenv("HINT_VAL_SHARDS"), val_path, multi_gpu_config.process_rank,
                         multi_gpu_config.num_processes, (hint_stats::i64)B * T, h->check_cpu_val);
        printf0("hints: mid layer %d, aux %.2f, hint lr %g, %zu hint parameters; validation rows on CPUs (%s)%s\n",
                h->mid_layer, h->aux, h->lr, h->n_params, h->cpu_val.describe.c_str(),
                h->cpu_val.pinned ? "" : "; the host rows NOT pinned");
    }
    if (!h->use_cpu_val || h->check_cpu_val) {
        h->val_build.setup(read_lines(getenv("HINT_VAL_SHARDS")), val_path, multi_gpu_config.process_rank,
                           multi_gpu_config.num_processes, env_int("HINT_VAL_THREADS", 4),
                           (hint_stats::i64)env_int("HINT_VAL_CHUNK_MB", 8) << 19);
        printf0("hints: mid layer %d, aux %.2f, hint lr %g, %zu hint parameters; validation rows over %zu shards (%.1f GB on "
                "this rank)\n", h->mid_layer, h->aux, h->lr, h->n_params, read_lines(getenv("HINT_VAL_SHARDS")).size(),
                2e-9 * h->val_build.n_tokens);
    }
    return h;
}

// the positions sorted by cell (a stable radix sort of 9-bit keys), for the cells' gradients
__global__ void hint_cell_keys_kernel(const int16_t* cell, uint32_t* keys, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) keys[i] = (uint32_t)cell[i];
}


// the rows of a batch: `rows` from row `first` on
static void hint_select(HintState* h, const hint_stats::Rows& rows, size_t first, cudaStream_t stream) {
    const int L = h->levels, E = h->entries;
    h->dev = HintBatch{rows.cell + first, rows.tokens + first * E, rows.weights + first * E, rows.tot + first * L,
                       rows.dist + first * L, rows.cnt + first * L};
    hint_cell_keys_kernel<<<CEIL_DIV(h->N, 256), 256, 0, stream>>>(h->dev.cell, h->sort_in, h->N);
    hint_stats::cub_run([&](void* t, size_t& b) {
        return cub::DeviceRadixSort::SortPairs(t, b, h->sort_in, h->sort_out, h->iota, h->order, h->N, 0, 9, stream);
    });
    cudaCheck(cudaGetLastError());
}

// ----------------------------------------------------------------------------
// on the clock: the training rows and the unigram init (at step 0), the validation rows (after the last step; with
// HINT_CPU_VAL only the wait for the corpus built beside the training, the rows looked up after the clock)

static void hint_build_train(HintState* h, cudaStream_t stream) {
    if (!h->use_cpu_val || h->check_cpu_val) h->val_build.start();  // the validation shards' loads, beside the training
    hint_stats::TrainBuildTimes tm;
    hint_stats::train_rows_build(h->train, tm, h->train_plan, multi_gpu_config.process_rank, h->params + h->o_base, stream);
    // the CPU build beside the training, once this rank's training rows are done (both read shards and fill memory:
    // started together, the training rows took up to twice as long)
    if (h->use_cpu_val) h->cpu_val.start();
    h->train_build_ms = 1000 * tm.total;
    printf0("hints: training rows in %.0f ms (read %.0f, prep %.0f, levels %.0f, tops %.0f; %lld stream positions, "
            "%lld hits)\n", 1000 * tm.total, 1000 * tm.load, 1000 * tm.prep, 1000 * tm.levels, 1000 * tm.tops,
            tm.stream, tm.hits);
}

// the CPU-built rows equal to the GPU build's at every position (HINT_CPU_VAL_CHECK)
template <typename T> static long long hint_count_diff(const T* a, const T* b, long long n) {
    std::vector<T> x(n), y(n);
    cudaCheck(cudaMemcpy(x.data(), a, n * sizeof(T), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(y.data(), b, n * sizeof(T), cudaMemcpyDeviceToHost));
    long long d = 0;
    for (long long i = 0; i < n; i++) d += x[i] != y[i];
    return d;
}

static void hint_build_val(HintState* h, cudaStream_t stream) {
    if (h->use_cpu_val) {
        h->cpu_val.wait();
        h->val_build_ms = 1000 * h->cpu_val.t_built;
        return;
    }
    hint_stats::ValBuild& b = h->val_build;
    b.finish(h->val, stream);
    h->val_build_ms = 1000 * b.t_total;
    printf0("hints: validation rows in %.0f ms (waiting for the loads %.0f, index %.0f, scans %.0f, merge %.0f, rows "
            "%.0f; the loads done %.0f ms after the clock started; level scans ms", 1000 * b.t_total, 1000 * b.t_wait,
            1000 * b.t_index, 1000 * b.t_scan, 1000 * b.t_merge, 1000 * b.t_rows, 1000 * (b.load_end.load() - b.t_start));
    for (int li = 0; li < hint_stats::NLEVELS; li++) printf0(" %.0f", b.level_ms[li]);
    printf0("; level merges ms");
    double prev = 0;
    for (int li = 0; li < hint_stats::NLEVELS; li++) { printf0(" %.0f", b.merge_ms[li] - prev); prev = b.merge_ms[li]; }
    printf0(")\n");
}

// after the clock, like the validation forward passes (HINT_CPU_VAL): the validation tokens looked up, the rows copied
// to the GPUs
static void hint_lookup_val(HintState* h, cudaStream_t stream) {
    if (!h->use_cpu_val) return;
    hint_stats::CpuValBuild& c = h->cpu_val;
    c.finish(h->val, stream);
    printf0("hints: validation rows (CPU) built %.0f ms (wall) after the build's start (waited for %.0f ms after the last "
            "step); after the clock, on the GPUs in %.0f ms (waiting %.0f: reading the validation tokens %.0f, the lookup "
            "%.0f; copies %.0f)\n", 1000 * c.t_built_end, 1000 * c.t_built, 1000 * c.t_total, 1000 * c.t_wait,
            1000 * c.head->read_s, 1000 * c.head->lookup_s, 1000 * c.t_copy);
    printf0("hints: cpu val build: %s\n", c.report.c_str());
    if (!h->check_cpu_val) return;
    hint_stats::Rows gpu;
    h->val_build.finish(gpu, stream);
    const long long N = hint_stats::VAL_TOKENS, L = hint_stats::NLEVELS, E = hint_stats::ENTRIES;
    const long long d[6] = {hint_count_diff(h->val.tot, gpu.tot, N * L), hint_count_diff(h->val.dist, gpu.dist, N * L),
                      hint_count_diff(h->val.cnt, gpu.cnt, N * L), hint_count_diff(h->val.cell, gpu.cell, N),
                      hint_count_diff(h->val.tokens, gpu.tokens, N * E), hint_count_diff(h->val.weights, gpu.weights, N * E)};
    printf("hints: CHECK rank %d: CPU vs GPU validation rows differ at tot %lld, dist %lld, cnt %lld, cell %lld, "
           "tokens %lld, weights %lld entries%s\n", multi_gpu_config.process_rank, d[0], d[1], d[2], d[3], d[4], d[5],
           d[0] + d[1] + d[2] + d[3] + d[4] + d[5] == 0 ? " (EQUAL)" : " (DIFFER)");
}

// the losses of a validation batch (rows from `first`) on the subsets; their labels (derived from the validation
// tokens) read at the first batch, after the clock
static void hint_val_subsets(HintState* h, const float* losses, size_t first) {
    if (h->subsets.empty() && getenv("HINT_VAL_SUBSETS")) {  // labels.npy: uint8 [VAL_TOKENS] after a version-1 header
        FILE* f = fopenCheck(getenv("HINT_VAL_SUBSETS"), "rb");
        unsigned char head[10];
        freadCheck(head, 1, 10, f);
        std::vector<char> rest(head[8] | (head[9] << 8));
        freadCheck(rest.data(), 1, rest.size(), f);
        h->subsets.resize(hint_stats::VAL_TOKENS);
        freadCheck(h->subsets.data(), 1, h->subsets.size(), f);
        fcloseCheck(f);
    }
    if (h->subsets.empty()) return;
    for (int i = 0; i < h->N; i++) {
        const int label = h->subsets[first + i];
        for (int s = 0; s < 3; s++) {
            if (s == 0 || (s == 1 && label >= 1) || (s == 2 && label == 2)) { h->subset_sum[s] += losses[i]; h->subset_n[s] += 1; }
        }
    }
}

// the run's result line (the clock: every step, its setup run before it, and the validation build after the last
// one, with HINT_CPU_VAL the wait for it), and the losses on the validation subsets (HINT_VAL_SUBSETS)
static void hint_report(HintState* h, float val_loss, double clock_s, int steps) {
    // after the clock: the builds' buffers freed
    cudaCheck(cudaFreeHost(h->train_plan.host));
    h->val_build.release();
    if (h->use_cpu_val) h->cpu_val.release();
    hint_stats::cache_release();
    printf0("FINAL val_loss:%.4f train_time_ms:%d steps:%d\n", val_loss, (int)(1000 * clock_s), steps);
    printf0("hints: on the clock %.0f ms: %d steps %.0f ms (their training rows %.0f ms, in step 0), then the validation "
            "%s %.0f ms\n", 1000 * clock_s, steps, 1000 * clock_s - h->val_build_ms, h->train_build_ms,
            h->use_cpu_val ? "build's wait" : "rows", h->val_build_ms);
    if (h->subsets.empty()) return;
    double sums[6] = {h->subset_sum[0], h->subset_sum[1], h->subset_sum[2], h->subset_n[0], h->subset_n[1], h->subset_n[2]};
#ifdef MULTI_GPU
    if (multi_gpu_config.num_processes > 1) {
        double* d = hint_stats::dalloc<double>(6);
        cudaCheck(cudaMemcpy(d, sums, sizeof(sums), cudaMemcpyHostToDevice));
        ncclCheck(ncclAllReduce(d, d, 6, ncclFloat64, ncclSum, multi_gpu_config.nccl_comm, 0));
        cudaCheck(cudaMemcpy(sums, d, sizeof(sums), cudaMemcpyDeviceToHost));
        hint_stats::dfree(d);
    }
#endif
    printf0("VAL_SUBSETS all:%.4f noexact:%.4f nonear:%.4f tokens:%.0f/%.0f/%.0f\n", sums[0] / sums[3], sums[1] / sums[4],
            sums[2] / sums[5], sums[3], sums[4], sums[5]);
}

// ----------------------------------------------------------------------------
// kernels: HintEmbedding forward

// hint[i] = (mean_e w_e * wte[tok_e] / ratio) * next_scale[cell] + cell_embed[cell], 0 where cell == 0
__global__ void hint_forward_kernel(float* hint, float* mean, const int16_t* cell, const uint16_t* tokens,
                                    const uint16_t* weights, const floatX* wte, const float* next_scale,
                                    const float* cell_embed, int C, int E, float inv_ratio) {
    const int i = blockIdx.x;
    const int c0 = cell[i];
    float* h = hint + (size_t)i * C;
    float* mn = mean + (size_t)i * C;
    if (c0 <= 0) {
        for (int c = threadIdx.x; c < C; c += blockDim.x) { h[c] = 0.0f; mn[c] = 0.0f; }
        return;
    }
    int tok[HINT_MAX_ENTRIES]; float w[HINT_MAX_ENTRIES];
    for (int e = 0; e < E; e++) {
        tok[e] = tokens[(size_t)i * E + e];
        w[e] = (float)weights[(size_t)i * E + e] / HINT_WEIGHT_ONE;
    }
    const float ns = next_scale[c0];
    const float* ce = cell_embed + (size_t)c0 * C;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float acc = 0.0f;
        for (int e = 0; e < E; e++) acc += w[e] * ((float)wte[(size_t)tok[e] * C + c] * inv_ratio);
        mn[c] = acc;
        h[c] = acc * ns + ce[c];
    }
}

// x += scale * hint (x in floatX, the residual stream or the block output added to it)
__global__ void hint_add_kernel(floatX* x, const float* hint, const float* scale, size_t n) {
    const float s = *scale;
    for (size_t k = blockIdx.x * (size_t)blockDim.x + threadIdx.x; k < n; k += (size_t)gridDim.x * blockDim.x) {
        x[k] = (floatX)((float)x[k] + s * hint[k]);
    }
}

// backward of a site: dhint (+)= scale * g, dscale += sum(hint * g)
__global__ void hint_site_backward_kernel(float* dhint, const floatX* g, const float* hint, const float* scale,
                                          float* dscale, size_t n, int accumulate) {
    const float s = *scale;
    float dot = 0.0f;
    for (size_t k = blockIdx.x * (size_t)blockDim.x + threadIdx.x; k < n; k += (size_t)gridDim.x * blockDim.x) {
        float gv = (float)g[k];
        float d = s * gv;
        dhint[k] = accumulate ? dhint[k] + d : d;
        dot += hint[k] * gv;
    }
    float block = blockReduce<warpReduceSum>(dot);
    if (threadIdx.x == 0) atomicAdd(dscale, block);
}

// the cells' gradients: dcell_embed[cell] += sum_i dhint[i], dnext_scale[cell] += sum_i <dhint[i], mean[i]> over
// the positions i of the cell. A block per HINT_CELL_CHUNK positions in cell order (order: the positions, cells: their
// cells, both sorted by cell), each cell's run within it summed and added atomically
#define HINT_CELL_CHUNK 128
__global__ void hint_cell_backward_kernel(float* dcell_embed, float* dnext_scale, const float* dhint, const float* mean,
                                          const int* order, const uint32_t* cells, int N, int C) {
    const int c = blockIdx.y * blockDim.x + threadIdx.x;
    const int a = blockIdx.x * HINT_CELL_CHUNK, b = min(N, a + HINT_CELL_CHUNK);
    if (cells[b - 1] == 0) return;  // no hint, no gradient (cell 0 sorts first)
    float acc = 0.0f, dot = 0.0f;
    for (int p = a; p < b; p++) {
        const uint32_t cell = cells[p];
        if (cell != 0 && c < C) {
            size_t k = (size_t)order[p] * C + c;
            float d = dhint[k];
            acc += d;
            dot += d * mean[k];
        }
        if (p + 1 == b || cells[p + 1] != cell) {  // the end of the cell's run (uniform over the block)
            if (cell != 0) {
                if (c < C) atomicAdd(dcell_embed + (size_t)cell * C + c, acc);
                float block = blockReduce<warpReduceSum>(dot, true);
                if (threadIdx.x == 0) atomicAdd(dnext_scale + cell, block);
            }
            acc = 0.0f; dot = 0.0f;
        }
    }
}

// ----------------------------------------------------------------------------
// kernels: NgramMix

__device__ __forceinline__ int hint_count_bin(float tot) {
    int b = 0;
    #pragma unroll
    for (int e = 0; e < 9; e++) b += tot > c_count_edges[e];
    return b;
}

__device__ __forceinline__ float log_sigmoid(float x) { return fminf(x, 0.0f) - log1pf(expf(-fabsf(x))); }

// log-sum-exp of the unigram logits (one block)
__global__ void hint_base_lse_kernel(float* out, const float* base, int V) {
    float mx = -INFINITY;
    for (int v = threadIdx.x; v < V; v += blockDim.x) mx = fmaxf(mx, base[v]);
    mx = blockReduce<warpReduceMax>(mx, true, -INFINITY);
    float s = 0.0f;
    for (int v = threadIdx.x; v < V; v += blockDim.x) s += expf(base[v] - mx);
    s = blockReduce<warpReduceSum>(s, true);
    if (threadIdx.x == 0) out[0] = mx + logf(s);
}

struct NgramParams {
    const float* base; const float* gamma; const float* gamma_level; const float* gamma_bin;
    const float* lam_logit; const float* lam_level; const float* lam_sparse; const float* lse;
};

// per position: the gate's logit a and log q(target) (hint_runtime.NgramMix.gates / log_q)
__device__ __forceinline__ void ngram_position(const NgramParams& P, const uint32_t* tot_r, const uint16_t* dist_r,
                                               const uint32_t* cnt_r, int target, int L,
                                               float* a_out, float* p_chain, float* g_out, float* logg_out,
                                               int* bins_out, int* gate_out, int* sparse_out) {
    int deepest = -1;
    for (int j = 0; j < L; j++) if (tot_r[j] > 0) deepest = j;
    int at = deepest < 0 ? 0 : deepest;
    int dbin = hint_count_bin((float)tot_r[at]);
    int sparse = (float)dist_r[at] == 1.0f;
    int d = deepest + 1;
    int gate = deepest >= 0 ? d * HINT_BINS + dbin : 0;
    int sp = deepest >= 0 ? d * 2 + sparse : 0;
    *gate_out = gate; *sparse_out = sp;
    *a_out = P.lam_logit[gate] + P.lam_level[gate / HINT_BINS] + P.lam_sparse[sp];
    float p = expf(P.base[target] - P.lse[0]);
    p_chain[0] = p;
    for (int j = 0; j < L; j++) {
        float tot = (float)tot_r[j];
        int b = hint_count_bin(tot);
        float lg = P.gamma[j * HINT_BINS + b] + P.gamma_level[j] + P.gamma_bin[b];
        float g = expf(fminf(fmaxf(lg, -20.0f), 20.0f));
        bins_out[j] = b; logg_out[j] = lg; g_out[j] = g;
        if (tot > 0.0f) {
            float gd = g * (float)dist_r[j];
            p = ((float)cnt_r[j] + gd * p) / (tot + gd);
        }
        p_chain[j + 1] = p;
    }
}

__global__ void ngram_forward_kernel(float* a_out, float* logq_out, NgramParams P, const uint32_t* tot,
                                     const uint16_t* dist, const uint32_t* cnt, const int* targets, int N, int L) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float pc[HINT_MAX_LEVELS + 1], g[HINT_MAX_LEVELS], lg[HINT_MAX_LEVELS]; int bins[HINT_MAX_LEVELS]; int gate, sp; float a;
    ngram_position(P, tot + (size_t)i * L, dist + (size_t)i * L, cnt + (size_t)i * L, targets[i], L, &a, pc, g, lg, bins, &gate, &sp);
    a_out[i] = a;
    logq_out[i] = logf(fmaxf(pc[L], 1e-30f));
}

// the n-gram parameters' gradients, given each position's posterior weight r of the LM in the mixture:
// dL/da = sigmoid(a) - r, dL/dlog q = -(1 - r), each times dloss (the mean's 1 / (B*T*accum))
__global__ void ngram_backward_kernel(float* grads, size_t o_base, size_t o_gamma, size_t o_gamma_level, size_t o_gamma_bin,
                                      size_t o_lam_logit, size_t o_lam_level, size_t o_lam_sparse, float* base_gsum,
                                      NgramParams P, const float* r_in, const uint32_t* tot, const uint16_t* dist,
                                      const uint32_t* cnt, const int* targets, int N, int L, float dloss) {
    __shared__ float s_gamma[HINT_MAX_LEVELS * HINT_BINS], s_gl[HINT_MAX_LEVELS], s_gb[HINT_BINS];
    __shared__ float s_ll[(HINT_MAX_LEVELS + 1) * HINT_BINS], s_lv[HINT_MAX_LEVELS + 1], s_ls[2 * (HINT_MAX_LEVELS + 1)];
    for (int k = threadIdx.x; k < HINT_MAX_LEVELS * HINT_BINS; k += blockDim.x) s_gamma[k] = 0.0f;
    for (int k = threadIdx.x; k < (HINT_MAX_LEVELS + 1) * HINT_BINS; k += blockDim.x) s_ll[k] = 0.0f;
    if (threadIdx.x < HINT_MAX_LEVELS) s_gl[threadIdx.x] = 0.0f;
    if (threadIdx.x < HINT_BINS) s_gb[threadIdx.x] = 0.0f;
    if (threadIdx.x < HINT_MAX_LEVELS + 1) s_lv[threadIdx.x] = 0.0f;
    if (threadIdx.x < 2 * (HINT_MAX_LEVELS + 1)) s_ls[threadIdx.x] = 0.0f;
    __syncthreads();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float g0 = 0.0f;
    if (i < N) {
        const uint32_t* tr = tot + (size_t)i * L;
        float pc[HINT_MAX_LEVELS + 1], g[HINT_MAX_LEVELS], lg[HINT_MAX_LEVELS]; int bins[HINT_MAX_LEVELS]; int gate, sp; float a;
        ngram_position(P, tr, dist + (size_t)i * L, cnt + (size_t)i * L, targets[i], L, &a, pc, g, lg, bins, &gate, &sp);
        float r = r_in[i];
        float sig = 1.0f / (1.0f + expf(-a));
        float ga = (sig - r) * dloss;
        atomicAdd(&s_ll[gate], ga);
        atomicAdd(&s_lv[gate / HINT_BINS], ga);
        atomicAdd(&s_ls[sp], ga);
        float glogq = -(1.0f - r) * dloss;
        float gp = pc[L] >= 1e-30f ? glogq / pc[L] : 0.0f;  // log(clamp_min(p, 1e-30))
        for (int j = L - 1; j >= 0; j--) {
            float t = (float)tr[j];
            if (t > 0.0f) {
                float gd = g[j] * (float)dist[(size_t)i * L + j];
                float den = t + gd;
                float in_range = (lg[j] >= -20.0f && lg[j] <= 20.0f) ? 1.0f : 0.0f;
                float dlg = gp * (pc[j] - pc[j + 1]) / den * gd * in_range;
                gp = gp * gd / den;
                atomicAdd(&s_gamma[j * HINT_BINS + bins[j]], dlg);
                atomicAdd(&s_gl[j], dlg);
                atomicAdd(&s_gb[bins[j]], dlg);
            }
        }
        g0 = gp * pc[0];  // dL/dlog p0, p0 = softmax(base)[target]
        atomicAdd(grads + o_base + targets[i], g0);
    }
    float block = blockReduce<warpReduceSum>(g0, true);
    if (threadIdx.x == 0) atomicAdd(base_gsum, block);
    __syncthreads();
    for (int k = threadIdx.x; k < L * HINT_BINS; k += blockDim.x) if (s_gamma[k] != 0.0f) atomicAdd(grads + o_gamma + k, s_gamma[k]);
    for (int k = threadIdx.x; k < (L + 1) * HINT_BINS; k += blockDim.x) if (s_ll[k] != 0.0f) atomicAdd(grads + o_lam_logit + k, s_ll[k]);
    if (threadIdx.x < L) atomicAdd(grads + o_gamma_level + threadIdx.x, s_gl[threadIdx.x]);
    if (threadIdx.x < HINT_BINS) atomicAdd(grads + o_gamma_bin + threadIdx.x, s_gb[threadIdx.x]);
    if (threadIdx.x < L + 1) atomicAdd(grads + o_lam_level + threadIdx.x, s_lv[threadIdx.x]);
    if (threadIdx.x < 2 * (L + 1)) atomicAdd(grads + o_lam_sparse + threadIdx.x, s_ls[threadIdx.x]);
}

// the dense part of the unigram logits' gradient: dbase[v] -= softmax(base)[v] * sum_i dL/dlog p0_i
__global__ void hint_base_finalize_kernel(float* dbase, const float* base, const float* scalars, int V) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < V) dbase[v] -= expf(base[v] - scalars[0]) * scalars[1];
}

// the fused classifier with the mixture: per row, the LM's nll from the softmax, the mixture's loss
// -logaddexp(logsig(a) - nll, logsig(-a) + log q), the LM's posterior r, and (training) dlogits =
// (softmax - onehot) * (r + aux) * dloss (llm.c's fused_classifier_kernel5 otherwise)
template <bool WriteDLogits>
__global__ void __launch_bounds__(1024, MAX_1024_THREADS_BLOCKS)
    fused_classifier_mix_kernel(floatX* logits, float* losses, float* lm_nll, float* r_out, const float* ng_a,
                                const float* ng_logq, const float dloss, const float aux, const int* targets,
                                int B, int T, int V, int P) {
    __shared__ float s_scale;
    int64_t idx = gridDim.x - (blockIdx.x + 1);
    int ix = targets[idx];
    SoftmaxParams sp = prepare_softmax_blockwide3(idx, logits, V, P);
    if (threadIdx.x == 0) {
        float nll = -((float)logits[idx * P + ix] - sp.Offset) + logf(1.0f / sp.Scale);  // -log softmax[target]
        float a = ng_a[idx];
        float A = log_sigmoid(a) - nll;
        float Bq = log_sigmoid(-a) + ng_logq[idx];
        float M = fmaxf(A, Bq) + log1pf(expf(-fabsf(A - Bq)));  // logaddexp
        float r = expf(A - M);
        losses[idx] -= M;
        lm_nll[idx] += nll;
        r_out[idx] = r;
        s_scale = (r + aux) * dloss;
    }
    __syncthreads();
    if (!WriteDLogits) return;
    const float scale = s_scale;
    const floatX* logits_vec = logits + idx * P;
    for (int i = threadIdx.x; i < V / x128::size; i += blockDim.x) {
        x128 packed_logits_vec = load128(logits_vec + i * x128::size);
        for (int k = 0; k < x128::size; ++k) {
            int element = i * x128::size + k;
            float prob = expf((float)packed_logits_vec[k] - sp.Offset) * sp.Scale;
            float indicator = (element == ix) ? 1.0f : 0.0f;
            packed_logits_vec[k] = (floatX)((prob - indicator) * scale);
        }
        store128cs(logits + idx * P + i * x128::size, packed_logits_vec);
    }
    int unaligned_start = V & ~(x128::size - 1);
    for (int i = threadIdx.x + unaligned_start; i < V; i++) {
        float prob = expf((float)logits_vec[i] - sp.Offset) * sp.Scale;
        float indicator = (i == ix) ? 1.0f : 0.0f;
        __stcs(logits + idx * P + i, (floatX)((prob - indicator) * scale));
    }
}

// ----------------------------------------------------------------------------
// the hint optimizer: torch.optim.Adam (betas 0.9 / 0.99, eps 1e-10, no weight decay), fp32

__global__ void hint_adam_kernel(float* p, const float* g, float* m, float* v, size_t n, float lr, float b1, float b2,
                                 float eps, float bc1, float bc2_sqrt) {
    size_t k = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (k >= n) return;
    float gk = g[k];
    float mk = b1 * m[k] + (1.0f - b1) * gk;
    float vk = b2 * v[k] + (1.0f - b2) * gk * gk;
    m[k] = mk; v[k] = vk;
    float denom = sqrtf(vk) / bc2_sqrt + eps;
    p[k] -= (lr / bc1) * mk / denom;
}

// ----------------------------------------------------------------------------
// launchers

static void hint_forward(HintState* h, const floatX* wte, cudaStream_t stream) {
    hint_forward_kernel<<<h->N, 128, 0, stream>>>(h->hint, h->mean, h->dev.cell, h->dev.tokens, h->dev.weights, wte,
                                                 h->params + h->o_next_scale, h->params + h->o_cell_embed, h->C,
                                                 h->entries, 1.0f / h->embed_ratio);
    cudaCheck(cudaGetLastError());
}

static void hint_add(HintState* h, floatX* x, int site, cudaStream_t stream) {
    size_t n = (size_t)h->N * h->C;
    hint_add_kernel<<<CEIL_DIV(n, 256 * 8), 256, 0, stream>>>(x, h->hint, h->params + h->o_site_scales + site, n);
    cudaCheck(cudaGetLastError());
}

static void hint_site_backward(HintState* h, const floatX* g, int site, int accumulate, cudaStream_t stream) {
    size_t n = (size_t)h->N * h->C;
    hint_site_backward_kernel<<<CEIL_DIV(n, 256 * 16), 256, 0, stream>>>(h->dhint, g, h->hint,
        h->params + h->o_site_scales + site, h->grads + h->o_site_scales + site, n, accumulate);
    cudaCheck(cudaGetLastError());
}

static void hint_cells_backward(HintState* h, cudaStream_t stream) {
    dim3 grid(CEIL_DIV(h->N, HINT_CELL_CHUNK), CEIL_DIV(h->C, 256));
    hint_cell_backward_kernel<<<grid, 256, 0, stream>>>(h->grads + h->o_cell_embed, h->grads + h->o_next_scale,
                                                        h->dhint, h->mean, h->order, h->sort_out, h->N, h->C);
    cudaCheck(cudaGetLastError());
}

static NgramParams hint_ngram_params(HintState* h) {
    float* p = h->params;
    return NgramParams{p + h->o_base, p + h->o_gamma, p + h->o_gamma_level, p + h->o_gamma_bin, p + h->o_lam_logit,
                       p + h->o_lam_level, p + h->o_lam_sparse, h->scalars};
}

// the n-gram side of the loss for a batch with `targets` (device): a, log q per position
static void hint_ngram_forward(HintState* h, const int* targets, cudaStream_t stream) {
    hint_base_lse_kernel<<<1, 1024, 0, stream>>>(h->scalars, h->params + h->o_base, h->V);
    ngram_forward_kernel<<<CEIL_DIV(h->N, 256), 256, 0, stream>>>(h->ng_a, h->ng_logq, hint_ngram_params(h),
        h->dev.tot, h->dev.dist, h->dev.cnt, targets, h->N, h->levels);
    cudaCheck(cudaGetLastError());
}

template <bool WriteDLogits>
static void hint_classifier(HintState* h, floatX* logits, float* losses, float dloss, const int* targets, int B, int T,
                            int V, int P, cudaStream_t stream) {
    fused_classifier_mix_kernel<WriteDLogits><<<B * T, 1024, 0, stream>>>(logits, losses, h->lm_nll, h->ng_r, h->ng_a,
        h->ng_logq, dloss, h->aux, targets, B, T, V, P);
    cudaCheck(cudaGetLastError());
}

static void hint_ngram_backward(HintState* h, const int* targets, float dloss, cudaStream_t stream) {
    ngram_backward_kernel<<<CEIL_DIV(h->N, 256), 256, 0, stream>>>(h->grads, h->o_base, h->o_gamma, h->o_gamma_level,
        h->o_gamma_bin, h->o_lam_logit, h->o_lam_level, h->o_lam_sparse, h->scalars + 1, hint_ngram_params(h), h->ng_r,
        h->dev.tot, h->dev.dist, h->dev.cnt, targets, h->N, h->levels, dloss);
    cudaCheck(cudaGetLastError());
}

static void hint_base_finalize(HintState* h, cudaStream_t stream) {
    hint_base_finalize_kernel<<<CEIL_DIV(h->V, 256), 256, 0, stream>>>(h->grads + h->o_base, h->params + h->o_base,
                                                                       h->scalars, h->V);
    cudaCheck(cudaGetLastError());
}

static void hint_zero_grads(HintState* h, cudaStream_t stream) {
    cudaCheck(cudaMemsetAsync(h->grads, 0, h->n_params * sizeof(float), stream));
    cudaCheck(cudaMemsetAsync(h->scalars + 1, 0, sizeof(float), stream));
}

// Adam step with the port's groups: the scales (next_scale, site_scales, gammas, lambdas) at 10x lr, cell_embed and
// the unigram logits at lr; lr = HINT_LR x `factor` (the model's lr over its peak: the record's schedule)
static void hint_update(HintState* h, float factor, cudaStream_t stream) {
    const float lr = h->lr * factor;
    h->adam_t += 1;
    const float b1 = 0.9f, b2 = 0.99f, eps = 1e-10f;
    float bc1 = 1.0f - powf(b1, (float)h->adam_t);
    float bc2_sqrt = sqrtf(1.0f - powf(b2, (float)h->adam_t));
    struct Seg { size_t o, n; float mult; };
    std::vector<Seg> segs = {{h->o_next_scale, (size_t)h->cells, 10.f}, {h->o_cell_embed, (size_t)h->cells * h->C, 1.f},
                             {h->o_site_scales, 3, 10.f}};
    segs.push_back({h->o_base, (size_t)h->V, 1.f});
    segs.push_back({h->o_gamma, h->n_params - h->o_gamma, 10.f});  // gamma .. lam_sparse are contiguous
    for (auto& s : segs) {
        hint_adam_kernel<<<CEIL_DIV(s.n, 256), 256, 0, stream>>>(h->params + s.o, h->grads + s.o, h->m + s.o, h->v + s.o,
                                                               s.n, lr * s.mult, b1, b2, eps, bc1, bc2_sqrt);
        cudaCheck(cudaGetLastError());
    }
}

#endif // LLMC_HINTS_CUH
