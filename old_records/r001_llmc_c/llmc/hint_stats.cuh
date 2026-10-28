/*
The hint rows' statistics, computed in the run on the GPUs (llmc/hints.cuh reads the rows):

- Rows: every training position's rows (this rank's positions, every step), strictly causal in the steps as
  exact_match's OnlineRows (old_records/online_hints.py): a step's query counts only the occurrences whose tokens (the
  k-context and its next token) were all trained in earlier steps. Built from the run's own training stream (each
  shard's trained prefix, a shard a part), on each rank's GPU, at the start of the clock.
- ValBuild: the validation rows (old_records.ValCountRows' count rows and ValHintRows' hint rows), counted over every
  training shard listed, the shards split over the ranks and the counts merged exactly: old_records/gpu_val_stats.py's
  InProcess (--fused --direct) in C++, its kernels (gpu_val_stats.cu) reused.

The definitions are exact_match's (kgram.rs, train_hints.rs, online.rs): a k-context is the k tokens ending at a
position, a query's is valid iff it lies in its segment (a segment starts at each BOS and at each part's start), an
occurrence is any k tokens within one part and the token after them. Rows, per position: the count rows tot / dist /
cnt (the occurrences of its context, their distinct next tokens, the ones followed by its target) at every level of
LEVELS, and the hint row (cell, the top ENTRIES next tokens by count then token, their weights) of its deepest level of
HINT_LEVELS with an occurrence.

On the GPU, parts (shards) are concatenated with a gap of SENT tokens (0xFFFF, not a GPT-2 token) before each, so a
context reaching past its part's start holds a SENT and matches nothing. A level's queries are indexed (CtxIndex: the
distinct contexts, a context's id its slot in a lookup table), every position of the parts looks its context up (the
hashed levels verified token by token, so every count is exact), and a level only looks up the positions whose
context matched at the previous level.
*/
#ifndef LLMC_HINT_STATS_CUH
#define LLMC_HINT_STATS_CUH

#include <atomic>
#include <string>
#include <thread>
#include <vector>
#include <fcntl.h>
#include <time.h>
#include <unistd.h>
#include <cub/cub.cuh>
#include "cuda_common.h"
#include "zero.cuh"

namespace hint_stats {

typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned short u16;
typedef long long i64;

constexpr u64 EMPTY = ~0ull;
constexpr u32 SENT = 0xFFFFu;
constexpr u64 BASE = 0x9E3779B97F4A7C15ull;
constexpr u32 NONE = 0xFFFFFFFFu;  // a position no step trains
constexpr int MAXPROBE = 4096;
constexpr int MAXW = 64;
constexpr i64 GAP = 32;            // SENT tokens before each part (>= the longest context)
constexpr u16 BOS = 50256;
constexpr int VOCAB = 50257;
constexpr int NLEVELS = 10;        // the count rows' levels (old_records.COUNT_LEVELS)
constexpr int LEVELS[NLEVELS] = {2, 3, 4, 5, 6, 8, 10, 12, 16, 20};
constexpr int NHINT = 7;           // the hint rows' levels, each its cell's length bucket (val_hints.LEVELS)
constexpr int HINT_LEVELS[NHINT] = {2, 3, 4, 5, 6, 10, 20};
constexpr int ENTRIES = 8;
constexpr int TOP = 1024;          // the validation cnt rule: the target's count if among the TOP most frequent
constexpr i64 VAL_TOKENS = 10485760;
constexpr int SHARD_HEADER = 1024; // bytes
constexpr int ALPHA_BITS = 10;     // the training rows' steps (< 1024)

static int hint_level_of(int li) {  // the hint level index of count level index li, or -1
    for (int h = 0; h < NHINT; h++) if (HINT_LEVELS[h] == LEVELS[li]) return h;
    return -1;
}

static double now_s() { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

// device memory for the builds: freed blocks are kept and reused (cudaMalloc and cudaFree synchronize, and cost
// milliseconds for big blocks); sizes rounded up to 1/8 powers of two; cache_release() frees them after the clock
static std::vector<std::pair<size_t, void*>> g_cache;  // free blocks
static std::vector<std::pair<void*, size_t>> g_live;   // blocks in use
static size_t round_size(size_t b) {
    if (b < 4096) return 4096;
    int e = 63 - __builtin_clzll(b);
    size_t step = (size_t)1 << (e >= 3 ? e - 3 : 0);
    return (b + step - 1) / step * step;
}
template <typename T> static T* dalloc(size_t n) {
    const size_t b = round_size((n > 0 ? n : 1) * sizeof(T));
    void* p = nullptr;
    size_t best = 0;
    for (size_t i = 0; i < g_cache.size(); i++)  // the smallest cached block that fits, within 2x
        if (g_cache[i].first >= b && g_cache[i].first <= 2 * b && (p == nullptr || g_cache[i].first < g_cache[best].first)) { p = g_cache[i].second; best = i; }
    size_t size = b;
    if (p) { size = g_cache[best].first; g_cache.erase(g_cache.begin() + best); }
    else if (cudaMalloc(&p, b) != cudaSuccess) {  // out of memory: give the cache back first
        (void)cudaGetLastError();
        for (auto& c : g_cache) cudaCheck(cudaFree(c.second));
        g_cache.clear();
        cudaCheck(cudaMalloc(&p, b));
    }
    g_live.push_back({p, size});
    return (T*)p;
}
template <typename T> static void dfree(T*& p) {
    if (!p) return;
    for (size_t i = 0; i < g_live.size(); i++)
        if (g_live[i].first == (void*)p) { g_cache.push_back({g_live[i].second, g_live[i].first}); g_live.erase(g_live.begin() + i); break; }
    p = nullptr;
}
static void cache_release() {
    for (auto& c : g_cache) cudaCheck(cudaFree(c.second));
    g_cache.clear();
}

// CUB's temporary storage, grown as needed
static void* g_temp = nullptr;
static size_t g_temp_bytes = 0;
template <typename F> static void cub_run(F f) {
    size_t bytes = 0;
    cudaCheck(f(nullptr, bytes));
    if (bytes > g_temp_bytes) {
        if (g_temp) cudaCheck(cudaFree(g_temp));
        g_temp_bytes = bytes + bytes / 4;
        if (cudaMalloc(&g_temp, g_temp_bytes) != cudaSuccess) {  // out of memory: give the cache back first
            (void)cudaGetLastError();
            for (auto& c : g_cache) cudaCheck(cudaFree(c.second));
            g_cache.clear();
            cudaCheck(cudaMalloc(&g_temp, g_temp_bytes));
        }
    }
    cudaCheck(f(g_temp, bytes));
}

template <typename T> static T d2h1(const T* p) { T v; cudaCheck(cudaMemcpy(&v, p, sizeof(T), cudaMemcpyDeviceToHost)); return v; }

static int bit_length(u64 v) { int b = 0; while (v) { b++; v >>= 1; } return b; }

static int grid_for(i64 n, int block) {
    i64 g = (n + block - 1) / block;
    return (int)(g < 1 ? 1 : (g > (1 << 30) ? (1 << 30) : g));
}

static std::vector<i64> shard_sizes(const std::vector<std::string>& paths) {
    std::vector<i64> sizes;
    for (auto& p : paths) {
        int header[256];
        FILE* f = fopenCheck(p.c_str(), "rb");
        freadCheck(header, sizeof(int), 256, f);
        fcloseCheck(f);
        if (header[0] != 20240520 || header[1] != 1) { fprintf(stderr, "invalid shard %s\n", p.c_str()); exit(EXIT_FAILURE); }
        sizes.push_back(header[2]);
    }
    return sizes;
}

// parts of sizes[i] tokens at starts[i] (32-aligned, >= GAP after the previous part's end), and the length
static i64 layout(const std::vector<i64>& sizes, std::vector<i64>& starts) {
    i64 at = GAP;
    starts.clear();
    for (i64 n : sizes) { starts.push_back(at); at = (at + n + GAP + 31) / 32 * 32; }
    return at;
}

// tokens [a, a + n) of shard `path` into host memory at dst
static void read_tokens(const std::string& path, i64 a, i64 n, u16* dst) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(EXIT_FAILURE); }
    size_t want = 2 * (size_t)n, got = 0;
    while (got < want) {
        ssize_t r = pread(fd, (char*)dst + got, want - got, SHARD_HEADER + 2 * a + got);
        if (r <= 0) { fprintf(stderr, "short read of %s\n", path.c_str()); exit(EXIT_FAILURE); }
        got += r;
    }
    close(fd);
}

// ----------------------------------------------------------------------------
// keys (gpu_val_stats.cu's): for k <= 4 the tokens packed (exact), else a polynomial hash of 64-bit token codes;
// both through splitmix64's bijective finalizer

__device__ __forceinline__ u64 mix64(u64 h) {
    h ^= h >> 30;
    h *= 0xbf58476d1ce4e5b9ull;
    h ^= h >> 27;
    h *= 0x94d049bb133111ebull;
    return h ^ (h >> 31);
}

__device__ __forceinline__ u64 tcode(u32 t) { return mix64((u64)t * 0x2545f4914f6cdd1dull + 0x632be59bd9b4e019ull) | 1ull; }

// the key of the K tokens ending just before `e` (e[-K..-1])
template <int K>
__device__ __forceinline__ u64 ctx_key(const u16* e) {
    u64 k;
    if (K <= 4) {
        u64 p = 0;
#pragma unroll
        for (int j = K; j >= 1; --j) p = (p << 16) | (u64)e[-j];
        k = mix64(p);
    } else {
        u64 h = 0, bp = 1;
#pragma unroll
        for (int j = 1; j <= K; ++j) { h += tcode(e[-j]) * bp; bp *= BASE; }
        k = mix64(h);
    }
    return k == EMPTY ? EMPTY - 1 : k;
}

#define HS_LEVEL_SWITCH(k, ...)                                \
    switch (k) {                                               \
        case 2: { constexpr int K = 2; __VA_ARGS__; } break;   \
        case 3: { constexpr int K = 3; __VA_ARGS__; } break;   \
        case 4: { constexpr int K = 4; __VA_ARGS__; } break;   \
        case 5: { constexpr int K = 5; __VA_ARGS__; } break;   \
        case 6: { constexpr int K = 6; __VA_ARGS__; } break;   \
        case 8: { constexpr int K = 8; __VA_ARGS__; } break;   \
        case 10: { constexpr int K = 10; __VA_ARGS__; } break; \
        case 12: { constexpr int K = 12; __VA_ARGS__; } break; \
        case 16: { constexpr int K = 16; __VA_ARGS__; } break; \
        case 20: { constexpr int K = 20; __VA_ARGS__; } break; \
        default: fprintf(stderr, "level %d\n", k); exit(EXIT_FAILURE); \
    }

// ----------------------------------------------------------------------------
// CtxIndex: one level's query contexts (gpu_val_stats.py's ValIndex): per position of x its context id (-1 where it
// is no query), and the lookup table (8-byte slots, a context's id is its slot) with each slot's representative
// position. The table's layout is canonical (the same on every rank for the same queries): the keys in order of home
// slot (then key) at slot_i = max(home_i, slot_{i-1} + 1), linear probing without wrap-around, then an empty slot.

struct CtxIndex {
    u64* table = nullptr;  // [ids]
    u32* rep = nullptr;    // [ids]
    int* ctx_of = nullptr; // [len]
    u64 cap = 0;           // the home slots' range
    i64 ids = 0, n = 0;    // the id space (table size), the distinct contexts
    void release() { dfree(table); dfree(rep); dfree(ctx_of); }
};

template <int K>
__global__ void ix_keys_k(const u16* x, const int* avail, i64 n, u64* out, unsigned char* flag) {
    i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x;
    if (i >= n) return;
    bool q = avail[i] >= K;
    out[i] = q ? ctx_key<K>(x + i + 1) : EMPTY;
    flag[i] = q;
}

__global__ void ix_gather_k(const u64* keys, const i64* pos, i64 m, u64* out) {
    i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x;
    if (i < m) out[i] = keys[pos[i]];
}

__global__ void ix_new_k(const u64* sk, i64 m, i64* isnew) {
    i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x;
    if (i < m) isnew[i] = (i == 0 || sk[i] != sk[i - 1]) ? 1 : 0;
}

// gid = inclusive sum of isnew (in place) - 1: the group's unique key and representative (its first position)
__global__ void ix_unique_k(const u64* sk, const i64* spos, const i64* csum, i64 m, u64* ukeys, i64* rep) {
    i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x;
    if (i >= m) return;
    if (i == 0 || sk[i] != sk[i - 1]) { ukeys[csum[i] - 1] = sk[i]; rep[csum[i] - 1] = spos[i]; }
}

// each query's k tokens against its group representative's: the mismatches (hash collisions) into *bad
__global__ void ix_verify_k(const u16* x, const i64* spos, const i64* csum, const i64* rep, i64 m, int k, u32* bad) {
    i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x;
    if (i >= m) return;
    i64 p = spos[i], r = rep[csum[i] - 1];
    for (int j = 0; j < k; ++j) if (x[p - j] != x[r - j]) { atomicAdd(bad, 1u); return; }
}

__global__ void ix_homes_k(const u64* keys, i64 n, u64 cap, i64* home) {
    i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x;
    if (i < n) home[i] = (i64)__umul64hi(keys[i], cap) - i;
}

__global__ void ix_place_k(const u64* ukeys, const i64* rep, const i64* slot_minus, i64 n, u64* table, u32* trep) {
    i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x;
    if (i >= n) return;
    i64 s = slot_minus[i] + i;
    table[s] = ukeys[i];
    trep[s] = (u32)rep[i];
}

__global__ void ix_ctx_of_k(const i64* spos, const i64* csum, const i64* slot_minus, i64 m, int* ctx_of) {
    i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x;
    if (i >= m) return;
    i64 g = csum[i] - 1;
    ctx_of[spos[i]] = (int)(slot_minus[g] + g);
}

// the index of the k-contexts ending at the positions i of x with avail[i] >= k (the k tokens x[i-k+1..i] lie in
// their segment); ids below 2^24 with ids24 (the validation's pair keys hold 24-bit ids)
static void ctx_index_build(CtxIndex& ix, const u16* x, const int* avail, i64 len, int k, bool ids24, cudaStream_t st) {
    u64* keys = dalloc<u64>(len);
    unsigned char* flag = dalloc<unsigned char>(len);
    i64* pos = dalloc<i64>(len);
    i64* nsel = dalloc<i64>(1);
    HS_LEVEL_SWITCH(k, (ix_keys_k<K><<<grid_for(len, 256), 256, 0, st>>>(x, avail, len, keys, flag)));
    cub_run([&](void* t, size_t& b) {
        return cub::DeviceSelect::Flagged(t, b, cub::CountingInputIterator<i64>(0), flag, pos, nsel, len, st);
    });
    const i64 m = d2h1(nsel);
    dfree(flag);
    u64* kq = dalloc<u64>(m);
    ix_gather_k<<<grid_for(m, 256), 256, 0, st>>>(keys, pos, m, kq);
    dfree(keys);
    u64* sk = dalloc<u64>(m);
    i64* spos = dalloc<i64>(m);
    cub_run([&](void* t, size_t& b) { return cub::DeviceRadixSort::SortPairs(t, b, kq, sk, pos, spos, m, 0, 64, st); });
    dfree(kq);
    dfree(pos);
    i64* csum = dalloc<i64>(m);
    ix_new_k<<<grid_for(m, 256), 256, 0, st>>>(sk, m, csum);
    cub_run([&](void* t, size_t& b) { return cub::DeviceScan::InclusiveSum(t, b, csum, csum, m, st); });
    const i64 n = m > 0 ? d2h1(csum + m - 1) : 0;
    u64* ukeys = dalloc<u64>(n);
    i64* rep = dalloc<i64>(n);
    ix_unique_k<<<grid_for(m, 256), 256, 0, st>>>(sk, spos, csum, m, ukeys, rep);
    u32* bad = dalloc<u32>(1);
    cudaCheck(cudaMemsetAsync(bad, 0, 4, st));
    ix_verify_k<<<grid_for(m, 256), 256, 0, st>>>(x, spos, csum, rep, m, k, bad);
    if (d2h1(bad)) { fprintf(stderr, "k=%d: %u query contexts collide with another's key\n", k, d2h1(bad)); exit(EXIT_FAILURE); }
    dfree(bad);
    u64 cap = 2 * (u64)n + 1024;
    if (ids24 && cap > (1ull << 24) - 4096) cap = (1ull << 24) - 4096;
    i64* slot_minus = dalloc<i64>(n);  // slot_i - i
    ix_homes_k<<<grid_for(n, 256), 256, 0, st>>>(ukeys, n, cap, slot_minus);
    cub_run([&](void* t, size_t& b) { return cub::DeviceScan::InclusiveScan(t, b, slot_minus, slot_minus, cub::Max(), n, st); });
    const i64 last = n > 0 ? d2h1(slot_minus + n - 1) + n - 1 : -1;
    const i64 size = ((i64)cap > last + 1 ? (i64)cap : last + 1) + 1;
    if (ids24 && size >= (1 << 24)) { fprintf(stderr, "k=%d: %lld contexts do not fit 24-bit ids\n", k, n); exit(EXIT_FAILURE); }
    ix.table = dalloc<u64>(size);
    ix.rep = dalloc<u32>(size);
    ix.ctx_of = dalloc<int>(len);
    cudaCheck(cudaMemsetAsync(ix.table, 0xFF, size * sizeof(u64), st));
    cudaCheck(cudaMemsetAsync(ix.rep, 0, size * sizeof(u32), st));
    cudaCheck(cudaMemsetAsync(ix.ctx_of, 0xFF, len * sizeof(int), st));
    ix_place_k<<<grid_for(n, 256), 256, 0, st>>>(ukeys, rep, slot_minus, n, ix.table, ix.rep);
    ix_ctx_of_k<<<grid_for(m, 256), 256, 0, st>>>(spos, csum, slot_minus, m, ix.ctx_of);
    cudaCheck(cudaGetLastError());
    ix.cap = cap; ix.ids = size; ix.n = n;
    dfree(sk); dfree(spos); dfree(csum); dfree(ukeys); dfree(rep); dfree(slot_minus); dfree(nsel);
}

// a position's context looked up (position p is the next token's; the context is tok[p-K..p-1]) and verified on the
// hashed levels against its slot's representative query (x holds the queries): its slot, or -1
template <int K, bool HASHED>
__device__ __forceinline__ i64 ix_lookup(const u16* __restrict__ tok, i64 p, const u64* __restrict__ vt, u64 vcap,
                                         const u32* __restrict__ vrep, const u16* __restrict__ x) {
    const u64 key = ctx_key<K>(tok + p);
    u64 s = __umul64hi(key, vcap);
    while (true) {
        const u64 v = vt[s];
        if (v == key) {
            if (HASHED) {
                const u16* a = tok + p - K;
                const u16* b = x + (i64)vrep[s] + 1 - K;
                bool same = true;
#pragma unroll
                for (int j = 0; j < K; ++j) same &= a[j] == b[j];
                if (!same) return -1;  // another context with the same key: no query's
            }
            return (i64)s;
        }
        if (v == EMPTY) return -1;
        ++s;  // no wrap-around: the table ends with an empty slot
    }
}

// ----------------------------------------------------------------------------
// The training rows of this rank's positions: row r is position r % (B*T) of the rank's micro-batch r / (B*T)
// (= step * accum + micro-step). A query at step s counts the occurrences with alpha <= s, alpha = 1 + the last step
// that trains (as an input or a target) one of its k + 1 tokens.

// rows (training or validation): per position its hint row and its count rows
struct Rows {
    i64 n = 0;  // rows
    int16_t* cell = nullptr; uint16_t* tokens = nullptr; uint16_t* weights = nullptr;  // [n], [n, 8], [n, 8]
    uint32_t* tot = nullptr; uint16_t* dist = nullptr; uint32_t* cnt = nullptr;        // [n, NLEVELS]
};

static void rows_alloc(Rows& r, i64 n, cudaStream_t st) {
    r.n = n;
    r.cell = dalloc<int16_t>(n); r.tokens = dalloc<uint16_t>(n * ENTRIES); r.weights = dalloc<uint16_t>(n * ENTRIES);
    r.tot = dalloc<uint32_t>(n * NLEVELS); r.dist = dalloc<uint16_t>(n * NLEVELS); r.cnt = dalloc<uint32_t>(n * NLEVELS);
    cudaCheck(cudaMemsetAsync(r.cell, 0, n * 2, st));
    cudaCheck(cudaMemsetAsync(r.tokens, 0, n * ENTRIES * 2, st));
    cudaCheck(cudaMemsetAsync(r.weights, 0, n * ENTRIES * 2, st));
    cudaCheck(cudaMemsetAsync(r.tot, 0, n * NLEVELS * 4, st));
    cudaCheck(cudaMemsetAsync(r.dist, 0, n * NLEVELS * 2, st));
    cudaCheck(cudaMemsetAsync(r.cnt, 0, n * NLEVELS * 4, st));
}

// one micro-batch of one rank: inputs [lo, lo + B*T) of shard `shard` (offsets in tokens), trained in `step`
struct Span { int step, micro, rank, shard; i64 offset; };

// llm.c's sequential DataLoader (dataloader_next_batch without shuffling), every rank's micro-batches of `steps` steps
static std::vector<Span> loader_plan(const std::vector<i64>& ntok, int nproc, i64 BT, int accum, int steps) {
    std::vector<Span> plan;
    for (int r = 0; r < nproc; r++) {
        size_t shard = 0, sample = 0;
        for (int b = 0; b < steps * accum; b++) {
            if (sample >= (size_t)((ntok[shard] - 1) / (nproc * BT))) {
                shard++; sample = 0;
                if (shard >= ntok.size()) { fprintf(stderr, "the training plan runs past the %zu shards\n", ntok.size()); exit(EXIT_FAILURE); }
            }
            plan.push_back({b / accum, b % accum, r, (int)shard, (i64)(sample * nproc * BT + r * BT)});
            sample++;
        }
    }
    return plan;
}

__global__ void tr_spans_k(const i64* lo, const int* step, const int* row0, i64 BT, u32* tau, u32* tau_in, int* row_of) {
    const i64 a = lo[blockIdx.x];
    const u32 s = (u32)step[blockIdx.x];
    const int r0 = row0[blockIdx.x];
    for (i64 i = threadIdx.x; i < BT; i += blockDim.x) {
        atomicMin(&tau[a + i], s);
        tau_in[a + i] = s;
        if (r0 >= 0) row_of[a + i] = r0 + (int)i;
    }
    if (threadIdx.x == 0) atomicMin(&tau[a + BT], s);  // the last input's target
}

__global__ void tr_segv_k(const u16* tok, i64 L, i64* v) {
    i64 p = blockIdx.x * (i64)blockDim.x + threadIdx.x;
    if (p < L) v[p] = tok[p] == BOS ? p : 0;
}

__global__ void tr_partstart_k(const i64* starts, int n, i64* v) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) v[starts[i]] = starts[i];
}

__global__ void tr_avail_k(const i64* seg, const int* row_of, i64 L, int* avail) {
    i64 p = blockIdx.x * (i64)blockDim.x + threadIdx.x;
    if (p < L) avail[p] = row_of[p] >= 0 ? (int)(p - seg[p] + 1) : 0;
}

__global__ void tr_unigram_k(const u16* tok, i64 n, u32* counts) {
    for (i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; i < n; i += (i64)gridDim.x * blockDim.x) {
        u16 t = tok[i];
        if (t < VOCAB) atomicAdd(&counts[t], 1u);
    }
}

// position p's occurrence (its context tok[p-K..p-1] and next token tok[p]) if its context is a query's and every
// one of its tokens is trained before the last step: (slot << 26 | next << 10 | alpha), or EMPTY
template <int K, bool HASHED>
__device__ __forceinline__ u64 tr_hit(const u16* __restrict__ tok, const u32* __restrict__ tau, i64 p, const u64* __restrict__ vt,
                                      u64 vcap, const u32* __restrict__ vrep, u32 steps) {
    const u32 nxt = tok[p];
    if (nxt == SENT) return EMPTY;
    const i64 s = ix_lookup<K, HASHED>(tok, p, vt, vcap, vrep, tok);
    if (s < 0) return EMPTY;
    u32 m = 0;
#pragma unroll
    for (int j = 0; j <= K; ++j) m = max(m, tau[p - j]);
    if (m == NONE || m + 1 >= steps) return EMPTY;
    return ((u64)s << (16 + ALPHA_BITS)) | ((u64)nxt << ALPHA_BITS) | (u64)(m + 1);
}

// dense levels: a lane per position, a warp per bitmap word (bm_in null: every position); the hits appended to emit
template <int K, bool HASHED>
__global__ void __launch_bounds__(256) tr_scan_k(const u16* __restrict__ tok, const u32* __restrict__ tau, i64 L,
                                                 const u32* __restrict__ bm_in, u32* __restrict__ bm_out,
                                                 const u64* __restrict__ vt, u64 vcap, const u32* __restrict__ vrep,
                                                 u32 steps, u64* emit, unsigned long long* emitted) {
    const int lane = threadIdx.x & 31;
    const i64 stride = (i64)gridDim.x * blockDim.x;
    for (i64 base = blockIdx.x * (i64)blockDim.x + (threadIdx.x & ~31); base < L; base += stride) {
        const i64 p = base + lane;
        const u32 word = bm_in ? bm_in[base >> 5] : 0xFFFFFFFFu;
        u64 key = EMPTY;
        if (((word >> lane) & 1u) && p < L) key = tr_hit<K, HASHED>(tok, tau, p, vt, vcap, vrep, steps);
        const bool hit = key != EMPTY;
        const u32 b = __ballot_sync(0xFFFFFFFFu, hit);
        if (lane == 0) bm_out[base >> 5] = b;
        if (b) {
            unsigned long long at = 0;
            if (lane == 0) at = atomicAdd(emitted, (unsigned long long)__popc(b));
            at = __shfl_sync(0xFFFFFFFFu, at, 0);
            if (hit) emit[at + __popc(b & ((1u << lane) - 1))] = key;
        }
    }
}

// sparse levels: a lane per bitmap word, its set bits in turn
template <int K, bool HASHED>
__global__ void __launch_bounds__(256) tr_scan_sparse_k(const u16* __restrict__ tok, const u32* __restrict__ tau, i64 L,
                                                        const u32* __restrict__ bm_in, u32* __restrict__ bm_out,
                                                        const u64* __restrict__ vt, u64 vcap, const u32* __restrict__ vrep,
                                                        u32 steps, u64* emit, unsigned long long* emitted) {
    const i64 nwords = (L + 31) >> 5;
    for (i64 w = blockIdx.x * (i64)blockDim.x + threadIdx.x; w < nwords; w += (i64)gridDim.x * blockDim.x) {
        u32 word = bm_in[w], out = 0;
        while (word) {
            const int b = __ffs(word) - 1;
            word &= word - 1;
            const i64 p = (w << 5) + b;
            if (p >= L) continue;
            const u64 key = tr_hit<K, HASHED>(tok, tau, p, vt, vcap, vrep, steps);
            if (key != EMPTY) { out |= 1u << b; emit[atomicAdd(emitted, 1ull)] = key; }
        }
        bm_out[w] = out;
    }
}

// per id, the range [start, end) of its entries in keys sorted by id (id = key >> shift); untouched ids keep 0, 0
__global__ void seg_ranges_k(const u64* keys, i64 n, int shift, u32* start, u32* end) {
    i64 j = blockIdx.x * (i64)blockDim.x + threadIdx.x;
    if (j >= n) return;
    u64 c = keys[j] >> shift;
    if (j == 0 || (keys[j - 1] >> shift) != c) start[c] = (u32)j;
    if (j == n - 1 || (keys[j + 1] >> shift) != c) end[c] = (u32)(j + 1);
}

// (id << 10 | alpha) of each hit, and the flags of the first hit of each (id, next token) run
__global__ void tr_time_keys_k(const u64* a1, i64 n, u64* a2, unsigned char* first) {
    i64 j = blockIdx.x * (i64)blockDim.x + threadIdx.x;
    if (j >= n) return;
    u64 v = a1[j];
    a2[j] = ((v >> (16 + ALPHA_BITS)) << ALPHA_BITS) | (v & ((1u << ALPHA_BITS) - 1));
    first[j] = j == 0 || (a1[j - 1] >> ALPHA_BITS) != (v >> ALPHA_BITS);
}

// the entries of keys[a, b) whose low ALPHA_BITS bits are <= s (sorted by them within the range)
__device__ __forceinline__ u32 count_le(const u64* keys, u32 a, u32 b, u32 s) {
    u32 lo = a, hi = b;
    while (lo < hi) { u32 m = (lo + hi) >> 1; if ((u32)(keys[m] & ((1u << ALPHA_BITS) - 1)) <= s) lo = m + 1; else hi = m; }
    return lo - a;
}

__device__ __forceinline__ u32 lower_bound(const u64* keys, u32 a, u32 b, u64 want) {
    while (a < b) { u32 m = (a + b) >> 1; if (keys[m] < want) a = m + 1; else b = m; }
    return a;
}

// per query of this level: tot, dist and cnt at level li; the deepest hint level so far and its context id
__global__ void tr_query_k(i64 L, const int* ctx_of, const int* row_of, const u32* tau_in, const u16* tok,
                           const u64* a1, const u64* a2, const u64* a3, const u32* start1, const u32* end1,
                           const u32* start3, const u32* end3, int li, int hl, u32* tot, u16* dist, u32* cnt,
                           signed char* hint_level, int* hint_ctx) {
    i64 p = blockIdx.x * (i64)blockDim.x + threadIdx.x;
    if (p >= L) return;
    const int c = ctx_of[p];
    if (c < 0) return;
    const u32 a = start1[c], b = end1[c];
    if (a >= b) return;
    const u32 s = tau_in[p];
    const u32 T = count_le(a2, a, b, s);
    if (T == 0) return;
    const u32 D = count_le(a3, start3[c], end3[c], s);
    const u64 want = ((u64)c << (16 + ALPHA_BITS)) | ((u64)tok[p + 1] << ALPHA_BITS);
    const u32 lo = lower_bound(a1, a, b, want);
    const u32 hi = lower_bound(a1, lo, b, want | (u64)(s + 1));  // s + 1 <= the steps < 2^ALPHA_BITS
    const i64 row = row_of[p];
    tot[row * NLEVELS + li] = T;
    dist[row * NLEVELS + li] = (u16)min(D, 65535u);
    cnt[row * NLEVELS + li] = hi - lo;
    if (hl >= 0) { hint_level[row] = (signed char)hl; hint_ctx[row] = c; }
}

// one hint level's snapshot keys (ctx << 10 | step) of the rows whose deepest hint level it is
__global__ void tr_snap_keys_k(i64 n, const signed char* hint_level, const int* hint_ctx, int hl, i64 rows_per_step,
                               u64* keys, unsigned char* flag) {
    i64 r = blockIdx.x * (i64)blockDim.x + threadIdx.x;
    if (r >= n) return;
    flag[r] = hint_level[r] == hl;
    keys[r] = ((u64)(u32)hint_ctx[r] << ALPHA_BITS) | (u64)(r / rows_per_step);
}

__device__ __forceinline__ bool better(u32 c, u32 t, u32 c2, u32 t2) { return c > c2 || (c == c2 && t < t2); }

// a warp per distinct snapshot (ctx, step): the top ENTRIES next tokens by count at that step (count descending, token
// ascending) over the context's (ctx, next token) runs of a1 (r1: their first indices, ranges start3 / end3)
__global__ void tr_top_k(const u64* snaps, i64 U, const u64* a1, const u32* r1, const u32* start3, const u32* end3,
                         const u32* end1, u16* top_tok, u32* top_cnt) {
    const i64 w = (blockIdx.x * (i64)blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (w >= U) return;
    const u64 snap = snaps[w];
    const u32 c = (u32)(snap >> ALPHA_BITS), s = (u32)(snap & ((1u << ALPHA_BITS) - 1));
    const u32 rs = start3[c], re = end3[c];
    u32 lc[ENTRIES], lt[ENTRIES];
    int nl = 0;
    for (u32 r = rs + lane; r < re; r += 32) {
        const u32 a = r1[r], b = r + 1 < re ? r1[r + 1] : end1[c];
        if ((u32)(a1[a] & ((1u << ALPHA_BITS) - 1)) > s) continue;
        const u32 cn = count_le(a1, a, b, s);
        const u32 t = (u32)((a1[a] >> ALPHA_BITS) & 0xFFFF);
        if (nl == ENTRIES && !better(cn, t, lc[ENTRIES - 1], lt[ENTRIES - 1])) continue;
        int i = nl < ENTRIES ? nl++ : ENTRIES - 1;
        while (i > 0 && better(cn, t, lc[i - 1], lt[i - 1])) { lc[i] = lc[i - 1]; lt[i] = lt[i - 1]; i--; }
        lc[i] = cn; lt[i] = t;
    }
    for (int j = 0; j < ENTRIES; j++) {
        const u64 mine = nl > 0 ? ((u64)lc[0] << 16) | (0xFFFFu - lt[0]) : 0;
        u64 best = mine;
        for (int o = 16; o > 0; o >>= 1) { u64 v = __shfl_xor_sync(0xFFFFFFFFu, best, o); best = v > best ? v : best; }
        if (best == 0) break;
        if (lane == 0) { top_tok[w * ENTRIES + j] = (u16)(0xFFFFu - (best & 0xFFFF)); top_cnt[w * ENTRIES + j] = (u32)(best >> 16); }
        if (mine == best) {  // tokens are distinct within a context: one lane
            for (int i = 1; i < nl; i++) { lc[i - 1] = lc[i]; lt[i - 1] = lt[i]; }
            nl--;
        }
    }
}

__constant__ double PURITY_EDGES[5] = {0.3, 0.5, 0.7, 0.85, 0.95};
__constant__ u64 COUNT_EDGES[6] = {2, 3, 5, 9, 17, 33};

// a hint row (old_records._val_rows / online.rs's answer): cell = ((1 + bucket) * 7 + count bin) * 6 + purity bin,
// the tokens and their weights (count / the top counts' sum * 65535, rounded half to even)
__device__ __forceinline__ void hint_row(int bucket, u64 total, const u16* tt, const u32* tc, int16_t* cell,
                                         uint16_t* tokens, uint16_t* weights) {
    double sum = 0;
    for (int r = 0; r < ENTRIES; r++) sum += (double)tc[r];
    int count_bin = 0;
    for (int e = 0; e < 6; ++e) count_bin += COUNT_EDGES[e] <= total;
    const double share = (double)tc[0] / (double)(total > 1 ? total : 1);
    int purity_bin = 0;
    for (int e = 0; e < 5; ++e) purity_bin += PURITY_EDGES[e] <= share;
    *cell = (int16_t)(((1 + bucket) * 7 + count_bin) * 6 + purity_bin);
    const double denom = sum > 1 ? sum : 1;
    for (int r = 0; r < ENTRIES; r++) {
        tokens[r] = tt[r];
        weights[r] = (uint16_t)rint((double)tc[r] / denom * 65535.0);
    }
}

// the rows of the sorted snapshot entries (svals: their rows, gid: their distinct snapshot)
__global__ void tr_hint_rows_k(i64 m, const u32* svals, const i64* csum, const u16* top_tok, const u32* top_cnt,
                               int hl, int li, Rows out) {
    i64 j = blockIdx.x * (i64)blockDim.x + threadIdx.x;
    if (j >= m) return;
    const i64 row = svals[j], u = csum[j] - 1;
    hint_row(hl, out.tot[row * NLEVELS + li], top_tok + u * ENTRIES, top_cnt + u * ENTRIES, out.cell + row,
             out.tokens + row * ENTRIES, out.weights + row * ENTRIES);
}

__global__ void compact_new_k(const u64* sk, const i64* csum, i64 m, u64* uniq) {
    i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x;
    if (i < m && (i == 0 || sk[i] != sk[i - 1])) uniq[csum[i] - 1] = sk[i];
}

// a hint level's per-level arrays, kept for its top-8 pass
struct LevelHits {
    i64 n = 0, ids = 0, runs = 0;
    u64* a1 = nullptr; u32* r1 = nullptr; u32* end1 = nullptr; u32* start3 = nullptr; u32* end3 = nullptr;
    void release() { dfree(a1); dfree(r1); dfree(end1); dfree(start3); dfree(end3); }
};

struct TrainBuildTimes { double load = 0, prep = 0, levels = 0, tops = 0, total = 0; i64 stream = 0, hits = 0; };

// the run's plan (every rank's micro-batches, `steps` steps of `accum` micro-steps of B*T tokens, from llm.c's
// sequential loader over `files`), the stream's parts (each read shard up to its last trained target; shard 0 whole,
// for the unigram) and a pinned buffer for them: made before the clock
struct TrainPlan {
    std::vector<std::string> files;
    std::vector<i64> ntok, sizes, starts;
    std::vector<Span> plan;
    int nshards = 0, nproc = 1, accum = 1, steps = 0;
    i64 BT = 0, L = 0;
    u16* host = nullptr;
};

static void train_plan(TrainPlan& p, const std::vector<std::string>& files, int nproc, i64 BT, int accum, int steps) {
    if (steps >= (1 << ALPHA_BITS) - 1) { fprintf(stderr, "train rows: %d steps, at most %d\n", steps, (1 << ALPHA_BITS) - 2); exit(EXIT_FAILURE); }
    p.files = files; p.nproc = nproc; p.BT = BT; p.accum = accum; p.steps = steps;
    p.ntok = shard_sizes(files);
    p.plan = loader_plan(p.ntok, nproc, BT, accum, steps);
    for (auto& s : p.plan) p.nshards = std::max(p.nshards, s.shard + 1);
    p.sizes.assign(p.nshards, 0);
    for (auto& s : p.plan) p.sizes[s.shard] = std::max(p.sizes[s.shard], s.offset + BT + 1);
    p.sizes[0] = std::max(p.sizes[0], p.ntok[0]);
    p.L = layout(p.sizes, p.starts);
    cudaCheck(cudaMallocHost(&p.host, p.L * sizeof(u16)));
}

// The training rows of rank `rank`'s micro-batches, and with unigram (a device [VOCAB] float array) the n-gram's
// unigram init from shard 0's token frequencies (hint_runtime: log((count + 0.5) / total)).
static void train_rows_build(Rows& out, TrainBuildTimes& tm, TrainPlan& tp, int rank, float* unigram, cudaStream_t st) {
    const double t0 = now_s();
    const std::vector<std::string>& files = tp.files;
    const std::vector<i64>& ntok = tp.ntok;
    const std::vector<Span>& plan = tp.plan;
    const std::vector<i64>& sizes = tp.sizes;
    const std::vector<i64>& starts = tp.starts;
    const int nshards = tp.nshards, accum = tp.accum, steps = tp.steps;
    const i64 BT = tp.BT, L = tp.L;
    u16* host = tp.host;
    tm.stream = L;
    {
        std::vector<std::thread> readers;
        const int R = 8;
        for (int t = 0; t < R; t++) readers.emplace_back([&, t]() {
            for (int j = 0; j < nshards; j++) {
                const i64 a = sizes[j] * t / R, b = sizes[j] * (t + 1) / R;
                read_tokens(files[j], a, b - a, host + starts[j] + a);
            }
        });
        for (auto& r : readers) r.join();
    }
    u16* tok = dalloc<u16>(L);
    cudaCheck(cudaMemsetAsync(tok, 0xFF, L * sizeof(u16), st));
    for (int j = 0; j < nshards; j++)
        cudaCheck(cudaMemcpyAsync(tok + starts[j], host + starts[j], sizes[j] * sizeof(u16), cudaMemcpyHostToDevice, st));
    cudaCheck(cudaStreamSynchronize(st));
    tm.load = now_s() - t0;
    if (unigram) {
        u32* counts = dalloc<u32>(VOCAB);
        cudaCheck(cudaMemsetAsync(counts, 0, VOCAB * 4, st));
        tr_unigram_k<<<1024, 256, 0, st>>>(tok + starts[0], ntok[0], counts);
        std::vector<u32> hc(VOCAB);
        cudaCheck(cudaMemcpy(hc.data(), counts, VOCAB * 4, cudaMemcpyDeviceToHost));
        double total = 0;
        for (int v = 0; v < VOCAB; v++) total += hc[v] + 0.5;
        std::vector<float> hu(VOCAB);
        for (int v = 0; v < VOCAB; v++) hu[v] = logf(fmaxf((float)((hc[v] + 0.5) / total), 1e-12f));
        cudaCheck(cudaMemcpy(unigram, hu.data(), VOCAB * 4, cudaMemcpyHostToDevice));
        dfree(counts);
    }
    // when each position is trained (tau: first as an input or a target; tau_in: as an input), and this rank's rows
    u32* tau = dalloc<u32>(L);
    u32* tau_in = dalloc<u32>(L);
    int* row_of = dalloc<int>(L);
    cudaCheck(cudaMemsetAsync(tau, 0xFF, L * 4, st));
    cudaCheck(cudaMemsetAsync(tau_in, 0xFF, L * 4, st));
    cudaCheck(cudaMemsetAsync(row_of, 0xFF, L * 4, st));
    {
        std::vector<i64> lo; std::vector<int> step, row0;
        for (auto& s : plan) {
            lo.push_back(starts[s.shard] + s.offset);
            step.push_back(s.step);
            row0.push_back(s.rank == rank ? (int)(((i64)s.step * accum + s.micro) * BT) : -1);
        }
        i64* dlo = dalloc<i64>(lo.size()); int* dstep = dalloc<int>(lo.size()); int* drow = dalloc<int>(lo.size());
        cudaCheck(cudaMemcpy(dlo, lo.data(), lo.size() * 8, cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(dstep, step.data(), lo.size() * 4, cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(drow, row0.data(), lo.size() * 4, cudaMemcpyHostToDevice));
        tr_spans_k<<<(int)lo.size(), 256, 0, st>>>(dlo, dstep, drow, BT, tau, tau_in, row_of);
        cudaCheck(cudaStreamSynchronize(st));
        dfree(dlo); dfree(dstep); dfree(drow);
    }
    // a query's k-context is valid iff it lies in its segment (from the last BOS at or before it, or its part's start)
    int* avail = dalloc<int>(L);
    {
        i64* v = dalloc<i64>(L);
        i64* dstarts = dalloc<i64>(nshards);
        cudaCheck(cudaMemcpy(dstarts, starts.data(), nshards * 8, cudaMemcpyHostToDevice));
        tr_segv_k<<<grid_for(L, 256), 256, 0, st>>>(tok, L, v);
        tr_partstart_k<<<1, 256, 0, st>>>(dstarts, nshards, v);
        cub_run([&](void* t, size_t& b) { return cub::DeviceScan::InclusiveScan(t, b, v, v, cub::Max(), L, st); });
        tr_avail_k<<<grid_for(L, 256), 256, 0, st>>>(v, row_of, L, avail);
        dfree(v); dfree(dstarts);
    }
    const i64 R = (i64)steps * accum * BT;
    rows_alloc(out, R, st);
    signed char* hint_level = dalloc<signed char>(R);
    int* hint_ctx = dalloc<int>(R);
    cudaCheck(cudaMemsetAsync(hint_level, 0xFF, R, st));
    const i64 nwords = L / 32;
    u32* bm[2] = {dalloc<u32>(nwords), dalloc<u32>(nwords)};
    u64* emit = dalloc<u64>(L);
    unsigned long long* emitted = dalloc<unsigned long long>(1);
    int sms = 0, dev = 0;
    cudaCheck(cudaGetDevice(&dev));
    cudaCheck(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    const int grid = sms * 32;
    LevelHits hits[NHINT];
    cudaCheck(cudaStreamSynchronize(st));
    tm.prep = now_s() - t0 - tm.load;
    // ---- per level (ascending): the queries' index, the scan, the hits sorted, the count rows
    bool none = false;
    for (int li = 0; li < NLEVELS && !none; li++) {
        const int k = LEVELS[li];
        CtxIndex ix;
        ctx_index_build(ix, tok, avail, L, k, false, st);
        cudaCheck(cudaMemsetAsync(emitted, 0, 8, st));
        const u32* bm_in = li == 0 ? nullptr : bm[(li - 1) % 2];
        u32* bm_out = bm[li % 2];
        const bool sparse = li > 0 && k > 5;
        if (k <= 4) {
            if (sparse) { HS_LEVEL_SWITCH(k, (tr_scan_sparse_k<K, false><<<grid, 256, 0, st>>>(tok, tau, L, bm_in, bm_out, ix.table, ix.cap, ix.rep, steps, emit, emitted))); }
            else { HS_LEVEL_SWITCH(k, (tr_scan_k<K, false><<<grid, 256, 0, st>>>(tok, tau, L, bm_in, bm_out, ix.table, ix.cap, ix.rep, steps, emit, emitted))); }
        } else {
            if (sparse) { HS_LEVEL_SWITCH(k, (tr_scan_sparse_k<K, true><<<grid, 256, 0, st>>>(tok, tau, L, bm_in, bm_out, ix.table, ix.cap, ix.rep, steps, emit, emitted))); }
            else { HS_LEVEL_SWITCH(k, (tr_scan_k<K, true><<<grid, 256, 0, st>>>(tok, tau, L, bm_in, bm_out, ix.table, ix.cap, ix.rep, steps, emit, emitted))); }
        }
        cudaCheck(cudaGetLastError());
        const i64 n = (i64)d2h1(emitted);
        tm.hits += n;
        if (n == 0) { ix.release(); none = true; break; }  // no occurrence of any query context: none deeper either
        const int ctx_bits = std::max(1, bit_length((u64)ix.ids - 1));
        u64* a1 = dalloc<u64>(n);
        cub_run([&](void* t, size_t& b) { return cub::DeviceRadixSort::SortKeys(t, b, emit, a1, n, 0, 16 + ALPHA_BITS + ctx_bits, st); });
        u32* start1 = dalloc<u32>(ix.ids); u32* end1 = dalloc<u32>(ix.ids);
        cudaCheck(cudaMemsetAsync(start1, 0, ix.ids * 4, st));
        cudaCheck(cudaMemsetAsync(end1, 0, ix.ids * 4, st));
        seg_ranges_k<<<grid_for(n, 256), 256, 0, st>>>(a1, n, 16 + ALPHA_BITS, start1, end1);
        // (id, alpha) of every hit sorted (tot), and of every (id, next token) run's first hit (dist; r1: the runs)
        u64* tmp = dalloc<u64>(n);
        unsigned char* first = dalloc<unsigned char>(n);
        tr_time_keys_k<<<grid_for(n, 256), 256, 0, st>>>(a1, n, tmp, first);
        u64* a3u = dalloc<u64>(n);
        u32* r1 = dalloc<u32>(n);
        i64* nsel = dalloc<i64>(1);
        cub_run([&](void* t, size_t& b) { return cub::DeviceSelect::Flagged(t, b, tmp, first, a3u, nsel, n, st); });
        const i64 runs = d2h1(nsel);
        cub_run([&](void* t, size_t& b) { return cub::DeviceSelect::Flagged(t, b, cub::CountingInputIterator<u32>(0), first, r1, nsel, n, st); });
        dfree(first);
        u64* a2 = dalloc<u64>(n);
        cub_run([&](void* t, size_t& b) { return cub::DeviceRadixSort::SortKeys(t, b, tmp, a2, n, 0, ALPHA_BITS + ctx_bits, st); });
        u64* a3 = tmp;  // reused
        cub_run([&](void* t, size_t& b) { return cub::DeviceRadixSort::SortKeys(t, b, a3u, a3, runs, 0, ALPHA_BITS + ctx_bits, st); });
        dfree(a3u);
        u32* start3 = dalloc<u32>(ix.ids); u32* end3 = dalloc<u32>(ix.ids);
        cudaCheck(cudaMemsetAsync(start3, 0, ix.ids * 4, st));
        cudaCheck(cudaMemsetAsync(end3, 0, ix.ids * 4, st));
        seg_ranges_k<<<grid_for(runs, 256), 256, 0, st>>>(a3, runs, ALPHA_BITS, start3, end3);
        const int hl = hint_level_of(li);
        tr_query_k<<<grid_for(L, 256), 256, 0, st>>>(L, ix.ctx_of, row_of, tau_in, tok, a1, a2, a3, start1, end1, start3,
                                                     end3, li, hl, out.tot, out.dist, out.cnt, hint_level, hint_ctx);
        cudaCheck(cudaGetLastError());
        dfree(a2); dfree(a3); dfree(start1); dfree(nsel);
        if (hl >= 0) {
            LevelHits& h = hits[hl];
            h.n = n; h.ids = ix.ids; h.runs = runs; h.a1 = a1; h.r1 = r1; h.end1 = end1; h.start3 = start3; h.end3 = end3;
        } else {
            dfree(a1); dfree(r1); dfree(end1); dfree(start3); dfree(end3);
        }
        ix.release();
    }
    cudaCheck(cudaStreamSynchronize(st));
    dfree(emit); dfree(emitted); dfree(bm[0]); dfree(bm[1]); dfree(avail); dfree(tau); dfree(tau_in); dfree(row_of);
    dfree(tok);
    tm.levels = now_s() - t0 - tm.load - tm.prep;
    // ---- hint rows: per hint level, the distinct (context, step) snapshots of the rows whose deepest level it is
    u64* keys = dalloc<u64>(R);
    unsigned char* flag = dalloc<unsigned char>(R);
    u64* kq = dalloc<u64>(R);
    u32* vq = dalloc<u32>(R);
    u64* sk = dalloc<u64>(R);
    u32* sv = dalloc<u32>(R);
    i64* nsel = dalloc<i64>(1);
    for (int hl = 0; hl < NHINT; hl++) {
        LevelHits& h = hits[hl];
        if (h.n == 0) continue;
        int li = 0;
        while (LEVELS[li] != HINT_LEVELS[hl]) li++;
        tr_snap_keys_k<<<grid_for(R, 256), 256, 0, st>>>(R, hint_level, hint_ctx, hl, (i64)accum * BT, keys, flag);
        cub_run([&](void* t, size_t& b) { return cub::DeviceSelect::Flagged(t, b, keys, flag, kq, nsel, R, st); });
        const i64 m = d2h1(nsel);
        cub_run([&](void* t, size_t& b) { return cub::DeviceSelect::Flagged(t, b, cub::CountingInputIterator<u32>(0), flag, vq, nsel, R, st); });
        if (m > 0) {
            const int ctx_bits = std::max(1, bit_length((u64)h.ids - 1));
            cub_run([&](void* t, size_t& b) { return cub::DeviceRadixSort::SortPairs(t, b, kq, sk, vq, sv, m, 0, ALPHA_BITS + ctx_bits, st); });
            i64* csum = dalloc<i64>(m);
            ix_new_k<<<grid_for(m, 256), 256, 0, st>>>(sk, m, csum);
            cub_run([&](void* t, size_t& b) { return cub::DeviceScan::InclusiveSum(t, b, csum, csum, m, st); });
            const i64 U = d2h1(csum + m - 1);
            u64* snaps = dalloc<u64>(U);
            compact_new_k<<<grid_for(m, 256), 256, 0, st>>>(sk, csum, m, snaps);
            u16* top_tok = dalloc<u16>(U * ENTRIES);
            u32* top_cnt = dalloc<u32>(U * ENTRIES);
            cudaCheck(cudaMemsetAsync(top_tok, 0, U * ENTRIES * 2, st));
            cudaCheck(cudaMemsetAsync(top_cnt, 0, U * ENTRIES * 4, st));
            tr_top_k<<<grid_for(U * 32, 256), 256, 0, st>>>(snaps, U, h.a1, h.r1, h.start3, h.end3, h.end1, top_tok, top_cnt);
            tr_hint_rows_k<<<grid_for(m, 256), 256, 0, st>>>(m, sv, csum, top_tok, top_cnt, hl, li, out);
            cudaCheck(cudaGetLastError());
            cudaCheck(cudaStreamSynchronize(st));
            dfree(csum); dfree(snaps); dfree(top_tok); dfree(top_cnt);
        }
        h.release();
    }
    cudaCheck(cudaStreamSynchronize(st));
    dfree(keys); dfree(flag); dfree(kq); dfree(vq); dfree(sk); dfree(sv); dfree(nsel); dfree(hint_level); dfree(hint_ctx);
    tm.total = now_s() - t0;
    tm.tops = tm.total - tm.load - tm.prep - tm.levels;
}

// ----------------------------------------------------------------------------
// ValBuild: gpu_val_stats.py's InProcess (--fused --direct). Each rank holds 1/world of the training shards in GPU
// memory and counts, per level, the (validation context, next token) pairs of its shards in a pair table; the ranks
// merge the counts (all-to-all to each context's owner rank, sort, sum); the owner computes its contexts' tot, dist,
// ranks and top-8, and the count rows of the validation positions with its contexts; then the hint rows, and every
// output summed over the ranks (each rank's part is disjoint).

// A pair table slot: the key (context id << 16 | next token, 40 bits) in the low bits, its count in the top 24 bits
// (EMPTY: never a key). A count that wraps past 2^24 - 1 adds one to the slot's byte in `wraps`.
constexpr int KEYBITS = 40;
constexpr u64 KEYMASK = (1ull << KEYBITS) - 1;
constexpr u64 ONE = 1ull << KEYBITS;

__device__ __forceinline__ void pair_add(u64* pk, u32* wraps, u64 cap, u64 key, u32* flags) {
    u64 s = __umul64hi(mix64(key), cap);
    for (int probe = 0; probe < MAXPROBE; ++probe) {
        u64 cur = pk[s];
        if (cur == EMPTY) {
            u64 prev = atomicCAS((unsigned long long*)&pk[s], EMPTY, key | ONE);
            if (prev == EMPTY) return;
            cur = prev;
        }
        if ((cur & KEYMASK) == key) {
            u64 old = atomicAdd((unsigned long long*)&pk[s], ONE);
            if ((old >> KEYBITS) == (1ull << (64 - KEYBITS)) - 1) atomicAdd(&wraps[s >> 2], 1u << (8 * (s & 3)));
            return;
        }
        s = s + 1 == cap ? 0 : s + 1;
    }
    atomicOr(&flags[0], 1u);
}

__device__ __forceinline__ u32 slot_count(const u32* wraps, u64 s, u64 v) {
    return (u32)(v >> KEYBITS) + (((wraps[s >> 2] >> (8 * (s & 3))) & 0xFFu) << (64 - KEYBITS));
}

// position p's pair key (context id << 16 | next token), or EMPTY
template <int K, bool HASHED>
__device__ __forceinline__ u64 val_probe(const u16* __restrict__ tok, i64 p, const u64* __restrict__ vt, u64 vcap,
                                         const u32* __restrict__ vrep, const u16* __restrict__ x) {
    const u32 nxt = tok[p];
    if (nxt == SENT) return EMPTY;
    const i64 s = ix_lookup<K, HASHED>(tok, p, vt, vcap, vrep, x);
    return s < 0 ? EMPTY : (((u64)s << 16) | nxt);
}

// dense levels: a lane per position, a warp per bitmap word (bm_in null: every position)
template <int K, bool HASHED>
__global__ void __launch_bounds__(256) val_scan_k(const u16* __restrict__ tok, i64 L, const u32* __restrict__ bm_in,
                                                  u32* __restrict__ bm_out, const u64* __restrict__ vt, u64 vcap,
                                                  const u32* __restrict__ vrep, const u16* __restrict__ x, u64* pk,
                                                  u32* pc, u64 cap, u32* flags) {
    const int lane = threadIdx.x & 31;
    const i64 stride = (i64)gridDim.x * blockDim.x;
    for (i64 base = blockIdx.x * (i64)blockDim.x + (threadIdx.x & ~31); base < L; base += stride) {
        const i64 p = base + lane;
        const u32 word = bm_in ? bm_in[base >> 5] : 0xFFFFFFFFu;
        u64 key = EMPTY;
        if (((word >> lane) & 1u) && p < L) key = val_probe<K, HASHED>(tok, p, vt, vcap, vrep, x);
        const bool hit = key != EMPTY;
        const u32 b = __ballot_sync(0xFFFFFFFFu, hit);
        if (lane == 0) bm_out[base >> 5] = b;
        if (hit) pair_add(pk, pc, cap, key, flags);
    }
}

// sparse levels: a lane per bitmap word, its set bits in turn
template <int K, bool HASHED>
__global__ void __launch_bounds__(256) val_scan_sparse_k(const u16* __restrict__ tok, i64 L, const u32* __restrict__ bm_in,
                                                         u32* __restrict__ bm_out, const u64* __restrict__ vt, u64 vcap,
                                                         const u32* __restrict__ vrep, const u16* __restrict__ x,
                                                         u64* pk, u32* pc, u64 cap, u32* flags) {
    const i64 nwords = (L + 31) >> 5;
    for (i64 w = blockIdx.x * (i64)blockDim.x + threadIdx.x; w < nwords; w += (i64)gridDim.x * blockDim.x) {
        u32 word = bm_in[w], out = 0;
        while (word) {
            const int b = __ffs(word) - 1;
            word &= word - 1;
            const i64 p = (w << 5) + b;
            if (p >= L) continue;
            const u64 key = val_probe<K, HASHED>(tok, p, vt, vcap, vrep, x);
            if (key != EMPTY) { out |= 1u << b; pair_add(pk, pc, cap, key, flags); }
        }
        bm_out[w] = out;
    }
}

__device__ __forceinline__ int owner_of(i64 c, const i64* b, int W) {
    int o = 0;
    while (o + 1 < W && c >= b[o + 1]) ++o;
    return o;
}

__global__ void part_count_k(const u64* pk, u64 cap, const i64* bounds, int W, unsigned long long* counts) {
    __shared__ unsigned long long h[MAXW];
    __shared__ i64 b[MAXW + 1];
    if (threadIdx.x < W) h[threadIdx.x] = 0;
    if (threadIdx.x <= W) b[threadIdx.x] = bounds[threadIdx.x];
    __syncthreads();
    for (u64 s = blockIdx.x * (u64)blockDim.x + threadIdx.x; s < cap; s += (u64)gridDim.x * blockDim.x) {
        u64 key = pk[s];
        if (key != EMPTY) atomicAdd(&h[owner_of((i64)((key & KEYMASK) >> 16), b, W)], 1ull);
    }
    __syncthreads();
    if (threadIdx.x < W && h[threadIdx.x]) atomicAdd(&counts[threadIdx.x], h[threadIdx.x]);
}

#define PS_ITEMS 8
__global__ void part_scatter_k(const u64* pk, const u32* pc, u64 cap, const i64* bounds, int W,
                               unsigned long long* cursor, u64* ok, u32* oc) {
    __shared__ u32 cnt[MAXW];
    __shared__ unsigned long long base[MAXW];
    __shared__ i64 b[MAXW + 1];
    if (threadIdx.x <= W) b[threadIdx.x] = bounds[threadIdx.x];
    const u64 tile_size = (u64)blockDim.x * PS_ITEMS;
    for (u64 tile = blockIdx.x * tile_size; tile < cap; tile += (u64)gridDim.x * tile_size) {
        if (threadIdx.x < W) cnt[threadIdx.x] = 0;
        __syncthreads();
        u64 key[PS_ITEMS];
        int o[PS_ITEMS];
        u32 local[PS_ITEMS];
#pragma unroll
        for (int j = 0; j < PS_ITEMS; ++j) {
            u64 s = tile + (u64)j * blockDim.x + threadIdx.x;
            key[j] = s < cap ? pk[s] : EMPTY;
            o[j] = -1;
            if (key[j] != EMPTY) {
                o[j] = owner_of((i64)((key[j] & KEYMASK) >> 16), b, W);
                local[j] = atomicAdd(&cnt[o[j]], 1u);
            }
        }
        __syncthreads();
        if (threadIdx.x < W && cnt[threadIdx.x]) base[threadIdx.x] = atomicAdd(&cursor[threadIdx.x], (unsigned long long)cnt[threadIdx.x]);
        __syncthreads();
#pragma unroll
        for (int j = 0; j < PS_ITEMS; ++j) {
            if (o[j] >= 0) {
                u64 s = tile + (u64)j * blockDim.x + threadIdx.x;
                u64 d = base[o[j]] + local[j];
                ok[d] = key[j] & KEYMASK;
                oc[d] = slot_count(pc, s, key[j]);
            }
        }
        __syncthreads();
    }
}
#undef PS_ITEMS

__global__ void sub_lo_k(u64* k, i64 n, u64 lo16) {
    i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x;
    if (i < n) k[i] -= lo16;
}

__global__ void seg_bounds_k(const u64* uk, i64 U, i64* start, i64* end) {
    i64 j = blockIdx.x * (i64)blockDim.x + threadIdx.x;
    if (j >= U) return;
    u64 c = uk[j] >> 16;
    if (j == 0 || (uk[j - 1] >> 16) != c) start[c] = j;
    if (j == U - 1 || (uk[j + 1] >> 16) != c) end[c] = j + 1;
}

__global__ void ctx_tot_k(const i64* start, const i64* end, const u32* agg, i64 nctx, u64* tot) {
    i64 c = blockIdx.x * (i64)blockDim.x + threadIdx.x;
    if (c >= nctx) return;
    u64 s = 0;
    for (i64 j = start[c]; j < end[c]; ++j) s += agg[j];
    tot[c] = s;
}

// the ranking key: context, then count descending; a stable sort keeps the token order (ascending) of ties
__global__ void rank_keys_k(const u64* uk, const u32* agg, i64 U, u64* key2, u32* val2) {
    i64 j = blockIdx.x * (i64)blockDim.x + threadIdx.x;
    if (j >= U) return;
    key2[j] = ((uk[j] >> 16) << 32) | (u64)(0xFFFFFFFFu - agg[j]);
    val2[j] = (u32)j;
}

__global__ void rank_scatter_k(const u32* sv, i64 U, const u64* uk, const u32* agg, const i64* start, u32* rank_of,
                               u16* top_tok, u32* top_cnt, int top) {
    i64 q = blockIdx.x * (i64)blockDim.x + threadIdx.x;
    if (q >= U) return;
    u32 j = sv[q];
    u64 c = uk[j] >> 16;
    i64 r = q - start[c];
    rank_of[j] = (u32)r;
    if (top_tok && r < top) {
        top_tok[c * top + r] = (u16)(uk[j] & 0xFFFF);
        top_cnt[c * top + r] = agg[j];
    }
}

// per validation position whose context id is in [lo, hi): tot, dist and cnt (the target's count if among the
// context's cnt_top most frequent next tokens), and bit `li` of mask where the context occurred
__global__ void pos_outputs_k(const int* ctx_of, i64 N, i64 lo, i64 hi, const u16* tgt, const u64* uk, const u32* agg,
                              const u32* rank_of, const i64* start, const i64* end, const u64* tot, int li, int cnt_top,
                              int* out_tot, int* out_dist, int* out_cnt, int* mask) {
    i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x;
    if (i >= N) return;
    i64 c = ctx_of[i];
    if (c < lo || c >= hi) return;
    c -= lo;
    i64 a = start[c], b = end[c];
    if (b <= a) return;
    out_tot[i] = (int)(u32)tot[c];
    out_dist[i] = (int)min(b - a, (i64)65535);
    u64 want = ((u64)c << 16) | tgt[i];
    while (a < b) {
        i64 m = (a + b) >> 1;
        if (uk[m] < want) a = m + 1; else b = m;
    }
    if (a < end[c] && uk[a] == want && rank_of[a] < (u32)cnt_top) out_cnt[i] = (int)agg[a];
    mask[i] |= 1 << li;
}

// the hint rows: per position its deepest hint level whose context occurred (mask), if this rank owns that context
struct HintLevelDesc { int li, bucket; const int* ctx_of; i64 lo, hi; const u16* top_tok; const u32* top_cnt; const u64* tot; };
struct HintDescs { HintLevelDesc d[NHINT]; };

__global__ void val_hint_rows_k(i64 N, const int* mask, HintDescs ds, int* rows) {
    i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x;
    if (i >= N) return;
    const int m = mask[i];
    for (int v = NHINT - 1; v >= 0; --v) {
        const HintLevelDesc& d = ds.d[v];
        if (!((m >> d.li) & 1)) continue;
        i64 c = d.ctx_of[i];
        if (c < d.lo || c >= d.hi) return;  // another rank's context
        c -= d.lo;
        int16_t cell; uint16_t tt[ENTRIES], ww[ENTRIES];
        hint_row(d.bucket, d.tot[c], d.top_tok + c * ENTRIES, d.top_cnt + c * ENTRIES, &cell, tt, ww);
        int* row = rows + i * 9;  // packed into int32s, summed over the ranks
        row[0] = cell;
        for (int r = 0; r < ENTRIES / 2; r++) {
            row[1 + r] = (int)((u32)tt[2 * r] | ((u32)tt[2 * r + 1] << 16));
            row[5 + r] = (int)((u32)ww[2 * r] | ((u32)ww[2 * r + 1] << 16));
        }
        return;
    }
}

// the count rows [3][NLEVELS][N] and the packed hint rows [N][9] (int32) into Rows' layout
__global__ void val_unpack_k(i64 N, const int* cnts, const int* rows, Rows out) {
    i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x;
    if (i >= N) return;
    for (int li = 0; li < NLEVELS; li++) {
        out.tot[i * NLEVELS + li] = (u32)cnts[(0 * NLEVELS + li) * N + i];
        out.dist[i * NLEVELS + li] = (u16)cnts[(1 * NLEVELS + li) * N + i];
        out.cnt[i * NLEVELS + li] = (u32)cnts[(2 * NLEVELS + li) * N + i];
    }
    const int* row = rows + i * 9;
    out.cell[i] = (int16_t)row[0];
    for (int r = 0; r < ENTRIES / 2; r++) {
        out.tokens[i * ENTRIES + 2 * r] = (u16)((u32)row[1 + r] & 0xFFFF);
        out.tokens[i * ENTRIES + 2 * r + 1] = (u16)((u32)row[1 + r] >> 16);
        out.weights[i * ENTRIES + 2 * r] = (u16)((u32)row[5 + r] & 0xFFFF);
        out.weights[i * ENTRIES + 2 * r + 1] = (u16)((u32)row[5 + r] >> 16);
    }
}

__global__ void val_avail_k(const u16* x, i64 n, i64* v) {
    i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x;
    if (i < n) v[i] = x[i] == BOS ? i : 0;
}

__global__ void val_avail2_k(const i64* seg, i64 n, int* avail) {
    i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x;
    if (i < n) avail[i] = (int)(i - seg[i] + 1);
}

// the distinct (context, next token) pairs of n training tokens per level, to size the pair tables: U(n) = u * (n /
// 4e8) ** beta (gpu_val_stats.PAIRS)
constexpr double PAIRS_U[NLEVELS] = {82.17e6, 62.30e6, 26.39e6, 8.99e6, 3.17e6, 0.550e6, 0.223e6, 0.162e6, 0.121e6, 0.102e6};
constexpr double PAIRS_BETA[NLEVELS] = {0.620, 0.667, 0.694, 0.706, 0.705, 0.671, 0.604, 0.577, 0.572, 0.581};

#ifdef MULTI_GPU
static void all_to_all(const void* send, const i64* send_off, const i64* send_n, void* recv, const i64* recv_off,
                       const i64* recv_n, ncclDataType_t type, size_t elem, int world, ncclComm_t comm, cudaStream_t st) {
    ncclCheck(ncclGroupStart());
    for (int r = 0; r < world; r++) {
        ncclCheck(ncclSend((const char*)send + send_off[r] * elem, send_n[r], type, r, comm, st));
        ncclCheck(ncclRecv((char*)recv + recv_off[r] * elem, recv_n[r], type, r, comm, st));
    }
    ncclCheck(ncclGroupEnd());
}
#endif

struct ValBuild {
    int rank = 0, world = 1;
    std::string val_path;
    std::vector<std::string> mine;
    std::vector<i64> sizes, starts;
    i64 L = 0, n_tokens = 0;
    u16* tok = nullptr;
    // the loads: `threads` threads, each its shards in chunks through two pinned buffers, on a stream of its own
    int threads = 4;
    i64 chunk = 4 << 20;  // tokens
    std::vector<u16*> pinned;
    std::vector<std::thread> workers;
    double t_start = 0;
    std::atomic<double> load_end{0.0};
    // the pair tables (allocated before the clock)
    u64* pk[NLEVELS] = {}; u32* pc[NLEVELS] = {}; u64 cap[NLEVELS] = {};
    u32* flags = nullptr;
    // timings (s), and each level's scan and merge (ms)
    double t_index = 0, t_scan = 0, t_merge = 0, t_rows = 0, t_total = 0, t_wait = 0;
    cudaEvent_t level_ev[NLEVELS + 1];
    float level_ms[NLEVELS] = {};
    double merge_ms[NLEVELS] = {};

    // before the clock: this rank's shards, their GPU buffer, the pinned buffers, the pair tables, NCCL's peers
    void setup(const std::vector<std::string>& shards, const std::string& val, int rank_, int world_, int threads_, i64 chunk_) {
        rank = rank_; world = world_; val_path = val; threads = threads_; chunk = chunk_;
        const size_t n = shards.size();
        for (size_t i = rank * n / world; i < (rank + 1) * n / world; i++) mine.push_back(shards[i]);
        sizes = shard_sizes(mine);
        L = layout(sizes, starts);
        for (i64 s : sizes) n_tokens += s;
        tok = dalloc<u16>(L);
        cudaCheck(cudaMemset(tok, 0xFF, L * sizeof(u16)));
        for (int t = 0; t < 2 * threads; t++) { u16* p; cudaCheck(cudaMallocHost(&p, chunk * sizeof(u16))); pinned.push_back(p); }
        for (int li = 0; li < NLEVELS; li++) {
            double est = std::min(1.15 * PAIRS_U[li] * pow(n_tokens / 4e8, PAIRS_BETA[li]), (double)n_tokens);
            cap[li] = std::max((u64)1 << 20, (u64)(est / 0.7));
            pk[li] = dalloc<u64>(cap[li]);
            pc[li] = dalloc<u32>((cap[li] + 3) / 4);
        }
        flags = dalloc<u32>(2);
        for (auto& e : level_ev) cudaCheck(cudaEventCreate(&e));
        int sms = 0, dev = 0;
        cudaCheck(cudaGetDevice(&dev));
        cudaCheck(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
        grid = sms * 32;
#ifdef MULTI_GPU
        if (world > 1) {  // the all-to-all's peer connections (seconds on 8 GPUs the first time), small and big messages
            const i64 big = 4 << 20;
            i64* a = dalloc<i64>(world * big);
            i64* b = dalloc<i64>(world * big);
            std::vector<i64> off(world), cnt(world, 256), cnt_big(world, big);
            for (int r = 0; r < world; r++) off[r] = r * big;
            all_to_all(a, off.data(), cnt.data(), b, off.data(), cnt.data(), ncclInt64, 8, world, multi_gpu_config.nccl_comm, 0);
            for (int rep = 0; rep < 2; rep++)
                all_to_all(a, off.data(), cnt_big.data(), b, off.data(), cnt_big.data(), ncclInt64, 8, world, multi_gpu_config.nccl_comm, 0);
            ncclCheck(ncclAllReduce(a, a, 256, ncclInt64, ncclSum, multi_gpu_config.nccl_comm, 0));
            cudaCheck(cudaDeviceSynchronize());
            dfree(a); dfree(b);
        }
#endif
    }

    // at the clock's start: the shards' loads, in the background beside the training
    void start() {
        t_start = now_s();
        int dev = 0;
        cudaCheck(cudaGetDevice(&dev));
        for (int t = 0; t < threads; t++) workers.emplace_back([this, t, dev]() {
            cudaCheck(cudaSetDevice(dev));
            cudaStream_t s;
            cudaCheck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
            cudaEvent_t used[2];
            bool pending[2] = {false, false};
            for (int b = 0; b < 2; b++) cudaCheck(cudaEventCreateWithFlags(&used[b], cudaEventDisableTiming));
            int j = 0;
            for (size_t i = t; i < mine.size(); i += threads) {
                for (i64 a = 0; a < sizes[i]; a += chunk, j++) {
                    const i64 c = std::min(chunk, sizes[i] - a);
                    const int b = j % 2;
                    if (pending[b]) cudaCheck(cudaEventSynchronize(used[b]));
                    read_tokens(mine[i], a, c, pinned[2 * t + b]);
                    cudaCheck(cudaMemcpyAsync(tok + starts[i] + a, pinned[2 * t + b], c * sizeof(u16), cudaMemcpyHostToDevice, s));
                    cudaCheck(cudaEventRecord(used[b], s));
                    pending[b] = true;
                }
            }
            cudaCheck(cudaStreamSynchronize(s));
            double e = now_s(), cur = load_end.load();
            while (e > cur && !load_end.compare_exchange_weak(cur, e)) {}
            for (int b = 0; b < 2; b++) cudaCheck(cudaEventDestroy(used[b]));
            cudaCheck(cudaStreamDestroy(s));
        });
    }

    // the validation tokens and their index, every level (on the clock)
    u16* x = nullptr;
    CtxIndex index[NLEVELS];
    void build_index(cudaStream_t st) {
        const double t0 = now_s();
        const i64 N = VAL_TOKENS;
        std::vector<u16> val(N + 1);
        read_tokens(val_path, 0, N + 1, val.data());
        x = dalloc<u16>(N + 1);
        cudaCheck(cudaMemcpy(x, val.data(), (N + 1) * sizeof(u16), cudaMemcpyHostToDevice));
        int* avail = dalloc<int>(N);
        i64* v = dalloc<i64>(N);
        val_avail_k<<<grid_for(N, 256), 256, 0, st>>>(x, N, v);
        cub_run([&](void* t, size_t& b) { return cub::DeviceScan::InclusiveScan(t, b, v, v, cub::Max(), N, st); });
        val_avail2_k<<<grid_for(N, 256), 256, 0, st>>>(v, N, avail);
        dfree(v);
        for (int li = 0; li < NLEVELS; li++) ctx_index_build(index[li], x, avail, N, LEVELS[li], true, st);
        dfree(avail);
        for (int li = 0; li < NLEVELS; li++) {
            cudaCheck(cudaMemsetAsync(pk[li], 0xFF, cap[li] * sizeof(u64), st));
            cudaCheck(cudaMemsetAsync(pc[li], 0, (cap[li] + 3) / 4 * sizeof(u32), st));
        }
        cudaCheck(cudaMemsetAsync(flags, 0, 8, st));
        bms[0] = dalloc<u32>(L / 32);
        bms[1] = dalloc<u32>(L / 32);
        cudaCheck(cudaMemsetAsync(bms[0], 0, L / 32 * 4, st));
        cudaCheck(cudaStreamSynchronize(st));
        t_index = now_s() - t0;
    }

    // level li's scan of tok[a, a + n) (a and n multiples of 32), after level li - 1's of the same positions
    u32* bms[2] = {nullptr, nullptr};
    int grid = 0;
    void scan(int li, i64 a, i64 n, cudaStream_t st) {
        const int k = LEVELS[li];
        CtxIndex& vi = index[li];
        const u32* bm_in = li == 0 ? nullptr : bms[(li - 1) % 2] + a / 32;
        u32* bm_out = bms[li % 2] + a / 32;
        const u16* t = tok + a;
        const int g = (int)std::min((i64)grid, std::max((i64)1, n / 256));
        if (li == 0 || k <= 5) {
            if (k <= 4) { HS_LEVEL_SWITCH(k, (val_scan_k<K, false><<<g, 256, 0, st>>>(t, n, bm_in, bm_out, vi.table, vi.cap, vi.rep, x, pk[li], pc[li], cap[li], flags))); }
            else { HS_LEVEL_SWITCH(k, (val_scan_k<K, true><<<g, 256, 0, st>>>(t, n, bm_in, bm_out, vi.table, vi.cap, vi.rep, x, pk[li], pc[li], cap[li], flags))); }
        } else {
            HS_LEVEL_SWITCH(k, (val_scan_sparse_k<K, true><<<g, 256, 0, st>>>(t, n, bm_in, bm_out, vi.table, vi.cap, vi.rep, x, pk[li], pc[li], cap[li], flags)));
        }
        cudaCheck(cudaGetLastError());
    }

    // after the last step, every rank: the index, the scans (each level over all the shards), the merges and the rows,
    // into `out` (VAL_TOKENS rows, every rank all of them)
    void finish(Rows& out, cudaStream_t st) {
        const double t0 = now_s();
        const i64 N = VAL_TOKENS;
        for (auto& th : workers) th.join();  // (each thread's copies are done when it ends)
        workers.clear();
        t_wait = now_s() - t0;
        build_index(st);
        for (int li = 0; li < NLEVELS; li++) {
            cudaCheck(cudaEventRecord(level_ev[li], st));
            scan(li, 0, L, st);
        }
        cudaCheck(cudaEventRecord(level_ev[NLEVELS], st));
        cudaCheck(cudaStreamSynchronize(st));
        for (int li = 0; li < NLEVELS; li++) cudaCheck(cudaEventElapsedTime(&level_ms[li], level_ev[li], level_ev[li + 1]));
        if (d2h1(flags)) { fprintf(stderr, "val build: a pair table overflowed\n"); exit(EXIT_FAILURE); }
        dfree(bms[0]); dfree(bms[1]);
        t_scan = now_s() - t0 - t_wait - t_index;
        const u16* tgt = x + 1;
        // ---- per level: the merge and the owner's outputs
        int* out_cnts = dalloc<int>(3 * NLEVELS * N);
        int* mask = dalloc<int>(N);
        cudaCheck(cudaMemsetAsync(out_cnts, 0, 3 * NLEVELS * N * sizeof(int), st));
        cudaCheck(cudaMemsetAsync(mask, 0, N * sizeof(int), st));
        u16* top_tok[NLEVELS] = {}; u32* top_cnt[NLEVELS] = {}; u64* tots[NLEVELS] = {}; i64 los[NLEVELS], his[NLEVELS];
        unsigned long long* counts = dalloc<unsigned long long>(world);
        unsigned long long* cursor = dalloc<unsigned long long>(world);
        i64* bounds = dalloc<i64>(world + 1);
        for (int li = 0; li < NLEVELS; li++) {
            const int hl = hint_level_of(li);
            CtxIndex& vi = index[li];
            std::vector<i64> bl(world + 1);
            for (int o = 0; o <= world; o++) bl[o] = vi.ids * o / world;
            const i64 lo = bl[rank], hi = bl[rank + 1];
            cudaCheck(cudaMemcpyAsync(bounds, bl.data(), (world + 1) * 8, cudaMemcpyHostToDevice, st));
            cudaCheck(cudaMemsetAsync(counts, 0, world * 8, st));
            part_count_k<<<2048, 256, 0, st>>>(pk[li], cap[li], bounds, world, counts);
            std::vector<i64> send_n(world), send_off(world), recv_n(world), recv_off(world);
            cudaCheck(cudaMemcpy(send_n.data(), counts, world * 8, cudaMemcpyDeviceToHost));
            i64 u_local = 0;
            for (int r = 0; r < world; r++) { send_off[r] = u_local; u_local += send_n[r]; }
            cudaCheck(cudaMemcpyAsync(cursor, send_off.data(), world * 8, cudaMemcpyHostToDevice, st));
            u64* send_k = dalloc<u64>(u_local);
            u32* send_c = dalloc<u32>(u_local);
            part_scatter_k<<<4096, 256, 0, st>>>(pk[li], pc[li], cap[li], bounds, world, cursor, send_k, send_c);
            u64* recv_k; u32* recv_c; i64 n_recv;
#ifdef MULTI_GPU
            if (world > 1) {
                i64* dn = dalloc<i64>(2 * world);
                cudaCheck(cudaMemcpyAsync(dn, send_n.data(), world * 8, cudaMemcpyHostToDevice, st));
                std::vector<i64> one(world, 1), idx(world);
                for (int r = 0; r < world; r++) idx[r] = r;
                all_to_all(dn, idx.data(), one.data(), dn + world, idx.data(), one.data(), ncclInt64, 8, world, multi_gpu_config.nccl_comm, st);
                cudaCheck(cudaMemcpy(recv_n.data(), dn + world, world * 8, cudaMemcpyDeviceToHost));
                dfree(dn);
                n_recv = 0;
                for (int r = 0; r < world; r++) { recv_off[r] = n_recv; n_recv += recv_n[r]; }
                recv_k = dalloc<u64>(n_recv);
                recv_c = dalloc<u32>(n_recv);
                all_to_all(send_k, send_off.data(), send_n.data(), recv_k, recv_off.data(), recv_n.data(), ncclUint64, 8, world, multi_gpu_config.nccl_comm, st);
                all_to_all(send_c, send_off.data(), send_n.data(), recv_c, recv_off.data(), recv_n.data(), ncclUint32, 4, world, multi_gpu_config.nccl_comm, st);
                cudaCheck(cudaStreamSynchronize(st));
                dfree(send_k); dfree(send_c);
            } else
#endif
            { recv_k = send_k; recv_c = send_c; n_recv = u_local; }
            const i64 n_own = hi - lo;
            const int ctx_bits = std::max(1, bit_length((u64)std::max(n_own, (i64)2) - 1));
            sub_lo_k<<<grid_for(n_recv, 256), 256, 0, st>>>(recv_k, n_recv, (u64)lo << 16);
            u64* sk = dalloc<u64>(n_recv);
            u32* sc = dalloc<u32>(n_recv);
            cub_run([&](void* t, size_t& b) { return cub::DeviceRadixSort::SortPairs(t, b, recv_k, sk, recv_c, sc, n_recv, 0, 16 + ctx_bits, st); });
            dfree(recv_k); dfree(recv_c);
            u64* uk = sk; u32* agg = sc; i64 U = n_recv;
            if (world > 1) {
                uk = dalloc<u64>(n_recv);
                agg = dalloc<u32>(n_recv);
                i64* nruns = dalloc<i64>(1);
                cub_run([&](void* t, size_t& b) { return cub::DeviceReduce::ReduceByKey(t, b, sk, uk, sc, agg, nruns, cub::Sum(), n_recv, st); });
                U = d2h1(nruns);
                dfree(nruns); dfree(sk); dfree(sc);
            }
            i64* start = dalloc<i64>(n_own); i64* end = dalloc<i64>(n_own);
            cudaCheck(cudaMemsetAsync(start, 0, n_own * 8, st));
            cudaCheck(cudaMemsetAsync(end, 0, n_own * 8, st));
            seg_bounds_k<<<grid_for(U, 256), 256, 0, st>>>(uk, U, start, end);
            u64* tot = dalloc<u64>(n_own);
            ctx_tot_k<<<grid_for(n_own, 256), 256, 0, st>>>(start, end, agg, n_own, tot);
            u64* key2 = dalloc<u64>(U); u32* val2 = dalloc<u32>(U); u64* k2s = dalloc<u64>(U); u32* sv = dalloc<u32>(U);
            rank_keys_k<<<grid_for(U, 256), 256, 0, st>>>(uk, agg, U, key2, val2);
            cub_run([&](void* t, size_t& b) { return cub::DeviceRadixSort::SortPairs(t, b, key2, k2s, val2, sv, U, 0, 32 + ctx_bits, st); });
            dfree(key2); dfree(val2); dfree(k2s);
            u32* rank_of = dalloc<u32>(U);
            if (hl >= 0) {
                top_tok[li] = dalloc<u16>(n_own * ENTRIES);
                top_cnt[li] = dalloc<u32>(n_own * ENTRIES);
                cudaCheck(cudaMemsetAsync(top_tok[li], 0, n_own * ENTRIES * 2, st));
                cudaCheck(cudaMemsetAsync(top_cnt[li], 0, n_own * ENTRIES * 4, st));
            }
            rank_scatter_k<<<grid_for(U, 256), 256, 0, st>>>(sv, U, uk, agg, start, rank_of, top_tok[li], top_cnt[li], ENTRIES);
            dfree(sv);
            pos_outputs_k<<<grid_for(N, 256), 256, 0, st>>>(vi.ctx_of, N, lo, hi, tgt, uk, agg, rank_of, start, end, tot, li, TOP,
                                                           out_cnts + (0 * NLEVELS + li) * N, out_cnts + (1 * NLEVELS + li) * N,
                                                           out_cnts + (2 * NLEVELS + li) * N, mask);
            cudaCheck(cudaGetLastError());
            cudaCheck(cudaStreamSynchronize(st));
            dfree(uk); dfree(agg); dfree(rank_of); dfree(start); dfree(end);
            if (hl >= 0) { tots[li] = tot; los[li] = lo; his[li] = hi; } else { dfree(tot); }
            merge_ms[li] = 1000 * (now_s() - t0 - t_wait - t_index - t_scan);  // cumulative
        }
        dfree(counts); dfree(cursor); dfree(bounds);
        t_merge = now_s() - t0 - t_wait - t_index - t_scan;
        // ---- the hint rows, and every output summed over the ranks
        int* rows = dalloc<int>(N * 9);
        cudaCheck(cudaMemsetAsync(rows, 0, N * 9 * sizeof(int), st));
#ifdef MULTI_GPU
        if (world > 1) ncclCheck(ncclAllReduce(mask, mask, N, ncclInt32, ncclSum, multi_gpu_config.nccl_comm, st));
#endif
        HintDescs ds;
        for (int h = 0; h < NHINT; h++) {
            int li = 0;
            while (LEVELS[li] != HINT_LEVELS[h]) li++;
            ds.d[h] = {li, h, index[li].ctx_of, los[li], his[li], top_tok[li], top_cnt[li], tots[li]};
        }
        val_hint_rows_k<<<grid_for(N, 256), 256, 0, st>>>(N, mask, ds, rows);
#ifdef MULTI_GPU
        if (world > 1) {
            ncclCheck(ncclGroupStart());
            ncclCheck(ncclAllReduce(out_cnts, out_cnts, 3 * NLEVELS * N, ncclInt32, ncclSum, multi_gpu_config.nccl_comm, st));
            ncclCheck(ncclAllReduce(rows, rows, N * 9, ncclInt32, ncclSum, multi_gpu_config.nccl_comm, st));
            ncclCheck(ncclGroupEnd());
        }
#endif
        rows_alloc(out, N, st);
        val_unpack_k<<<grid_for(N, 256), 256, 0, st>>>(N, out_cnts, rows, out);
        cudaCheck(cudaStreamSynchronize(st));
        for (int li = 0; li < NLEVELS; li++) { dfree(top_tok[li]); dfree(top_cnt[li]); dfree(tots[li]); index[li].release(); }
        dfree(out_cnts); dfree(mask); dfree(rows); dfree(x);
        t_rows = now_s() - t0 - t_wait - t_index - t_scan - t_merge;
        t_total = now_s() - t0;
    }

    void release() {
        dfree(tok);
        for (int li = 0; li < NLEVELS; li++) { dfree(pk[li]); dfree(pc[li]); }
        dfree(flags);
        for (auto p : pinned) cudaCheck(cudaFreeHost(p));
        pinned.clear();
    }
};

}  // namespace hint_stats

#endif  // LLMC_HINT_STATS_CUH
