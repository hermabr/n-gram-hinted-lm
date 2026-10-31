// GPU kernels of track_1_short/counts/build.py: exact n-gram counts of the query contexts in training shards held in
// GPU memory. Built with nvcc and called through ctypes; every buffer is a torch tensor.
//
// A training occurrence of a k-context is any k tokens within one shard and the token after them (it may cross a BOS);
// a query position's k-context is the k tokens ending at it, valid iff they lie in its segment. On the GPU the shards
// are concatenated with a gap of SENT tokens (0xFFFF, not a GPT-2 token) before each, so a context reaching back past
// its shard's start holds a SENT and matches nothing, and a position whose next token is SENT is skipped.
//
// A k-context's key: for k <= 4 its tokens packed (exact), else a polynomial hash of 64-bit token codes; both through
// splitmix64's bijective finalizer. A training context whose hashed key equals a query context's is compared token by
// token with that context's representative query position, so every count is exact.
#include <cstdint>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned short u16;
typedef long long i64;

#define EMPTY 0xFFFFFFFFFFFFFFFFull
#define SENT 0xFFFFu
#define BASE 0x9E3779B97F4A7C15ull
#define MAXPROBE 4096
#define MAXW 64

__device__ __forceinline__ u64 mix64(u64 h) {
    h ^= h >> 30;
    h *= 0xbf58476d1ce4e5b9ull;
    h ^= h >> 27;
    h *= 0x94d049bb133111ebull;
    return h ^ (h >> 31);
}

__device__ __forceinline__ u64 tcode(u32 t) { return mix64((u64)t * 0x2545f4914f6cdd1dull + 0x632be59bd9b4e019ull) | 1ull; }

// The key of the K tokens ending just before `e` (e[-K..-1]).
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
        for (int j = 1; j <= K; ++j) {
            h += tcode(e[-j]) * bp;
            bp *= BASE;
        }
        k = mix64(h);
    }
    return k == EMPTY ? EMPTY - 1 : k;
}

// build.py's LEVELS
#define LEVEL_SWITCH(k, ...)                                   \
    switch (k) {                                               \
        case 2: { constexpr int K = 2; __VA_ARGS__; } break;   \
        case 3: { constexpr int K = 3; __VA_ARGS__; } break;   \
        case 4: { constexpr int K = 4; __VA_ARGS__; } break;   \
        case 5: { constexpr int K = 5; __VA_ARGS__; } break;   \
        case 6: { constexpr int K = 6; __VA_ARGS__; } break;   \
        case 10: { constexpr int K = 10; __VA_ARGS__; } break; \
        case 20: { constexpr int K = 20; __VA_ARGS__; } break; \
        default: return -1;                                    \
    }

static int grid_for(i64 n, int block) {
    i64 g = (n + block - 1) / block;
    if (g < 1) g = 1;
    if (g > (1 << 30)) g = 1 << 30;
    return (int)g;
}

// ---------------------------------------------------------------------------------------------------------------
// Query side
// ---------------------------------------------------------------------------------------------------------------

template <int K>
__global__ void val_keys_k(const u16* x, const int* avail, i64 n, u64* out) {
    i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = avail[i] >= K ? ctx_key<K>(x + i + 1) : EMPTY;
}

extern "C" int val_keys(const u16* x, const int* avail, i64 n, int k, u64* out, void* stream) {
    LEVEL_SWITCH(k, (val_keys_k<K><<<grid_for(n, 256), 256, 0, (cudaStream_t)stream>>>(x, avail, n, out)));
    return (int)cudaGetLastError();
}

// Each valid position's k tokens against its group representative's: the mismatches (hash collisions) into *bad.
__global__ void val_verify_k(const u16* x, const i64* spos, const i64* gid, const i64* rep, i64 n, int k,
                             unsigned long long* bad) {
    i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x;
    if (i >= n) return;
    i64 p = spos[i], r = rep[gid[i]];
    for (int j = 0; j < k; ++j)
        if (x[p - j] != x[r - j]) {
            atomicAdd(bad, 1ull);
            return;
        }
}

extern "C" int val_verify(const u16* x, const i64* spos, const i64* gid, const i64* rep, i64 n, int k, u64* bad,
                          void* stream) {
    val_verify_k<<<grid_for(n, 256), 256, 0, (cudaStream_t)stream>>>(x, spos, gid, rep, n, k,
                                                                      (unsigned long long*)bad);
    return (int)cudaGetLastError();
}

// The lookup table's layout is canonical, the same on every rank (a context's slot is its id): the keys in order of
// home slot (then key) at slot_i = max(home_i, slot_{i-1} + 1), which is linear probing without wrap-around; the
// table ends with an empty slot. val_homes gives the home slots (of `cap`) of keys.
__global__ void val_homes_k(const u64* keys, i64 n, u64 cap, i64* home) {
    i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x;
    if (i < n) home[i] = (i64)__umul64hi(keys[i], cap);
}

extern "C" int val_homes(const u64* keys, i64 n, u64 cap, i64* home, void* stream) {
    if (n > 0) val_homes_k<<<grid_for(n, 256), 256, 0, (cudaStream_t)stream>>>(keys, n, cap, home);
    return (int)cudaGetLastError();
}

// ---------------------------------------------------------------------------------------------------------------
// Training scan: per level, the positions whose previous level matched (bm_in; all when null), their key looked up
// in the level's query table, verified, and (context id, next token) counted in the pair table.
// ---------------------------------------------------------------------------------------------------------------

// A pair table slot: the key (context id << 16 | next token, 41 bits) in the low bits, its count in the top 23 bits
// (EMPTY: all ones, never a key). A count that wraps past 2^23 - 1 adds one to the slot's byte in `wraps`.
#define KEYBITS 41
#define KEYMASK ((1ull << KEYBITS) - 1)
#define ONE (1ull << KEYBITS)

// Open addressing from the key's hash, probed linearly; flags[0] set if the table is full.
__device__ __forceinline__ void pair_add(u64* pk, u32* wraps, u64 cap, u64 key, u32* flags) {
    u64 i = __umul64hi(mix64(key), cap);
    for (int probe = 0; probe < MAXPROBE; ++probe) {
        u64 cur = pk[i];
        if (cur == EMPTY) {
            u64 prev = atomicCAS((unsigned long long*)&pk[i], EMPTY, key | ONE);
            if (prev == EMPTY) return;
            cur = prev;
        }
        if ((cur & KEYMASK) == key) {
            u64 old = atomicAdd((unsigned long long*)&pk[i], ONE);
            if ((old >> KEYBITS) == (1ull << (64 - KEYBITS)) - 1) atomicAdd(&wraps[i >> 2], 1u << (8 * (i & 3)));
            return;
        }
        i = i + 1 == cap ? 0 : i + 1;
    }
    atomicOr(&flags[0], 1u);
}

__device__ __forceinline__ u32 slot_count(const u32* wraps, u64 s, u64 v) {
    return (u32)(v >> KEYBITS) + (((wraps[s >> 2] >> (8 * (s & 3))) & 0xFFu) << (64 - KEYBITS));
}

// Position p's k-context looked up and verified (hashed levels): its pair key (context id << 16 | next token), or
// EMPTY.
template <int K>
__device__ __forceinline__ u64 probe_key(const u16* __restrict__ tok, i64 p, const u64* __restrict__ vt, u64 vcap,
                                         const u32* __restrict__ vrep, const u16* __restrict__ x) {
    // (positions are never within K of the buffer's start: it starts with GAP SENT tokens, a position whose next
    // token is SENT is skipped, and a context reaching into a gap holds a SENT, which no query context does)
    const u32 nxt = tok[p];
    if (nxt == SENT) return EMPTY;
    const u64 key = ctx_key<K>(tok + p);
    u64 s = __umul64hi(key, vcap);
    while (true) {
        const u64 v = vt[s];
        if (v == key) {
            if (K > 4) {
                const u16* a = tok + p - K;
                const u16* b = x + (i64)vrep[s] + 1 - K;
                bool same = true;
#pragma unroll
                for (int j = 0; j < K; ++j) same &= a[j] == b[j];
                if (!same) return EMPTY;
            }
            return (s << 16) | nxt;
        }
        if (v == EMPTY) return EMPTY;
        ++s;  // no wrap-around: the table ends with an empty slot
    }
}

// Dense levels: a lane per position, a warp per bitmap word (bm_in null: every position).
template <int K>
__global__ void __launch_bounds__(256) scan_k(const u16* __restrict__ tok, i64 L, const u32* __restrict__ bm_in,
                                              u32* __restrict__ bm_out, const u64* __restrict__ vt, u64 vcap, const u32* __restrict__ vrep,
                                              const u16* __restrict__ x, u64* pk, u32* pc, u64 cap, u32* flags) {
    const int lane = threadIdx.x & 31;
    const i64 stride = (i64)gridDim.x * blockDim.x;
    for (i64 base = blockIdx.x * (i64)blockDim.x + (threadIdx.x & ~31); base < L; base += stride) {
        const i64 p = base + lane;
        const i64 w = base >> 5;
        u32 stop = lane == 0 ? *(volatile u32*)&flags[0] : 0u;  // the pair table overflowed: counts() raises
        if (__shfl_sync(0xFFFFFFFFu, stop, 0)) break;
        const u32 word = bm_in ? bm_in[w] : 0xFFFFFFFFu;
        u64 key = EMPTY;
        if (word && ((word >> lane) & 1u) && p < L) key = probe_key<K>(tok, p, vt, vcap, vrep, x);
        const bool hit = key != EMPTY;
        const u32 b = __ballot_sync(0xFFFFFFFFu, hit);
        if (lane == 0) bm_out[w] = b;
        if (hit) pair_add(pk, pc, cap, key, flags);
    }
}

// Sparse levels: a lane per bitmap word, its set bits in turn.
template <int K>
__global__ void __launch_bounds__(256) scan_sparse_k(const u16* __restrict__ tok, i64 L, const u32* __restrict__ bm_in,
                                                     u32* __restrict__ bm_out, const u64* __restrict__ vt, u64 vcap, const u32* __restrict__ vrep,
                                                     const u16* __restrict__ x, u64* pk, u32* pc, u64 cap, u32* flags) {
    const i64 nwords = (L + 31) >> 5;
    for (i64 w = blockIdx.x * (i64)blockDim.x + threadIdx.x; w < nwords; w += (i64)gridDim.x * blockDim.x) {
        if (*(volatile u32*)&flags[0]) break;
        u32 word = bm_in[w], out = 0;
        while (word) {
            const int b = __ffs(word) - 1;
            word &= word - 1;
            const i64 p = (w << 5) + b;
            if (p >= L) continue;
            const u64 key = probe_key<K>(tok, p, vt, vcap, vrep, x);
            if (key != EMPTY) {
                out |= 1u << b;
                pair_add(pk, pc, cap, key, flags);
            }
        }
        bm_out[w] = out;
    }
}

extern "C" int scan_level(const u16* tok, i64 L, int k, const u32* bm_in, u32* bm_out, const u64* vt, u64 vcap,
                          const u32* vrep, const u16* x, u64* pk, u32* pc, u64 cap, u32* flags, int grid, int sparse,
                          void* stream) {
    cudaStream_t st = (cudaStream_t)stream;
    if (sparse) {
        LEVEL_SWITCH(k, (scan_sparse_k<K><<<grid, 256, 0, st>>>(tok, L, bm_in, bm_out, vt, vcap, vrep, x, pk, pc, cap, flags)));
    } else {
        LEVEL_SWITCH(k, (scan_k<K><<<grid, 256, 0, st>>>(tok, L, bm_in, bm_out, vt, vcap, vrep, x, pk, pc, cap, flags)));
    }
    return (int)cudaGetLastError();
}

// ---------------------------------------------------------------------------------------------------------------
// Local aggregation: the pair table's entries by owner rank (owner o holds context ids [bounds[o], bounds[o + 1])).
// ---------------------------------------------------------------------------------------------------------------

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

extern "C" int part_count(const u64* pk, u64 cap, const i64* bounds, int W, u64* counts, void* stream) {
    if (W > MAXW) return -1;
    part_count_k<<<2048, 256, 0, (cudaStream_t)stream>>>(pk, cap, bounds, W, (unsigned long long*)counts);
    return (int)cudaGetLastError();
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

extern "C" int part_scatter(const u64* pk, const u32* pc, u64 cap, const i64* bounds, int W, u64* cursor, u64* ok,
                            u32* oc, void* stream) {
    if (W > MAXW) return -1;
    part_scatter_k<<<4096, 256, 0, (cudaStream_t)stream>>>(pk, pc, cap, bounds, W, (unsigned long long*)cursor, ok, oc);
    return (int)cudaGetLastError();
}

// ---------------------------------------------------------------------------------------------------------------
// Owner side: CUB sorts and reductions, per-context segments, top-8, and per-position outputs.
// ---------------------------------------------------------------------------------------------------------------

// With temp == null: the temp bytes needed (returned through *temp_bytes); else the sort.
extern "C" int sort_pairs(void* temp, size_t* temp_bytes, const u64* kin, u64* kout, const u32* vin, u32* vout, i64 n,
                          int begin_bit, int end_bit, void* stream) {
    return (int)cub::DeviceRadixSort::SortPairs(temp, *temp_bytes, (const unsigned long long*)kin,
                                                (unsigned long long*)kout, vin, vout, n, begin_bit, end_bit,
                                                (cudaStream_t)stream);
}

// Inclusive running maximum (torch.cummax is ~50x slower on 10M elements).
extern "C" int inclusive_max(void* temp, size_t* temp_bytes, const i64* in, i64* out, i64 n, void* stream) {
    return (int)cub::DeviceScan::InclusiveScan(temp, *temp_bytes, (const long long*)in, (long long*)out, cub::Max(), n,
                                               (cudaStream_t)stream);
}

extern "C" int reduce_by_key(void* temp, size_t* temp_bytes, const u64* kin, u64* kout, const u32* vin, u32* vout,
                             i64* nruns, i64 n, void* stream) {
    return (int)cub::DeviceReduce::ReduceByKey(temp, *temp_bytes, (const unsigned long long*)kin,
                                               (unsigned long long*)kout, vin, vout, nruns, cuda::std::plus<u32>(), n,
                                               (cudaStream_t)stream);
}

__global__ void seg_bounds_k(const u64* uk, i64 U, i64* start, i64* end) {
    i64 j = blockIdx.x * (i64)blockDim.x + threadIdx.x;
    if (j >= U) return;
    u64 c = uk[j] >> 16;
    if (j == 0 || (uk[j - 1] >> 16) != c) start[c] = j;
    if (j == U - 1 || (uk[j + 1] >> 16) != c) end[c] = j + 1;
}

extern "C" int seg_bounds(const u64* uk, i64 U, i64* start, i64* end, void* stream) {
    if (U > 0) seg_bounds_k<<<grid_for(U, 256), 256, 0, (cudaStream_t)stream>>>(uk, U, start, end);
    return (int)cudaGetLastError();
}

__global__ void ctx_tot_k(const i64* start, const i64* end, const u32* agg, i64 nctx, u64* tot) {
    i64 c = blockIdx.x * (i64)blockDim.x + threadIdx.x;
    if (c >= nctx) return;
    u64 s = 0;
    for (i64 j = start[c]; j < end[c]; ++j) s += agg[j];
    tot[c] = s;
}

extern "C" int ctx_tot(const i64* start, const i64* end, const u32* agg, i64 nctx, u64* tot, void* stream) {
    if (nctx > 0) ctx_tot_k<<<grid_for(nctx, 256), 256, 0, (cudaStream_t)stream>>>(start, end, agg, nctx, tot);
    return (int)cudaGetLastError();
}

// The ranking key: context, then count descending; a stable sort keeps the token order (ascending) of ties.
__global__ void rank_keys_k(const u64* uk, const u32* agg, i64 U, u64* key2, u32* val2) {
    i64 j = blockIdx.x * (i64)blockDim.x + threadIdx.x;
    if (j >= U) return;
    key2[j] = ((uk[j] >> 16) << 32) | (u64)(0xFFFFFFFFu - agg[j]);
    val2[j] = (u32)j;
}

extern "C" int rank_keys(const u64* uk, const u32* agg, i64 U, u64* key2, u32* val2, void* stream) {
    if (U > 0) rank_keys_k<<<grid_for(U, 256), 256, 0, (cudaStream_t)stream>>>(uk, agg, U, key2, val2);
    return (int)cudaGetLastError();
}

// Each context's `top` most frequent next tokens and their counts, from the ranked pairs `sv`.
__global__ void rank_scatter_k(const u32* sv, i64 U, const u64* uk, const u32* agg, const i64* start, u16* top_tok,
                               u32* top_cnt, int top) {
    i64 q = blockIdx.x * (i64)blockDim.x + threadIdx.x;
    if (q >= U) return;
    u32 j = sv[q];
    u64 c = uk[j] >> 16;
    i64 r = q - start[c];
    if (r < top) {
        top_tok[c * top + r] = (u16)(uk[j] & 0xFFFF);
        top_cnt[c * top + r] = agg[j];
    }
}

extern "C" int rank_scatter(const u32* sv, i64 U, const u64* uk, const u32* agg, const i64* start, u16* top_tok,
                            u32* top_cnt, int top, void* stream) {
    if (U > 0)
        rank_scatter_k<<<grid_for(U, 256), 256, 0, (cudaStream_t)stream>>>(sv, U, uk, agg, start, top_tok, top_cnt, top);
    return (int)cudaGetLastError();
}

// Per query position whose context id is in [lo, hi): tot, dist and cnt (the target's count).
__global__ void pos_outputs_k(const int* ctx_of, i64 N, i64 lo, i64 hi, const u16* tgt, const u64* uk, const u32* agg,
                              const i64* start, const i64* end, const u64* tot, int* out_tot, int* out_dist, int* out_cnt) {
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
    if (a < end[c] && uk[a] == want) out_cnt[i] = (int)agg[a];
}

extern "C" int pos_outputs(const int* ctx_of, i64 N, i64 lo, i64 hi, const u16* tgt, const u64* uk, const u32* agg,
                           const i64* start, const i64* end, const u64* tot, int* out_tot, int* out_dist, int* out_cnt,
                           void* stream) {
    pos_outputs_k<<<grid_for(N, 256), 256, 0, (cudaStream_t)stream>>>(ctx_of, N, lo, hi, tgt, uk, agg, start, end, tot,
                                                                       out_tot, out_dist, out_cnt);
    return (int)cudaGetLastError();
}
