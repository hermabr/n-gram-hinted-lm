// GPU kernels of gpu_val_stats.py: the validation-side n-gram statistics (val_hints.ValContextCounts' counts, as the
// count rows and the hint rows read them) from training shards held in GPU memory. Built with nvcc into
// gpu_val_stats_kernels.so (see gpu_val_stats.py's build()) and called through ctypes; every buffer is a torch tensor.
//
// Definitions are exact_match/src/kgram.rs's: a training occurrence of a k-context is any k tokens within one shard
// and the token after them (it may cross a BOS); a validation position's k-context is the k tokens ending at it,
// valid iff they lie in its segment. On the GPU the shards are concatenated with a gap of SENT tokens (0xFFFF, not a
// GPT-2 token) before each, so a context reaching back past its shard's start holds a SENT and matches nothing, and
// a position whose next token is SENT is skipped.
//
// A k-context's key: for k <= 4 its tokens packed (exact), else a polynomial hash of 64-bit token codes; both through
// splitmix64's bijective finalizer. A training context whose hashed key equals a validation context's is compared
// token by token with that context's representative validation position, so every count is exact.
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

#define LEVEL_SWITCH(k, ...)                                   \
    switch (k) {                                               \
        case 1: { constexpr int K = 1; __VA_ARGS__; } break;   \
        case 2: { constexpr int K = 2; __VA_ARGS__; } break;   \
        case 3: { constexpr int K = 3; __VA_ARGS__; } break;   \
        case 4: { constexpr int K = 4; __VA_ARGS__; } break;   \
        case 5: { constexpr int K = 5; __VA_ARGS__; } break;   \
        case 6: { constexpr int K = 6; __VA_ARGS__; } break;   \
        case 7: { constexpr int K = 7; __VA_ARGS__; } break;   \
        case 8: { constexpr int K = 8; __VA_ARGS__; } break;   \
        case 10: { constexpr int K = 10; __VA_ARGS__; } break; \
        case 12: { constexpr int K = 12; __VA_ARGS__; } break; \
        case 16: { constexpr int K = 16; __VA_ARGS__; } break; \
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
// Validation side
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
// in the level's validation table, verified, and (context id, next token) counted in the pair table.
// ---------------------------------------------------------------------------------------------------------------

// A pair table slot: the key (context id << 16 | next token, 40 bits) in the low bits, its count in the top 24 bits
// (EMPTY: all ones, never a key). A count that wraps past 2^24 - 1 adds one to the slot's byte in `wraps`.
#define KEYBITS 40
#define KEYMASK ((1ull << KEYBITS) - 1)
#define ONE (1ull << KEYBITS)
#ifdef TEST_WRAP  // tests only: every count starts near the wrap
#define COUNT_OFF ((1u << (64 - KEYBITS)) - 64u)
#else
#define COUNT_OFF 0u
#endif

// The table is 2^pbits sub-tables of sub_cap slots: a key's sub-table is its hash's top pbits bits (pbits = 0: one
// table), its home slot within by the hash's other bits, probed linearly within the sub-table.
__device__ __forceinline__ void pair_add(u64* pk, u32* wraps, u64 sub_cap, int pbits, u64 key, u32* flags) {
    const u64 h = mix64(key);
    const u64 base = pbits ? (h >> (64 - pbits)) * sub_cap : 0;
    u64 s = __umul64hi(h << pbits, sub_cap);
    for (int probe = 0; probe < MAXPROBE; ++probe) {
        const u64 i = base + s;
        u64 cur = pk[i];
        if (cur == EMPTY) {
            u64 prev = atomicCAS((unsigned long long*)&pk[i], EMPTY, key | (ONE * (1ull + COUNT_OFF)));
            if (prev == EMPTY) return;
            cur = prev;
        }
        if ((cur & KEYMASK) == key) {
            u64 old = atomicAdd((unsigned long long*)&pk[i], ONE);
            if ((old >> KEYBITS) == (1ull << (64 - KEYBITS)) - 1) atomicAdd(&wraps[i >> 2], 1u << (8 * (i & 3)));
            return;
        }
        s = s + 1 == sub_cap ? 0 : s + 1;
    }
    atomicOr(&flags[0], 1u);
}

__device__ __forceinline__ u32 slot_count(const u64* pk, const u32* wraps, u64 s, u64 v) {
    return (u32)(v >> KEYBITS) + (((wraps[s >> 2] >> (8 * (s & 3))) & 0xFFu) << (64 - KEYBITS)) - COUNT_OFF;
}

// Position p's k-context looked up and verified (hashed levels): its pair key (context id << 16 | next token), or
// EMPTY.
template <int K, bool HASHED>
__device__ __forceinline__ u64 probe_key(const u16* __restrict__ tok, i64 p, const u64* __restrict__ vt, u64 vcap,
                                         const u32* __restrict__ vrep, const u16* __restrict__ x, u32* flags) {
    // (positions are never within K of the buffer's start: it starts with GAP SENT tokens, a position whose next
    // token is SENT is skipped, and a context reaching into a gap holds a SENT, which no validation context does)
    const u32 nxt = tok[p];
    if (nxt == SENT) return EMPTY;
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
                if (!same) {
                    atomicAdd(&flags[1], 1u);
                    return EMPTY;
                }
            }
            return (s << 16) | nxt;
        }
        if (v == EMPTY) return EMPTY;
        ++s;  // no wrap-around: the table ends with an empty slot
    }
}

// Dense levels: a lane per position, a warp per bitmap word (bm_in null: every position). Each hit's pair is counted
// in the table directly, or with EMIT appended to `emit` (its count in emitted[0]) for the partitioned aggregation.
template <int K, bool HASHED, bool EMIT>
__global__ void __launch_bounds__(256) scan_k(const u16* __restrict__ tok, i64 L, const u32* __restrict__ bm_in,
                                              u32* __restrict__ bm_out, const u64* __restrict__ vt, u64 vcap, const u32* __restrict__ vrep,
                                              const u16* __restrict__ x, u64* pk, u32* pc, u64 sub_cap, int pbits,
                                              u32* flags, u64* emit, unsigned long long* emitted) {
    const int lane = threadIdx.x & 31;
    const i64 stride = (i64)gridDim.x * blockDim.x;
    for (i64 base = blockIdx.x * (i64)blockDim.x + (threadIdx.x & ~31); base < L; base += stride) {
        const i64 p = base + lane;
        const i64 w = base >> 5;
        u32 stop = lane == 0 ? *(volatile u32*)&flags[0] : 0u;  // the pair table overflowed: the level reruns
        if (__shfl_sync(0xFFFFFFFFu, stop, 0)) break;
        const u32 word = bm_in ? bm_in[w] : 0xFFFFFFFFu;
        u64 key = EMPTY;
        if (word && ((word >> lane) & 1u) && p < L) key = probe_key<K, HASHED>(tok, p, vt, vcap, vrep, x, flags);
        const bool hit = key != EMPTY;
        const u32 b = __ballot_sync(0xFFFFFFFFu, hit);
        if (lane == 0) bm_out[w] = b;
        if (EMIT) {
            if (b) {
                unsigned long long at = 0;
                if (lane == 0) at = atomicAdd(emitted, (unsigned long long)__popc(b));
                at = __shfl_sync(0xFFFFFFFFu, at, 0);
                if (hit) emit[at + __popc(b & ((1u << lane) - 1))] = key;
            }
        } else if (hit) {
            pair_add(pk, pc, sub_cap, pbits, key, flags);
        }
    }
}

// Sparse levels: a lane per bitmap word, its set bits in turn, counted directly.
template <int K, bool HASHED>
__global__ void __launch_bounds__(256) scan_sparse_k(const u16* __restrict__ tok, i64 L, const u32* __restrict__ bm_in,
                                                     u32* __restrict__ bm_out, const u64* __restrict__ vt, u64 vcap, const u32* __restrict__ vrep,
                                                     const u16* __restrict__ x, u64* pk, u32* pc, u64 sub_cap,
                                                     int pbits, u32* flags) {
    const i64 nwords = (L + 31) >> 5;
    for (i64 w = blockIdx.x * (i64)blockDim.x + threadIdx.x; w < nwords; w += (i64)gridDim.x * blockDim.x) {
        if (*(volatile u32*)&flags[0]) break;
        u32 word = bm_in[w], out = 0;
        while (word) {
            const int b = __ffs(word) - 1;
            word &= word - 1;
            const i64 p = (w << 5) + b;
            if (p >= L) continue;
            const u64 key = probe_key<K, HASHED>(tok, p, vt, vcap, vrep, x, flags);
            if (key != EMPTY) {
                out |= 1u << b;
                pair_add(pk, pc, sub_cap, pbits, key, flags);
            }
        }
        bm_out[w] = out;
    }
}

// Medium-density levels: a warp per 32 bitmap words, each lane one word; per round every lane with bits left probes
// its next set bit, and the round's hits are appended together.
template <int K, bool HASHED>
__global__ void __launch_bounds__(256) scan_words_emit_k(const u16* __restrict__ tok, i64 L, const u32* __restrict__ bm_in,
                                                         u32* __restrict__ bm_out, const u64* __restrict__ vt, u64 vcap,
                                                         const u32* __restrict__ vrep, const u16* __restrict__ x,
                                                         u32* flags, u64* emit, unsigned long long* emitted) {
    const int lane = threadIdx.x & 31;
    const i64 nwords = (L + 31) >> 5;
    const i64 warp = (blockIdx.x * (i64)blockDim.x + threadIdx.x) >> 5;
    const i64 nwarps = ((i64)gridDim.x * blockDim.x) >> 5;
    for (i64 wbase = warp * 32; wbase < nwords; wbase += nwarps * 32) {
        const i64 w = wbase + lane;
        u32 word = w < nwords ? bm_in[w] : 0u, out = 0;
        while (__any_sync(0xFFFFFFFFu, word)) {
            u64 key = EMPTY;
            int b = 0;
            if (word) {
                b = __ffs(word) - 1;
                word &= word - 1;
                const i64 p = (w << 5) + b;
                if (p < L) key = probe_key<K, HASHED>(tok, p, vt, vcap, vrep, x, flags);
            }
            const bool hit = key != EMPTY;
            if (hit) out |= 1u << b;
            const u32 m = __ballot_sync(0xFFFFFFFFu, hit);
            if (m) {
                unsigned long long at = 0;
                if (lane == 0) at = atomicAdd(emitted, (unsigned long long)__popc(m));
                at = __shfl_sync(0xFFFFFFFFu, at, 0);
                if (hit) emit[at + __popc(m & ((1u << lane) - 1))] = key;
            }
        }
        if (w < nwords) bm_out[w] = out;
    }
}

// mode 0: dense, counted directly; 1: sparse, counted directly; 2: dense, emitted; 3: by words, emitted
extern "C" int scan_level(const u16* tok, i64 L, int k, const u32* bm_in, u32* bm_out, const u64* vt, u64 vcap,
                          const u32* vrep,
                          const u16* x, u64* pk, u32* pc, u64 sub_cap, int pbits, u32* flags, int grid, int mode,
                          u64* emit, u64* emitted, void* stream) {
    if (grid <= 0) grid = grid_for(L, 256);
    cudaStream_t st = (cudaStream_t)stream;
    const u64* t = vt;
    unsigned long long* em = (unsigned long long*)emitted;
    if (mode == 1 && bm_in) {
        if (k <= 4) {
            LEVEL_SWITCH(k, (scan_sparse_k<K, false><<<grid, 256, 0, st>>>(tok, L, bm_in, bm_out, t, vcap, vrep, x, pk, pc, sub_cap, pbits, flags)));
        } else {
            LEVEL_SWITCH(k, (scan_sparse_k<K, true><<<grid, 256, 0, st>>>(tok, L, bm_in, bm_out, t, vcap, vrep, x, pk, pc, sub_cap, pbits, flags)));
        }
    } else if (mode == 3 && bm_in) {
        if (k <= 4) {
            LEVEL_SWITCH(k, (scan_words_emit_k<K, false><<<grid, 256, 0, st>>>(tok, L, bm_in, bm_out, t, vcap, vrep, x, flags, emit, em)));
        } else {
            LEVEL_SWITCH(k, (scan_words_emit_k<K, true><<<grid, 256, 0, st>>>(tok, L, bm_in, bm_out, t, vcap, vrep, x, flags, emit, em)));
        }
    } else if (mode == 2) {
        if (k <= 4) {
            LEVEL_SWITCH(k, (scan_k<K, false, true><<<grid, 256, 0, st>>>(tok, L, bm_in, bm_out, t, vcap, vrep, x, pk, pc, sub_cap, pbits, flags, emit, em)));
        } else {
            LEVEL_SWITCH(k, (scan_k<K, true, true><<<grid, 256, 0, st>>>(tok, L, bm_in, bm_out, t, vcap, vrep, x, pk, pc, sub_cap, pbits, flags, emit, em)));
        }
    } else if (k <= 4) {
        LEVEL_SWITCH(k, (scan_k<K, false, false><<<grid, 256, 0, st>>>(tok, L, bm_in, bm_out, t, vcap, vrep, x, pk, pc, sub_cap, pbits, flags, emit, em)));
    } else {
        LEVEL_SWITCH(k, (scan_k<K, true, false><<<grid, 256, 0, st>>>(tok, L, bm_in, bm_out, t, vcap, vrep, x, pk, pc, sub_cap, pbits, flags, emit, em)));
    }
    return (int)cudaGetLastError();
}

// The emitted keys by sub-table (their hash's top pbits bits): counts, then a scatter into each sub-table's range.
// Per-bin counters live in dynamic shared memory (up to 2^14 bins where the device allows).
__global__ void subpart_count_k(const u64* keys, i64 n, int pbits, unsigned long long* counts) {
    extern __shared__ u32 h[];
    const int P = 1 << pbits;
    for (int b = threadIdx.x; b < P; b += blockDim.x) h[b] = 0;
    __syncthreads();
    for (i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; i < n; i += (i64)gridDim.x * blockDim.x)
        atomicAdd(&h[mix64(keys[i]) >> (64 - pbits)], 1u);
    __syncthreads();
    for (int b = threadIdx.x; b < P; b += blockDim.x)
        if (h[b]) atomicAdd(&counts[b], (unsigned long long)h[b]);
}

#define SP_ITEMS 16
__global__ void subpart_scatter_k(const u64* keys, i64 n, int pbits, unsigned long long* cursor, u64* out) {
    extern __shared__ u32 sm[];
    const int P = 1 << pbits;
    u32* cnt = sm;
    u32* base = sm + P;  // a chunk's offsets fit 32 bits
    const i64 tile_size = (i64)blockDim.x * SP_ITEMS;
    for (i64 tile = blockIdx.x * tile_size; tile < n; tile += (i64)gridDim.x * tile_size) {
        for (int b = threadIdx.x; b < P; b += blockDim.x) cnt[b] = 0;
        __syncthreads();
        u64 key[SP_ITEMS];
        int o[SP_ITEMS];
        u32 local[SP_ITEMS];
#pragma unroll
        for (int j = 0; j < SP_ITEMS; ++j) {
            const i64 i = tile + (i64)j * blockDim.x + threadIdx.x;
            o[j] = -1;
            if (i < n) {
                key[j] = keys[i];
                o[j] = (int)(mix64(key[j]) >> (64 - pbits));
                local[j] = atomicAdd(&cnt[o[j]], 1u);
            }
        }
        __syncthreads();
        for (int b = threadIdx.x; b < P; b += blockDim.x)
            if (cnt[b]) base[b] = (u32)atomicAdd(&cursor[b], (unsigned long long)cnt[b]);
        __syncthreads();
#pragma unroll
        for (int j = 0; j < SP_ITEMS; ++j)
            if (o[j] >= 0) out[(u64)base[o[j]] + local[j]] = key[j];
        __syncthreads();
    }
}

__global__ void aggregate_k(const u64* keys, i64 n, u64* pk, u32* wraps, u64 sub_cap, int pbits, u32* flags) {
    for (i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; i < n; i += (i64)gridDim.x * blockDim.x) {
        if (*(volatile u32*)&flags[0]) return;  // a sub-table overflowed: the level reruns
        pair_add(pk, wraps, sub_cap, pbits, keys[i], flags);
    }
}

// The most sub-table bits the grouping kernels support on this device (their counters in shared memory).
extern "C" int max_pbits() {
    int dev = 0, smem = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    int b = 1;
    while (b < 14 && (8 << (b + 1)) <= smem) ++b;
    return b;
}

static int set_smem(const void* fn, size_t bytes) {
    if (bytes > 48 * 1024) return (int)cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes);
    return 0;
}

extern "C" int subpart_count(const u64* keys, i64 n, int pbits, u64* counts, void* stream) {
    if (pbits < 1 || pbits > 14) return -1;
    size_t bytes = 4u << pbits;
    int e = set_smem((const void*)subpart_count_k, bytes);
    if (e) return e;
    if (n > 0)
        subpart_count_k<<<1024, 256, bytes, (cudaStream_t)stream>>>(keys, n, pbits, (unsigned long long*)counts);
    return (int)cudaGetLastError();
}

extern "C" int subpart_scatter(const u64* keys, i64 n, int pbits, u64* cursor, u64* out, void* stream) {
    if (pbits < 1 || pbits > 14) return -1;
    size_t bytes = 8u << pbits;
    int e = set_smem((const void*)subpart_scatter_k, bytes);
    if (e) return e;
    if (n > 0)
        subpart_scatter_k<<<1024, 256, bytes, (cudaStream_t)stream>>>(keys, n, pbits, (unsigned long long*)cursor, out);
    return (int)cudaGetLastError();
}

extern "C" int aggregate(const u64* keys, i64 n, u64* pk, u32* wraps, u64 sub_cap, int pbits, u32* flags, int grid,
                         void* stream) {
    if (n > 0) aggregate_k<<<grid > 0 ? grid : grid_for(n, 256), 256, 0, (cudaStream_t)stream>>>(keys, n, pk, wraps, sub_cap, pbits, flags);
    return (int)cudaGetLastError();
}

__global__ void popcount_k(const u32* bm, i64 n, unsigned long long* out) {
    unsigned long long c = 0;
    for (i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; i < n; i += (i64)gridDim.x * blockDim.x) c += __popc(bm[i]);
    for (int o = 16; o > 0; o >>= 1) c += __shfl_down_sync(0xFFFFFFFFu, c, o);
    if ((threadIdx.x & 31) == 0 && c) atomicAdd(out, c);
}

extern "C" int popcount(const u32* bm, i64 n, u64* out, void* stream) {
    popcount_k<<<1024, 256, 0, (cudaStream_t)stream>>>(bm, n, (unsigned long long*)out);
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
                oc[d] = slot_count(pk, pc, s, key[j]);
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
// Owner side: CUB sorts and reductions, per-context segments, ranks, top-8, and per-position outputs.
// ---------------------------------------------------------------------------------------------------------------

// With temp == null: the temp bytes needed (returned through *temp_bytes); else the sort.
extern "C" int sort_pairs(void* temp, size_t* temp_bytes, const u64* kin, u64* kout, const u32* vin, u32* vout, i64 n,
                          int begin_bit, int end_bit, void* stream) {
    cudaError_t e = cub::DeviceRadixSort::SortPairs(temp, *temp_bytes, (const unsigned long long*)kin,
                                                    (unsigned long long*)kout, vin, vout, n, begin_bit, end_bit,
                                                    (cudaStream_t)stream);
    return (int)e;
}

// Inclusive running maximum (torch.cummax is ~50x slower on 10M elements).
extern "C" int inclusive_max(void* temp, size_t* temp_bytes, const i64* in, i64* out, i64 n, void* stream) {
    cudaError_t e = cub::DeviceScan::InclusiveScan(temp, *temp_bytes, (const long long*)in, (long long*)out,
                                                   cub::Max(), n, (cudaStream_t)stream);
    return (int)e;
}

extern "C" int reduce_by_key(void* temp, size_t* temp_bytes, const u64* kin, u64* kout, const u32* vin, u32* vout,
                             i64* nruns, i64 n, void* stream) {
    cudaError_t e = cub::DeviceReduce::ReduceByKey(temp, *temp_bytes, (const unsigned long long*)kin,
                                                   (unsigned long long*)kout, vin, vout, nruns, cuda::std::plus<u32>(), n,
                                                   (cudaStream_t)stream);
    return (int)e;
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

extern "C" int rank_scatter(const u32* sv, i64 U, const u64* uk, const u32* agg, const i64* start, u32* rank_of,
                            u16* top_tok, u32* top_cnt, int top, void* stream) {
    if (U > 0)
        rank_scatter_k<<<grid_for(U, 256), 256, 0, (cudaStream_t)stream>>>(sv, U, uk, agg, start, rank_of, top_tok,
                                                                            top_cnt, top);
    return (int)cudaGetLastError();
}

// Per validation position whose context id is in [lo, hi): tot, dist and cnt (the target's count if among the
// context's cnt_top most frequent next tokens), and bit `li` of mask where the context occurred.
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

extern "C" int pos_outputs(const int* ctx_of, i64 N, i64 lo, i64 hi, const u16* tgt, const u64* uk, const u32* agg,
                           const u32* rank_of, const i64* start, const i64* end, const u64* tot, int li, int cnt_top,
                           int* out_tot, int* out_dist, int* out_cnt, int* mask, void* stream) {
    pos_outputs_k<<<grid_for(N, 256), 256, 0, (cudaStream_t)stream>>>(ctx_of, N, lo, hi, tgt, uk, agg, rank_of, start,
                                                                       end, tot, li, cnt_top, out_tot, out_dist,
                                                                       out_cnt, mask);
    return (int)cudaGetLastError();
}

// One hint variant's rows (old_records._val_rows with one slot): per position its deepest level of the variant whose
// context occurred (mask), if this rank owns that context: cell, top-8 tokens and weights, packed into 9 int32
// (cell, 4 x two tokens, 4 x two weights). The variant's levels ascend; per level v, desc[v * 8 + ...] holds: mask
// bit, length bucket, ctx_of pointer, lo, hi, top tokens pointer, top counts pointer, totals pointer.
__constant__ double PURITY_EDGES[5] = {0.3, 0.5, 0.7, 0.85, 0.95};
__constant__ u64 COUNT_EDGES[6] = {2, 3, 5, 9, 17, 33};

__global__ void variant_rows_k(i64 N, const int* mask, int nv, const i64* desc, int* rows) {
    i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x;
    if (i >= N) return;
    int m = mask[i];
    for (int v = nv - 1; v >= 0; --v) {
        const i64* d = desc + v * 8;
        if (!((m >> d[0]) & 1)) continue;
        const int* ctx_of = (const int*)d[2];
        i64 c = ctx_of[i];
        if (c < d[3] || c >= d[4]) return;  // another rank's context
        c -= d[3];
        const u16* tt = (const u16*)d[5] + c * 8;
        const u32* tc = (const u32*)d[6] + c * 8;
        u64 total = ((const u64*)d[7])[c];
        u32 cnt[8];
        double sum = 0;
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            cnt[r] = tc[r];
            sum += (double)cnt[r];
        }
        int count_bin = 0;
#pragma unroll
        for (int e = 0; e < 6; ++e) count_bin += COUNT_EDGES[e] <= total;
        double share = (double)cnt[0] / (double)(total > 1 ? total : 1);
        int purity_bin = 0;
#pragma unroll
        for (int e = 0; e < 5; ++e) purity_bin += PURITY_EDGES[e] <= share;
        int cell = ((1 + (int)d[1]) * 7 + count_bin) * 6 + purity_bin;
        double denom = sum > 1 ? sum : 1;
        int* row = rows + i * 9;
        row[0] = cell;
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            row[1 + r] = (int)((u32)tt[2 * r] | ((u32)tt[2 * r + 1] << 16));
            u32 w0 = (u32)rint((double)cnt[2 * r] / denom * 65535.0);
            u32 w1 = (u32)rint((double)cnt[2 * r + 1] / denom * 65535.0);
            row[5 + r] = (int)(w0 | (w1 << 16));
        }
        return;
    }
}

extern "C" int variant_rows(i64 N, const int* mask, int nv, const i64* desc, int* rows, void* stream) {
    variant_rows_k<<<grid_for(N, 256), 256, 0, (cudaStream_t)stream>>>(N, mask, nv, desc, rows);
    return (int)cudaGetLastError();
}
