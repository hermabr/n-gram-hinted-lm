//! SlotIndexV1: the first SlotIndex (2026-09-29), modded-nanogpt branch approxO1, commit 13fc6f12 ("SlotIndex: docs
//! for the final design; defaults block=true, slot_cap=64"), exact_match/src/slotindex.rs, with the helpers it took
//! from that commit's lib.rs (key, P, BOS, HEADER, Output, fence, Map, populate, Corpus, pieces) copied in. Report:
//! artifacts/sep29-approxO1-index.md.
//!
//! Entry (u64, one per corpus position with a 6-token context after its segment's latest BOS, but for a leading
//! BOS): [ slot: 14 | tag: 48 - 14 - fb | fingerprint fields: fb | next token: 16 ]. The key is the 6-gram's
//! polynomial prefix-hash difference times an odd constant; its top 16 bits pick one of 65,536 partitions, the next
//! 14 the slot. A level's field is the top bits of its context's polynomial hash (never 0), or 0 when the context
//! crosses its shard's start or holds a BOS but as its first token.
//!
//! Build:
//!   1. scan: each thread scans its contiguous share of the corpus once in 256-position blocks (a serial prefix-hash
//!      loop two tokens per step, then the key and each level's field in loops that vectorize) and appends each entry
//!      to its partition through a one-line write-combining buffer per partition; a full line goes out with one
//!      non-temporal store into the thread's current 32-line chunk for that partition (chained per (thread,
//!      partition)).
//!   2. layout: partition sizes and bases (serial, < 0.01 s).
//!   3. finish: per partition (largest first, rayon par_iter), a counting sort by the 2^14 slot bits: a histogram
//!      walk over the chains, a prefix sum capped at slot_cap entries per slot (the first in corpus order stay), a
//!      placement walk into a cache-resident buffer, non-temporal stores of it into the partition's region of the
//!      flat index, and the slot directory (u32 per slot).
//!
//! The recommended configuration (13fc6f12's new() defaults) is fixed: block=true, partition_bits=16, slot_bits=14,
//! chunk_lines=32, key_mode=0 (multiply), fp_mode=0 (polynomial), avx512=true (64-byte non-temporal stores and the
//! AVX-512 pass 1 when the CPU has AVX-512F; else the 16-byte stores and generic code, as the original), entry_bytes=8,
//! pipeline/null/stage/radix/one_walk/pair/nta off, pf_lines=1, stage_entries=2^21, tokens_per_shard=101M. The option
//! fields and their branches are kept as in the original.
//!
//! Adaptations:
//! - levels [10, 20] with fingerprint bits [12, 6] (the record's), not [10, 16, 32] / [10, 5, 5]: entry
//!   [slot 14 | tag 16 | fps 18 | token 16].
//! - slot_cap = cap = 288 (the original's default 64). The record keeps at most 2 x 144 = 288 entries per key (a hot
//!   key fills its two buckets; 144 slots per bucket at the record's corpus size). v1's nearest equivalent is the per
//!   slot cap: a slot holds one key's entries plus the few colliding keys', and the build keeps its first slot_cap in
//!   corpus order. cap (matches used per lookup) is lookup-only.
//! - The corpus is the first `tokens` tokens of `files` (crate::shards; the last shard's range is cut there), mapped
//!   as in the original (private read-only mappings `stride` tokens apart; only the used part of each file).
//! - The rayon global pool (RAYON_NUM_THREADS) is replaced by crate::pinned_pool(threads, cpus): pass 1's per-thread
//!   writers, pass 2's par_iter and the prefault run on it.
//! - create() = new() + prefault() (the original's separate pre-clock call: MADV_POPULATE_WRITE of every buffer, and
//!   MADV_POPULATE_READ of the corpus pages); build() = build(), returning the entries kept (reachable), dropped (by
//!   the slot cap), bytes (index + directory as allocated) and the original's phase times.
//! - Lookups (query_all, lookup_items, set_caps, partition_sizes, slot_histogram) are dropped; digest() (off the
//!   clock, verification only) hashes each partition's directory and reachable entries.
//!
//! Verified (2026-10-09) against 13fc6f12 built from `git archive` with levels=[10, 20], bits=[12, 6],
//! slot_cap=cap=288 and the same digest: identical kept counts and digests for 1 whole shard (login node, the
//! non-AVX-512 path, 8 and 5 threads) and 2 whole shards (miso4, AVX-512, 48 threads); the index does not depend on
//! the thread count.
use crate::{pinned_pool, shards};
use pyo3::{exceptions::PyValueError, prelude::*};
use rayon::prelude::*;
use std::collections::HashMap;
use std::os::fd::AsRawFd;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

// From 13fc6f12's lib.rs.
const BOS: u16 = 50256;
const P: u64 = 0x9e3779b185ebca87;
const HEADER: usize = 512; // a shard file's header, in tokens

fn key(mut h: u64) -> u64 {
    h ^= h >> 30;
    h = h.wrapping_mul(0xbf58476d1ce4e5b9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94d049bb133111eb);
    h ^ (h >> 31)
}

#[derive(Clone, Copy)]
struct Output<T>(*mut T);
unsafe impl<T> Sync for Output<T> {}
unsafe impl<T> Send for Output<T> {}
impl<T: Copy> Output<T> {
    unsafe fn slice<'a>(self, offset: usize, len: usize) -> &'a mut [T] {
        std::slice::from_raw_parts_mut(self.0.add(offset), len)
    }
}

fn fence() {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_sfence()
    };
}

/// Private anonymous memory (13fc6f12's Map, its anon constructor only).
struct Map<T> {
    ptr: *mut T,
    len: usize,
}
unsafe impl<T> Send for Map<T> {}
unsafe impl<T> Sync for Map<T> {}

impl<T> Map<T> {
    /// Private memory in transparent huge pages, zero until written.
    fn anon(len: usize) -> std::io::Result<Self> {
        let bytes = (len * std::mem::size_of::<T>()).max(1);
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                bytes,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error());
        }
        unsafe { libc::madvise(ptr, bytes, libc::MADV_HUGEPAGE) };
        Ok(Self { ptr: ptr.cast(), len })
    }

    fn get(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl<T> Drop for Map<T> {
    fn drop(&mut self) {
        unsafe { libc::munmap(self.ptr.cast(), (self.len * std::mem::size_of::<T>()).max(1)) };
    }
}

/// madvise(advice) over [ptr, ptr + bytes), widened to whole pages, in parallel 256 MB pieces.
fn populate(ptr: usize, bytes: usize, advice: libc::c_int) -> std::io::Result<()> {
    const CHUNK: usize = 1 << 28;
    let (start, end) = (ptr & !4095, ptr + bytes);
    (start..end).step_by(CHUNK).collect::<Vec<_>>().par_iter().try_for_each(|&at| {
        if unsafe { libc::madvise(at as *mut libc::c_void, CHUNK.min(end - at), advice) } != 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(())
    })
}

/// Read-only private mappings of shard files, `stride` tokens apart in one reserved range: token k of file i sits
/// at position i * stride + HEADER + k, and ranges[i] spans file i's tokens. The page cache is the corpus, so
/// nothing is copied, and every process shares its pages.
struct Corpus {
    ptr: *mut u16,
    len: usize,
    ranges: Vec<(usize, usize)>,
}
unsafe impl Send for Corpus {}
unsafe impl Sync for Corpus {}

impl Corpus {
    /// Port: the first `tokens` tokens of `files` (crate::shards checks the headers and cuts the last one); each
    /// file is mapped up to its last used token.
    fn map(files: &[String], stride: usize, tokens: usize) -> PyResult<Self> {
        let shards = shards(files, tokens)?;
        let len = shards.len() * stride;
        let flags = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE;
        let ptr = unsafe { libc::mmap(std::ptr::null_mut(), len * 2, libc::PROT_READ, flags, -1, 0) };
        if ptr == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error().into());
        }
        let mut corpus = Self { ptr: ptr.cast(), len, ranges: Vec::new() };
        for (i, shard) in shards.iter().enumerate() {
            let n = shard.len;
            if HEADER + n > stride {
                return Err(PyValueError::new_err("shard exceeds tokens_per_shard"));
            }
            let at = unsafe { corpus.ptr.add(i * stride) };
            let flags = libc::MAP_PRIVATE | libc::MAP_FIXED;
            let mapped = unsafe { libc::mmap(at.cast(), (HEADER + n) * 2, libc::PROT_READ, flags, shard.file.as_raw_fd(), 0) };
            if mapped == libc::MAP_FAILED {
                return Err(std::io::Error::last_os_error().into());
            }
            corpus.ranges.push((i * stride + HEADER, i * stride + HEADER + n));
        }
        Ok(corpus)
    }

    fn get(&self) -> &[u16] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Map the pages of `ranges` into this process ahead of use.
    fn populate(&self, ranges: &[(usize, usize)]) -> std::io::Result<()> {
        ranges.iter().try_for_each(|&(lo, hi)| {
            populate(self.ptr as usize + lo * 2, (hi - lo) * 2, libc::MADV_POPULATE_READ)
        })
    }
}

impl Drop for Corpus {
    fn drop(&mut self) {
        unsafe { libc::munmap(self.ptr.cast(), self.len * 2) };
    }
}

/// Thread t's share of `ranges` split evenly across `threads`, as (range start, range end, start, end): the thread
/// owns the positions p with start < p <= end.
fn pieces(ranges: &[(usize, usize)], t: usize, threads: usize) -> Vec<(usize, usize, usize, usize)> {
    let total: usize = ranges.iter().map(|&(lo, hi)| hi - lo).sum();
    let (a, b) = (total * t / threads, total * (t + 1) / threads);
    let mut offset = 0;
    let mut out = Vec::new();
    for &(lo, hi) in ranges {
        let clip = |x: usize| lo + x.clamp(offset, offset + hi - lo) - offset;
        if clip(a) < clip(b) {
            out.push((lo, hi, clip(a), clip(b)));
        }
        offset += hi - lo;
    }
    out
}

// 13fc6f12's slotindex.rs from here on (build path).
const MIN: usize = 6;
const RING: usize = 128; // prefix hashes kept by a scan, a power of two above the longest level
const MASK: usize = RING - 1;
const MAX_LEVELS: usize = 4;
const LINE: usize = 8; // entries per 64-byte line

pub const KEY_MUL: u8 = 0;
pub const KEY_MIX: u8 = 1;
pub const FP_POLY: u8 = 0;
pub const FP_MUL: u8 = 1;

#[derive(Clone)]
struct Hasher {
    n: usize,
    len: [usize; MAX_LEVELS],
    bits: [u32; MAX_LEVELS],
    shift: [u32; MAX_LEVELS], // bit offset of level k's field in an entry
    pow: [u64; MAX_LEVELS],
    mul: [u64; MAX_LEVELS],
    p6: u64,
    p2: u64,
    m6: u64,
}

impl Hasher {
    fn new(levels: &[usize], bits: &[u32]) -> Self {
        let mut h = Self {
            n: levels.len(),
            len: [0; MAX_LEVELS],
            bits: [1; MAX_LEVELS],
            shift: [0; MAX_LEVELS],
            pow: [0; MAX_LEVELS],
            mul: [0; MAX_LEVELS],
            p6: P.wrapping_pow(MIN as u32),
            p2: P.wrapping_mul(P),
            m6: key(0x6a09e667f3bcc909) | 1,
        };
        let mut at = 16;
        for (k, (&l, &b)) in levels.iter().zip(bits).enumerate() {
            h.len[k] = l;
            h.bits[k] = b;
            h.shift[k] = at;
            at += b;
            h.pow[k] = P.wrapping_pow(l as u32);
            h.mul[k] = key(0x2545f4914f6cdd1d ^ (k as u64 + 1).wrapping_mul(P)) | 1;
        }
        h
    }

    /// Scan tokens[begin..end] and call emit(pos, key, fingerprint fields) for each pos = i + 1 with
    /// first < pos < stop whose 6-token context lies after the latest BOS (or `begin`) but for a leading BOS.
    /// Returns the largest token seen.
    #[inline(always)]
    fn scan<const KM: u8, const FM: u8>(
        &self,
        tokens: &[u16],
        begin: usize,
        end: usize,
        first: usize,
        stop: usize,
        mut emit: impl FnMut(usize, u64, u64),
    ) -> u16 {
        let mut ring = [0u64; RING];
        let (mut g, mut seg, mut worst) = (0u64, begin, 0u16);
        for i in begin..end {
            let token = tokens[i];
            worst = worst.max(token);
            if token == BOS {
                seg = i;
            }
            g = g.wrapping_mul(P).wrapping_add(token as u64 + 1);
            let pos = i + 1;
            ring[pos & MASK] = g;
            if pos < seg + MIN || pos <= first || pos >= stop {
                continue;
            }
            let v6 = g.wrapping_sub(ring[(pos - MIN) & MASK].wrapping_mul(self.p6));
            let h = if KM == KEY_MIX { key(v6) } else { v6.wrapping_mul(self.m6) };
            let avail = pos - seg;
            let mut fps = 0u64;
            for k in 0..MAX_LEVELS {
                if k < self.n {
                    let mut v = g.wrapping_sub(ring[pos.wrapping_sub(self.len[k]) & MASK].wrapping_mul(self.pow[k]));
                    if FM == FP_MUL {
                        v = v.wrapping_mul(self.mul[k]);
                    }
                    let f = (v >> (64 - self.bits[k])).max(1);
                    fps |= (f * (avail >= self.len[k]) as u64) << self.shift[k];
                }
            }
            emit(pos, h, fps);
        }
        worst
    }

    /// As scan (same keys and fields), in blocks of BLOCK positions: a serial loop of the prefix hashes and context
    /// lengths, then loops over the block for the key and each level that vectorize, then the emits.
    #[inline(always)]
    fn scan_block<const KM: u8, const FM: u8>(
        &self,
        tokens: &[u16],
        begin: usize,
        end: usize,
        first: usize,
        stop: usize,
        mut emit: impl FnMut(usize, u64, u64),
    ) -> u16 {
        const B: usize = 256;
        let mut g = [0u64; RING + B]; // g[RING + j]: the prefix hash at pos = i0 + j + 1
        let mut av = [0u32; B]; // pos - latest BOS (or begin)
        let mut hh = [0u64; B];
        let mut ff = [0u64; B];
        let (mut gc, mut seg, mut worst) = (0u64, begin, 0u16);
        let mut i0 = begin;
        while i0 < end {
            let n = B.min(end - i0);
            // Two tokens per step: g2 = g0 P^2 + (a0 P + a1) keeps one multiply-add on the dependency chain per pair.
            let mut j = 0;
            while j + 1 < n {
                let (t0, t1) = (tokens[i0 + j], tokens[i0 + j + 1]);
                worst = worst.max(t0.max(t1));
                if t0 == BOS {
                    seg = i0 + j;
                }
                av[j] = (i0 + j + 1 - seg) as u32;
                if t1 == BOS {
                    seg = i0 + j + 1;
                }
                av[j + 1] = (i0 + j + 2 - seg) as u32;
                let (a0, a1) = (t0 as u64 + 1, t1 as u64 + 1);
                g[RING + j] = gc.wrapping_mul(P).wrapping_add(a0);
                gc = gc.wrapping_mul(self.p2).wrapping_add(a0.wrapping_mul(P).wrapping_add(a1));
                g[RING + j + 1] = gc;
                j += 2;
            }
            if j < n {
                let token = tokens[i0 + j];
                worst = worst.max(token);
                if token == BOS {
                    seg = i0 + j;
                }
                gc = gc.wrapping_mul(P).wrapping_add(token as u64 + 1);
                g[RING + j] = gc;
                av[j] = (i0 + j + 1 - seg) as u32;
            }
            let (cur, back6) = (&g[RING..RING + B], &g[RING - MIN..RING - MIN + B]);
            for j in 0..B {
                let v6 = cur[j].wrapping_sub(back6[j].wrapping_mul(self.p6));
                hh[j] = if KM == KEY_MIX { key(v6) } else { v6.wrapping_mul(self.m6) };
                ff[j] = 0;
            }
            for k in 0..self.n {
                let (l, pw, mul, b, sh) = (self.len[k], self.pow[k], self.mul[k], self.bits[k], self.shift[k]);
                let back = &g[RING - l..RING - l + B];
                for j in 0..B {
                    let mut v = cur[j].wrapping_sub(back[j].wrapping_mul(pw));
                    if FM == FP_MUL {
                        v = v.wrapping_mul(mul);
                    }
                    let f = (v >> (64 - b)).max(1);
                    ff[j] |= if av[j] >= l as u32 { f << sh } else { 0 };
                }
            }
            for j in 0..n {
                let pos = i0 + j + 1;
                if av[j] >= MIN as u32 && pos > first && pos < stop {
                    emit(pos, hh[j], ff[j]);
                }
            }
            g.copy_within(n..n + RING, 0);
            i0 += n;
        }
        worst
    }

}

/// Copy one 64-byte line with non-temporal stores (dst 64-byte aligned, src 16-byte aligned).
#[inline(always)]
unsafe fn stream_line<const W: bool>(dst: *mut u64, src: *const u64) {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;
        if W {
            _mm512_stream_si512(dst.cast(), _mm512_loadu_si512(src.cast()));
        } else {
            for i in 0..4 {
                _mm_stream_si128((dst as *mut __m128i).add(i), _mm_loadu_si128((src as *const __m128i).add(i)));
            }
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    std::ptr::copy_nonoverlapping(src, dst, LINE);
}

#[inline(always)]
fn prefetch<T>(ptr: *const T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
        _mm_prefetch(ptr as *const i8, _MM_HINT_T0);
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = ptr;
}

#[inline(always)]
fn prefetch_nta<T>(ptr: *const T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::{_mm_prefetch, _MM_HINT_NTA};
        _mm_prefetch(ptr as *const i8, _MM_HINT_NTA);
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = ptr;
}

fn wide_stores() -> bool {
    #[cfg(target_arch = "x86_64")]
    return std::arch::is_x86_feature_detected!("avx512f");
    #[cfg(not(target_arch = "x86_64"))]
    false
}

/// A thread's pass-1 state: one write-combining line per partition, and its chunk chains.
struct Writer {
    buf: Map<u64>,  // parts * LINE entries
    used: Vec<u8>,  // entries in each partition's line
    cur: Vec<u32>,  // next arena line of each partition's current chunk; 0 = no chunk yet
    head: Vec<u32>, // first chunk of each partition's chain
    nch: Vec<u32>,  // chunks in each chain
    last: Vec<u32>, // entries in the last chunk of each chain (after the pass)
    lo: u32,        // this thread's chunk ids: [lo, hi)
    hi: u32,
    entries: usize,
}

/// Pass-2 scratch of a thread.
struct Finisher {
    pairs: Vec<[u32; 2]>,
    small: Vec<u8>,
    hist: Vec<u32>,
    cursor: Vec<u32>,
    out: Map<u64>,
    stage: Map<u64>,
    slots: Vec<u16>,
}

#[derive(Clone, Copy)]
struct Shape {
    pb: u32,
    sb: u32,
    fb: u32,
    cl: u32, // lines per chunk
    x: bool, // 10-byte pass-1 entries (6 per line: six u64 then six u16), else 8-byte (8 per line)
}

#[allow(dead_code)] // key_slot and want: the lookup's
impl Shape {
    #[inline(always)]
    fn part(&self, h: u64) -> usize {
        (h >> (64 - self.pb)) as usize
    }
    #[inline(always)]
    fn low_mask(&self) -> u64 {
        (1u64 << (16 + self.fb)) - 1
    }
    /// The slot of a key: the sb key bits below the partition bits.
    #[inline(always)]
    fn key_slot(&self, h: u64) -> usize {
        ((h << self.pb) >> (64 - self.sb)) as usize
    }
    /// The final entry's high bits (tag) of a key, shifted down: what a lookup compares.
    #[inline(always)]
    fn want(&self, h: u64) -> u64 {
        let k = h << self.pb;
        let k = if self.x { k << 16 } else { k };
        k >> (16 + self.fb)
    }
    /// Pass-1 entry of a key: (u64 part, u16 slot word). 8-byte: [slot | tag | fps | token], the slot word unused.
    /// 10-byte: the u64 part is [tag | fps | token] (the final entry) and the u16 word the key's next 16 bits.
    #[inline(always)]
    fn entry<const X: bool>(&self, h: u64, low: u64) -> (u64, u16) {
        let k = h << self.pb;
        if X {
            (((k << 16) & !self.low_mask()) | low, (k >> 48) as u16)
        } else {
            ((k & !self.low_mask()) | low, 0)
        }
    }
    #[inline(always)]
    fn slot8(&self, e: u64) -> usize {
        (e >> (64 - self.sb)) as usize
    }
    #[inline(always)]
    fn slot10(&self, w: u16) -> usize {
        (w >> (16 - self.sb)) as usize
    }
}

const fn per_line(x: bool) -> usize {
    if x {
        6
    } else {
        LINE
    }
}

#[pyclass]
pub struct SlotIndexV1 {
    hs: Hasher,
    #[pyo3(get)]
    levels: Vec<usize>,
    #[pyo3(get)]
    bits: Vec<u32>,
    #[pyo3(get)]
    cap: usize,
    #[pyo3(get)]
    slot_cap: usize,
    #[pyo3(get)]
    key_mode: u8,
    #[pyo3(get)]
    fp_mode: u8,
    #[pyo3(get)]
    pipeline: bool,
    #[pyo3(get)]
    wide: bool,
    #[pyo3(get)]
    null: bool, // profiling: pass 1 computes entries but never appends them
    #[pyo3(get)]
    pf_lines: usize, // lines of the next chunk prefetched when pass 2 starts a chunk
    #[pyo3(get)]
    stage: bool, // pass 2's histogram walk copies the partition into a cache-resident staging buffer
    #[pyo3(get)]
    block: bool, // pass 1 scans in vectorized blocks
    #[pyo3(get)]
    radix: bool,
    #[pyo3(get)]
    nta: bool, // plain pass 2's placement walk prefetches chunks non-temporally
    #[pyo3(get)]
    pair: bool, // plain pass 2 keeps each slot's cursor and end side by side
    #[pyo3(get)]
    one_walk: bool, // radix: one walk of the chains into fixed-capacity staging buckets // pass 2 by two counting passes (64 buckets, then in-L1 slots), uncapped (lookups take slot_cap)
    finish_ns: [AtomicUsize; 3],
    shape: Shape,
    corpus: Corpus,
    ranges: Vec<(usize, usize)>,
    threads: usize,
    pool: std::sync::Arc<rayon::ThreadPool>, // port: the pinned build pool
    arena: Map<u64>,
    next: Map<u32>,
    writers: Vec<Writer>,
    finishers: Vec<Finisher>,
    index: Map<u64>,
    dir: Map<u32>,
    base: Vec<u64>,
    counts: Vec<u64>,
    ready: bool,
    #[pyo3(get)]
    entries: usize,
    #[pyo3(get)]
    kept: usize,
    #[pyo3(get)]
    chunks_used: usize,
}

impl SlotIndexV1 {
    fn nslots(&self) -> usize {
        1 << self.shape.sb
    }

    /// Pass 1 over thread t's share, monomorphized per hash modes, pipelining and store width.
    #[inline(always)]
    fn pass1_body<const KM: u8, const FM: u8, const PIPE: bool, const W: bool, const X: bool>(&self, w: &mut Writer, t: usize) -> Result<(), String> {
        let per = per_line(X);
        let (corpus, hs, sh) = (self.corpus.get(), &self.hs, self.shape);
        let longest = hs.len[..hs.n].iter().copied().max().unwrap_or(MIN).max(MIN);
        let arena = self.arena.ptr;
        let next = self.next.ptr;
        let cl = sh.cl;
        let buf = w.buf.ptr;
        let Writer { used, cur, head, nch, last, lo, hi, entries, .. } = w;
        let mut alloc = *lo;
        let limit = *hi;
        let mut overflow = false;
        let mut count = 0usize;
        let mut worst = 0u16;
        let mut pend_p = [0usize; LINE * 2];
        let mut pend_e = [0u64; LINE * 2];
        let mut pend_w = [0u16; LINE * 2];
        let mut npend = 0usize;
        let null = self.null;
        let mut sink = 0u64;
        let mut append = |p: usize, e: u64, sw: u16, overflow: &mut bool| unsafe {
            let u = *used.get_unchecked(p) as usize;
            *buf.add(p * LINE + u) = e;
            if X {
                *(buf.add(p * LINE) as *mut u16).add(4 * per + u) = sw;
            }
            if u == per - 1 {
                *used.get_unchecked_mut(p) = 0;
                let mut c = *cur.get_unchecked(p);
                if c % cl == 0 {
                    if alloc >= limit {
                        *overflow = true;
                        return;
                    }
                    let id = alloc;
                    alloc += 1;
                    if c == 0 {
                        head[p] = id;
                    } else {
                        *next.add(((c - 1) / cl) as usize) = id;
                    }
                    nch[p] += 1;
                    c = id * cl;
                }
                stream_line::<W>(arena.add(c as usize * LINE), buf.add(p * LINE));
                *cur.get_unchecked_mut(p) = c + 1;
            } else {
                *used.get_unchecked_mut(p) = (u + 1) as u8;
            }
        };
        for (lo_r, hi_r, start, end) in pieces(&self.ranges, t, self.threads) {
            let begin = start.saturating_sub(longest).max(lo_r);
            let body = |pos: usize, h: u64, fps: u64| {
                let p = sh.part(h);
                let (e, sw) = sh.entry::<X>(h, fps | corpus[pos] as u64);
                count += 1;
                if null {
                    sink = sink.wrapping_add(e ^ (p as u64) ^ sw as u64);
                } else if PIPE {
                    prefetch(unsafe { buf.add(p * LINE) });
                    pend_p[npend] = p;
                    pend_e[npend] = e;
                    pend_w[npend] = sw;
                    npend += 1;
                    if npend == pend_p.len() {
                        for j in 0..npend {
                            append(pend_p[j], pend_e[j], pend_w[j], &mut overflow);
                        }
                        npend = 0;
                    }
                } else {
                    append(p, e, sw, &mut overflow);
                }
            };
            let wv = if self.block {
                hs.scan_block::<KM, FM>(corpus, begin, end, start, hi_r, body)
            } else {
                hs.scan::<KM, FM>(corpus, begin, end, start, hi_r, body)
            };
            worst = worst.max(wv);
        }
        for j in 0..npend {
            append(pend_p[j], pend_e[j], pend_w[j], &mut overflow);
        }
        // The partial lines, then each chain's last chunk fill.
        let parts = 1usize << sh.pb;
        for p in 0..parts {
            let u = used[p] as usize;
            if u > 0 {
                let mut c = cur[p];
                if c % cl == 0 {
                    if alloc >= limit {
                        overflow = true;
                        break;
                    }
                    let id = alloc;
                    alloc += 1;
                    if c == 0 {
                        head[p] = id;
                    } else {
                        unsafe { *next.add(((c - 1) / cl) as usize) = id };
                    }
                    nch[p] += 1;
                    c = id * cl;
                }
                unsafe { std::ptr::copy_nonoverlapping(buf.add(p * LINE), arena.add(c as usize * LINE), LINE) };
                cur[p] = c + 1;
                used[p] = 0;
                w_last(last, p, c + 1, cl, per, u);
            } else if cur[p] != 0 {
                w_last(last, p, cur[p], cl, per, per);
            } else {
                last[p] = 0;
            }
        }
        fence();
        *entries = count + (sink == 1) as usize;
        if worst > BOS {
            return Err("invalid GPT-2 token".into());
        }
        if overflow {
            return Err("chunk region overflow".into());
        }
        Ok(())
    }

    fn pass1<const KM: u8, const FM: u8>(&self, w: &mut Writer, t: usize) -> Result<(), String> {
        if self.shape.x {
            self.pass1_x::<KM, FM, true>(w, t)
        } else {
            self.pass1_x::<KM, FM, false>(w, t)
        }
    }

    fn pass1_x<const KM: u8, const FM: u8, const X: bool>(&self, w: &mut Writer, t: usize) -> Result<(), String> {
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx512f,avx512dq,avx512vl,avx512bw")]
        unsafe fn wide<const KM: u8, const FM: u8, const PIPE: bool, const X: bool>(s: &SlotIndexV1, w: &mut Writer, t: usize) -> Result<(), String> {
            s.pass1_body::<KM, FM, PIPE, true, X>(w, t)
        }
        match (self.pipeline, self.wide) {
            #[cfg(target_arch = "x86_64")]
            (true, true) => unsafe { wide::<KM, FM, true, X>(self, w, t) },
            #[cfg(target_arch = "x86_64")]
            (false, true) => unsafe { wide::<KM, FM, false, X>(self, w, t) },
            (true, _) => self.pass1_body::<KM, FM, true, false, X>(w, t),
            _ => self.pass1_body::<KM, FM, false, false, X>(w, t),
        }
    }

    /// Call f(slot, final entry) for each entry of thread t's chain of partition p, in order.
    #[inline(always)]
    fn chain<const X: bool, const NT: bool>(&self, t: usize, p: usize, mut f: impl FnMut(usize, u64)) {
        let w = &self.writers[t];
        let n = w.nch[p];
        if n == 0 {
            return;
        }
        let sh = self.shape;
        let per = per_line(X);
        let words = sh.cl as usize * LINE; // u64 words per chunk
        let (arena, next) = (self.arena.get(), self.next.get());
        let mut c = w.head[p] as usize;
        for i in 0..n {
            let len = if i + 1 == n { w.last[p] as usize } else { sh.cl as usize * per };
            if i + 1 < n {
                let nc = arena.as_ptr().wrapping_add(next[c] as usize * words);
                if NT {
                    // Non-temporal prefetch of the whole next chunk: into L1 without displacing the L2-resident output.
                    for l in 0..sh.cl as usize {
                        prefetch_nta(nc.wrapping_add(l * LINE));
                    }
                } else {
                    for l in 0..self.pf_lines {
                        prefetch(nc.wrapping_add(l * LINE));
                    }
                }
            }
            let chunk = &arena[c * words..(c + 1) * words];
            if X {
                for (l, line) in chunk.chunks_exact(LINE).enumerate().take(len.div_ceil(per)) {
                    let k = per.min(len - l * per);
                    let sw = unsafe { std::slice::from_raw_parts(line.as_ptr() as *const u16, 4 * LINE) };
                    for j in 0..k {
                        f(sh.slot10(sw[4 * per + j]), line[j]);
                    }
                }
            } else {
                for &e in &chunk[..len] {
                    f(sh.slot8(e), e);
                }
            }
            if i + 1 < n {
                c = next[c] as usize;
            }
        }
    }

    /// Pass 2 for partition p: counting sort by slot, capped, into the index; the directory. Returns entries kept.
    fn finish<const W: bool, const X: bool>(&self, p: usize, f: &mut Finisher) -> usize {
        let s = self.nslots();
        let cap = self.slot_cap as u32;
        let t0 = Instant::now();
        let Finisher { hist, cursor, out, stage, slots, .. } = f;
        hist.fill(0);
        let staged = self.stage && self.counts[p] as usize <= stage.len;
        if staged {
            let (st, mut k) = (stage.ptr, 0usize);
            for t in 0..self.threads {
                self.chain::<X, false>(t, p, |i, e| unsafe {
                    *hist.get_unchecked_mut(i) += 1;
                    *st.add(k) = e;
                    if X {
                        *slots.get_unchecked_mut(k) = i as u16;
                    }
                    k += 1;
                });
            }
        } else {
            for t in 0..self.threads {
                self.chain::<X, false>(t, p, |i, _| unsafe { *hist.get_unchecked_mut(i) += 1 });
            }
        }
        let t1 = Instant::now();
        let dir = unsafe { Output(self.dir.ptr).slice(p * (s + 1), s + 1) };
        let mut acc = 0u32;
        for i in 0..s {
            dir[i] = acc;
            cursor[i] = acc;
            acc += hist[i].min(cap);
            hist[i] = acc;
        }
        dir[s] = acc;
        let n = acc as usize;
        if n > out.len {
            // Only with slot_cap * 2^sb above the buffer: cannot happen (sized at construction).
            return 0;
        }
        let o = out.ptr;
        if self.pair && !staged {
            // Cursor and end of a slot side by side: one cache line access per placed entry.
            let ce = &mut f.pairs;
            for i in 0..s {
                ce[i] = [dir[i], dir[i + 1]];
            }
            for t in 0..self.threads {
                self.chain::<X, false>(t, p, |i, e| unsafe {
                    let [c, end] = *ce.get_unchecked(i);
                    if c < end {
                        *o.add(c as usize) = e;
                        ce.get_unchecked_mut(i)[0] = c + 1;
                    }
                });
            }
        }
        let Finisher { hist, cursor, stage, slots, .. } = f;
        let mut place = |i: usize, e: u64| {
            let c = unsafe { *cursor.get_unchecked(i) };
            if c < unsafe { *hist.get_unchecked(i) } {
                unsafe {
                    *o.add(c as usize) = e;
                    *cursor.get_unchecked_mut(i) = c + 1;
                }
            }
        };
        if self.pair && !staged {
        } else if staged {
            let st = &stage.get()[..self.counts[p] as usize];
            if X {
                for (k, &e) in st.iter().enumerate() {
                    place(unsafe { *slots.get_unchecked(k) } as usize, e);
                }
            } else {
                for &e in st {
                    place(self.shape.slot8(e), e);
                }
            }
        } else {
            for t in 0..self.threads {
                if self.nta {
                    self.chain::<X, true>(t, p, &mut place);
                } else {
                    self.chain::<X, false>(t, p, &mut place);
                }
            }
        }
        let t2 = Instant::now();
        let dst = unsafe { self.index.ptr.add(self.base[p] as usize) };
        for l in 0..n.div_ceil(LINE) {
            unsafe { stream_line::<W>(dst.add(l * LINE), o.add(l * LINE)) };
        }
        let t3 = Instant::now();
        for (a, d) in self.finish_ns.iter().zip([t1 - t0, t2 - t1, t3 - t2]) {
            a.fetch_add(d.as_nanos() as usize, Ordering::Relaxed);
        }
        n
    }

    /// Pass 2 for partition p by two counting passes, uncapped: count the partition's 64 top-slot-bit buckets (a
    /// walk of its chains), scatter it into them (a second walk, into a cache-resident staging buffer), then per
    /// bucket (~2,400 entries, in L1) count and place by the remaining slot bits into the output buffer, whose
    /// lines go to the index with non-temporal stores. dir[s] = start of slot s; lookups take its first slot_cap.
    fn finish_radix<const W: bool, const X: bool>(&self, p: usize, f: &mut Finisher) -> usize {
        const TOP: u32 = 6;
        let sb = self.shape.sb;
        let low = sb - TOP;
        let s = self.nslots();
        let n = self.counts[p] as usize;
        let t0 = Instant::now();
        let Finisher { small, hist, cursor, out, stage, .. } = f;
        if n > stage.len || n > out.len {
            return usize::MAX; // caller falls back to finish()
        }
        let (st, sm) = (stage.ptr, small.as_mut_ptr());
        // top: the buckets' output offsets (prefix sums); base: where each bucket sits in the staging buffer.
        let mut top = [0u32; (1 << TOP) + 1];
        let mut base = [0u32; 1 << TOP];
        let mut at = [0u32; 1 << TOP];
        let mut done = false;
        // One walk: fixed-capacity staging buckets (1.25x the mean + 64); a partition whose buckets overflow (a hot
        // key) is redone below with a counting walk first.
        let bcap = (n >> TOP) + (n >> (TOP + 2)) + 64;
        if self.one_walk && bcap << TOP <= stage.len {
            for b in 0..1 << TOP {
                base[b] = (b * bcap) as u32;
                at[b] = base[b];
            }
            for t in 0..self.threads {
                self.chain::<X, false>(t, p, |i, e| unsafe {
                    let b = i >> low;
                    let c = *at.get_unchecked(b) as usize;
                    if c < (b + 1) * bcap {
                        *st.add(c) = e;
                        if X {
                            *sm.add(c) = (i & ((1 << low) - 1)) as u8;
                        }
                    }
                    *at.get_unchecked_mut(b) = c as u32 + 1;
                });
            }
            done = (0..1 << TOP).all(|b| (at[b] - base[b]) as usize <= bcap);
            if done {
                for b in 0..1 << TOP {
                    top[b + 1] = top[b] + (at[b] - base[b]);
                }
            }
        }
        if !done {
            top = [0u32; (1 << TOP) + 1];
            for t in 0..self.threads {
                self.chain::<X, false>(t, p, |i, _| top[(i >> low) + 1] += 1);
            }
            for b in 0..1 << TOP {
                top[b + 1] += top[b];
            }
            base.copy_from_slice(&top[..1 << TOP]);
            at.copy_from_slice(&top[..1 << TOP]);
            for t in 0..self.threads {
                self.chain::<X, false>(t, p, |i, e| unsafe {
                    let b = i >> low;
                    let c = *at.get_unchecked(b) as usize;
                    *st.add(c) = e;
                    if X {
                        *sm.add(c) = (i & ((1 << low) - 1)) as u8;
                    }
                    *at.get_unchecked_mut(b) = c as u32 + 1;
                });
            }
        }
        let t1 = Instant::now();
        let dir = unsafe { Output(self.dir.ptr).slice(p * (s + 1), s + 1) };
        let (o, sh) = (out.ptr, self.shape);
        let m = (1usize << low) - 1;
        let lh = &mut hist[..1 << low];
        let cur = &mut cursor[..1 << low];
        for b in 0..1usize << TOP {
            let (lo, len, sbase) = (top[b] as usize, (top[b + 1] - top[b]) as usize, base[b] as usize);
            let es = unsafe { std::slice::from_raw_parts(st.add(sbase), len) };
            let ss = unsafe { std::slice::from_raw_parts(sm.add(sbase), if X { len } else { 0 }) };
            let slot_lo = |k: usize, e: u64| if X { unsafe { *ss.get_unchecked(k) as usize } } else { sh.slot8(e) & m };
            lh.fill(0);
            for (k, &e) in es.iter().enumerate() {
                unsafe { *lh.get_unchecked_mut(slot_lo(k, e)) += 1 };
            }
            let mut acc = lo as u32;
            for j in 0..=m {
                dir[(b << low) + j] = acc;
                cur[j] = acc;
                acc += lh[j];
            }
            for (k, &e) in es.iter().enumerate() {
                unsafe {
                    let j = slot_lo(k, e);
                    let c = *cur.get_unchecked(j);
                    *o.add(c as usize) = e;
                    *cur.get_unchecked_mut(j) = c + 1;
                }
            }
        }
        dir[s] = n as u32;
        let t2 = Instant::now();
        let dst = unsafe { self.index.ptr.add(self.base[p] as usize) };
        for l in 0..n.div_ceil(LINE) {
            unsafe { stream_line::<W>(dst.add(l * LINE), o.add(l * LINE)) };
        }
        let t3 = Instant::now();
        for (a, d) in self.finish_ns.iter().zip([t1 - t0, t2 - t1, t3 - t2]) {
            a.fetch_add(d.as_nanos() as usize, Ordering::Relaxed);
        }
        n
    }

    fn finish_any<const W: bool, const X: bool>(&self, p: usize, f: &mut Finisher) -> usize {
        if self.radix {
            let n = self.finish_radix::<W, X>(p, f);
            if n != usize::MAX {
                return n.min(self.capped(p));
            }
        }
        self.finish::<W, X>(p, f)
    }

    /// Entries of partition p that lookups can reach (at most slot_cap per slot), from its directory.
    fn capped(&self, p: usize) -> usize {
        let s = self.nslots();
        let d = &self.dir.get()[p * (s + 1)..(p + 1) * (s + 1)];
        d.windows(2).map(|w| ((w[1] - w[0]) as usize).min(self.slot_cap)).sum()
    }

    fn pass2(&mut self) -> usize {
        let parts = 1usize << self.shape.pb;
        let mut order: Vec<usize> = (0..parts).collect();
        order.sort_unstable_by_key(|&p| std::cmp::Reverse(self.counts[p]));
        let kept = AtomicUsize::new(0);
        let fins: Vec<std::sync::Mutex<Finisher>> = std::mem::take(&mut self.finishers).into_iter().map(std::sync::Mutex::new).collect();
        let this = &*self;
        let wide = this.wide;
        order.par_iter().for_each(|&p| {
            let i = rayon::current_thread_index().unwrap_or(0) % fins.len();
            let mut f = fins[i].lock().unwrap();
            #[cfg(target_arch = "x86_64")]
            #[target_feature(enable = "avx512f")]
            unsafe fn wide_finish<const X: bool>(s: &SlotIndexV1, p: usize, f: &mut Finisher) -> usize {
                s.finish_any::<true, X>(p, f)
            }
            let n = match (wide, this.shape.x) {
                #[cfg(target_arch = "x86_64")]
                (true, true) => unsafe { wide_finish::<true>(this, p, &mut f) },
                #[cfg(target_arch = "x86_64")]
                (true, false) => unsafe { wide_finish::<false>(this, p, &mut f) },
                (_, true) => this.finish_any::<false, true>(p, &mut f),
                _ => this.finish_any::<false, false>(p, &mut f),
            };
            kept.fetch_add(n, Ordering::Relaxed);
        });
        fence();
        self.finishers = fins.into_iter().map(|m| m.into_inner().unwrap()).collect();
        kept.into_inner()
    }

}

/// Split a big mapping into 256 MB VMAs by flagging every other piece MADV_DONTDUMP (no effect on paging or huge
/// pages): Slurm's periodic smaps walk holds a VMA's lock while it walks it, and a fault in a 100 GB VMA would wait
/// for the whole walk (about a second).
fn split<T>(m: Map<T>) -> std::io::Result<Map<T>> {
    const PIECE: usize = 1 << 28;
    let bytes = m.len * std::mem::size_of::<T>();
    let start = (m.ptr as usize + 4095) & !4095;
    for offset in (PIECE..bytes.saturating_sub(start - m.ptr as usize)).step_by(2 * PIECE) {
        let len = PIECE.min(bytes - offset - (start - m.ptr as usize));
        if unsafe { libc::madvise((start + offset) as *mut libc::c_void, len, libc::MADV_DONTDUMP) } != 0 {
            return Err(std::io::Error::last_os_error());
        }
    }
    Ok(m)
}

fn w_last(last: &mut [u32], p: usize, end_line: u32, cl: u32, per: usize, in_last_line: usize) {
    // end_line is one past the chain's last written line; its chunk starts at (end_line - 1) / cl * cl.
    let start = (end_line - 1) / cl * cl;
    last[p] = (end_line - 1 - start) * per as u32 + in_last_line as u32;
}


// 13fc6f12's new() defaults (the recommended configuration), with the port's levels, bits and caps.
const LEVELS: [usize; 2] = [10, 20];
const BITS: [u32; 2] = [12, 6];
const PARTITION_BITS: u32 = 16;
const SLOT_BITS: u32 = 14;
const SLOT_CAP: usize = 288;
const CAP: usize = 288;
const CHUNK_LINES: u32 = 32;
const KEY_MODE: u8 = KEY_MUL;
const FP_MODE: u8 = FP_POLY;
const PIPELINE: bool = false;
const AVX512: bool = true;
const ENTRY_BYTES: usize = 8;
const NULL: bool = false;
const PF_LINES: usize = 1;
const STAGE: bool = false;
const STAGE_ENTRIES: usize = 1 << 21;
const BLOCK: bool = true;
const RADIX: bool = false;
const ONE_WALK: bool = false;
const PAIR: bool = false;
const NTA: bool = false;
const TOKENS_PER_SHARD: usize = 101_000_000;

#[pymethods]
impl SlotIndexV1 {
    /// Off the clock: 13fc6f12's new() for the first `tokens` tokens of `files` on `threads` threads pinned to `cpus`,
    /// then its prefault() (every buffer, and the corpus pages).
    #[staticmethod]
    fn create(py: Python<'_>, files: Vec<String>, tokens: usize, threads: usize, cpus: Vec<usize>) -> PyResult<Self> {
        let (levels, bits) = (LEVELS.to_vec(), BITS.to_vec());
        let (partition_bits, slot_bits, slot_cap, cap, chunk_lines) = (PARTITION_BITS, SLOT_BITS, SLOT_CAP, CAP, CHUNK_LINES);
        let (key_mode, fp_mode, pipeline, avx512, entry_bytes, null) = (KEY_MODE, FP_MODE, PIPELINE, AVX512, ENTRY_BYTES, NULL);
        let (pf_lines, stage, stage_entries, block, radix, one_walk, pair, nta) = (PF_LINES, STAGE, STAGE_ENTRIES, BLOCK, RADIX, ONE_WALK, PAIR, NTA);
        let x = match entry_bytes {
            8 => false,
            10 => true,
            _ => return Err(PyValueError::new_err("entry_bytes must be 8 or 10")),
        };
        let fb: u32 = bits.iter().sum();
        if levels.len() != bits.len() || levels.len() > MAX_LEVELS || bits.iter().any(|&b| b == 0 || b > 24) {
            return Err(PyValueError::new_err("need one bit width (1-24) per level, at most 4 levels"));
        }
        if levels.iter().any(|&l| l <= MIN || l >= RING) || levels.windows(2).any(|w| w[0] >= w[1]) {
            return Err(PyValueError::new_err("levels must ascend within (6, 128)"));
        }
        let tag_bits = if x { 48 - fb as i64 } else { 48 - slot_bits as i64 - fb as i64 };
        if !(8..=20).contains(&partition_bits) || !(4..=if x { 16 } else { 20 }).contains(&slot_bits) || tag_bits < 8 {
            return Err(PyValueError::new_err("need partition_bits in [8, 20], slot_bits in [4, 20] (16 with 10-byte entries), and >= 8 tag bits"));
        }
        if files.is_empty() || cap == 0 || slot_cap == 0 || slot_cap > u16::MAX as usize || !chunk_lines.is_power_of_two() || key_mode > 1 || fp_mode > 1 {
            return Err(PyValueError::new_err("need shards, cap, slot_cap in [1, 65535], power-of-two chunk_lines, modes 0/1"));
        }
        if threads == 0 {
            return Err(PyValueError::new_err("need at least one thread"));
        }
        let stride = (HEADER + TOKENS_PER_SHARD).next_multiple_of(2048);
        let pool = std::sync::Arc::new(pinned_pool(threads, &cpus, "slotv1")?);
        let index = py.detach(|| -> PyResult<Self> {
            let corpus = Corpus::map(&files, stride, tokens)?;
            let ranges = corpus.ranges.clone();
            let parts = 1usize << partition_bits;
            let ch = chunk_lines as usize * per_line(x); // entries per chunk
            // Thread t's chunk region: its share's entries in whole chunks plus one partial chunk per partition.
            let mut writers = Vec::with_capacity(threads);
            let mut next_id = 1usize; // chunk 0 is never used (cur 0 means no chunk)
            let mut total = 0usize;
            for t in 0..threads {
                let n: usize = pieces(&ranges, t, threads).iter().map(|&(_, _, s, e)| e - s).sum();
                total += n;
                let size = n.div_ceil(ch) + parts + 1;
                writers.push(Writer {
                    buf: Map::anon(parts * LINE)?,
                    used: vec![0; parts],
                    cur: vec![0; parts],
                    head: vec![0; parts],
                    nch: vec![0; parts],
                    last: vec![0; parts],
                    lo: next_id as u32,
                    hi: (next_id + size) as u32,
                    entries: 0,
                });
                next_id += size;
            }
            if next_id * chunk_lines as usize >= u32::MAX as usize {
                return Err(PyValueError::new_err("arena exceeds u32 line indexes"));
            }
            let s = 1usize << slot_bits;
            let out_len = (s * slot_cap).min(total + LINE).next_multiple_of(LINE);
            let finishers = (0..threads)
                .map(|_| -> std::io::Result<Finisher> {
                    Ok(Finisher {
                        hist: vec![0; s],
                        cursor: vec![0; s],
                        out: Map::anon(out_len)?,
                        stage: Map::anon(if stage || radix { stage_entries } else { 1 })?,
                        slots: vec![0; if stage && x { stage_entries } else { 0 }],
                        small: vec![0; if radix && x { stage_entries } else { 0 }],
                        pairs: vec![[0; 2]; s],
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Self {
                hs: Hasher::new(&levels, &bits),
                levels,
                bits,
                cap,
                slot_cap,
                key_mode,
                fp_mode,
                pipeline,
                wide: avx512 && wide_stores(),
                null,
                pf_lines: pf_lines.min(chunk_lines as usize),
                stage,
                block,
                radix,
                one_walk,
                pair,
                nta,
                finish_ns: Default::default(),
                shape: Shape { pb: partition_bits, sb: slot_bits, fb, cl: chunk_lines, x },
                threads,
                arena: split(Map::anon(next_id * chunk_lines as usize * LINE)?)?,
                next: Map::anon(next_id)?,
                writers,
                finishers,
                index: split(Map::anon(total + parts * LINE)?)?,
                dir: split(Map::anon(parts * (s + 1))?)?,
                base: vec![0; parts],
                counts: vec![0; parts],
                ready: false,
                entries: 0,
                kept: 0,
                chunks_used: 0,
                corpus,
                ranges,
                pool,
            })
        })?;
        // 13fc6f12's prefault(), on the build pool.
        let ix = &index;
        py.detach(move || -> std::io::Result<()> {
            ix.pool.install(move || {
                let index = ix;
                let w = libc::MADV_POPULATE_WRITE;
                populate(index.arena.ptr as usize, index.arena.len * 8, w)?;
                populate(index.next.ptr as usize, index.next.len * 4, w)?;
                populate(index.index.ptr as usize, index.index.len * 8, w)?;
                populate(index.dir.ptr as usize, index.dir.len * 4, w)?;
                for wr in &index.writers {
                    populate(wr.buf.ptr as usize, wr.buf.len * 8, w)?;
                }
                for f in &index.finishers {
                    populate(f.out.ptr as usize, f.out.len * 8, w)?;
                    populate(f.stage.ptr as usize, f.stage.len * 8, w)?;
                }
                index.corpus.populate(&index.ranges)
            })
        })
        .map_err(|e| PyValueError::new_err(format!("prefault: {e}")))?;
        Ok(index)
    }

    /// On the clock: 13fc6f12's build(), pass 1 (scan), layout (partition bases), pass 2 (finish), on the pinned
    /// pool. Returns entries (kept), dropped (by the slot cap), bytes (index + directory as allocated), the arena's
    /// bytes as allocated and used, and the original's phase times (seconds; the finish_*_cpu are CPU seconds per
    /// thread).
    fn build(&mut self, py: Python<'_>) -> PyResult<HashMap<String, f64>> {
        if self.ready {
            return Err(PyValueError::new_err("already built"));
        }
        let pool = self.pool.clone();
        let phases = py.detach(|| {
            pool.install(|| -> Result<Vec<(String, f64)>, String> {
                let t0 = Instant::now();
                let mut writers = std::mem::take(&mut self.writers);
                let (km, fm) = (self.key_mode, self.fp_mode);
                let this = &*self;
                writers.par_iter_mut().enumerate().try_for_each(|(t, w)| match (km, fm) {
                    (KEY_MUL, FP_POLY) => this.pass1::<KEY_MUL, FP_POLY>(w, t),
                    (KEY_MUL, _) => this.pass1::<KEY_MUL, FP_MUL>(w, t),
                    (_, FP_POLY) => this.pass1::<KEY_MIX, FP_POLY>(w, t),
                    _ => this.pass1::<KEY_MIX, FP_MUL>(w, t),
                })?;
                self.writers = writers;
                let t1 = Instant::now();
                let parts = 1usize << self.shape.pb;
                let ch = (self.shape.cl as usize * per_line(self.shape.x)) as u64;
                let mut at = 0u64;
                self.entries = 0;
                self.chunks_used = 0;
                for p in 0..parts {
                    let mut n = 0u64;
                    for w in &self.writers {
                        if w.nch[p] > 0 {
                            n += (w.nch[p] as u64 - 1) * ch + w.last[p] as u64;
                        }
                    }
                    self.counts[p] = n;
                    self.base[p] = at;
                    at += n.next_multiple_of(LINE as u64);
                }
                for w in &self.writers {
                    self.entries += w.entries;
                    self.chunks_used += w.nch.iter().map(|&c| c as usize).sum::<usize>();
                }
                if at as usize > self.index.len {
                    return Err("index overflow".into());
                }
                let t2 = Instant::now();
                self.kept = self.pass2();
                let t3 = Instant::now();
                self.ready = true;
                let per_thread = |i: usize| self.finish_ns[i].swap(0, Ordering::Relaxed) as f64 * 1e-9 / self.threads as f64;
                Ok(vec![
                    ("scan".into(), (t1 - t0).as_secs_f64()),
                    ("layout".into(), (t2 - t1).as_secs_f64()),
                    ("finish".into(), (t3 - t2).as_secs_f64()),
                    ("finish_hist_cpu".into(), per_thread(0)),
                    ("finish_place_cpu".into(), per_thread(1)),
                    ("finish_copy_cpu".into(), per_thread(2)),
                ])
            })
        });
        let phases = phases.map_err(PyValueError::new_err)?;
        let mut out: HashMap<String, f64> = phases.into_iter().map(|(k, v)| (format!("{k}_s"), v)).collect();
        let ch = self.shape.cl as usize * LINE * 8;
        out.insert("entries".into(), self.kept as f64);
        out.insert("dropped".into(), (self.entries - self.kept) as f64);
        out.insert("bytes".into(), (self.index.len * 8 + self.dir.len * 4) as f64);
        out.insert("index_bytes".into(), (self.index.len * 8) as f64);
        out.insert("directory_bytes".into(), (self.dir.len * 4) as f64);
        out.insert("arena_bytes".into(), (self.arena.len * 8) as f64);
        out.insert("arena_used_bytes".into(), (self.chunks_used * ch) as f64);
        out.insert("threads".into(), self.threads as f64);
        Ok(out)
    }

    /// Off the clock, verification: (entries kept, digest of every partition's directory, digest of every partition's
    /// reachable entries, i.e. index[base[p]..base[p] + dir[p][2^sb]]), order-sensitive.
    fn digest(&self, py: Python<'_>) -> PyResult<(usize, u64, u64)> {
        if !self.ready {
            return Err(PyValueError::new_err("not built"));
        }
        let s = self.nslots();
        let mix = |h: u64, v: u64| key(h ^ v.wrapping_mul(P)).wrapping_add(v);
        Ok(py.detach(|| {
            self.pool.install(|| {
                let (dir, index) = (self.dir.get(), self.index.get());
                let per: Vec<(usize, u64, u64)> = (0..1usize << self.shape.pb)
                    .into_par_iter()
                    .map(|p| {
                        let d = &dir[p * (s + 1)..(p + 1) * (s + 1)];
                        let n = d[s] as usize;
                        let b = self.base[p] as usize;
                        let hd = d.iter().fold(p as u64, |h, &v| mix(h, v as u64));
                        let he = index[b..b + n].iter().fold(p as u64, |h, &v| mix(h, v));
                        (n, key(hd), key(he))
                    })
                    .collect();
                per.iter().fold((0, 0, 0), |(n, a, b), &(m, x, y)| (n + m, mix(a, x), mix(b, y)))
            })
        }))
    }
}
