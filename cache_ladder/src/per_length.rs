//! PerLengthTables: SlotIndex v1's build (slot_v1.rs, commit 13fc6f12) applied to the Sep 16 idea of one table per
//! context length: three slot tables, for the 6-, 10- and 20-token contexts, each keyed by its own context's hash.
//!
//! Entry (u64, one per table per corpus position whose k-token context lies in its shard, after the segment's latest
//! BOS but for a leading BOS): [ slot: sb | tag: 48 - sb | next token: 16 ]. The key of a k-token context is its
//! polynomial prefix-hash difference times an odd constant (the 6-gram's is v1's key, the 10- and 20-gram's v1's two
//! level multipliers); its top pb bits pick the partition (implicit in the entry), the next sb = 30 - pb the slot, so
//! a table has 2^30 slots. pb = 15 (32,768 partitions), or 16 above 2^34 positions (17.2B tokens).
//!
//! Build: the three tables one after another, each by v1's passes, into its own flat index and slot directory, the
//! arena reused:
//!   1. scan: each thread scans its contiguous share of the corpus in 256-position blocks (v1's serial prefix-hash
//!      loop two tokens per step, then the table's key in a loop that vectorizes) and appends each entry to its
//!      partition through a one-line write-combining buffer per partition; a full line goes out with one non-temporal
//!      store into the thread's current 64-line chunk for that partition (chained per (thread, partition)). No
//!      counting pass.
//!   2. layout: partition sizes and bases in the table's index (serial).
//!   3. finish: per partition (largest first, rayon par_iter), a counting sort by the slot bits: a histogram walk
//!      over the chains, a prefix sum capped at 288 entries per slot (the first in corpus order stay), a placement
//!      walk into a cache-resident buffer, non-temporal stores of it into the partition's region of the table's index,
//!      and the slot directory (u32 per slot).
//!
//! From v1: the entry but for its fingerprint fields, the hashes, the scan, the write-combining lines and chunk
//! chains, the capped counting sort, and the helpers (Map, populate, Corpus, pieces, split, Output, fence,
//! stream_line, Shape), with the configuration fixed at 13fc6f12's defaults and its option fields and their branches
//! dropped. From the Sep 16 per-length tables (bench_joint.cpp, this rung's earlier port): one table per length, so
//! the 10- and 20-token contexts are keyed by their own hash instead of being 12- and 6-bit fingerprints in the
//! 6-gram's entry; its count pass, 24-byte rows, full radix sort and mode reduction are gone (a table stores next
//! tokens, not modes).
//!
//! What changed from v1, besides the three tables:
//! - one scan per table, not one scan writing all three: three times v1's write-combining lines per thread would
//!   not stay in cache, and the arena holds one table's entries (8 bytes per token) instead of three;
//! - 2^15 partitions instead of 2^16 (2^16 again above 17.2B tokens), chunks of 64 lines instead of 32: v1's 4 MB of
//!   write-combining lines per thread is twice a core's L2 (the scan of the three tables took 0.6 s per billion
//!   tokens with them, 0.42 with 2 MB), while bigger partitions make the finish's placement walk leave the cache
//!   (2^15 partitions are slower than 2^16 above about 17B tokens); measured on miso1 (2026-10-10, 48 threads);
//! - a table keeps the first 288 entries per slot, so a hot 6-gram's longer contexts are kept beyond its first 288
//!   occurrences, while their own slots have room.
//!
//! Memory: the three indexes (8 bytes per token each) and the arena (8) at once, 32 bytes per token, plus the
//! directories (12.9 GB) and the arena's chunk slack (one chunk per thread and partition, 6.4 GB; 12.9 GB with 2^16
//! partitions). digest() and slots() are off the clock, for verification.
use crate::{pinned_pool, shards};
use pyo3::{exceptions::PyValueError, prelude::*};
use rayon::prelude::*;
use std::collections::HashMap;
use std::os::fd::AsRawFd;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

// From slot_v1.rs (13fc6f12's lib.rs).
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

    fn populate(&self) -> std::io::Result<()> {
        populate(self.ptr as usize, self.len * std::mem::size_of::<T>(), libc::MADV_POPULATE_WRITE)
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
/// at position i * stride + HEADER + k, and ranges[i] spans file i's tokens.
struct Corpus {
    ptr: *mut u16,
    len: usize,
    ranges: Vec<(usize, usize)>,
}
unsafe impl Send for Corpus {}
unsafe impl Sync for Corpus {}

impl Corpus {
    /// The first `tokens` tokens of `files` (crate::shards checks the headers and cuts the last one); each file is
    /// mapped up to its last used token.
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

fn wide_stores() -> bool {
    #[cfg(target_arch = "x86_64")]
    return std::arch::is_x86_feature_detected!("avx512f");
    #[cfg(not(target_arch = "x86_64"))]
    false
}

// The tables and v1's configuration.
const KS: [usize; 3] = [6, 10, 20];
const NK: usize = KS.len();
const RING: usize = 32; // prefix hashes kept by a scan, a power of two above the longest length
const LINE: usize = 8; // entries per 64-byte line
const KEY_BITS: u32 = 30; // partition and slot bits
const BIG: usize = 1 << 34; // positions above which a table has 2^16 partitions, not 2^15
const SLOT_CAP: u32 = 288;
const CL: u32 = 64; // lines per chunk
const TOKENS_PER_SHARD: usize = 101_000_000;

/// The key multiplier of table k: v1's 6-gram multiplier, then its level multipliers.
fn multiplier(k: usize) -> u64 {
    if k == 0 {
        key(0x6a09e667f3bcc909) | 1
    } else {
        key(0x2545f4914f6cdd1d ^ (k as u64).wrapping_mul(P)) | 1
    }
}

/// A table's split of the key's top 30 bits into partition and slot bits (v1's Shape).
#[derive(Clone, Copy)]
struct Shape {
    pb: u32,
    sb: u32,
}

impl Shape {
    fn parts(&self) -> usize {
        1 << self.pb
    }
    fn slots(&self) -> usize {
        1 << self.sb
    }
    #[inline(always)]
    fn part(&self, h: u64) -> usize {
        (h >> (64 - self.pb)) as usize
    }
    /// The entry of a key and next token: [slot | tag | token], the partition bits shifted out.
    #[inline(always)]
    fn entry(&self, h: u64, token: u16) -> u64 {
        ((h << self.pb) & !0xffff) | token as u64
    }
    #[inline(always)]
    fn slot(&self, e: u64) -> usize {
        (e >> (64 - self.sb)) as usize
    }
}

/// v1's scan_block for one table (context length l, P^l, key multiplier): call emit(pos, key) for each pos = i + 1
/// with first < pos < stop whose l-token context lies after the latest BOS (or `begin`) but for a leading BOS.
/// Returns the largest token seen.
#[inline(always)]
fn scan_block(tokens: &[u16], l: usize, pw: u64, mul: u64, begin: usize, end: usize, first: usize, stop: usize, mut emit: impl FnMut(usize, u64)) -> u16 {
    const B: usize = 256;
    let p2 = P.wrapping_mul(P);
    let mut g = [0u64; RING + B]; // g[RING + j]: the prefix hash at pos = i0 + j + 1
    let mut av = [0u32; B]; // pos - latest BOS (or begin)
    let mut hh = [0u64; B];
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
            gc = gc.wrapping_mul(p2).wrapping_add(a0.wrapping_mul(P).wrapping_add(a1));
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
        let (cur, back) = (&g[RING..RING + B], &g[RING - l..RING - l + B]);
        for j in 0..B {
            hh[j] = cur[j].wrapping_sub(back[j].wrapping_mul(pw)).wrapping_mul(mul);
        }
        for j in 0..n {
            let pos = i0 + j + 1;
            if av[j] >= l as u32 && pos > first && pos < stop {
                emit(pos, hh[j]);
            }
        }
        g.copy_within(n..n + RING, 0);
        i0 += n;
    }
    worst
}

/// A thread's pass-1 state: one write-combining line per partition, and its chunk chains (reset per table).
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
    hist: Vec<u32>,
    cursor: Vec<u32>,
    out: Map<u64>,
}

/// A finished table: its flat index, slot directory (slots + 1 u32 per partition) and partition bases.
struct Table {
    index: Map<u64>,
    dir: Map<u32>,
    base: Vec<u64>,
    entries: usize,
    kept: usize,
}

#[pyclass]
pub struct PerLengthTables {
    shape: Shape,
    corpus: Corpus,
    ranges: Vec<(usize, usize)>,
    threads: usize,
    pool: std::sync::Arc<rayon::ThreadPool>,
    wide: bool,
    arena: Map<u64>,
    next: Map<u32>,
    writers: Vec<Writer>,
    finishers: Vec<std::sync::Mutex<Finisher>>,
    counts: Vec<u64>,
    tables: Vec<Table>,
    finish_ns: [AtomicUsize; 3],
    chunks_used: usize,
    ready: bool,
}

impl PerLengthTables {
    /// Pass 1 of table k over thread t's share.
    #[inline(always)]
    fn pass1_body<const W: bool>(&self, k: usize, w: &mut Writer, t: usize) -> Result<(), String> {
        let (corpus, sh) = (self.corpus.get(), self.shape);
        let (l, pw, mul) = (KS[k], P.wrapping_pow(KS[k] as u32), multiplier(k));
        let longest = KS[NK - 1];
        let (arena, next, cl) = (self.arena.ptr, self.next.ptr, CL);
        let buf = w.buf.ptr;
        let Writer { used, cur, head, nch, last, lo, hi, entries, .. } = w;
        used.fill(0);
        cur.fill(0);
        nch.fill(0);
        let mut alloc = *lo;
        let limit = *hi;
        let mut overflow = false;
        let mut count = 0usize;
        let mut worst = 0u16;
        let mut append = |p: usize, e: u64, overflow: &mut bool| unsafe {
            let u = *used.get_unchecked(p) as usize;
            *buf.add(p * LINE + u) = e;
            if u == LINE - 1 {
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
            let body = |pos: usize, h: u64| {
                count += 1;
                append(sh.part(h), sh.entry(h, corpus[pos]), &mut overflow);
            };
            worst = worst.max(scan_block(corpus, l, pw, mul, begin, end, start, hi_r, body));
        }
        // The partial lines, then each chain's last chunk fill.
        for p in 0..sh.parts() {
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
                last[p] = w_last(c + 1, cl, u);
            } else if cur[p] != 0 {
                last[p] = w_last(cur[p], cl, LINE);
            } else {
                last[p] = 0;
            }
        }
        fence();
        *entries = count;
        if worst > BOS {
            return Err("invalid GPT-2 token".into());
        }
        if overflow {
            return Err("chunk region overflow".into());
        }
        Ok(())
    }

    fn pass1(&self, k: usize, w: &mut Writer, t: usize) -> Result<(), String> {
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx512f,avx512dq,avx512vl,avx512bw")]
        unsafe fn wide(s: &PerLengthTables, k: usize, w: &mut Writer, t: usize) -> Result<(), String> {
            s.pass1_body::<true>(k, w, t)
        }
        #[cfg(target_arch = "x86_64")]
        if self.wide {
            return unsafe { wide(self, k, w, t) };
        }
        self.pass1_body::<false>(k, w, t)
    }

    /// Call f(slot, entry) for each entry of thread t's chain of partition p, in order.
    #[inline(always)]
    fn chain(&self, t: usize, p: usize, mut f: impl FnMut(usize, u64)) {
        let w = &self.writers[t];
        let n = w.nch[p];
        if n == 0 {
            return;
        }
        let (words, sh) = (CL as usize * LINE, self.shape); // u64 words per chunk
        let (arena, next) = (self.arena.get(), self.next.get());
        let mut c = w.head[p] as usize;
        for i in 0..n {
            let len = if i + 1 == n { w.last[p] as usize } else { words };
            if i + 1 < n {
                prefetch(arena.as_ptr().wrapping_add(next[c] as usize * words));
            }
            for &e in &arena[c * words..c * words + len] {
                f(sh.slot(e), e);
            }
            if i + 1 < n {
                c = next[c] as usize;
            }
        }
    }

    /// Pass 2 for partition p of a table: counting sort by slot, capped, into the table's index; the directory.
    /// Returns entries kept.
    fn finish<const W: bool>(&self, table: &Table, p: usize, f: &mut Finisher) -> usize {
        let t0 = Instant::now();
        let Finisher { hist, cursor, out } = f;
        hist.fill(0);
        for t in 0..self.threads {
            self.chain(t, p, |i, _| unsafe { *hist.get_unchecked_mut(i) += 1 });
        }
        let t1 = Instant::now();
        let s = self.shape.slots();
        let dir = unsafe { Output(table.dir.ptr).slice(p * (s + 1), s + 1) };
        let mut acc = 0u32;
        for i in 0..s {
            dir[i] = acc;
            cursor[i] = acc;
            acc += hist[i].min(SLOT_CAP);
            hist[i] = acc;
        }
        dir[s] = acc;
        let n = acc as usize;
        let o = out.ptr;
        for t in 0..self.threads {
            self.chain(t, p, |i, e| unsafe {
                let c = *cursor.get_unchecked(i);
                if c < *hist.get_unchecked(i) {
                    *o.add(c as usize) = e;
                    *cursor.get_unchecked_mut(i) = c + 1;
                }
            });
        }
        let t2 = Instant::now();
        let dst = unsafe { table.index.ptr.add(table.base[p] as usize) };
        for l in 0..n.div_ceil(LINE) {
            unsafe { stream_line::<W>(dst.add(l * LINE), o.add(l * LINE)) };
        }
        let t3 = Instant::now();
        for (a, d) in self.finish_ns.iter().zip([t1 - t0, t2 - t1, t3 - t2]) {
            a.fetch_add(d.as_nanos() as usize, Ordering::Relaxed);
        }
        n
    }

    fn pass2(&self, k: usize) -> usize {
        let mut order: Vec<usize> = (0..self.shape.parts()).collect();
        order.sort_unstable_by_key(|&p| std::cmp::Reverse(self.counts[p]));
        let kept = AtomicUsize::new(0);
        let table = &self.tables[k];
        order.par_iter().for_each(|&p| {
            let i = rayon::current_thread_index().unwrap_or(0) % self.finishers.len();
            let mut f = self.finishers[i].lock().unwrap();
            #[cfg(target_arch = "x86_64")]
            #[target_feature(enable = "avx512f")]
            unsafe fn wide_finish(s: &PerLengthTables, table: &Table, p: usize, f: &mut Finisher) -> usize {
                s.finish::<true>(table, p, f)
            }
            #[cfg(target_arch = "x86_64")]
            let n = if self.wide { unsafe { wide_finish(self, table, p, &mut f) } } else { self.finish::<false>(table, p, &mut f) };
            #[cfg(not(target_arch = "x86_64"))]
            let n = self.finish::<false>(table, p, &mut f);
            kept.fetch_add(n, Ordering::Relaxed);
        });
        fence();
        kept.into_inner()
    }

    /// Table k's scan, layout and finish; returns their times.
    fn build_table(&mut self, k: usize) -> Result<[f64; 3], String> {
        let t0 = Instant::now();
        let mut writers = std::mem::take(&mut self.writers);
        let this = &*self;
        writers.par_iter_mut().enumerate().try_for_each(|(t, w)| this.pass1(k, w, t))?;
        self.writers = writers;
        let t1 = Instant::now();
        let ch = (CL as usize * LINE) as u64;
        let (mut at, mut chunks) = (0u64, 0usize);
        let table = &mut self.tables[k];
        for p in 0..self.shape.parts() {
            let mut n = 0u64;
            for w in &self.writers {
                if w.nch[p] > 0 {
                    n += (w.nch[p] as u64 - 1) * ch + w.last[p] as u64;
                }
            }
            self.counts[p] = n;
            table.base[p] = at;
            at += n.next_multiple_of(LINE as u64);
        }
        for w in &self.writers {
            table.entries += w.entries;
            chunks += w.nch.iter().map(|&c| c as usize).sum::<usize>();
        }
        self.chunks_used = self.chunks_used.max(chunks);
        if at as usize > table.index.len {
            return Err("index overflow".into());
        }
        let t2 = Instant::now();
        self.tables[k].kept = self.pass2(k);
        let t3 = Instant::now();
        Ok([(t1 - t0).as_secs_f64(), (t2 - t1).as_secs_f64(), (t3 - t2).as_secs_f64()])
    }
}

/// Entries in the last chunk of a chain, from one past its last written line and the entries in that line.
fn w_last(end_line: u32, cl: u32, in_last_line: usize) -> u32 {
    let start = (end_line - 1) / cl * cl;
    (end_line - 1 - start) * LINE as u32 + in_last_line as u32
}

#[pymethods]
impl PerLengthTables {
    /// Off the clock: map the first `tokens` tokens of `files`, allocate the arena (one table's entries), the three
    /// indexes and directories and the threads' buffers for `threads` threads pinned to `cpus`, and fault them all in
    /// (v1's new() + prefault()).
    #[staticmethod]
    fn create(py: Python<'_>, files: Vec<String>, tokens: usize, threads: usize, cpus: Vec<usize>) -> PyResult<Self> {
        if files.is_empty() || threads == 0 {
            return Err(PyValueError::new_err("need shards and at least one thread"));
        }
        let stride = (HEADER + TOKENS_PER_SHARD).next_multiple_of(2048);
        let pool = std::sync::Arc::new(pinned_pool(threads, &cpus, "perlength")?);
        let index = py.detach(|| -> PyResult<Self> {
            let corpus = Corpus::map(&files, stride, tokens)?;
            let ranges = corpus.ranges.clone();
            let positions: usize = ranges.iter().map(|&(lo, hi)| hi - lo).sum();
            let pb = if positions > BIG { 16 } else { 15 };
            let shape = Shape { pb, sb: KEY_BITS - pb };
            let (parts, s) = (shape.parts(), shape.slots());
            let ch = CL as usize * LINE; // entries per chunk
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
            if next_id * CL as usize >= u32::MAX as usize {
                return Err(PyValueError::new_err("arena exceeds u32 line indexes"));
            }
            let out_len = (s * SLOT_CAP as usize).min(total + LINE).next_multiple_of(LINE);
            let finishers = (0..threads)
                .map(|_| -> std::io::Result<std::sync::Mutex<Finisher>> { Ok(std::sync::Mutex::new(Finisher { hist: vec![0; s], cursor: vec![0; s], out: Map::anon(out_len)? })) })
                .collect::<Result<Vec<_>, _>>()?;
            let tables = (0..NK)
                .map(|_| -> std::io::Result<Table> {
                    Ok(Table {
                        index: split(Map::anon(total + parts * LINE)?)?,
                        dir: split(Map::anon(parts * (s + 1))?)?,
                        base: vec![0; parts],
                        entries: 0,
                        kept: 0,
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Self {
                shape,
                threads,
                wide: wide_stores(),
                arena: split(Map::anon(next_id * CL as usize * LINE)?)?,
                next: Map::anon(next_id)?,
                writers,
                finishers,
                counts: vec![0; parts],
                tables,
                finish_ns: Default::default(),
                chunks_used: 0,
                ready: false,
                corpus,
                ranges,
                pool,
            })
        })?;
        let ix = &index;
        py.detach(move || -> std::io::Result<()> {
            ix.pool.install(move || {
                ix.arena.populate()?;
                ix.next.populate()?;
                for t in &ix.tables {
                    t.index.populate()?;
                    t.dir.populate()?;
                }
                for w in &ix.writers {
                    w.buf.populate()?;
                }
                for f in &ix.finishers {
                    f.lock().unwrap().out.populate()?;
                }
                ix.corpus.populate(&ix.ranges)
            })
        })
        .map_err(|e| PyValueError::new_err(format!("prefault: {e}")))?;
        Ok(index)
    }

    /// On the clock: the 6-, 10- and 20-gram tables, each by scan, layout and finish, on the pinned pool. Returns
    /// entries (kept, all tables) and dropped (by the slot cap), the same per table (entries6, dropped6, ...), bytes
    /// (indexes + directories as allocated), the arena's bytes as allocated and used (its fullest table), and the
    /// phase times summed over the tables and per table (seconds; the finish_*_cpu are CPU seconds per thread).
    fn build(&mut self, py: Python<'_>) -> PyResult<HashMap<String, f64>> {
        if self.ready {
            return Err(PyValueError::new_err("already built"));
        }
        let pool = self.pool.clone();
        let times = py.detach(|| pool.install(|| (0..NK).map(|k| self.build_table(k)).collect::<Result<Vec<_>, _>>()));
        let times = times.map_err(PyValueError::new_err)?;
        self.ready = true;
        let mut out = HashMap::new();
        for (i, phase) in ["scan", "layout", "finish"].iter().enumerate() {
            out.insert(format!("{phase}_s"), times.iter().map(|t| t[i]).sum());
            for (k, t) in KS.iter().zip(&times) {
                out.insert(format!("{phase}{k}_s"), t[i]);
            }
        }
        for (i, name) in ["hist", "place", "copy"].iter().enumerate() {
            out.insert(format!("finish_{name}_cpu_s"), self.finish_ns[i].swap(0, Ordering::Relaxed) as f64 * 1e-9 / self.threads as f64);
        }
        for (k, t) in KS.iter().zip(&self.tables) {
            out.insert(format!("entries{k}"), t.kept as f64);
            out.insert(format!("dropped{k}"), (t.entries - t.kept) as f64);
        }
        let (index, dir) = (self.tables.iter().map(|t| t.index.len * 8).sum::<usize>(), self.tables.iter().map(|t| t.dir.len * 4).sum::<usize>());
        out.insert("entries".into(), self.tables.iter().map(|t| t.kept).sum::<usize>() as f64);
        out.insert("dropped".into(), self.tables.iter().map(|t| t.entries - t.kept).sum::<usize>() as f64);
        out.insert("bytes".into(), (index + dir) as f64);
        out.insert("index_bytes".into(), index as f64);
        out.insert("directory_bytes".into(), dir as f64);
        out.insert("arena_bytes".into(), (self.arena.len * 8) as f64);
        out.insert("arena_used_bytes".into(), (self.chunks_used * CL as usize * LINE * 8) as f64);
        out.insert("partition_bits".into(), self.shape.pb as f64);
        out.insert("threads".into(), self.threads as f64);
        Ok(out)
    }

    /// Off the clock, verification: (entries kept, digest of every table's directories, digest of its reachable
    /// entries, i.e. index[base[p]..base[p] + dir[p][2^sb]]) per table, order-sensitive.
    fn digest(&self, py: Python<'_>) -> PyResult<Vec<(usize, u64, u64)>> {
        if !self.ready {
            return Err(PyValueError::new_err("not built"));
        }
        let mix = |h: u64, v: u64| key(h ^ v.wrapping_mul(P)).wrapping_add(v);
        let s = self.shape.slots();
        Ok(py.detach(|| {
            self.pool.install(|| {
                self.tables
                    .iter()
                    .map(|table| {
                        let (dir, index) = (table.dir.get(), table.index.get());
                        let per: Vec<(usize, u64, u64)> = (0..self.shape.parts())
                            .into_par_iter()
                            .map(|p| {
                                let d = &dir[p * (s + 1)..(p + 1) * (s + 1)];
                                let (n, b) = (d[s] as usize, table.base[p] as usize);
                                let hd = d.iter().fold(p as u64, |h, &v| mix(h, v as u64));
                                let he = index[b..b + n].iter().fold(p as u64, |h, &v| mix(h, v));
                                (n, key(hd), key(he))
                            })
                            .collect();
                        per.iter().fold((0, 0, 0), |(n, a, b), &(m, x, y)| (n + m, mix(a, x), mix(b, y)))
                    })
                    .collect()
            })
        }))
    }

    /// Off the clock, verification: table i's (0, 1, 2 for 6, 10, 20 tokens) non-empty slots in order, as their
    /// global ids (partition << sb | slot, the key's top 30 bits), their entry counts, and all its reachable entries
    /// in index order.
    fn slots<'py>(&self, py: Python<'py>, i: usize) -> PyResult<(Bound<'py, numpy::PyArray1<u32>>, Bound<'py, numpy::PyArray1<u32>>, Bound<'py, numpy::PyArray1<u64>>)> {
        if !self.ready || i >= NK {
            return Err(PyValueError::new_err("not built, or no such table"));
        }
        let table = &self.tables[i];
        let (dir, index) = (table.dir.get(), table.index.get());
        let (mut ids, mut lens, mut entries) = (Vec::new(), Vec::new(), Vec::new());
        let (sb, s) = (self.shape.sb, self.shape.slots());
        for p in 0..self.shape.parts() {
            let d = &dir[p * (s + 1)..(p + 1) * (s + 1)];
            for i in 0..s {
                if d[i + 1] > d[i] {
                    ids.push(((p << sb) | i) as u32);
                    lens.push(d[i + 1] - d[i]);
                }
            }
            let b = table.base[p] as usize;
            entries.extend_from_slice(&index[b..b + d[s] as usize]);
        }
        use numpy::IntoPyArray;
        Ok((ids.into_pyarray(py), lens.into_pyarray(py), entries.into_pyarray(py)))
    }
}
