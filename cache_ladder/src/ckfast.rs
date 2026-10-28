//! FastChecksumIndex: the corpus-free checksum index build with its speed options, ported from
//! modded-nanogpt commit 6f334e2b (branch ckfast), exact_match/src/ckfast.rs (with the helpers it used from that
//! commit's exact_match/src/lib.rs), fixed to its best measured option set "best-dir"
//! (artifacts/sep29-ckfast-bench.py: hash=fast scatter=line flush=128 sortnt=1 fastcount=1 order=dir bits=12 cap=64,
//! no threshold, numa=0, pipeline=0; artifacts/sep29-ckfast-miso4-17658915.txt: 19.3 s at 16 threads, 18.8 s at 32,
//! over 10.26B tokens with six levels). Only that configuration's code paths are kept; the others are dropped.
//!
//! The index: every position with a 6-gram before it (inside its BOS segment) and a next token gets an entry
//! tag u32 | fp6 u16 | next u16 | one 12-bit fingerprint per context length in `LEVELS`, packed little-endian. The
//! 6-gram key's top 12 bits pick one of 4096 partitions, its next 30 bits are the tag (the top 16 of which index the
//! partition's directory), and fp6 is the key's low 16 bits. A level fingerprint is the top 12 bits of
//! (P-polynomial hash of the context) * M_k (odd), computed from the prefix-hash ring the 6-gram key already keeps
//! (2 multiplies per level), 0 if the segment is shorter than the context (the hash's 0 moved to 1). Built in three
//! phases on the pool's threads, each thread owning an equal share of the corpus:
//!   count    each thread counts its share's entries per partition (one loop, token check fused);
//!   scatter  partition p holds thread 0's segment, thread 1's, ... (line-aligned); each thread hashes its share
//!            and appends to per-partition 192-byte buffers flushed 128 bytes (whole lines) at a time with
//!            non-temporal stores;
//!   sort     partitions taken dynamically (work stealing): gather by the tag's top 8 bits into 256 buckets through
//!            write-combining buffers with non-temporal stores, then a stable 8-bit counting sort of each bucket by
//!            the next 8 tag bits (the directory's 16 bits in all; buckets of <= 256 entries by the full tag), in a
//!            cache-resident buffer, copied to the index with non-temporal stores; the directory is read off that
//!            pass's histogram. A lookup scans its directory slot for the tag and keeps the first 64 matches.
//!
//! Adaptations (everything else is the original's code):
//! - levels = [10, 20] (the record SlotIndex's contexts) instead of the default [8, 12, 18, 26, 40, 64]; field width 12
//!   bits as measured. Entries are 8 + ceil(2 * 12 / 8) = 11 bytes (best-dir had 17). The levels stay runtime data of
//!   the Hasher, as in the original; the field width stays a compile-time constant, as there.
//! - The pool: `pinned_pool(threads, cpus, ...)` from lib.rs, instead of the process's global rayon pool
//!   (RAYON_NUM_THREADS). Every phase, and the untimed prefault, runs in it.
//! - Input: the first `tokens` tokens of `files`. The original's reader (Corpus::map: read-only private mappings of
//!   whole shards, `stride` = 101M + 512 tokens apart in one reserved range) is kept, with the last shard's range cut
//!   so the ranges hold exactly `tokens` tokens. Contexts never cross shards, as in the original.
//! - The entries and the directory are MAP_SHARED mappings of /dev/shm files, as in the original, at a unique path
//!   /dev/shm/cache-ladder-ckfast-<pid>-<n>.{entries,directory}; as in the original each file is unlinked right after
//!   it is mapped (in create()), so dropping the index frees the memory and nothing is left in /dev/shm.
//! - Timing split as the original's benchmark: create() is its constructor + prefault() (`alloc`, off the clock:
//!   mapping the shards, allocating and touching the entries, directory and per-thread buffers, MADV_POPULATE_READ
//!   of the corpus pages); build() is count + scatter + sort (`build`, on the clock).
//! - Debug only: query() (the original's lookup, check6 on, for one input stream) and digest() (a hash of every
//!   partition's sorted entries and directory), to check the port against the original.
use crate::{pinned_pool, BOS};
use numpy::{ndarray::Array2, IntoPyArray, PyReadonlyArray1};
use pyo3::{exceptions::PyValueError, prelude::*};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::os::fd::AsRawFd;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

// --- The best-dir option set and the port's levels.
const LEVELS: [usize; 2] = [10, 20];
const FP_BITS: u32 = 12; // field width (compile-time, as the original's b = 12 engine)
const S: usize = 8 + (LEVELS.len() * FP_BITS as usize).div_ceil(8); // entry bytes: 11
const CAP: usize = 64; // lookup: first matches kept per slot
const FLUSH: usize = 128; // bytes per non-temporal flush, line scatter and sortnt gather
const TOKENS_PER_SHARD: usize = 101_000_000; // the original's default (the corpus mapping's stride)

// --- Constants of the original lib.rs.
const STOP: u16 = u16::MAX;
const P: u64 = 0x9e3779b185ebca87;
const HEADER: usize = 512; // a shard file's header, in tokens
const BITS: usize = 12; // partition bits
const PARTITIONS: usize = 1 << BITS;
const TAG_MASK: u64 = (1 << 30) - 1;
const DIR_BITS: u32 = 16;
const DIR: usize = (1 << DIR_BITS) + 1;
const PREFETCH_BATCH: usize = 16;

// --- Constants of the original ckfast.rs.
const MIN: usize = 6;
const RING: usize = 128; // prefix hashes kept by a scan, a power of two above the longest context
const MASK: usize = RING - 1;
const MAX_LEVELS: usize = 8;
const BURST: usize = 16; // ChecksumIndex's burst, in entries (only sizes the index's slack here)
const LINE: usize = 64;
const SMALL_BYTES: usize = 1 << 19; // each of the two cache-resident sort buffers

// ===================================================================================== helpers of the original lib.rs

fn key(mut h: u64) -> u64 {
    h ^= h >> 30;
    h = h.wrapping_mul(0xbf58476d1ce4e5b9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94d049bb133111eb);
    h ^ (h >> 31)
}

fn fence() {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_sfence()
    };
}

#[inline(always)]
fn prefetch<T>(ptr: *const T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
        _mm_prefetch(ptr as *const i8, _MM_HINT_T0);
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = ptr;
}

fn prefault<T>(v: &mut [T]) {
    let bytes = unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, std::mem::size_of_val(v)) };
    bytes.par_chunks_mut(1 << 24).for_each(|chunk| {
        for i in (0..chunk.len()).step_by(4096) {
            unsafe { std::ptr::write_volatile(chunk.as_mut_ptr().add(i), 0) };
        }
    });
}

/// Calls emit(end, hash) for each 6-gram (n-gram) of `tokens` inside a BOS segment: the original's lookup windows.
fn windows(tokens: &[u16], n: usize, mut emit: impl FnMut(usize, u64)) {
    let (mut hash, mut available) = (0u64, 0usize);
    let power = P.wrapping_pow(n as u32);
    for (i, &token) in tokens.iter().enumerate() {
        if token == STOP {
            hash = 0;
            available = 0;
            continue;
        }
        if token == BOS {
            hash = 0;
            available = 0;
        }
        hash = hash.wrapping_mul(P).wrapping_add(token as u64 + 1);
        if available >= n {
            hash = hash.wrapping_sub((tokens[i - n] as u64 + 1).wrapping_mul(power));
        }
        available += 1;
        if available >= n {
            emit(i + 1, hash);
        }
    }
}

fn top_tokens(tokens: &mut Vec<i32>) -> Vec<(i32, i32)> {
    tokens.sort_unstable();
    let mut counts: Vec<(i32, i32)> = Vec::new();
    for &token in tokens.iter() {
        if counts.last().is_some_and(|&(t, _)| t == token) {
            counts.last_mut().unwrap().1 += 1;
        } else {
            counts.push((token, 1));
        }
    }
    counts.sort_unstable_by_key(|&(token, count)| (std::cmp::Reverse(count), token));
    counts.truncate(4);
    counts
}

/// The top-4 of `tokens` (count desc, token asc) as a row: tokens, then counts, -1 padded.
fn row(tokens: &mut Vec<i32>) -> [i32; 8] {
    let mut next = [-1; 8];
    for (i, (token, count)) in top_tokens(tokens).into_iter().enumerate() {
        next[i] = token;
        next[4 + i] = count;
    }
    next
}

/// The query's segments: split at its start, end and every BOS (the original's segments() with no boundaries).
fn segments(x: &[u16]) -> Vec<(usize, usize)> {
    let mut starts = vec![0, x.len()];
    starts.extend(x.iter().enumerate().filter_map(|(i, &v)| (v == BOS).then_some(i)));
    starts.sort_unstable();
    starts.dedup();
    starts.windows(2).map(|w| (w[0], w[1])).collect()
}

/// Partition, directory index and 30-bit tag of a key.
fn slot(h: u64) -> (usize, usize, u64) {
    let p = (h >> (64 - BITS)) as usize;
    let tag = (h >> (64 - BITS - 30)) & TAG_MASK;
    (p, p * DIR + (tag >> (30 - DIR_BITS)) as usize, tag)
}

/// A mapping of memory: a shared file (here /dev/shm, as the original) or private anonymous memory.
struct Map<T> {
    ptr: *mut T,
    len: usize,
}
unsafe impl<T> Send for Map<T> {}
unsafe impl<T> Sync for Map<T> {}

impl<T> Map<T> {
    fn open(path: &str, len: usize) -> std::io::Result<Self> {
        let bytes = len * std::mem::size_of::<T>();
        let file = std::fs::OpenOptions::new().read(true).write(true).create(true).open(path)?;
        file.set_len(bytes as u64)?;
        let ptr = unsafe {
            libc::mmap(std::ptr::null_mut(), bytes, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_SHARED, file.as_raw_fd(), 0)
        };
        if ptr == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error());
        }
        // Split the mapping into 256 MB VMAs by flagging every other piece. A page-table walk of this process
        // (Slurm reads smaps every 30 s) holds the mmap lock for a whole VMA; one 100 GB VMA stalls every page
        // fault and mmap in the process for about a second, a 256 MB one for a few ms.
        const PIECE: usize = 1 << 28;
        for offset in (PIECE..bytes).step_by(2 * PIECE) {
            let len = PIECE.min(bytes - offset);
            if unsafe { libc::madvise(ptr.cast::<u8>().add(offset).cast(), len, libc::MADV_NOHUGEPAGE) } != 0 {
                return Err(std::io::Error::last_os_error());
            }
        }
        Ok(Self { ptr: ptr.cast(), len })
    }

    /// Private memory in transparent huge pages, zero until written.
    fn anon(len: usize) -> std::io::Result<Self> {
        let bytes = (len * std::mem::size_of::<T>()).max(1);
        let ptr = unsafe {
            libc::mmap(std::ptr::null_mut(), bytes, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, -1, 0)
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

    fn get_mut(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
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

fn open_shard(path: &str) -> PyResult<(File, usize)> {
    let mut file = File::open(path)?;
    let mut header = [0u8; 1024];
    file.read_exact(&mut header)?;
    let word = |i: usize| i32::from_le_bytes(header[i..i + 4].try_into().unwrap());
    let n = word(8);
    if word(0) != 20240520 || word(4) != 1 || n < 0 || file.metadata()?.len() != 1024 + n as u64 * 2 {
        return Err(PyValueError::new_err(format!("invalid shard: {path}")));
    }
    Ok((file, n as usize))
}

/// Read-only private mappings of shard files, `stride` tokens apart in one reserved range: token k of file i sits
/// at position i * stride + HEADER + k, and ranges[i] spans file i's tokens (the port: the last file's range cut so
/// the ranges hold exactly `tokens` tokens). The page cache is the corpus, so nothing is copied.
struct Corpus {
    ptr: *mut u16,
    len: usize,
    ranges: Vec<(usize, usize)>,
}
unsafe impl Send for Corpus {}
unsafe impl Sync for Corpus {}

impl Corpus {
    fn map(files: &[String], tokens: usize, stride: usize) -> PyResult<Self> {
        let mut used = Vec::new();
        let mut left = tokens;
        for path in files {
            if left == 0 {
                break;
            }
            let (file, n) = open_shard(path)?;
            if HEADER + n > stride {
                return Err(PyValueError::new_err("shard exceeds tokens_per_shard"));
            }
            let take = n.min(left);
            left -= take;
            used.push((file, n, take));
        }
        if left > 0 {
            return Err(PyValueError::new_err(format!("the files hold {} tokens, fewer than {tokens}", tokens - left)));
        }
        let len = used.len().max(1) * stride;
        let flags = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE;
        let ptr = unsafe { libc::mmap(std::ptr::null_mut(), len * 2, libc::PROT_READ, flags, -1, 0) };
        if ptr == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error().into());
        }
        let mut corpus = Self { ptr: ptr.cast(), len, ranges: Vec::new() };
        for (i, (file, n, take)) in used.into_iter().enumerate() {
            let at = unsafe { corpus.ptr.add(i * stride) };
            let flags = libc::MAP_PRIVATE | libc::MAP_FIXED;
            let mapped = unsafe { libc::mmap(at.cast(), (HEADER + n) * 2, libc::PROT_READ, flags, file.as_raw_fd(), 0) };
            if mapped == libc::MAP_FAILED {
                return Err(std::io::Error::last_os_error().into());
            }
            corpus.ranges.push((i * stride + HEADER, i * stride + HEADER + take));
        }
        Ok(corpus)
    }

    fn get(&self) -> &[u16] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Map the pages of `ranges` into this process ahead of use.
    fn populate(&self, ranges: &[(usize, usize)]) -> std::io::Result<()> {
        ranges.iter().try_for_each(|&(lo, hi)| populate(self.ptr as usize + lo * 2, (hi - lo) * 2, libc::MADV_POPULATE_READ))
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

// ======================================================================================== the original ckfast.rs

/// Per-level constants of the fingerprints (hash=fast).
struct Hasher {
    n: usize,
    b: u32,
    len: [usize; MAX_LEVELS],
    pow: [u64; MAX_LEVELS], // P^L
    mul: [u64; MAX_LEVELS], // odd multiplier
    shift: [u32; MAX_LEVELS],
    p6: u64,
}

impl Hasher {
    fn new(levels: &[usize], bits: usize) -> Self {
        let mut h = Self {
            n: levels.len(),
            b: bits as u32,
            len: [0; MAX_LEVELS],
            pow: [0; MAX_LEVELS],
            mul: [0; MAX_LEVELS],
            shift: [0; MAX_LEVELS],
            p6: P.wrapping_pow(MIN as u32),
        };
        for (k, &l) in levels.iter().enumerate() {
            h.len[k] = l;
            h.shift[k] = (k * bits) as u32;
            h.mul[k] = key(0x2545f4914f6cdd1d ^ (k as u64 + 1).wrapping_mul(P)) | 1;
            h.pow[k] = P.wrapping_pow(l as u32);
        }
        h
    }

    /// Level k's 64-bit value from the P prefix hashes at the context's end (g) and start (rg).
    #[inline(always)]
    fn level(&self, k: usize, g: u64, rg: u64) -> u64 {
        g.wrapping_sub(rg.wrapping_mul(self.pow[k])).wrapping_mul(self.mul[k])
    }
}

/// The top `b` bits of a level fingerprint, 0 (the invalid marker) moved to 1.
#[inline(always)]
fn top(v: u64, b: u32) -> u64 {
    (v >> (64 - b)).max(1)
}

#[inline(always)]
fn tag(e: &[u8; S]) -> u32 {
    u32::from_le_bytes([e[0], e[1], e[2], e[3]])
}

#[inline(always)]
fn fields(e: &[u8; S]) -> u128 {
    let mut b = [0u8; 16];
    b[..S - 8].copy_from_slice(&e[8..]);
    u128::from_le_bytes(b)
}

unsafe fn ents<'a>(base: *mut u8, byte: usize, len: usize) -> &'a mut [[u8; S]] {
    std::slice::from_raw_parts_mut(base.add(byte).cast::<[u8; S]>(), len)
}

/// Non-temporal copy of whole 16-byte units; dst 16-byte aligned. Call `fence` before others read it.
#[inline(always)]
unsafe fn stream(dst: *mut u8, src: *const u8, bytes: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::{__m128i, _mm_loadu_si128, _mm_stream_si128};
        for i in (0..bytes).step_by(16) {
            _mm_stream_si128(dst.add(i).cast::<__m128i>(), _mm_loadu_si128(src.add(i).cast()));
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    std::ptr::copy_nonoverlapping(src, dst, bytes);
}

/// Copy with non-temporal stores for the whole lines of dst, ordinary ones for the partial lines at either end.
unsafe fn nt_copy(dst: *mut u8, src: *const u8, bytes: usize) {
    let head = ((LINE - dst as usize % LINE) % LINE).min(bytes);
    std::ptr::copy_nonoverlapping(src, dst, head);
    let body = (bytes - head) / LINE * LINE;
    stream(dst.add(head), src.add(head), body);
    std::ptr::copy_nonoverlapping(src.add(head + body), dst.add(head + body), bytes - head - body);
}

/// One stable counting-sort pass of src into dst on `digit` (below hist.len()); hist ends as each digit's end.
#[inline(always)]
fn radix(src: &[[u8; S]], dst: &mut [[u8; S]], digit: impl Fn(&[u8; S]) -> usize, hist: &mut [usize]) {
    hist.fill(0);
    for x in src {
        hist[digit(x)] += 1;
    }
    let mut sum = 0;
    for a in hist.iter_mut() {
        let n = *a;
        *a = sum;
        sum += n;
    }
    for x in src {
        let d = digit(x);
        dst[hist[d]] = *x;
        hist[d] += 1;
    }
}

/// Line-aligned writer (scatter=line): per partition a byte buffer of `lb` bytes, `flush` bytes (whole lines)
/// streamed at a time to a line-aligned destination, the rest (under one entry) moved to the buffer's front.
struct LineSink {
    buf: *mut u8,
    lb: usize,
    flush: usize,
    used: Vec<usize>,
    at: Vec<usize>,
    out: *mut u8,
}

impl LineSink {
    #[inline(always)]
    fn push(&mut self, p: usize, head: u64, fields: u128) {
        let u = self.used[p];
        unsafe {
            let b = self.buf.add(p * self.lb);
            // 24 bytes from u: the buffer's slack takes what runs past the entry.
            b.add(u).cast::<u64>().write_unaligned(head);
            b.add(u + 8).cast::<u128>().write_unaligned(fields);
            let u = u + S;
            if u >= self.flush {
                stream(self.out.add(self.at[p]), b, self.flush);
                self.at[p] += self.flush;
                std::ptr::copy_nonoverlapping(b.add(self.flush), b, 32);
                self.used[p] = u - self.flush;
            } else {
                self.used[p] = u;
            }
        }
    }

    fn finish(&mut self) {
        for p in 0..PARTITIONS {
            unsafe { std::ptr::copy_nonoverlapping(self.buf.add(p * self.lb), self.out.add(self.at[p]), self.used[p]) };
        }
        fence();
    }
}

/// Scan one piece (lo, hi, start, end) of the corpus and push the entry of every position start < pos <= end with a
/// 6-gram in its segment and a next token. B: the field width as a compile-time constant.
#[inline(always)]
fn scan<const B: u32>(corpus: &[u16], (lo, hi, start, end): (usize, usize, usize, usize), hs: &Hasher, sink: &mut LineSink) {
    let begin = start.saturating_sub(hs.len[hs.n - 1]).max(lo);
    let mut rg = [0u64; RING];
    let (mut g, mut seg) = (0u64, begin);
    let (n, b) = (hs.n, B);
    for i in begin..end {
        let token = corpus[i];
        if token == BOS {
            seg = i;
        } else if token == STOP {
            seg = i + 1;
        }
        g = g.wrapping_mul(P).wrapping_add(token as u64 + 1);
        let pos = i + 1;
        rg[pos & MASK] = g;
        if pos <= start || pos >= hi || pos < seg + MIN || corpus[pos] == STOP {
            continue;
        }
        let at6 = (pos - MIN) & MASK;
        let h = key(g.wrapping_sub(rg[at6].wrapping_mul(hs.p6)));
        let p = (h >> (64 - BITS)) as usize;
        let tag = (h >> (64 - BITS - 30)) & TAG_MASK;
        let fp6 = h & 0xffff;
        let avail = pos - seg;
        let mut v = 0u128;
        for k in 0..MAX_LEVELS {
            if k == n {
                break;
            }
            let l = hs.len[k];
            let at = pos.wrapping_sub(l) & MASK;
            let x = top(hs.level(k, g, rg[at]), b);
            let x = if avail >= l { x } else { 0 };
            v |= (x as u128) << (k as u32 * b);
        }
        sink.push(p, tag | fp6 << 32 | (corpus[pos] as u64) << 48, v);
    }
}

/// The count of one piece, as scan's positions, into counts[partition]; false if the piece holds an invalid token.
fn count_piece(corpus: &[u16], (lo, hi, start, end): (usize, usize, usize, usize), p6: u64, counts: &mut [usize]) -> bool {
    let begin = start.saturating_sub(MIN).max(lo);
    let mut rg = [0u64; 8];
    let (mut g, mut seg, mut bad) = (0u64, begin, false);
    for i in begin..end {
        let token = corpus[i];
        bad |= token > BOS && token != STOP;
        if token == BOS {
            seg = i;
        } else if token == STOP {
            seg = i + 1;
        }
        g = g.wrapping_mul(P).wrapping_add(token as u64 + 1);
        let pos = i + 1;
        rg[pos & 7] = g;
        if pos <= start || pos >= hi || pos < seg + MIN || corpus[pos] == STOP {
            continue;
        }
        let h = key(g.wrapping_sub(rg[(pos - MIN) & 7].wrapping_mul(p6)));
        counts[(h >> (64 - BITS)) as usize] += 1;
    }
    !bad
}

/// A thread's private buffers.
struct Local {
    counts: Vec<usize>,
    wc: Map<u8>,
    tmp: Map<u8>,
    small: Map<u8>,
    small2: Map<u8>,
    gwc: Map<u8>,
}

fn touch(v: &mut [u8]) {
    for i in (0..v.len()).step_by(4096) {
        unsafe { std::ptr::write_volatile(v.as_mut_ptr().add(i), 0) };
    }
}

#[derive(Clone, Copy)]
struct Ptr<T>(*mut T);
unsafe impl<T> Sync for Ptr<T> {}
unsafe impl<T> Send for Ptr<T> {}

struct Engine {
    hs: Hasher,
    corpus: Corpus,
    ranges: Vec<(usize, usize)>,
    entries: Map<u8>,
    directory: Map<u32>,
    locals: Vec<Mutex<Local>>,
    counts: Vec<Vec<usize>>,
    seg: Vec<Vec<usize>>, // [thread][partition] byte offset of the thread's segment
    offsets: Vec<usize>,  // entry index of each partition
    lengths: Vec<usize>,
    limits: [usize; 3], // largest bucket sorted by comparison, through the small buffers; largest partition in the scratch
    tmp_bytes: usize,
    threads: usize,
    local_bytes: usize,
}

static INSTANCE: AtomicUsize = AtomicUsize::new(0);

impl Engine {
    const SMALL: usize = SMALL_BYTES / S;

    /// The original's constructor; call inside the pool (`threads` is its size).
    fn new(path: &str, files: &[String], tokens: usize, threads: usize) -> PyResult<Self> {
        let stride = (HEADER + TOKENS_PER_SHARD).next_multiple_of(2048);
        let corpus = Corpus::map(files, tokens, stride)?;
        let ranges = corpus.ranges.clone();
        let n: usize = ranges.iter().map(|&(lo, hi)| hi - lo).sum();
        let lb = FLUSH + LINE;
        let tmp_bytes = (2 * n.div_ceil(PARTITIONS) + 4096) * S + 256 * LINE;
        let hs = Hasher::new(&LEVELS, FP_BITS as usize);
        let locals = (0..threads)
            .map(|_| -> std::io::Result<Mutex<Local>> {
                Ok(Mutex::new(Local {
                    counts: vec![0; PARTITIONS],
                    wc: Map::anon(PARTITIONS * lb)?,
                    tmp: Map::anon(tmp_bytes)?,
                    small: Map::anon(SMALL_BYTES + LINE)?,
                    small2: Map::anon(SMALL_BYTES + LINE)?,
                    gwc: Map::anon(256 * (FLUSH + LINE))?,
                }))
            })
            .collect::<std::io::Result<Vec<_>>>()?;
        let local_bytes = threads * (PARTITIONS * lb + tmp_bytes + 2 * (SMALL_BYTES + LINE) + 256 * (FLUSH + LINE));
        let entries = (n + threads * PARTITIONS * BURST + PARTITIONS * LINE + LINE) * S;
        let e = Map::<u8>::open(&format!("{path}.entries"), entries)?;
        std::fs::remove_file(format!("{path}.entries"))?;
        let d = Map::<u32>::open(&format!("{path}.directory"), PARTITIONS * DIR)?;
        std::fs::remove_file(format!("{path}.directory"))?;
        Ok(Self {
            hs,
            corpus,
            ranges,
            entries: e,
            directory: d,
            locals,
            counts: vec![vec![0; PARTITIONS]; threads],
            seg: vec![vec![0; PARTITIONS]; threads],
            offsets: vec![0; PARTITIONS + 1],
            lengths: vec![0; PARTITIONS],
            limits: [256, Self::SMALL, usize::MAX],
            tmp_bytes,
            threads,
            local_bytes,
        })
    }

    /// The original's prefault (numa off); call inside the pool.
    fn prefault(&mut self) -> std::io::Result<()> {
        prefault(self.entries.get_mut());
        prefault(self.directory.get_mut());
        self.locals.par_iter().for_each(|l| {
            let mut l = l.lock().unwrap();
            let l = &mut *l;
            for m in [&mut l.wc, &mut l.tmp, &mut l.small, &mut l.small2, &mut l.gwc] {
                touch(m.get_mut());
            }
        });
        self.corpus.populate(&self.ranges)
    }

    /// fastcount: one loop per piece; call inside the pool.
    fn count(&mut self) -> Result<usize, String> {
        let (corpus, ranges, threads, locals) = (self.corpus.get(), &self.ranges, self.threads, &self.locals);
        let p6 = self.hs.p6;
        let results = rayon::broadcast(|ctx| {
            let t = ctx.index();
            let mut l = locals[t].lock().unwrap();
            let counts = &mut l.counts;
            counts.fill(0);
            let mut ok = true;
            for piece in pieces(ranges, t, threads) {
                ok &= count_piece(corpus, piece, p6, counts);
            }
            if !ok {
                return Err("invalid GPT-2 token".to_string());
            }
            Ok(())
        });
        results.into_iter().collect::<Result<Vec<()>, String>>()?;
        for (t, l) in self.locals.iter().enumerate() {
            self.counts[t].copy_from_slice(&l.lock().unwrap().counts);
        }
        Ok(self.counts.iter().flatten().sum())
    }

    /// Partition p holds thread 0's segment, thread 1's, ...: line-aligned segments in partitions that start at a
    /// multiple of 64 entries.
    fn layout(&mut self) -> Result<(), String> {
        let mut end = 0usize; // bytes
        for p in 0..PARTITIONS {
            end = end.next_multiple_of(LINE * S);
            self.offsets[p] = end / S;
            self.lengths[p] = 0;
            for t in 0..self.threads {
                let c = self.counts[t][p];
                end = end.next_multiple_of(LINE);
                self.seg[t][p] = end;
                end += c * S;
                self.lengths[p] += c;
            }
        }
        end = end.next_multiple_of(S);
        self.offsets[PARTITIONS] = end / S;
        if end > self.entries.len || self.lengths.iter().any(|&n| n > u32::MAX as usize) {
            return Err("partition layout exceeds the index".into());
        }
        Ok(())
    }

    /// scatter=line flush=128, hash=fast, b=12 (pipeline off); call inside the pool.
    fn scatter(&mut self) -> Result<(), String> {
        self.layout()?;
        let (corpus, ranges, threads, locals) = (self.corpus.get(), &self.ranges, self.threads, &self.locals);
        let (out, seg, hs) = (Ptr(self.entries.ptr), &self.seg, &self.hs);
        rayon::broadcast(|ctx| {
            let t = ctx.index();
            let out = out;
            let l = locals[t].lock().unwrap();
            let at = seg[t].clone();
            let used = vec![0usize; PARTITIONS];
            let mut sink = LineSink { buf: l.wc.ptr, lb: FLUSH + LINE, flush: FLUSH, used, at, out: out.0 };
            for piece in pieces(ranges, t, threads) {
                scan::<FP_BITS>(corpus, piece, hs, &mut sink);
            }
            sink.finish();
            drop(l);
        });
        Ok(())
    }

    /// Partitions taken dynamically (work stealing, numa off); call inside the pool.
    fn sort(&mut self) {
        let this = &*self;
        (0..PARTITIONS).into_par_iter().with_max_len(1).for_each(|p| {
            let i = rayon::current_thread_index().unwrap();
            let mut l = this.locals[i].lock().unwrap();
            this.sort_partition(p, &mut l);
            fence();
        });
    }

    /// Sort partition p by the directory's 16 bits (order=dir, sortnt), stably, and fill its directory.
    fn sort_partition(&self, p: usize, l: &mut Local) {
        let n = self.lengths[p];
        let base = self.entries.ptr;
        let part_byte = self.offsets[p] * S;
        let directory = unsafe { std::slice::from_raw_parts_mut(self.directory.ptr.add(p * DIR), DIR) };
        let segs = || (0..self.threads).map(move |t| (self.seg[t][p], self.counts[t][p]));
        if n * S + 256 * LINE > self.tmp_bytes || n > self.limits[2] {
            // Compact the segments to the partition's start and sort it by tag, which suits either order.
            let mut at = part_byte;
            for (off, len) in segs() {
                unsafe { std::ptr::copy(base.add(off), base.add(at), len * S) };
                at += len * S;
            }
            let part = unsafe { ents(base, part_byte, n) };
            part.sort_by_key(|e| tag(e));
            dir_scan(part, 0, 0, 1 << 16, directory);
            directory[1 << 16] = n as u32;
            return;
        }
        // Gather into 256 buckets by the tag's top 8 bits.
        let mut starts = [0usize; 257];
        for (off, len) in segs() {
            for e in unsafe { ents(base, off, len) }.iter() {
                starts[(tag(e) >> 22) as usize + 1] += 1;
            }
        }
        for b in 0..256 {
            starts[b + 1] += starts[b];
        }
        let tmp = l.tmp.ptr;
        let mut tb = [0usize; 256]; // bytes
        // sortnt: through write-combining buffers with non-temporal stores, buckets line-aligned in the scratch.
        {
            let mut at = 0;
            for b in 0..256 {
                tb[b] = at;
                at = (at + (starts[b + 1] - starts[b]) * S).next_multiple_of(LINE);
            }
            let (flush, lb, g) = (FLUSH, FLUSH + LINE, l.gwc.ptr);
            let mut used = [0usize; 256];
            let mut cur = tb;
            for (off, len) in segs() {
                for e in unsafe { ents(base, off, len) }.iter() {
                    let b = (tag(e) >> 22) as usize;
                    unsafe {
                        let w = g.add(b * lb);
                        w.add(used[b]).cast::<[u8; S]>().write_unaligned(*e);
                        used[b] += S;
                        if used[b] >= flush {
                            stream(tmp.add(cur[b]), w, flush);
                            cur[b] += flush;
                            std::ptr::copy_nonoverlapping(w.add(flush), w, 32);
                            used[b] -= flush;
                        }
                    }
                }
            }
            for b in 0..256 {
                unsafe { std::ptr::copy_nonoverlapping(g.add(b * lb), tmp.add(cur[b]), used[b]) };
            }
            fence();
        }
        let part = unsafe { ents(base, part_byte, n) };
        let small = unsafe { ents(l.small.ptr, 0, Self::SMALL) };
        let mut hist = vec![0usize; 2048];
        let dd = |e: &[u8; S]| (tag(e) >> 14 & 255) as usize;
        for b in 0..256 {
            let (lo, hi) = (starts[b], starts[b + 1]);
            let m = hi - lo;
            let dirb = &mut directory[b * 256..b * 256 + 256];
            let src = unsafe { ents(tmp, tb[b], m) };
            let dst = &mut part[lo..hi];
            if m == 0 {
                dirb.fill(lo as u32);
            } else if m <= self.limits[0] {
                dst.copy_from_slice(src);
                dst.sort_by_key(|e| tag(e));
                dir_scan(dst, lo, b * 256, 256, dirb);
            } else {
                // order=dir: one 8-bit pass on the directory's low 8 bits (no threshold: slots stay unsorted).
                let h = &mut hist[..256];
                if m <= self.limits[1] {
                    radix(src, &mut small[..m], dd, h);
                    unsafe { nt_copy(dst.as_mut_ptr().cast(), small.as_ptr().cast(), m * S) };
                } else {
                    radix(src, dst, dd, h);
                }
                for j in 0..256 {
                    dirb[j] = (lo + if j == 0 { 0 } else { h[j - 1] }) as u32;
                }
            }
        }
        directory[1 << 16] = n as u32;
    }

    /// The original's lookup (hash=fast, order=dir without threshold), for one segment `x`.
    fn lookup(&self, x: &[u16], check6: bool, length: &mut [i32], next: &mut [[i32; 8]], rejected: &mut [i32]) {
        let total = self.offsets[PARTITIONS];
        let entries = unsafe { ents(self.entries.ptr, 0, total) };
        let (directory, offsets, hs, cap) = (self.directory.get(), &self.offsets, &self.hs, CAP);
        let (b, n, mask) = (hs.b, hs.n, (1u128 << hs.b) - 1);
        let mut items = Vec::with_capacity(x.len());
        windows(x, MIN, |offset, h| items.push((offset, key(h))));
        let mut g = vec![0u64; x.len() + 1];
        for (i, &t) in x.iter().enumerate() {
            g[i + 1] = g[i].wrapping_mul(P).wrapping_add(t as u64 + 1);
        }
        let mut found = [(0usize, 0usize); PREFETCH_BATCH];
        let mut best_tokens = Vec::with_capacity(cap);
        let mut q = [0u64; MAX_LEVELS];
        for batch in items.chunks(PREFETCH_BATCH) {
            for &(_, h) in batch {
                prefetch(directory.as_ptr().wrapping_add(slot(h).1));
            }
            for (i, &(_, h)) in batch.iter().enumerate() {
                let (p, d, _) = slot(h);
                found[i] = (offsets[p] + directory[d] as usize, offsets[p] + directory[d + 1] as usize);
                prefetch(entries.as_ptr().wrapping_add(found[i].0));
            }
            for (i, &(offset, h)) in batch.iter().enumerate() {
                let t = slot(h).2 as u32;
                let (lo, hi) = found[i];
                if lo == hi {
                    continue;
                }
                let mut ready = false;
                let fp6 = h as u16;
                let (mut best, mut dropped, mut taken) = (0, 0, 0);
                best_tokens.clear();
                let mut each = |e: &[u8; S]| {
                    if !ready {
                        ready = true;
                        for k in 0..n {
                            let l = hs.len[k];
                            q[k] = if l > offset { 0 } else { top(hs.level(k, g[offset], g[offset - l]), b) };
                        }
                    }
                    if check6 && u16::from_le_bytes([e[4], e[5]]) != fp6 {
                        dropped += 1;
                        return;
                    }
                    let v = fields(e);
                    let mut bucket = MIN;
                    for k in 0..n {
                        let s = ((v >> hs.shift[k]) & mask) as u64;
                        let l = hs.len[k];
                        if l > offset || s == 0 || s != q[k] {
                            break;
                        }
                        bucket = l;
                    }
                    if bucket > best {
                        best = bucket;
                        best_tokens.clear();
                    }
                    if bucket == best {
                        best_tokens.push(u16::from_le_bytes([e[6], e[7]]) as i32);
                    }
                };
                for e in &entries[lo..hi] {
                    if tag(e) == t {
                        each(e);
                        taken += 1;
                        if taken == cap {
                            break;
                        }
                    }
                }
                if taken == 0 {
                    continue;
                }
                length[offset - 1] = best as i32;
                next[offset - 1] = row(&mut best_tokens);
                rejected[offset - 1] = dropped;
            }
        }
    }
}

/// dir[k - first] = at + first index of `part` whose tag's top 16 bits are >= k, for k in first..first + len.
fn dir_scan(part: &[[u8; S]], at: usize, first: usize, len: usize, dir: &mut [u32]) {
    let mut k = first;
    for (i, e) in part.iter().enumerate() {
        let d = (tag(e) >> 14) as usize;
        while k <= d && k < first + len {
            dir[k - first] = (at + i) as u32;
            k += 1;
        }
    }
    while k < first + len {
        dir[k - first] = (at + part.len()) as u32;
        k += 1;
    }
}

/// The corpus-free checksum index of modded-nanogpt 6f334e2b's FastChecksumIndex, best-dir options, levels [10, 20].
#[pyclass]
pub struct FastChecksumIndex {
    engine: Engine,
    pool: rayon::ThreadPool,
    create_s: f64,
    entries: Option<usize>,
}

#[pymethods]
impl FastChecksumIndex {
    /// Untimed (the original's constructor + prefault): maps the shards, allocates the /dev/shm index files (unlinked
    /// at once) and the per-thread buffers, and touches all of them and the corpus pages.
    #[staticmethod]
    fn create(py: Python<'_>, files: Vec<String>, tokens: usize, threads: usize, cpus: Vec<usize>) -> PyResult<Self> {
        if threads == 0 {
            return Err(PyValueError::new_err("threads must be positive"));
        }
        let t0 = std::time::Instant::now();
        let pool = pinned_pool(threads, &cpus, "ckfast")?;
        let path = format!("/dev/shm/cache-ladder-ckfast-{}-{}", std::process::id(), INSTANCE.fetch_add(1, Ordering::Relaxed));
        let engine = py.detach(|| {
            pool.install(|| -> PyResult<Engine> {
                let mut e = Engine::new(&path, &files, tokens, threads)?;
                e.prefault()?;
                Ok(e)
            })
        })?;
        Ok(Self { engine, pool, create_s: t0.elapsed().as_secs_f64(), entries: None })
    }

    /// Timed: count, scatter, sort (the original's `build`).
    fn build(&mut self, py: Python<'_>) -> PyResult<HashMap<String, f64>> {
        if self.entries.is_some() {
            return Err(PyValueError::new_err("already built"));
        }
        let (engine, pool) = (&mut self.engine, &self.pool);
        let (entries, times) = py
            .detach(|| {
                pool.install(|| -> Result<(usize, [f64; 3]), String> {
                    let t0 = std::time::Instant::now();
                    let entries = engine.count()?;
                    let t1 = std::time::Instant::now();
                    engine.scatter()?;
                    let t2 = std::time::Instant::now();
                    engine.sort();
                    let t3 = std::time::Instant::now();
                    Ok((entries, [(t1 - t0).as_secs_f64(), (t2 - t1).as_secs_f64(), (t3 - t2).as_secs_f64()]))
                })
            })
            .map_err(PyValueError::new_err)?;
        self.entries = Some(entries);
        let e = &self.engine;
        let laid = e.offsets[PARTITIONS] * S;
        let directory = e.directory.len * 4;
        let tokens: usize = e.ranges.iter().map(|&(lo, hi)| hi - lo).sum();
        Ok(HashMap::from([
            ("entries".into(), entries as f64),
            ("bytes".into(), (laid + directory) as f64),
            ("entry_bytes".into(), S as f64),
            ("entries_bytes".into(), laid as f64),
            ("directory_bytes".into(), directory as f64),
            ("mapped_bytes".into(), (e.entries.len + directory + e.local_bytes) as f64),
            ("tokens".into(), tokens as f64),
            ("threads".into(), e.threads as f64),
            ("count_s".into(), times[0]),
            ("scatter_s".into(), times[1]),
            ("sort_s".into(), times[2]),
            ("create_s".into(), self.create_s),
        ]))
    }

    /// Debug: the original's query (check6 on) over one input stream split at BOS: (length, next [n, 8], rejected).
    fn query<'py>(&self, py: Python<'py>, inputs: PyReadonlyArray1<i32>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        if self.entries.is_none() {
            return Err(PyValueError::new_err("index has not been built"));
        }
        let inputs = inputs.as_slice()?;
        if inputs.iter().any(|&v| !(0..=BOS as i32).contains(&v)) {
            return Err(PyValueError::new_err("invalid GPT-2 token"));
        }
        let x: Vec<u16> = inputs.iter().map(|&v| v as u16).collect();
        let (engine, pool) = (&self.engine, &self.pool);
        let (length, next, rejected) = py.detach(|| {
            pool.install(|| {
                let results: Vec<_> = segments(&x)
                    .par_iter()
                    .map(|&(start, end)| {
                        let m = end - start;
                        let (mut length, mut next, mut rejected) = (vec![0; m], vec![[-1; 8]; m], vec![0; m]);
                        engine.lookup(&x[start..end], true, &mut length, &mut next, &mut rejected);
                        (start, length, next, rejected)
                    })
                    .collect();
                let (mut length, mut rejected) = (vec![0; x.len()], vec![0; x.len()]);
                let mut next = Array2::from_elem((x.len(), 8), -1);
                for (start, l, ids, r) in results {
                    length[start..start + l.len()].copy_from_slice(&l);
                    rejected[start..start + r.len()].copy_from_slice(&r);
                    next.as_slice_mut().unwrap()[start * 8..(start + ids.len()) * 8].copy_from_slice(ids.as_flattened());
                }
                (length, next, rejected)
            })
        });
        Ok(vec![length.into_pyarray(py).into_any(), next.into_pyarray(py).into_any(), rejected.into_pyarray(py).into_any()])
    }

    /// Debug: a 64-bit digest of every partition's sorted entries and directory (independent of the thread count).
    fn digest(&self, py: Python<'_>) -> PyResult<u64> {
        if self.entries.is_none() {
            return Err(PyValueError::new_err("index has not been built"));
        }
        let (e, pool) = (&self.engine, &self.pool);
        Ok(py.detach(|| {
            pool.install(|| {
                (0..PARTITIONS)
                    .into_par_iter()
                    .map(|p| {
                        let part = unsafe { ents(e.entries.ptr, e.offsets[p] * S, e.lengths[p]) };
                        let mut h = key(p as u64 ^ (part.len() as u64) << 20);
                        for x in part.iter() {
                            h = key(h ^ u64::from_le_bytes(x[..8].try_into().unwrap()));
                            h = key(h ^ fields(x) as u64);
                        }
                        for &d in &e.directory.get()[p * DIR..(p + 1) * DIR] {
                            h = h.wrapping_mul(P).wrapping_add(d as u64);
                        }
                        key(h)
                    })
                    .reduce(|| 0u64, |a, b| a.wrapping_add(b))
            })
        }))
    }
}
