//! SlotIndexBranchFree: the record's SlotIndex as of modded-nanogpt commit 51029638 ("Scale and accelerate
//! exact-match slot indexing", 2026-10-05, faster-larger-cache lineage): branch-free AVX-512 inner loops.
//!
//! This is that commit's exact_match/src/slot.rs build path, with the helpers it took from that commit's lib.rs
//! (mmap, Map, touch, fence, the AVX-512 check, MIN/LEVELS/STOP) copied in. It stores the same data as the final rung
//! (exact_match/src/slot.rs in this repo): one u64 entry per training position with a MIN-gram in its shard,
//!
//!   next token (16 bits) | level-10 fingerprint (12) | level-20 fingerprint (6) | key check (30) ,
//!
//! in 2^14 partitions x 2^14 buckets of `slots` entries (slots from the corpus size at LOAD 0.8), each entry in its
//! home bucket b1 while it has room, else in b2, else dropped. Build:
//!   1. scan: thread t preads the 64K-token pieces t, t + T, ..., hashes 4096 positions at a time (NH hashes with
//!      AVX-512F/BW/DQ/VL; not yet the IFMA hash), and scatters every entry through a per-thread write-combining burst
//!      of one cache line per partition into 2048-entry blocks of a private huge-page arena, with non-temporal stores.
//!   2. finish: per partition (largest first, handed out dynamically), place its entries into b1 / b2 with selects,
//!      staging only each bucket's filling line and streaming it to the table the moment it is complete.
//!
//! It predates three changes the final rung has, and has none of them: the arena's shared block counter in cache
//! lines of its own (1802a267; here it is a plain AtomicUsize field of the builder, next to the arena), mbind
//! MPOL_LOCAL on every mapping (f315854f; here mmap only madvises), and the IFMA multiply-shift hash (9ceb0b20).
//!
//! Adaptations (none touch the loops):
//! - Input: the first `tokens` tokens of `files` (crate::shards), not whole shards: the last shard's token count is
//!   cut, so its pieces stop there. The scan still preads the shard files as the original did.
//! - Threads and CPUs come from create's arguments; the pool is crate::pinned_pool (pins with a broadcast instead of
//!   the original's start handler: the same placement).
//! - The table, a new file in /dev/shm in the original (shared with the other ranks), is a memfd made in create
//!   (shared memory as well; miso's shmem_enabled is "never", so either is in 4 KB pages), freed on drop.
//! - create() is the original's create (allocate, map the table, fault in buffers, arena and table on the pool); the
//!   lookup pool, attach, query and release are dropped.
//! - build() returns statistics: place() also sums each partition's bucket fills (it reads every fill at its end
//!   anyway) for the kept count; the scan and finish phases are timed with Instant.
//! - Off the clock, for verification: table_path (the memfd as /proc/self/fd/N) to compare the table with the
//!   original's.
//!
//! Verified on miso4 (2026-10-09): the table is bit-identical to 51029638's (built from `git archive`) for 1 and 5
//! whole shards at 48 pinned threads (the table depends on the thread count, as in the original).
use crate::{pinned_pool, shards, Shard, HEADER_BYTES};
use pyo3::{exceptions::PyValueError, prelude::*};
use rayon::prelude::*;
use std::arch::x86_64::*;
use std::collections::HashMap;
use std::fs::File;
use std::hint::select_unpredictable;
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering::Relaxed};
use std::sync::{LazyLock, Mutex};
use std::time::Instant;

// From 51029638's lib.rs.
const STOP: u16 = u16::MAX;
const MIN: usize = 6;
const LEVELS: [usize; 2] = [10, 20];

/// The build loops are written for AVX-512 (the record's hosts have it).
static AVX512: LazyLock<bool> = LazyLock::new(|| {
    is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512dq")
        && is_x86_feature_detected!("avx512vl")
        && is_x86_feature_detected!("bmi2")
});

/// mmap read-write, in transparent huge pages, split into 256 MB VMAs (alternate ones flagged MADV_DONTDUMP) so that
/// a page-table walk (Slurm reads smaps every 30 s) holds the mmap lock only briefly per VMA. Not inherited by a fork.
/// (51029638's lib.rs: no mbind.)
fn mmap(at: *mut libc::c_void, bytes: usize, flags: libc::c_int, fd: libc::c_int) -> std::io::Result<*mut libc::c_void> {
    const VMA: usize = 1 << 28;
    let ptr = unsafe { libc::mmap(at, bytes, libc::PROT_READ | libc::PROT_WRITE, flags, fd, 0) };
    if ptr == libc::MAP_FAILED {
        return Err(std::io::Error::last_os_error());
    }
    unsafe { libc::madvise(ptr, bytes, libc::MADV_HUGEPAGE) };
    unsafe { libc::madvise(ptr, bytes, libc::MADV_DONTFORK) };
    for off in (VMA..bytes).step_by(2 * VMA) {
        unsafe { libc::madvise(ptr.cast::<u8>().add(off).cast(), VMA.min(bytes - off), libc::MADV_DONTDUMP) };
    }
    Ok(ptr)
}

/// A mapping of `len` T's, unmapped on drop.
struct Map<T> {
    ptr: *mut T,
    len: usize,
    bytes: usize,
}
unsafe impl<T> Send for Map<T> {}
unsafe impl<T> Sync for Map<T> {}

impl<T> Map<T> {
    /// Private memory, zero until written, starting at a huge page.
    fn new(len: usize) -> std::io::Result<Self> {
        const HUGE: usize = 1 << 21;
        let bytes = (len * std::mem::size_of::<T>()).max(1).next_multiple_of(HUGE);
        let flags = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE;
        let raw = mmap(std::ptr::null_mut(), bytes + HUGE, flags, -1)? as usize;
        let ptr = raw.next_multiple_of(HUGE);
        unsafe {
            libc::munmap(raw as *mut libc::c_void, ptr - raw);
            libc::munmap((ptr + bytes) as *mut libc::c_void, raw + HUGE - ptr);
        }
        Ok(Self { ptr: ptr as *mut T, len, bytes })
    }
}

impl<T> std::ops::Deref for Map<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl<T> std::ops::DerefMut for Map<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl<T> Drop for Map<T> {
    fn drop(&mut self) {
        unsafe { libc::munmap(self.ptr.cast(), self.bytes) };
    }
}

fn fence() {
    unsafe { _mm_sfence() };
}

/// Write one byte per page of `v`, so later writes take no page faults.
fn touch<T>(v: &mut [T]) {
    let bytes = v.as_mut_ptr() as *mut u8;
    for i in (0..std::mem::size_of_val(v)).step_by(4096) {
        unsafe { std::ptr::write_volatile(bytes.add(i), 0) };
    }
}

// 51029638's slot.rs from here on (build path).
const BITS: [u32; 2] = [12, 6]; // fingerprint bits per level
const SHIFTS: [u32; 2] = [16, 28]; // where each level's fingerprint sits in an entry
const CHECK_SHIFT: u32 = 34;
const KBITS: u32 = 64 - CHECK_SHIFT; // key check bits
const PBITS: u32 = 14; // partitions: 2^14
const NBITS: u32 = 14; // buckets per partition: 2^14
const BUCKETS: usize = 1 << (PBITS + NBITS);
const LOAD: f64 = 0.8; // entries per slot (10.26B tokens: 48 slots, 103 GB; 41.2B: 192, 412 GB)
const BACK: usize = LEVELS[1]; // tokens before a context end that its hash reads
const STEP: usize = 1 << 12; // positions hashed at once
const PIECE: usize = 1 << 16; // context ends per piece of the scan (one pread)
const BURST: usize = 8; // entries per write-combining burst and per staging line: one cache line
const BLOCK: usize = 2048; // entries per arena block, a multiple of BURST
const CHUNK: usize = 64; // entries between the queued line stores of the scatter and the placement
const NONE: u32 = u32::MAX;
const K0: u64 = 0x9e3779b97f4a7c15;
const K1: u64 = 0xc2b2ae3d27d4eb4f;
const K2: u64 = 0xff51afd7ed558ccd;

const fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

/// The multiplier of word w of level k.
const fn mul(k: u64, w: u64) -> u64 {
    splitmix((k + 1) << 32 | w) | 1
}

/// Per level, the words its fingerprint hashes as (offset o, multiplier): word q(j - o) holds the 4 tokens before
/// j - o, so level 10 covers the tokens 7-10 before the context end j and level 20 the tokens 11-20.
const LEVEL10: [(usize, u64); 1] = [(6, mul(0, 0))];
const LEVEL20: [(usize, u64); 3] = [(10, mul(1, 0)), (14, mul(1, 1)), (16, mul(1, 2))];
const FINALS: [u64; 2] = [splitmix(0xabcdef) | 1, splitmix(0xabcdef ^ 1) | 1];

#[inline(always)]
fn check(e: u64) -> u64 {
    (e >> CHECK_SHIFT) & ((1 << KBITS) - 1)
}

#[inline(always)]
fn b1(check: u64) -> usize {
    (check >> (KBITS - NBITS)) as usize
}

#[inline(always)]
fn b2(check: u64) -> usize {
    (check.wrapping_mul(K0) >> (64 - NBITS)) as usize
}

/// Bucket slots for a corpus of `tokens`: a multiple of BURST, below 256 (a bucket's fill is a byte).
fn slots_for(tokens: usize) -> PyResult<usize> {
    let slots = (tokens as f64 / (BUCKETS as f64 * LOAD) / BURST as f64).ceil() as usize * BURST;
    if slots >= 256 {
        return Err(PyValueError::new_err(format!("{tokens} tokens need {slots} slots per bucket, more than 248")));
    }
    Ok(slots.max(BURST))
}

/// pread exactly `buf` at byte `offset` of `file`.
fn pread<T>(file: &File, buf: &mut [T], offset: usize) -> std::io::Result<()> {
    let bytes = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, std::mem::size_of_val(buf)) };
    file.read_exact_at(bytes, offset as u64)
}

/// Scratch of one hashing step.
struct Hashes {
    q: Vec<u64>,
    e: Vec<u64>,
    part: Vec<u32>,
}

impl Hashes {
    fn new() -> Self {
        Self { q: vec![0; STEP + BACK + 8], e: vec![0; STEP], part: vec![0; STEP] }
    }

    /// The entries (with the next token if `next`) and partitions of the n contexts ending at t[j0..j0 + n]
    /// (exclusive ends): context r ends at t[j0 + r], which is its next token. Needs j0 >= BACK.
    fn hash(&mut self, t: &[u16], j0: usize, n: usize, next: bool) {
        assert!(*AVX512);
        unsafe { self.hash_avx512(t, j0, n, next) }
    }

    /// q[i] holds the 4 tokens before t[j0 - BACK + i] as one word, 8 words at a time by one word permute of 11
    /// tokens; then per position the key (an NH hash of its last 6 tokens, words j and j - 2), its partition and
    /// check, and the level fingerprints (NH terms with one 32-bit add for both halves, then a 32x32-bit multiply).
    #[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl")]
    unsafe fn hash_avx512(&mut self, t: &[u16], j0: usize, n: usize, next: bool) {
        const { assert!(STEP % 8 == 0 && CHECK_SHIFT + KBITS == 64 && SHIFTS[0] < 32 - BITS[0] && SHIFTS[1] >= 32 - BITS[1]) };
        let base = j0 - BACK;
        assert!(t.len() >= base + n + BACK - 1 + next as usize && n <= STEP);
        let (tp, qp, ep, pp) = (t.as_ptr(), self.q.as_mut_ptr(), self.e.as_mut_ptr(), self.part.as_mut_ptr());
        // Lane k of 8 takes words k..k + 4 of the 11 tokens from base + i - 4.
        static IDX: [u16; 32] = [0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6, 4, 5, 6, 7, 5, 6, 7, 8, 6, 7, 8, 9, 7, 8, 9, 10];
        let idx = _mm512_loadu_si512(IDX.as_ptr().cast());
        let mut i = 4;
        while i < n + BACK {
            let avail = t.len() - (base + i - 4);
            let mask = if avail >= 11 { 0x7ff } else { (1u32 << avail) - 1 };
            let w = _mm512_maskz_loadu_epi16(mask, tp.add(base + i - 4).cast());
            _mm512_storeu_si512(qp.add(i).cast(), _mm512_permutexvar_epi16(idx, w));
            i += 8;
        }
        let nh = |q: __m512i, m: u64| {
            let s = _mm512_add_epi32(q, _mm512_set1_epi64(m as i64));
            _mm512_mul_epu32(s, _mm512_srli_epi64::<32>(s))
        };
        // In asm: LLVM sees that only the low 32 product bits are used and turns vpmuludq into the 3-uop vpmullq.
        let fold = |a: __m512i, fin: u64| {
            let (x, f) = (_mm512_xor_si512(a, _mm512_srli_epi64::<32>(a)), _mm512_set1_epi64(fin as u32 as i64));
            let p: __m512i;
            std::arch::asm!("vpmuludq {p}, {x}, {f}", p = lateout(zmm_reg) p, x = in(zmm_reg) x, f = in(zmm_reg) f, options(pure, nomem, nostack));
            p
        };
        let (k2, keep) = (_mm512_set1_epi64(K2 as i64), _mm512_set1_epi64(!((1u64 << CHECK_SHIFT) - 1) as i64));
        let (m10, m20) = (_mm512_set1_epi64((((1u64 << BITS[0]) - 1) << SHIFTS[0]) as i64), _mm512_set1_epi64((((1u64 << BITS[1]) - 1) << SHIFTS[1]) as i64));
        const R10: u32 = 32 - BITS[0] - SHIFTS[0];
        const L20: u32 = SHIFTS[1] - (32 - BITS[1]);
        for r in (0..n).step_by(8) {
            let q = |o: usize| _mm512_loadu_si512(qp.add(BACK + r - o).cast());
            let mut h = _mm512_add_epi64(nh(q(0), K0), nh(q(2), K1));
            h = _mm512_xor_si512(h, _mm512_srli_epi64::<29>(h));
            h = _mm512_mullo_epi64(h, k2);
            h = _mm512_xor_si512(h, _mm512_srli_epi64::<32>(h));
            _mm256_storeu_si256(pp.add(r).cast(), _mm512_cvtepi64_epi32(_mm512_srli_epi64::<{ 64 - PBITS }>(h)));
            let x10 = fold(nh(q(LEVEL10[0].0), LEVEL10[0].1), FINALS[0]);
            let a20 = _mm512_add_epi64(_mm512_add_epi64(nh(q(LEVEL20[0].0), LEVEL20[0].1), nh(q(LEVEL20[1].0), LEVEL20[1].1)), nh(q(LEVEL20[2].0), LEVEL20[2].1));
            let x20 = fold(a20, FINALS[1]);
            let mut e = _mm512_or_si512(
                _mm512_and_si512(_mm512_slli_epi64::<PBITS>(h), keep),
                _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi64::<R10>(x10), m10), _mm512_and_si512(_mm512_slli_epi64::<L20>(x20), m20)),
            );
            if next {
                let lanes = (n - r).min(8);
                let nt = _mm_maskz_loadu_epi16(((1u32 << lanes) - 1) as u8, tp.add(j0 + r).cast());
                e = _mm512_or_si512(e, _mm512_cvtepu16_epi64(nt));
            }
            _mm512_storeu_si512(ep.add(r).cast(), e);
        }
    }
}

/// One scan thread's state: its bursts, the current block of each partition, its block log, and after the scan its
/// blocks by partition (partition p's are ks[off[p]..off[p + 1]], in allocation order).
struct Scatter {
    buf: Map<u64>,
    used: Map<u8>,
    block: Vec<u32>,
    fill: Vec<u32>,
    log: Vec<(u32, u32)>, // (partition, block) in allocation order
    local: std::ops::Range<usize>, // this thread's own blocks
    off: Vec<u32>,
    ks: Vec<u32>,
}

impl Scatter {
    /// Where partition p's next entries go: its current block, or a new one if that is full (a partition without a
    /// block has fill BLOCK, so one test covers both).
    #[inline(always)]
    fn dst(&mut self, p: usize, builder: &Builder) -> std::io::Result<*mut u64> {
        if self.fill[p] as usize == BLOCK {
            self.next_block(p, builder)?;
        }
        Ok(unsafe { builder.arena.ptr.add(self.block[p] as usize * BLOCK + self.fill[p] as usize) })
    }

    /// A new block: the thread's own, else one from the shared rest of the arena.
    #[cold]
    #[inline(never)]
    fn next_block(&mut self, p: usize, builder: &Builder) -> std::io::Result<()> {
        let b = self.local.next().unwrap_or_else(|| builder.overflow.fetch_add(1, Relaxed));
        if b >= builder.arena.len() / BLOCK {
            return Err(std::io::Error::other("scan arena is full"));
        }
        (self.block[p], self.fill[p]) = (b as u32, 0);
        self.log.push((p as u32, b as u32));
        Ok(())
    }

    /// Scatter entries `e` (partitions `part`) without a branch on a burst filling up: each entry is inserted into its
    /// burst line in a register (a 64 B load and a plain 64 B store, volatile: LLVM would make it a masked store, which
    /// a later load of the line cannot forward from), the line is also copied to a small queue whose index advances
    /// (by the lane bit) only when the burst is full, and every CHUNK entries the queued lines are streamed to their
    /// blocks, in order.
    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,bmi2")]
    unsafe fn scatter(&mut self, b: &Builder, e: &[u64], part: &[u32]) -> std::io::Result<()> {
        let (buf, used) = (self.buf.ptr, self.used.ptr);
        let mut q = [_mm512_setzero_si512(); CHUNK];
        let mut qp = [0u32; CHUNK];
        let (qv, qpp) = (q.as_mut_ptr(), qp.as_mut_ptr());
        let m = e.len().min(part.len());
        let (ep, pp) = (e.as_ptr(), part.as_ptr());
        let mut i = 0;
        while i < m {
            let mut n = 0usize;
            macro_rules! one {
                ($j:expr) => {{
                    let (x, p) = (*ep.add($j), *pp.add($j) as usize);
                    let u = *used.add(p) as u32;
                    let line = buf.add(p * BURST);
                    let bit = 1u32 << u;
                    let v = _mm512_mask_set1_epi64(_mm512_load_si512(line.cast()), bit as u8, x as i64);
                    std::ptr::write_volatile(line.cast::<__m512i>(), v);
                    *used.add(p) = ((u + 1) % BURST as u32) as u8;
                    *qv.add(n) = v;
                    *qpp.add(n) = p as u32;
                    n += (bit >> (BURST - 1)) as usize;
                }};
            }
            if m - i >= CHUNK {
                for j in 0..CHUNK {
                    one!(i + j);
                }
                i += CHUNK;
            } else {
                while i < m {
                    one!(i);
                    i += 1;
                }
            }
            for x in 0..n {
                let p = *qpp.add(x) as usize;
                let d = self.dst(p, b)?;
                _mm512_stream_si512(d.cast(), *qv.add(x));
                *self.fill.get_unchecked_mut(p) += BURST as u32;
            }
        }
        Ok(())
    }

    /// After the scan: the partial bursts, every current block's length, and the blocks by partition.
    fn close(&mut self, b: &Builder) -> std::io::Result<()> {
        for p in 0..1 << PBITS {
            let u = self.used[p] as usize;
            if u > 0 {
                let dst = self.dst(p, b)?;
                unsafe { std::ptr::copy_nonoverlapping(self.buf.ptr.add(p * BURST), dst, u) };
                self.fill[p] += u as u32;
            }
            if self.block[p] != NONE {
                b.lens[self.block[p] as usize].store(self.fill[p], Relaxed);
            }
        }
        fence();
        self.off.fill(0);
        for &(p, _) in &self.log {
            self.off[p as usize + 1] += 1;
        }
        for p in 0..1 << PBITS {
            self.off[p + 1] += self.off[p];
        }
        self.ks.resize(self.log.len(), 0);
        let mut at = self.off.clone();
        for &(p, k) in &self.log {
            self.ks[at[p as usize] as usize] = k;
            at[p as usize] += 1;
        }
        Ok(())
    }
}

/// Place partition p's entries (its arena `blocks`, in order) and stream its buckets to the table at `dst`. Both fill
/// counts are read and the bucket chosen with selects; a fill stops at `slots`, and an entry that fits neither bucket
/// goes to b2 with fill `slots`: lane 0 of the staging line of a full bucket, which is never streamed again (slots
/// is a multiple of 8). The entry is inserted into its bucket's staging line `sp` in a register (as in the scatter),
/// and the line is also copied to a small queue whose index advances only when the line is complete; every CHUNK
/// entries the queued lines are streamed to the table, then at the end the partial last lines (zeros past the
/// fill). Never-written lines stay zero (the table is zero from create). Returns the entries kept (port: the sum of
/// the final fills).
#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,bmi2")]
unsafe fn place(b: &Builder, blocks: &[u32], sp: *mut u64, fp: *mut u8, dst: *mut u64) -> usize {
    let slots = b.slots;
    let mut q = [_mm512_setzero_si512(); CHUNK];
    let mut qo = [0u32; CHUNK];
    let (qv, qop) = (q.as_mut_ptr(), qo.as_mut_ptr());
    for &k in blocks {
        let len = b.lens[k as usize].load(Relaxed) as usize;
        let mut s = b.arena.ptr.add(k as usize * BLOCK) as *const u64;
        let end = s.add(len);
        while s < end {
            let mut n = 0usize;
            macro_rules! one {
                ($s:expr) => {{
                    let e = *$s;
                    let (h1, h2) = (b1(check(e)), b2(check(e)));
                    let (f1, f2) = (*fp.add(h1) as usize, *fp.add(h2) as usize);
                    let in1 = f1 < slots;
                    let h = select_unpredictable(in1, h1, h2);
                    let f = select_unpredictable(in1, f1, f2);
                    let line = sp.add(h * BURST);
                    let bit = 1u32 << (f % BURST);
                    let v = _mm512_mask_set1_epi64(_mm512_load_si512(line.cast()), bit as u8, e as i64);
                    std::ptr::write_volatile(line.cast::<__m512i>(), v);
                    *fp.add(h) = (f + (f < slots) as usize) as u8;
                    *qv.add(n) = v;
                    *qop.add(n) = (h * slots + (f & !(BURST - 1))) as u32; // the line's offset in the slice
                    n += (bit >> (BURST - 1)) as usize;
                }};
            }
            if end.offset_from(s) >= CHUNK as isize {
                for j in 0..CHUNK {
                    one!(s.add(j));
                }
                s = s.add(CHUNK);
            } else {
                while s < end {
                    one!(s);
                    s = s.add(1);
                }
            }
            for x in 0..n {
                _mm512_stream_si512(dst.add(*qop.add(x) as usize).cast(), *qv.add(x));
            }
        }
    }
    let mut kept = 0;
    for h in 0..1usize << NBITS {
        let f = *fp.add(h) as usize;
        kept += f;
        if f % BURST != 0 {
            let mask = ((1u32 << (f % BURST)) - 1) as u8;
            _mm512_stream_si512(dst.add(h * slots + f - f % BURST).cast(), _mm512_maskz_loadu_epi64(mask, sp.add(h * BURST).cast()));
        }
        *fp.add(h) = 0;
    }
    kept
}

/// The build state (51029638's Builder; `files` are the shards, cut to the requested tokens).
struct Builder {
    files: Vec<Shard>,
    sizes: Vec<usize>,
    pieces: Vec<(usize, usize)>, // (file, first context end) of each scan piece
    slots: usize,
    pool: rayon::ThreadPool,
    arena: Map<u64>,
    lens: Vec<AtomicU32>,   // each arena block's entries
    overflow: AtomicUsize, // the next arena block past the threads' own
    table: Map<u64>,
    scatters: Vec<Mutex<Scatter>>,
    finishes: Vec<Mutex<(Map<u64>, Map<u8>)>>, // a thread's staging line and fill count of every bucket of a slice
}

#[pyclass]
pub struct SlotIndexBranchFree {
    file: File, // the table's memfd
    slots: usize,
    builder: Builder,
    built: AtomicBool,
}

#[pymethods]
impl SlotIndexBranchFree {
    /// Off the clock: 51029638's create for the first `tokens` tokens of `files` (the table a memfd), faulting in
    /// the per-thread buffers, the arena and the table on the build pool.
    #[staticmethod]
    fn create(py: Python<'_>, files: Vec<String>, tokens: usize, threads: usize, cpus: Vec<usize>) -> PyResult<Self> {
        if !*AVX512 {
            return Err(PyValueError::new_err("SlotIndexBranchFree needs AVX-512"));
        }
        if threads == 0 {
            return Err(PyValueError::new_err("need at least one thread"));
        }
        let files = shards(&files, tokens)?;
        let sizes: Vec<usize> = files.iter().map(|s| s.len).collect();
        let slots = slots_for(sizes.iter().sum())?;
        let table_len = BUCKETS * slots;
        // Full blocks, one partial block per (thread, partition), 2% slack. A thread takes blocks from its own
        // first-touch slice (half its fair share, so a slow thread strands little) and then from the shared rest.
        let full = sizes.iter().map(|&n| n.saturating_sub(MIN)).sum::<usize>().div_ceil(BLOCK);
        let local = full / 2 / threads;
        let arena = Map::new((full * 51 / 50 + threads * (1 << PBITS) + 64) * BLOCK)?;
        let name = c"cache_ladder-slot-branch-free";
        let fd = unsafe { libc::syscall(libc::SYS_memfd_create, name.as_ptr(), libc::MFD_CLOEXEC) };
        if fd < 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        let file = unsafe { File::from_raw_fd(fd as i32) };
        file.set_len((table_len * 8) as u64)?;
        let table = mmap(std::ptr::null_mut(), table_len * 8, libc::MAP_SHARED, file.as_raw_fd())?;
        let builder = Builder {
            pieces: sizes.iter().enumerate().flat_map(|(f, &n)| (MIN..n).step_by(PIECE).map(move |a| (f, a))).collect(),
            files,
            sizes,
            slots,
            pool: pinned_pool(threads, &cpus, "slotbranchfree")?,
            lens: (0..arena.len() / BLOCK).map(|_| AtomicU32::new(BLOCK as u32)).collect(),
            arena,
            overflow: AtomicUsize::new(threads * local),
            table: Map { ptr: table.cast(), len: table_len, bytes: table_len * 8 },
            scatters: (0..threads)
                .map(|t| {
                    let (buf, used) = (Map::new((1 << PBITS) * BURST)?, Map::new(1 << PBITS)?);
                    let (block, fill) = (vec![NONE; 1 << PBITS], vec![BLOCK as u32; 1 << PBITS]);
                    let blocks = 2 * full / threads + (1 << PBITS); // ample: presized, not grown on the clock
                    let (log, ks) = (Vec::with_capacity(blocks), Vec::with_capacity(blocks));
                    let off = vec![0; (1 << PBITS) + 1];
                    Ok(Mutex::new(Scatter { buf, used, block, fill, log, local: t * local..(t + 1) * local, off, ks }))
                })
                .collect::<std::io::Result<_>>()?,
            finishes: (0..threads).map(|_| Ok(Mutex::new((Map::new((1 << NBITS) * BURST)?, Map::new(1 << NBITS)?)))).collect::<std::io::Result<_>>()?,
        };
        // Fault in the per-thread buffers first, while 2 MB pages are still easy to get (on a fragmented host, after
        // hundreds of GB of arena and table they fell back to 4 KB pages, and the scan's burst stores then missed the
        // TLB), then the arena (each thread its own slice first) and the table, which stays mapped.
        let b = &builder;
        py.detach(|| {
            b.pool.broadcast(|ctx| {
                let (i, n) = (ctx.index(), ctx.num_threads());
                {
                    let (mut scatter, mut finish) = (b.scatters[i].lock().unwrap(), b.finishes[i].lock().unwrap());
                    let scatter = &mut *scatter;
                    touch(&mut scatter.buf);
                    touch(&mut scatter.used);
                    touch(scatter.log.spare_capacity_mut());
                    touch(scatter.ks.spare_capacity_mut());
                    touch(&mut finish.0);
                    touch(&mut finish.1);
                }
                let part = |m: &Map<u64>, a: usize, b: usize| unsafe { std::slice::from_raw_parts_mut(m.ptr.add(a), b - a) };
                let (own, rest) = (local * BLOCK, b.arena.len() - n * local * BLOCK);
                touch(part(&b.arena, i * own, (i + 1) * own));
                touch(part(&b.arena, n * own + rest * i / n, n * own + rest * (i + 1) / n));
                touch(part(&b.table, table_len * i / n, table_len * (i + 1) / n));
            });
        });
        Ok(Self { file, slots, builder, built: AtomicBool::new(false) })
    }

    /// On the clock: 51029638's build (scan, then finish). Returns entries (kept), dropped, bytes (the table), the
    /// arena's bytes, slots, threads, and the phases' seconds.
    fn build(&mut self, py: Python<'_>) -> PyResult<HashMap<String, f64>> {
        if self.built.swap(true, Relaxed) {
            return Err(PyValueError::new_err("already built (the arena and scatter state are single-use)"));
        }
        let b = &self.builder;
        let (scan_s, finish_s, total, kept) = py.detach(|| -> std::io::Result<(f64, f64, usize, usize)> {
            let t0 = Instant::now();
            // 1. Scan.
            let scanned = b.pool.broadcast(|ctx| -> std::io::Result<()> {
                let mut guard = b.scatters[ctx.index()].lock().unwrap();
                let sc = &mut *guard;
                let (mut hashes, mut tokens) = (Hashes::new(), vec![STOP; PIECE + BACK]);
                for &(f, a) in b.pieces.iter().skip(ctx.index()).step_by(ctx.num_threads()) {
                    // tokens[BACK + r] = shard token a + r; tokens before the shard's start are STOP.
                    let (n, lo) = ((a + PIECE).min(b.sizes[f]) - a, a.saturating_sub(BACK));
                    let pad = BACK - (a - lo);
                    tokens[..pad].fill(STOP);
                    pread(&b.files[f].file, &mut tokens[pad..BACK + n], HEADER_BYTES + lo * 2)?;
                    for r in (0..n).step_by(STEP) {
                        let m = STEP.min(n - r);
                        hashes.hash(&tokens, BACK + r, m, true);
                        unsafe { sc.scatter(b, &hashes.e[..m], &hashes.part[..m])? };
                    }
                }
                sc.close(b)
            });
            scanned.into_iter().collect::<std::io::Result<()>>()?;
            let t1 = Instant::now();

            // 2. Finish: the partitions largest first, each placed into the table.
            let scatters: Vec<_> = b.scatters.iter().map(|s| s.lock().unwrap()).collect();
            let size = |p: usize| -> usize {
                scatters.iter().flat_map(|s| &s.ks[s.off[p] as usize..s.off[p + 1] as usize]).map(|&k| b.lens[k as usize].load(Relaxed) as usize).sum()
            };
            let sizes: Vec<usize> = b.pool.install(|| (0..1 << PBITS).into_par_iter().map(size).collect());
            let mut order: Vec<usize> = (0..1 << PBITS).collect();
            order.sort_by_key(|&p| std::cmp::Reverse(sizes[p]));
            let next = AtomicUsize::new(0);
            let kept = AtomicUsize::new(0);
            b.pool.broadcast(|ctx| {
                let mut guard = b.finishes[ctx.index()].lock().unwrap();
                let (stage, fill) = &mut *guard;
                let (mut blocks, mut mine) = (Vec::new(), 0);
                while let Some(&p) = order.get(next.fetch_add(1, Relaxed)) {
                    blocks.clear();
                    for s in &scatters {
                        blocks.extend_from_slice(&s.ks[s.off[p] as usize..s.off[p + 1] as usize]);
                    }
                    mine += unsafe { place(b, &blocks, stage.ptr, fill.ptr, b.table.ptr.add(p * (1 << NBITS) * b.slots)) };
                }
                fence();
                kept.fetch_add(mine, Relaxed);
            });
            let t2 = Instant::now();
            Ok(((t1 - t0).as_secs_f64(), (t2 - t1).as_secs_f64(), sizes.iter().sum(), kept.into_inner()))
        })?;
        Ok(HashMap::from([
            ("entries".into(), kept as f64),
            ("dropped".into(), (total - kept) as f64),
            ("bytes".into(), (b.table.len * 8) as f64),
            ("arena_bytes".into(), (b.arena.len() * 8) as f64),
            ("slots".into(), self.slots as f64),
            ("threads".into(), b.pool.current_num_threads() as f64),
            ("scan_s".into(), scan_s),
            ("finish_s".into(), finish_s),
        ]))
    }

    /// Off the clock, for verification: a path to the table (the memfd), valid while this index lives.
    #[getter]
    fn table_path(&self) -> String {
        format!("/proc/{}/fd/{}", std::process::id(), self.file.as_raw_fd())
    }
}
