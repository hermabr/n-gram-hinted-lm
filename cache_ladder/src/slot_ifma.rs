//! SlotIndexIfma: the record's SlotIndex as of modded-nanogpt commit 9025a886 ("Delay validation index timing until
//! after the clock stops", 2026-10-07, oct7-cleaned-up-record lineage), the record until 2026-10-20: slot_branch_free's
//! build with the IFMA hash, mbind MPOL_LOCAL and the padded block counter.
//!
//! This is that commit's exact_match/src/slot.rs build path, with the helpers it took from that commit's lib.rs
//! (mmap with mbind, madvise, Map, touch, fence, the AVX-512 check, MIN/LEVELS/STOP) copied in. It stores the same
//! data as the final rung (exact_match/src/slot.rs in this repo): one u64 entry per training position with a MIN-gram in
//! its shard,
//!
//!   next token (16 bits) | level-10 fingerprint (12) | level-20 fingerprint (6) | key check (30) ,
//!
//! in 2^14 partitions x 2^14 buckets of `slots` entries (slots from the corpus size at LOAD 0.8), each entry in its
//! home bucket b1 while it has room, else in b2, else dropped. Build:
//!   1. scan: thread t preads the 64K-token pieces t, t + T, ..., hashes 4096 positions at a time (multiply-add-shift
//!      hashes mod 2^52 with AVX-512 IFMA on byte-permuted 3-token words, VBMI), and scatters every entry through a
//!      per-thread write-combining burst of one cache line per partition into 2048-entry blocks of a private huge-page
//!      arena, with non-temporal stores.
//!   2. finish: per partition (largest first, handed out dynamically), place its entries into b1 / b2 with selects,
//!      staging only each bucket's filling line and streaming it to the table the moment it is complete.
//!
//! Over slot_branch_free (51029638) it has three changes: the IFMA multiply-shift hash (9ceb0b20; the hash is new, so
//! the table's bits differ), every mapping mbind MPOL_LOCAL, out of automatic NUMA balancing (f315854f), and the
//! arena's shared block counter in cache lines of its own (1802a267). It predates the final rung's fused hash and
//! scatter (16 positions one group ahead of their inserts), its tail stealing of the scan, its 8192-entry blocks, and
//! its placement of one 8 B store per entry with a u16 queue of completed lines and an arena prefetch.
//!
//! Adaptations (none touch the loops):
//! - Input: the first `tokens` tokens of `files` (crate::shards), not whole shards: the last shard's token count is
//!   cut, so its pieces stop there. The scan still preads the shard files as the original did.
//! - Threads and CPUs come from create's arguments; the pool is crate::pinned_pool (pins with a broadcast as the
//!   original's did, without its CPU_SETSIZE check).
//! - The table, a new or empty file in /dev/shm in the original (shared with the other ranks), is a memfd made in
//!   create (shared memory as well; miso's shmem_enabled is "never", so either is in 4 KB pages), freed on drop.
//! - create() is the original's create (allocate, map the table, fault in buffers, arena and table on the pool); the
//!   lookup pool, attach, query and release are dropped (and with them agree, the lookup's level count).
//! - build() returns statistics: place() also sums each partition's bucket fills (it reads every fill at its end
//!   anyway) for the kept count; the scan and finish phases are timed with Instant.
//! - Off the clock, for verification: table_path (the memfd as /proc/self/fd/N) to compare the table with the
//!   original's.
//!
//! Verified on miso3 (2026-10-20): the table is bit-identical to 9025a886's (built from `git archive`) for 3 whole
//! shards at 16 pinned threads and 5 at 31, and builds as fast (0.21-0.24 s vs 0.22 s; 0.34 s vs 0.34 s).
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

// From 9025a886's lib.rs.
const STOP: u16 = u16::MAX;
const MIN: usize = 6;
const LEVELS: [usize; 2] = [10, 20];

/// The build loops are written for AVX-512 with IFMA and VBMI (the record's hosts have it: Sapphire Rapids, Zen 4).
static AVX512: LazyLock<bool> = LazyLock::new(|| {
    is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512dq")
        && is_x86_feature_detected!("avx512vl")
        && is_x86_feature_detected!("avx512ifma")
        && is_x86_feature_detected!("avx512vbmi")
        && is_x86_feature_detected!("bmi2")
});

/// mmap read-write, in transparent huge pages, split into 256 MB VMAs (alternate ones flagged MADV_DONTDUMP) so that
/// a page-table walk (Slurm reads smaps every 30 s) holds the mmap lock only briefly per VMA. Not inherited by a fork,
/// where every write here would copy a page. An explicit local policy keeps first-touch placement and takes the mapping
/// out of automatic NUMA balancing, which migrated huge pages under the build (7.2-8.9 s at 35B tokens, 6.7 s without).
fn mmap(bytes: usize, flags: libc::c_int, fd: libc::c_int) -> std::io::Result<*mut libc::c_void> {
    const VMA: usize = 1 << 28;
    let ptr = unsafe { libc::mmap(std::ptr::null_mut(), bytes, libc::PROT_READ | libc::PROT_WRITE, flags, fd, 0) };
    if ptr == libc::MAP_FAILED {
        return Err(std::io::Error::last_os_error());
    }
    madvise(ptr, bytes, libc::MADV_HUGEPAGE)?;
    madvise(ptr, bytes, libc::MADV_DONTFORK)?;
    // A warning only: containers without CAP_SYS_NICE (Docker's default seccomp profile) may not set a policy.
    const MPOL_LOCAL: libc::c_long = 4;
    if unsafe { libc::syscall(libc::SYS_mbind, ptr, bytes, MPOL_LOCAL, std::ptr::null::<libc::c_ulong>(), 0 as libc::c_ulong, 0 as libc::c_uint) } != 0 {
        eprintln!("mbind(MPOL_LOCAL) failed, NUMA balancing may migrate this mapping: {}", std::io::Error::last_os_error());
    }
    for off in (VMA..bytes).step_by(2 * VMA) {
        madvise(unsafe { ptr.cast::<u8>().add(off) }.cast(), VMA.min(bytes - off), libc::MADV_DONTDUMP)?;
    }
    Ok(ptr)
}

/// madvise, failing on an error.
fn madvise(ptr: *mut libc::c_void, bytes: usize, advice: libc::c_int) -> std::io::Result<()> {
    if unsafe { libc::madvise(ptr, bytes, advice) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
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
        let raw = mmap(bytes + HUGE, flags, -1)? as usize;
        let ptr = raw.next_multiple_of(HUGE);
        for (at, len) in [(raw, ptr - raw), (ptr + bytes, raw + HUGE - ptr)] {
            if len > 0 && unsafe { libc::munmap(at as *mut libc::c_void, len) } != 0 {
                return Err(std::io::Error::last_os_error());
            }
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


// 9025a886's slot.rs from here on (build path).
const BITS: [u32; 2] = [12, 6]; // fingerprint bits per level
const SHIFTS: [u32; 2] = [16, 28]; // where each level's fingerprint sits in an entry
const CHECK_SHIFT: u32 = 34;
const KBITS: u32 = 64 - CHECK_SHIFT; // key check bits
const PBITS: u32 = 14; // partitions: 2^14
const NBITS: u32 = 14; // buckets per partition: 2^14
const BUCKETS: usize = 1 << (PBITS + NBITS);
const LOAD: f64 = 0.8; // entries per slot (10.26B tokens: 48 slots, 103 GB; 41.2B: 192, 412 GB)
const BACK: usize = LEVELS[1]; // tokens before a context end that hash_avx512 reads
const STEP: usize = 1 << 12; // positions hashed at once
const PIECE: usize = 1 << 16; // context ends per piece of the scan (one pread)
const BURST: usize = 8; // entries per write-combining burst and per staging line: one cache line
const BLOCK: usize = 2048; // entries per arena block, a multiple of BURST
const CHUNK: usize = 64; // entries between the queued line stores of the scatter and the placement
const NONE: u32 = u32::MAX;
const K0: u64 = 0x9e3779b97f4a7c15; // 2^64 / the golden ratio: b2 is a Fibonacci hash of the check

/// The splitmix64 generator's output at `x`: its golden-ratio increment, then its finalizer (Steele, Lea and Flood).
const fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

/// The odd 52-bit multiplier and the 52-bit addend of IFMA word i, from splitmix64 at two arbitrary seeds (one for
/// the multipliers, one for the addends). hash_avx512's multipliers: 0-1 the key's words, 8 its high products, 2-3
/// level 10's words, 4-7 level 20's; its addends: 0 the key, 1 level 10, 2 level 20.
const fn m52(i: u64) -> u64 {
    (splitmix(0x1f3a_0000 + i) >> 12) | 1
}
const fn a52(i: u64) -> u64 {
    splitmix(0x2e4b_0000 + i) >> 12
}

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

/// vpermb (1 uop; LLVM turns it into the 2-uop vpermw when the bytes move in pairs).
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
#[inline]
unsafe fn vpermb(idx: __m512i, x: __m512i) -> __m512i {
    let d: __m512i;
    std::arch::asm!("vpermb {d}, {i}, {x}", d = lateout(zmm_reg) d, i = in(zmm_reg) idx, x = in(zmm_reg) x, options(pure, nomem, nostack));
    d
}

/// Scratch of one hashing step.
struct Hashes {
    e: Vec<u64>,
    part: Vec<u32>,
}

impl Hashes {
    fn new() -> Self {
        Self { e: vec![0; STEP], part: vec![0; STEP] }
    }

    /// The entries (with the next token if `next`) and partitions of the n contexts ending at t[j0..j0 + n]
    /// (exclusive ends): context r ends at t[j0 + r], which is its next token. Needs j0 >= BACK.
    fn hash(&mut self, t: &[u16], j0: usize, n: usize, next: bool) {
        assert!(*AVX512);
        unsafe { self.hash_avx512(t, j0, n, next) }
    }

    /// Multiply-add-shift hashes mod 2^52 (one vpmadd52luq per word) of 3-token words, word o of a context the 3
    /// tokens before its end - o: the key of words 0 and 3 (also the high halves of their products, then one more
    /// IFMA; bits 38-51 the partition, 8-37 the check), the level
    /// fingerprints of words 6, 7 (tokens 7-10) and 8, 11, 14, 17 (tokens 9-20), their top bits. Per 8 positions one
    /// load of the 32 tokens from 20 before the first context end; a word (lane: 3 tokens, then a copy of the first,
    /// as IFMA reads 52 bits) and the next token are byte permutes of it, and words 8, 11, 14 are the previous 8
    /// positions' 0, 3, 6.
    #[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma,avx512vbmi")]
    unsafe fn hash_avx512(&mut self, t: &[u16], j0: usize, n: usize, next: bool) {
        const { assert!(STEP % 16 == 0 && CHECK_SHIFT + KBITS == 64 && KBITS + PBITS == 44 && BITS[0] == 12 && BITS[1] == 6 && SHIFTS[0] == 16 && SHIFTS[1] == 28) };
        assert!(j0 >= BACK && t.len() >= j0 + n - 1 + next as usize && n <= STEP);
        // The offsets o of the permuted words; permute 8 is the next token.
        const OS: [usize; 8] = [0, 3, 6, 7, 17, 8, 11, 14];
        let idx: [__m512i; 9] = std::array::from_fn(|w| {
            let mut b = [0u8; 64];
            for k in 0..8 {
                for i in 0..4 {
                    let at = if w < 8 { k + BACK - 3 - OS[w] + i % 3 } else { k + BACK };
                    (b[8 * k + 2 * i], b[8 * k + 2 * i + 1]) = (2 * at as u8, 2 * at as u8 + 1);
                }
            }
            _mm512_loadu_si512(b.as_ptr().cast())
        });
        let (tp, ep, pp) = (t.as_ptr(), self.e.as_mut_ptr(), self.part.as_mut_ptr());
        let keep = _mm512_set1_epi64(!((1u64 << CHECK_SHIFT) - 1) as i64);
        let m10 = _mm512_set1_epi64((!((1u64 << CHECK_SHIFT) - 1) | ((1u64 << BITS[0]) - 1) << SHIFTS[0]) as i64);
        let low = _mm512_set1_epi64(0xffff);
        let pmask = _mm512_set1_epi32((1 << PBITS) - 1);
        let pidx = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
        let madd = |acc: __m512i, w: __m512i, i: u64| _mm512_madd52lo_epu64(acc, w, _mm512_set1_epi64(m52(i) as i64));
        let a = |i: u64| _mm512_set1_epi64(a52(i) as i64);
        let load = |r: usize| {
            let (start, avail) = (j0 + r - BACK, t.len() - (j0 + r - BACK));
            if avail >= 32 { _mm512_loadu_si512(tp.add(start).cast()) } else { _mm512_maskz_loadu_epi16((1u32 << avail) - 1, tp.add(start).cast()) }
        };
        // Entries r..r + 8 from the previous 8 positions' words 0, 3, 6; returns the key rotated (check at
        // CHECK_SHIFT, partition at 0) and its words 0, 3, 6.
        let eight = |r: usize, p: [__m512i; 3]| -> (__m512i, [__m512i; 3]) {
            let x = load(r);
            let w = |i: usize| vpermb(idx[i], x);
            let (w0, w3, w6) = (w(0), w(1), w(2));
            // the key: low and high 52 bits of the products, combined (alone, the low sum mod 2^52 of tokens at
            // 16-bit offsets collides at 2^-20 for contexts differing only in tokens 1 and 4)
            let hi = |acc: __m512i, w: __m512i, i: u64| _mm512_madd52hi_epu64(acc, w, _mm512_set1_epi64(m52(i) as i64));
            let key = madd(madd(madd(a(0), w0, 0), w3, 1), hi(hi(_mm512_setzero_si512(), w0, 0), w3, 1), 8);
            let f10 = madd(madd(a(1), w6, 2), w(3), 3);
            let f20 = madd(madd(madd(madd(a(2), p[0], 4), p[1], 5), p[2], 6), w(4), 7);
            let rot = _mm512_rol_epi64::<{ 64 - (52 - PBITS as i32) }>(key);
            let x10 = _mm512_srli_epi64::<{ 52 - BITS[0] - SHIFTS[0] }>(f10);
            let x20 = _mm512_srli_epi64::<{ 52 - BITS[1] - SHIFTS[1] }>(f20);
            let e = _mm512_ternarylogic_epi64::<0xca>(keep, rot, x10); // keep ? rot : x10
            let e = _mm512_ternarylogic_epi64::<0xca>(m10, e, x20);
            let e = if next { _mm512_ternarylogic_epi64::<0xca>(low, vpermb(idx[8], x), e) } else { _mm512_andnot_si512(low, e) };
            _mm512_storeu_si512(ep.add(r).cast(), e);
            (rot, [w0, w3, w6])
        };
        let x = load(0);
        let mut p = [vpermb(idx[5], x), vpermb(idx[6], x), vpermb(idx[7], x)];
        for r in (0..n).step_by(16) {
            let (lo, q) = eight(r, p);
            let (hi, q) = eight(r + 8, q);
            p = q;
            _mm512_storeu_si512(pp.add(r).cast(), _mm512_and_si512(_mm512_permutex2var_epi32(lo, pidx, hi), pmask));
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
        let b = self.local.next().unwrap_or_else(|| builder.overflow.0.fetch_add(1, Relaxed));
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

/// A value in cache lines of its own (a pair: the adjacent-line prefetcher fetches them together). The arena's shared
/// block counter, which every scan thread bumps, made the build a third slower (30B tokens: 10.4 s, not 7.8) when it
/// shared a line with the arena's pointer, which every burst reads: in about half the runs, depending on where Python
/// allocated the index.
#[repr(align(128))]
struct Padded<T>(T);

/// The build state (9025a886's Builder; `files` are the shards, cut to the requested tokens).
struct Builder {
    files: Vec<Shard>,
    sizes: Vec<usize>,
    pieces: Vec<(usize, usize)>, // (file, first context end) of each scan piece
    slots: usize,
    pool: rayon::ThreadPool,
    arena: Map<u64>,
    lens: Vec<AtomicU32>,   // each arena block's entries
    overflow: Box<Padded<AtomicUsize>>, // the next arena block past the lanes' own, in lines of its own
    table: Map<u64>,
    scatters: Vec<Mutex<Scatter>>,
    finishes: Vec<Mutex<(Map<u64>, Map<u8>)>>, // a thread's staging line and fill count of every bucket of a slice
    built: AtomicBool, // build ran (the arena and scatter state are single-use)
}

#[pyclass]
pub struct SlotIndexIfma {
    file: File, // the table's memfd
    slots: usize,
    builder: Builder,
}

#[pymethods]
impl SlotIndexIfma {
    /// Off the clock: 9025a886's create for the first `tokens` tokens of `files` (the table a memfd), faulting in the
    /// per-thread buffers, the arena and the table on the build pool.
    #[staticmethod]
    fn create(py: Python<'_>, files: Vec<String>, tokens: usize, threads: usize, cpus: Vec<usize>) -> PyResult<Self> {
        if !*AVX512 {
            return Err(PyValueError::new_err("SlotIndexIfma needs AVX-512 with IFMA and VBMI"));
        }
        if threads == 0 {
            return Err(PyValueError::new_err("need at least one thread"));
        }
        let files = shards(&files, tokens)?;
        let sizes: Vec<usize> = files.iter().map(|s| s.len).collect();
        let slots = slots_for(sizes.iter().sum())?;
        let table_len = BUCKETS * slots;
        // Each (thread, partition) fills whole blocks and at most one partial one: at most `full` + threads * 2^PBITS
        // blocks in all. A thread takes blocks from its own first-touch slice of the arena first, then from the shared
        // rest. The slice is half a thread's fair share: the static piece split gives every thread about its fair share,
        // so each uses its slice up and strands none. The 2% and 64 blocks on top are slack for an uneven split; past
        // them the scan fails ("scan arena is full").
        let full = sizes.iter().map(|&n| n.saturating_sub(MIN)).sum::<usize>().div_ceil(BLOCK);
        let local = full / 2 / threads;
        let arena = Map::new((full * 51 / 50 + threads * (1 << PBITS) + 64) * BLOCK)?;
        let name = c"cache_ladder-slot-ifma";
        let fd = unsafe { libc::syscall(libc::SYS_memfd_create, name.as_ptr(), libc::MFD_CLOEXEC) };
        if fd < 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        let file = unsafe { File::from_raw_fd(fd as i32) };
        file.set_len((table_len * 8) as u64)?;
        let table = mmap(table_len * 8, libc::MAP_SHARED, file.as_raw_fd())?;
        let builder = Builder {
            pieces: sizes.iter().enumerate().flat_map(|(f, &n)| (MIN..n).step_by(PIECE).map(move |a| (f, a))).collect(),
            files,
            sizes,
            slots,
            pool: pinned_pool(threads, &cpus, "slotifma")?,
            lens: (0..arena.len() / BLOCK).map(|_| AtomicU32::new(BLOCK as u32)).collect(),
            arena,
            overflow: Box::new(Padded(AtomicUsize::new(threads * local))),
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
            built: AtomicBool::new(false),
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
        Ok(Self { file, slots, builder })
    }

    /// On the clock: 9025a886's build (scan, then finish). Returns entries (kept), dropped, bytes (the table), the
    /// arena's bytes, slots, threads, and the phases' seconds.
    fn build(&self, py: Python<'_>) -> PyResult<HashMap<String, f64>> {
        let b = &self.builder;
        if b.built.swap(true, Relaxed) {
            return Err(PyValueError::new_err("already built (the arena and scatter state are single-use)"));
        }
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
