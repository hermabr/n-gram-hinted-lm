//! SlotIndexOnePass: the record's SlotIndex (exact_match/src/slot.rs) built in one pass instead of two, to measure what
//! the two-stage build (scan into a partitioned arena, then place each partition into its buckets) gains over writing
//! every entry straight into its bucket.
//!
//! The same table as the record: one u64 entry per training position with a MIN-gram in its shard (the record's IFMA
//! hash, copied), 2^14 partitions x 2^14 buckets of `slots` entries, an entry in its home bucket b1 while it has room,
//! else in b2, else dropped. The same scan as well (thread t preads the 64K-token pieces t, t + T, ..., hashes 4096
//! positions at a time), but each hashed step goes straight into the table: the global b1 and b2 of every entry are
//! computed with AVX-512, then every entry is inserted by reading both fill counts, choosing the bucket with selects and
//! storing the entry at the fill (a dropped entry goes to a private sink line instead). There is no branch on the data.
//! The table's random lines and fill counts are prefetched (prefetchw) ahead: the fills of entry i + 2 * dist, and the
//! table line where entry i + dist would go given the fills then. dist 0: no prefetch.
//!
//! The threads share the table and the fill counts. Modes:
//! - "racy": u8 fills, read and written with relaxed (plain) loads and stores. Two threads inserting into the same
//!   bucket at the same moment can write the same slot, losing an entry; `counts` reports how many (`lost`).
//! - "atomic": u32 fills; b1's fill is read, then the chosen bucket's fill is claimed with a fetch_add (lock xadd), so
//!   no entry is lost. A fill counts on past `slots`; the bucket holds min(fill, slots).
//! - "simd": as "racy" with u32 fills, 16 entries at a time: both fills gathered, the buckets chosen with blends, the
//!   entries and new fills scattered. A group where two entries choose the same bucket (vpconflictd) is inserted one
//!   entry at a time instead.
//! - "hash": the scan and bucket computation only, no inserts: the floor of the one-pass build.
//!
//! create(files, tokens, threads, cpus, mode, dist, huge) is the record's create (the table in shared memory, faulted
//! in on the build pool with mbind MPOL_LOCAL) plus the fill counts, also faulted in; build() is the timed part. The
//! record's table is a shared-memory file, in 4 KB pages on miso (shmem_enabled "never"), so nearly every random insert
//! misses the TLB; huge: the table in private memory in 2 MB pages instead (as the fills always are), to separate the
//! TLB's share of the one-pass cost from the DRAM's. table_path is then an empty file.
use crate::{pinned_pool, shards, Shard, HEADER_BYTES};
use pyo3::{exceptions::PyValueError, prelude::*};
use rayon::prelude::*;
use std::arch::x86_64::*;
use std::collections::HashMap;
use std::fs::File;
use std::hint::select_unpredictable;
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicU8, AtomicUsize, Ordering::Relaxed};
use std::sync::LazyLock;
use std::time::Instant;

// From the record's lib.rs.
const STOP: u16 = u16::MAX;
const MIN: usize = 6;
const LEVELS: [usize; 2] = [10, 20];

static AVX512: LazyLock<bool> = LazyLock::new(|| {
    is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512dq")
        && is_x86_feature_detected!("avx512vl")
        && is_x86_feature_detected!("avx512cd")
        && is_x86_feature_detected!("avx512ifma")
        && is_x86_feature_detected!("avx512vbmi")
        && is_x86_feature_detected!("bmi2")
});

/// The record's lib.rs mmap: read-write, transparent huge pages, MPOL_LOCAL, 256 MB VMAs.
fn mmap(bytes: usize, flags: libc::c_int, fd: libc::c_int) -> std::io::Result<*mut libc::c_void> {
    const VMA: usize = 1 << 28;
    let ptr = unsafe { libc::mmap(std::ptr::null_mut(), bytes, libc::PROT_READ | libc::PROT_WRITE, flags, fd, 0) };
    if ptr == libc::MAP_FAILED {
        return Err(std::io::Error::last_os_error());
    }
    unsafe { libc::madvise(ptr, bytes, libc::MADV_HUGEPAGE) };
    unsafe { libc::madvise(ptr, bytes, libc::MADV_DONTFORK) };
    const MPOL_LOCAL: libc::c_long = 4;
    if unsafe { libc::syscall(libc::SYS_mbind, ptr, bytes, MPOL_LOCAL, std::ptr::null::<libc::c_ulong>(), 0 as libc::c_ulong, 0 as libc::c_uint) } != 0 {
        eprintln!("mbind(MPOL_LOCAL) failed: {}", std::io::Error::last_os_error());
    }
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
        let raw = mmap(bytes + HUGE, flags, -1)? as usize;
        let ptr = raw.next_multiple_of(HUGE);
        unsafe {
            libc::munmap(raw as *mut libc::c_void, ptr - raw);
            libc::munmap((ptr + bytes) as *mut libc::c_void, raw + HUGE - ptr);
        }
        Ok(Self { ptr: ptr as *mut T, len, bytes })
    }

    /// Elements a..b, for faulting in a thread's slice.
    fn part(&self, a: usize, b: usize) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.add(a), b - a) }
    }
}

impl<T> Drop for Map<T> {
    fn drop(&mut self) {
        unsafe { libc::munmap(self.ptr.cast(), self.bytes) };
    }
}

/// Write one byte per page of `v`, so later writes take no page faults.
fn touch<T>(v: &mut [T]) {
    let bytes = v.as_mut_ptr() as *mut u8;
    for i in (0..std::mem::size_of_val(v)).step_by(4096) {
        unsafe { std::ptr::write_volatile(bytes.add(i), 0) };
    }
}

// The record's slot.rs from here to Hashes (constants, hash).
const BITS: [u32; 2] = [12, 6];
const SHIFTS: [u32; 2] = [16, 28];
const CHECK_SHIFT: u32 = 34;
const KBITS: u32 = 64 - CHECK_SHIFT;
const PBITS: u32 = 14;
const NBITS: u32 = 14;
const BUCKETS: usize = 1 << (PBITS + NBITS);
const LOAD: f64 = 0.8;
const BACK: usize = LEVELS[1];
const STEP: usize = 1 << 12;
const PIECE: usize = 1 << 16;
const BURST: usize = 8;
const K0: u64 = 0x9e3779b97f4a7c15;

const fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}
const fn m52(i: u64) -> u64 {
    (splitmix(0x1f3a_0000 + i) >> 12) | 1
}
const fn a52(i: u64) -> u64 {
    splitmix(0x2e4b_0000 + i) >> 12
}

fn slots_for(tokens: usize) -> PyResult<usize> {
    let slots = (tokens as f64 / (BUCKETS as f64 * LOAD) / BURST as f64).ceil() as usize * BURST;
    if slots >= 256 {
        return Err(PyValueError::new_err(format!("{tokens} tokens need {slots} slots per bucket, more than 248")));
    }
    Ok(slots.max(BURST))
}

fn pread<T>(file: &File, buf: &mut [T], offset: usize) -> std::io::Result<()> {
    let bytes = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, std::mem::size_of_val(buf)) };
    file.read_exact_at(bytes, offset as u64)
}

#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
#[inline]
unsafe fn vpermb(idx: __m512i, x: __m512i) -> __m512i {
    let d: __m512i;
    std::arch::asm!("vpermb {d}, {i}, {x}", d = lateout(zmm_reg) d, i = in(zmm_reg) idx, x = in(zmm_reg) x, options(pure, nomem, nostack));
    d
}

/// Scratch of one hashing step (the record's), plus each entry's global b1 and b2 (partition << NBITS | bucket), padded
/// for 16-lane loads.
struct Hashes {
    e: Vec<u64>,
    part: Vec<u32>,
    g1: Vec<u32>,
    g2: Vec<u32>,
}

impl Hashes {
    fn new() -> Self {
        Self { e: vec![0; STEP], part: vec![0; STEP], g1: vec![0; STEP + 16], g2: vec![0; STEP + 16] }
    }

    /// The record's hash_avx512, verbatim.
    #[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma,avx512vbmi")]
    unsafe fn hash_avx512(&mut self, t: &[u16], j0: usize, n: usize, next: bool) {
        const { assert!(STEP % 16 == 0 && CHECK_SHIFT + KBITS == 64 && KBITS + PBITS == 44 && BITS[0] == 12 && BITS[1] == 6 && SHIFTS[0] == 16 && SHIFTS[1] == 28) };
        assert!(j0 >= BACK && t.len() >= j0 + n - 1 + next as usize && n <= STEP);
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
        let eight = |r: usize, p: [__m512i; 3]| -> (__m512i, [__m512i; 3]) {
            let x = load(r);
            let w = |i: usize| vpermb(idx[i], x);
            let (w0, w3, w6) = (w(0), w(1), w(2));
            let hi = |acc: __m512i, w: __m512i, i: u64| _mm512_madd52hi_epu64(acc, w, _mm512_set1_epi64(m52(i) as i64));
            let key = madd(madd(madd(a(0), w0, 0), w3, 1), hi(hi(_mm512_setzero_si512(), w0, 0), w3, 1), 8);
            let f10 = madd(madd(a(1), w6, 2), w(3), 3);
            let f20 = madd(madd(madd(madd(a(2), p[0], 4), p[1], 5), p[2], 6), w(4), 7);
            let rot = _mm512_rol_epi64::<{ 64 - (52 - PBITS as i32) }>(key);
            let x10 = _mm512_srli_epi64::<{ 52 - BITS[0] - SHIFTS[0] }>(f10);
            let x20 = _mm512_srli_epi64::<{ 52 - BITS[1] - SHIFTS[1] }>(f20);
            let e = _mm512_ternarylogic_epi64::<0xca>(keep, rot, x10);
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

    /// The global b1 and b2 of the first n entries, 8 at a time (the record's b1 and b2 of the check, in the entry's
    /// partition).
    #[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl")]
    unsafe fn buckets(&mut self, n: usize) {
        let (ep, pp, g1, g2) = (self.e.as_ptr(), self.part.as_ptr(), self.g1.as_mut_ptr(), self.g2.as_mut_ptr());
        let k0 = _mm512_set1_epi64(K0 as i64);
        for r in (0..n).step_by(8) {
            let e = _mm512_loadu_si512(ep.add(r).cast());
            let p = _mm512_slli_epi64::<{ NBITS }>(_mm512_cvtepu32_epi64(_mm256_loadu_si256(pp.add(r).cast())));
            let check = _mm512_srli_epi64::<{ CHECK_SHIFT }>(e);
            let h1 = _mm512_or_si512(p, _mm512_srli_epi64::<{ KBITS - NBITS }>(check));
            let h2 = _mm512_or_si512(p, _mm512_srli_epi64::<{ 64 - NBITS }>(_mm512_mullo_epi64(check, k0)));
            _mm256_storeu_si256(g1.add(r).cast(), _mm512_cvtepi64_epi32(h1));
            _mm256_storeu_si256(g2.add(r).cast(), _mm512_cvtepi64_epi32(h2));
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Racy,
    Atomic,
    Simd,
    Hash,
}

/// What every insert loop needs: the table, the fills (u8 for Racy, u32 otherwise), the slots per bucket, a thread's
/// sink line for dropped entries, and the prefetch distance.
#[derive(Clone, Copy)]
struct Ctx {
    table: *mut u64,
    fills: *mut u8,
    slots: usize,
    sink: *mut u64,
    dist: usize,
}

/// prefetchw (in asm: _MM_HINT_ET0 needs the unstable prfchw target feature, else LLVM emits prefetcht0).
#[inline(always)]
unsafe fn prefetchw<T>(p: *const T) {
    std::arch::asm!("prefetchw [{p}]", p = in(reg) p, options(nostack, preserves_flags, readonly));
}

#[inline(always)]
unsafe fn store(dst: *mut u64, e: u64) {
    (*(dst as *const AtomicU64)).store(e, Relaxed);
}

/// A u8 or u32 fill count, read and written with relaxed loads and stores (plain movs).
trait Fill {
    const BYTES: usize;
    unsafe fn get(fills: *mut u8, h: usize) -> usize;
    unsafe fn set(fills: *mut u8, h: usize, f: usize);
}
struct U8;
struct U32;
impl Fill for U8 {
    const BYTES: usize = 1;
    #[inline(always)]
    unsafe fn get(fills: *mut u8, h: usize) -> usize {
        (*(fills.add(h) as *const AtomicU8)).load(Relaxed) as usize
    }
    #[inline(always)]
    unsafe fn set(fills: *mut u8, h: usize, f: usize) {
        (*(fills.add(h) as *const AtomicU8)).store(f as u8, Relaxed)
    }
}
impl Fill for U32 {
    const BYTES: usize = 4;
    #[inline(always)]
    unsafe fn get(fills: *mut u8, h: usize) -> usize {
        (*(fills.cast::<AtomicU32>().add(h))).load(Relaxed) as usize
    }
    #[inline(always)]
    unsafe fn set(fills: *mut u8, h: usize, f: usize) {
        (*(fills.cast::<AtomicU32>().add(h))).store(f as u32, Relaxed)
    }
}

/// Where entry i goes now: its bucket (b1 if its fill is below slots, else b2) and that bucket's fill.
#[inline(always)]
unsafe fn target<F: Fill>(c: &Ctx, g1: *const u32, g2: *const u32, i: usize) -> (usize, usize) {
    let (h1, h2) = (*g1.add(i) as usize, *g2.add(i) as usize);
    let (f1, f2) = (F::get(c.fills, h1), F::get(c.fills, h2));
    let in1 = f1 < c.slots;
    (select_unpredictable(in1, h1, h2), select_unpredictable(in1, f1, f2))
}

/// Insert entry i with plain loads and stores; returns 1 if it was kept.
#[inline(always)]
unsafe fn insert_one<F: Fill>(c: &Ctx, e: *const u64, g1: *const u32, g2: *const u32, i: usize) -> usize {
    let (h, f) = target::<F>(c, g1, g2, i);
    let keep = f < c.slots;
    store(select_unpredictable(keep, c.table.add(h * c.slots + f), c.sink), *e.add(i));
    F::set(c.fills, h, f + keep as usize);
    keep as usize
}

/// Prefetch for entry i: the fills of entry i + 2 * dist, the table line of entry i + dist.
#[inline(always)]
unsafe fn ahead<F: Fill>(c: &Ctx, g1: *const u32, g2: *const u32, i: usize, m: usize) {
    let j = (i + 2 * c.dist).min(m - 1);
    prefetchw(c.fills.add(*g1.add(j) as usize * F::BYTES));
    prefetchw(c.fills.add(*g2.add(j) as usize * F::BYTES));
    let (h, f) = target::<F>(c, g1, g2, (i + c.dist).min(m - 1));
    prefetchw(c.table.add(h * c.slots + f));
}

/// "racy": the m entries one at a time with u8 fills. Returns the entries kept (as this thread saw them).
#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,bmi2")]
unsafe fn insert_racy(c: &Ctx, h: &Hashes, m: usize) -> usize {
    let (e, g1, g2) = (h.e.as_ptr(), h.g1.as_ptr(), h.g2.as_ptr());
    let mut kept = 0;
    for i in 0..m {
        if c.dist > 0 {
            ahead::<U8>(c, g1, g2, i, m);
        }
        kept += insert_one::<U8>(c, e, g1, g2, i);
    }
    kept
}

/// "atomic": the m entries one at a time with u32 fills, claiming a slot with fetch_add. Returns the entries kept.
#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,bmi2")]
unsafe fn insert_atomic(c: &Ctx, h: &Hashes, m: usize) -> usize {
    let (e, g1, g2) = (h.e.as_ptr(), h.g1.as_ptr(), h.g2.as_ptr());
    let fills = c.fills.cast::<AtomicU32>();
    let mut kept = 0;
    for i in 0..m {
        if c.dist > 0 {
            ahead::<U32>(c, g1, g2, i, m);
        }
        let (h1, h2) = (*g1.add(i) as usize, *g2.add(i) as usize);
        let in1 = ((*fills.add(h1)).load(Relaxed) as usize) < c.slots;
        let h = select_unpredictable(in1, h1, h2);
        let f = (*fills.add(h)).fetch_add(1, Relaxed) as usize;
        let keep = f < c.slots;
        store(select_unpredictable(keep, c.table.add(h * c.slots + f), c.sink), *e.add(i));
        kept += keep as usize;
    }
    kept
}

/// "simd": the m entries 16 at a time with u32 fills (gather both fills, blend, scatter the entries and the new fills),
/// a group with two entries in one bucket one entry at a time. Prefetches as the scalar loops, a group at a time.
#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx512cd,bmi2")]
unsafe fn insert_simd(c: &Ctx, h: &Hashes, m: usize) -> usize {
    let (e, g1, g2) = (h.e.as_ptr(), h.g1.as_ptr(), h.g2.as_ptr());
    let fills = c.fills.cast::<i32>();
    let vslots32 = _mm512_set1_epi32(c.slots as i32);
    let vslots64 = _mm512_set1_epi64(c.slots as i64);
    let one = _mm512_set1_epi32(1);
    let mut kept = 0;
    for i0 in (0..m).step_by(16) {
        let lanes = (m - i0).min(16);
        if c.dist > 0 {
            for j in 0..16 {
                ahead::<U32>(c, g1, g2, i0 + j, m);
            }
        }
        let k: __mmask16 = ((1u32 << lanes) - 1) as u16;
        let v1 = _mm512_maskz_loadu_epi32(k, g1.add(i0).cast());
        let v2 = _mm512_maskz_loadu_epi32(k, g2.add(i0).cast());
        let f1 = _mm512_mask_i32gather_epi32::<4>(_mm512_setzero_si512(), k, v1, fills);
        let f2 = _mm512_mask_i32gather_epi32::<4>(_mm512_setzero_si512(), k, v2, fills);
        let in1 = _mm512_cmplt_epu32_mask(f1, vslots32);
        let hv = _mm512_mask_blend_epi32(in1, v2, v1);
        let fv = _mm512_mask_blend_epi32(in1, f2, f1);
        let conflict = _mm512_conflict_epi32(hv);
        if _mm512_mask_test_epi32_mask(k, conflict, conflict) != 0 {
            for j in 0..lanes {
                kept += insert_one::<U32>(c, e, g1, g2, i0 + j);
            }
            continue;
        }
        let keep = k & _mm512_cmplt_epu32_mask(fv, vslots32);
        let off = |h: __m256i, f: __m256i| _mm512_add_epi64(_mm512_mul_epu32(_mm512_cvtepu32_epi64(h), vslots64), _mm512_cvtepu32_epi64(f));
        let (hlo, hhi) = (_mm512_castsi512_si256(hv), _mm512_extracti64x4_epi64::<1>(hv));
        let (flo, fhi) = (_mm512_castsi512_si256(fv), _mm512_extracti64x4_epi64::<1>(fv));
        let elo = _mm512_maskz_loadu_epi64(k as u8, e.add(i0).cast());
        let ehi = _mm512_maskz_loadu_epi64((k >> 8) as u8, e.add(i0 + 8).cast());
        _mm512_mask_i64scatter_epi64::<8>(c.table.cast(), keep as u8, off(hlo, flo), elo);
        _mm512_mask_i64scatter_epi64::<8>(c.table.cast(), (keep >> 8) as u8, off(hhi, fhi), ehi);
        _mm512_mask_i32scatter_epi32::<4>(fills, keep, hv, _mm512_add_epi32(fv, one));
        kept += keep.count_ones() as usize;
    }
    kept
}

struct Builder {
    files: Vec<Shard>,
    sizes: Vec<usize>,
    pieces: Vec<(usize, usize)>,
    slots: usize,
    pool: rayon::ThreadPool,
    table: Map<u64>,
    fills: Map<u8>, // BUCKETS u8 ("racy") or u32 counts
}

#[pyclass]
pub struct SlotIndexOnePass {
    file: File,
    mode: Mode,
    dist: usize,
    builder: Builder,
    built: AtomicBool,
    claimed: AtomicUsize, // entries the threads kept, as each saw it
    checksum: AtomicUsize, // "hash": the xor of the buckets, so the hashing is not optimized away
}

#[pymethods]
impl SlotIndexOnePass {
    /// Off the clock: the record's create for the first `tokens` tokens of `files` (the table a memfd, faulted in on
    /// the build pool) and the fill counts, also faulted in.
    #[staticmethod]
    #[pyo3(signature = (files, tokens, threads, cpus, mode = "racy", dist = 16, huge = false))]
    fn create(py: Python<'_>, files: Vec<String>, tokens: usize, threads: usize, cpus: Vec<usize>, mode: &str, dist: usize, huge: bool) -> PyResult<Self> {
        if !*AVX512 {
            return Err(PyValueError::new_err("SlotIndexOnePass needs AVX-512 with CD, IFMA and VBMI"));
        }
        let mode = match mode {
            "racy" => Mode::Racy,
            "atomic" => Mode::Atomic,
            "simd" => Mode::Simd,
            "hash" => Mode::Hash,
            _ => return Err(PyValueError::new_err(format!("unknown mode {mode:?}"))),
        };
        if threads == 0 {
            return Err(PyValueError::new_err("need at least one thread"));
        }
        let files = shards(&files, tokens)?;
        let sizes: Vec<usize> = files.iter().map(|s| s.len).collect();
        let slots = slots_for(sizes.iter().sum())?;
        let table_len = BUCKETS * slots;
        let name = c"cache_ladder-slot-one-pass";
        let fd = unsafe { libc::syscall(libc::SYS_memfd_create, name.as_ptr(), libc::MFD_CLOEXEC) };
        if fd < 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        let file = unsafe { File::from_raw_fd(fd as i32) };
        let table = if huge {
            Map::new(table_len)?
        } else {
            file.set_len((table_len * 8) as u64)?;
            Map { ptr: mmap(table_len * 8, libc::MAP_SHARED, file.as_raw_fd())?.cast(), len: table_len, bytes: table_len * 8 }
        };
        let fill_bytes = if mode == Mode::Racy { 1 } else { 4 };
        let builder = Builder {
            pieces: sizes.iter().enumerate().flat_map(|(f, &n)| (MIN..n).step_by(PIECE).map(move |a| (f, a))).collect(),
            files,
            sizes,
            slots,
            pool: pinned_pool(threads, &cpus, "slotonepass")?,
            table,
            fills: Map::new(BUCKETS * fill_bytes)?,
        };
        let b = &builder;
        py.detach(|| {
            b.pool.broadcast(|ctx| {
                let (i, n) = (ctx.index(), ctx.num_threads());
                touch(b.fills.part(b.fills.len * i / n, b.fills.len * (i + 1) / n));
                touch(b.table.part(table_len * i / n, table_len * (i + 1) / n));
            });
        });
        Ok(Self { file, mode, dist, builder, built: AtomicBool::new(false), claimed: AtomicUsize::new(0), checksum: AtomicUsize::new(0) })
    }

    /// On the clock: the one-pass build. Returns the table's bytes, slots, threads, and the build's seconds.
    fn build(&self, py: Python<'_>) -> PyResult<HashMap<String, f64>> {
        if self.built.swap(true, Relaxed) {
            return Err(PyValueError::new_err("already built"));
        }
        let (b, mode, dist) = (&self.builder, self.mode, self.dist);
        let seconds = py.detach(|| -> std::io::Result<f64> {
            let t0 = Instant::now();
            let done = b.pool.broadcast(|ctx| -> std::io::Result<()> {
                let (mut hashes, mut tokens) = (Hashes::new(), vec![STOP; PIECE + BACK]);
                let mut sink = [0u64; 8];
                let c = Ctx { table: b.table.ptr, fills: b.fills.ptr, slots: b.slots, sink: sink.as_mut_ptr(), dist };
                let (mut kept, mut sum) = (0, 0usize);
                for &(f, a) in b.pieces.iter().skip(ctx.index()).step_by(ctx.num_threads()) {
                    let (n, lo) = ((a + PIECE).min(b.sizes[f]) - a, a.saturating_sub(BACK));
                    let pad = BACK - (a - lo);
                    tokens[..pad].fill(STOP);
                    pread(&b.files[f].file, &mut tokens[pad..BACK + n], HEADER_BYTES + lo * 2)?;
                    for r in (0..n).step_by(STEP) {
                        let m = STEP.min(n - r);
                        unsafe {
                            hashes.hash_avx512(&tokens, BACK + r, m, true);
                            hashes.buckets(m);
                            kept += match mode {
                                Mode::Racy => insert_racy(&c, &hashes, m),
                                Mode::Atomic => insert_atomic(&c, &hashes, m),
                                Mode::Simd => insert_simd(&c, &hashes, m),
                                Mode::Hash => {
                                    sum ^= hashes.g1[..m].iter().zip(&hashes.g2[..m]).fold(0, |s, (&x, &y)| s ^ (x as usize) ^ (y as usize) << 28);
                                    0
                                }
                            };
                        }
                    }
                }
                std::hint::black_box(&sink);
                self.claimed.fetch_add(kept, Relaxed);
                self.checksum.fetch_xor(sum, Relaxed);
                Ok(())
            });
            done.into_iter().collect::<std::io::Result<()>>()?;
            Ok(t0.elapsed().as_secs_f64())
        })?;
        Ok(HashMap::from([
            ("bytes".into(), (b.table.len * 8) as f64),
            ("fill_bytes".into(), b.fills.len as f64),
            ("slots".into(), b.slots as f64),
            ("threads".into(), b.pool.current_num_threads() as f64),
            ("dist".into(), self.dist as f64),
            ("build_s".into(), seconds),
        ]))
    }

    /// Off the clock, after build: positions (entries offered), entries (kept in the table: the fills, capped at
    /// slots), claimed (kept as the threads saw it), lost (claimed - entries: slots written twice by racing threads)
    /// and dropped (positions - claimed: both buckets full).
    fn counts(&self, py: Python<'_>) -> HashMap<String, f64> {
        let b = &self.builder;
        let (slots, racy) = (b.slots, self.mode == Mode::Racy);
        let entries: usize = py.detach(|| {
            b.pool.install(|| {
                (0..BUCKETS)
                    .into_par_iter()
                    .with_min_len(1 << 16)
                    .map(|h| unsafe { if racy { U8::get(b.fills.ptr, h) } else { U32::get(b.fills.ptr, h) } }.min(slots))
                    .sum()
            })
        });
        let positions: usize = b.sizes.iter().map(|&n| n.saturating_sub(MIN)).sum();
        let claimed = self.claimed.load(Relaxed);
        HashMap::from([
            ("positions".into(), positions as f64),
            ("entries".into(), entries as f64),
            ("claimed".into(), claimed as f64),
            ("lost".into(), claimed as f64 - entries as f64),
            ("dropped".into(), (positions - claimed) as f64),
            ("checksum".into(), self.checksum.load(Relaxed) as f64),
        ])
    }

    /// Off the clock: a path to the table (the memfd), valid while this index lives.
    #[getter]
    fn table_path(&self) -> String {
        format!("/proc/{}/fd/{}", std::process::id(), self.file.as_raw_fd())
    }
}
