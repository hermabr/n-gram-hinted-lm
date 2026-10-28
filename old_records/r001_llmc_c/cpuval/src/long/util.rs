//! The long-level engine's helpers, as val_build.rs has them: kgram's context hash, the filters' key and
//! blocked Bloom filter; and the threads' occurrence lists in block chains carved from arena chunks faulted before
//! the clock.
use rayon::prelude::*;
use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;

pub(crate) const TOP: usize = 1024; // cnt: the target's count if among its context's TOP most frequent next tokens
pub(crate) const KEEP: usize = 8; // the hint rows' entries
pub(crate) const PARTS_LOG2: u32 = 12; // ~4096 reduce partitions per level
pub(crate) const TABLE_LOAD: f64 = 0.5; // the context tables' maximum load
pub(crate) const FILTER_BITS: u32 = 28; // the 8-contexts' filter
pub(crate) const P: u64 = 0x9e3779b185ebca87; // kgram's context hash
pub(crate) const M: u64 = 0x9e3779b97f4a7c15; // the filters' mixer
pub(crate) const FP: u64 = (1 << 40) - 1; // the tables' fingerprints
pub(crate) const COUNTS: [usize; 6] = [2, 3, 5, 9, 17, 33]; // the hint cell's bin edges of a context's total
pub(crate) const PURITIES: [f64; 5] = [0.3, 0.5, 0.7, 0.85, 0.95]; // and of its top next token's share of it

/// A raw pointer that threads write disjoint parts of.
#[derive(Clone, Copy)]
pub(crate) struct Output<T>(pub(crate) *mut T);
unsafe impl<T> Sync for Output<T> {}
unsafe impl<T> Send for Output<T> {}
impl<T: Copy> Output<T> {
    pub(crate) unsafe fn set(self, offset: usize, value: T) {
        *self.0.add(offset) = value;
    }
}

/// kgram's context hash (nonzero).
#[inline]
pub(crate) fn hash(tokens: &[u16]) -> u64 {
    let mut h = tokens.iter().fold(0u64, |h, &t| h.wrapping_mul(P).wrapping_add(t as u64 + 1));
    h ^= h >> 30;
    h = h.wrapping_mul(0xbf58476d1ce4e5b9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94d049bb133111eb);
    (h ^ (h >> 31)) | 1
}

/// The filters' key of 4 to 8 tokens: two unaligned 8-byte loads (the first and the last 4 tokens) mixed by two
/// multiplications.
#[inline]
pub(crate) fn fkey(tokens: &[u16]) -> u64 {
    let n = tokens.len();
    debug_assert!((4..=8).contains(&n));
    let p = tokens.as_ptr();
    let a = unsafe { std::ptr::read_unaligned(p.cast::<u64>()) };
    let b = unsafe { std::ptr::read_unaligned(p.add(n - 4).cast::<u64>()) };
    (a.wrapping_mul(0x9e3779b97f4a7c15) ^ b ^ n as u64).wrapping_mul(0xd6e8feb86659fd93)
}

/// `n` zeros (zeroed in parallel) in a Vec whose pages are faulted as transparent huge pages, interleaved over the NUMA
/// nodes: the random probes of big tables then hit the TLB, and every node serves its share of them.
pub(crate) fn huge_zeroed<T: Copy + Default + Send + Sync>(n: usize) -> Vec<T> {
    let mut v: Vec<T> = Vec::with_capacity(n);
    let bytes = n * std::mem::size_of::<T>();
    if bytes >= 1 << 21 {
        let start = (v.as_mut_ptr() as usize).next_multiple_of(4096);
        let end = (v.as_mut_ptr() as usize + bytes) & !4095;
        if end > start {
            unsafe { libc::madvise(start as *mut libc::c_void, end - start, libc::MADV_HUGEPAGE) };
            super::numa::place(start as *mut u8, end - start, None);
        }
    }
    unsafe { v.set_len(n) };
    v.par_chunks_mut(1 << 16).for_each(|c| c.fill(T::default()));
    v
}

/// A Vec's reserved pages (whole pages inside it) interleaved over the NUMA nodes (and so left alone by the automatic
/// NUMA balancing); before they are faulted.
pub(crate) fn interleave<T>(v: &mut Vec<T>) {
    let start = (v.as_mut_ptr() as usize).next_multiple_of(4096);
    let end = (v.as_mut_ptr() as usize + v.capacity() * std::mem::size_of::<T>()) & !4095;
    if end > start {
        super::numa::place(start as *mut u8, end - start, None);
    }
}

/// A blocked Bloom filter: two bits in one 64-bit word per key.
pub(crate) struct Filter {
    pub(crate) words: Vec<u64>,
    shift: u32,
}

impl Filter {
    /// The filter in `words` (zeroed, a power of two of them).
    pub(crate) fn with_words(words: Vec<u64>, keys: impl ParallelIterator<Item = u64>) -> Self {
        let shift = 64 - words.len().trailing_zeros();
        {
            let atomics: &[AtomicU64] = unsafe { &*(words.as_slice() as *const [u64] as *const [AtomicU64]) };
            keys.for_each(|h| {
                let (w, b) = Self::at(h, shift);
                atomics[w].fetch_or(b, Ordering::Relaxed);
            });
        }
        Self { words, shift }
    }

    #[inline]
    fn at(h: u64, shift: u32) -> (usize, u64) {
        let m = h.wrapping_mul(M);
        ((m >> shift) as usize, 1u64 << ((m >> 20) & 63) | 1u64 << ((m >> 26) & 63))
    }

    #[inline]
    pub(crate) fn prefetch(&self, h: u64) {
        let (w, _) = Self::at(h, self.shift);
        unsafe { _mm_prefetch(self.words.as_ptr().wrapping_add(w).cast(), _MM_HINT_T0) };
    }

    #[inline]
    pub(crate) fn test(&self, h: u64) -> bool {
        let (w, b) = Self::at(h, self.shift);
        unsafe { *self.words.get_unchecked(w) & b == b }
    }
}

/// Arena chunks (64 MB, transparent huge pages) faulted before the clock, per NUMA node, shared by the threads' lists
/// (a thread takes its node's); a list that needs more maps a fresh one (counted: none should be needed on the clock).
pub(crate) struct Pool {
    free: Vec<Mutex<Vec<usize>>>, // per node
    home: Vec<(usize, usize)>,    // (chunk, its node), sorted
    all: Mutex<Vec<(usize, usize)>>,
    map: Vec<usize>, // per CPU its node
    pub(crate) fresh: AtomicUsize,
}

pub(crate) const CHUNK: usize = 8 << 20; // u64s per arena chunk (64 MB)

impl Pool {
    /// `bytes` of chunks, split over the nodes, faulted now (in parallel).
    pub(crate) fn new(bytes: usize) -> Self {
        let map = super::numa::cpu_nodes();
        let nodes = map.iter().max().map_or(1, |&n| n + 1);
        let per = bytes.div_ceil(CHUNK * 8).div_ceil(nodes);
        let free: Vec<Mutex<Vec<usize>>> = (0..nodes)
            .map(|node| {
                Mutex::new(
                    (0..per)
                        .into_par_iter()
                        .map(|_| {
                            let p = super::numa::map_on(CHUNK * 8, (nodes > 1).then_some(node)) as usize;
                            unsafe { std::slice::from_raw_parts_mut(p as *mut u64, CHUNK) }.fill(0);
                            p
                        })
                        .collect(),
                )
            })
            .collect();
        let all = free.iter().flat_map(|f| f.lock().unwrap().iter().map(|&p| (p, CHUNK)).collect::<Vec<_>>()).collect();
        let mut home: Vec<(usize, usize)> = free.iter().enumerate().flat_map(|(n, f)| f.lock().unwrap().iter().map(|&p| (p, n)).collect::<Vec<_>>()).collect();
        home.sort_unstable();
        Self { free, home, all: Mutex::new(all), map, fresh: AtomicUsize::new(0) }
    }

    /// A chunk of at least `words` u64s, of the calling thread's node if any is left (else of another node's).
    fn take(&self, words: usize) -> (*mut u64, usize) {
        if words <= CHUNK {
            let me = super::numa::this_node(&self.map).min(self.free.len() - 1);
            for k in 0..self.free.len() {
                if let Some(p) = self.free[(me + k) % self.free.len()].lock().unwrap().pop() {
                    return (p as *mut u64, CHUNK);
                }
            }
        }
        self.fresh.fetch_add(1, Ordering::Relaxed);
        let n = words.max(CHUNK);
        let p = super::numa::map_on(n * 8, None) as usize;
        self.all.lock().unwrap().push((p, n));
        (p as *mut u64, n)
    }

    /// Chunks back for another lookup, each to its node's list (fresh ones to the first).
    fn give(&self, chunks: &mut Vec<(*mut u64, usize)>) {
        for (p, n) in chunks.drain(..) {
            if n == CHUNK {
                let node = self.home.binary_search_by_key(&(p as usize), |h| h.0).map_or(0, |i| self.home[i].1);
                self.free[node].lock().unwrap().push(p as usize);
            }
        }
    }

    /// Chunks handed out at the moment (for the memory report).
    pub(crate) fn chunks(&self) -> usize {
        self.all.lock().unwrap().len()
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        for &(p, n) in self.all.lock().unwrap().iter() {
            unsafe { libc::munmap(p as *mut libc::c_void, (n * 8).next_multiple_of(1 << 21)) };
        }
    }
}

/// Per thread, per level, per partition: the occurrences found, in chains of blocks (64 entries, doubling up to 2^16;
/// each block's first two words link the previous block of its chain) carved from arena chunks of the pool: no
/// allocation in the lookup.
pub(crate) struct Lists {
    hot: Vec<Hot>,      // per (level, partition)
    first: Vec<usize>,  // per level its first hot entry
    chunks: Vec<(*mut u64, usize)>,
    at: *mut u64,
    left: usize,
    pool: std::sync::Arc<Pool>,
}

#[derive(Clone, Copy)]
struct Hot {
    cur: *mut u64,
    len: u32,
    cap: u32,
}

unsafe impl Send for Lists {}
unsafe impl Sync for Lists {}

impl Lists {
    pub(crate) fn new(parts: &[usize], pool: std::sync::Arc<Pool>) -> Self {
        let mut first = vec![0];
        for &p in parts {
            first.push(first.last().unwrap() + p);
        }
        let hot = vec![Hot { cur: std::ptr::null_mut(), len: 0, cap: 0 }; *first.last().unwrap()];
        Self { hot, first, chunks: Vec::with_capacity(4096), at: std::ptr::null_mut(), left: 0, pool }
    }

    /// Empty again, its chunks back in the pool.
    pub(crate) fn reset(&mut self) {
        self.hot.fill(Hot { cur: std::ptr::null_mut(), len: 0, cap: 0 });
        self.pool.give(&mut self.chunks);
        self.at = std::ptr::null_mut();
        self.left = 0;
    }

    #[inline]
    pub(crate) fn push(&mut self, i: usize, p: usize, v: u64) {
        let k = self.first[i] + p;
        let h = unsafe { self.hot.get_unchecked_mut(k) };
        if h.len == h.cap {
            self.grow(k);
        }
        let h = unsafe { self.hot.get_unchecked_mut(k) };
        unsafe { *h.cur.add(h.len as usize) = v };
        h.len += 1;
    }

    #[cold]
    #[inline(never)]
    fn grow(&mut self, k: usize) {
        let h = self.hot[k];
        let cap = if h.cap == 0 { 64 } else { (h.cap * 2).min(1 << 16) } as usize;
        if self.left < cap + 2 {
            let (p, n) = self.pool.take(cap + 2);
            self.chunks.push((p, n));
            self.at = p;
            self.left = n;
        }
        let b = self.at;
        self.at = self.at.wrapping_add(cap + 2);
        self.left -= cap + 2;
        unsafe {
            *b = h.cur as u64;
            *b.add(1) = h.len as u64;
        }
        self.hot[k] = Hot { cur: unsafe { b.add(2) }, len: 0, cap: cap as u32 };
    }

    /// Level i, partition p's blocks (pointer, length), newest first.
    #[inline]
    pub(crate) fn for_blocks(&self, i: usize, p: usize, mut f: impl FnMut(&[u64])) {
        let h = self.hot[self.first[i] + p];
        let (mut cur, mut len) = (h.cur, h.len as usize);
        while !cur.is_null() {
            f(unsafe { std::slice::from_raw_parts(cur, len) });
            unsafe {
                len = *cur.sub(1) as usize;
                cur = *cur.sub(2) as *mut u64;
            }
        }
    }
}
