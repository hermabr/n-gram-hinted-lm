//! The short levels without the validation tokens: per level k (a k-token context) the counts of every (context, next
//! token) pair in the first K_k training shards, built beside the training, then looked up for the validation
//! positions after the clock. The rows equal val_build.rs's (ApproxValCountRows / ApproxValHintRows with
//! `level_shards`) at these levels.
//!
//! A k-context's 16k bits (tokens + 1, the first token highest) are mixed by a bijection of 16k-bit integers into its
//! key H: equal keys are equal contexts, and H's top bits are uniform. An occurrence is the integer H << 16 | next.
//!   1. scatter: the shards' positions (contexts within a shard; a context with a BOS after its first token is never a
//!      validation context and is skipped) appended to their partition (H's top bits) in the threads' block lists
//!      (when they would outgrow the scratch, in rounds over ranges of the partitions, each reading all K_k shards);
//!   2. aggregate: per partition its occurrences sorted (2^SUB_BITS sub-buckets by H's next bits, each sorted; level
//!      2 over many shards: counted in a cache-resident hash table first), so the contexts are sorted by H and the
//!      pairs by H then token, into a compact part: per context the bits of H below the directory and its run's start;
//!      per pair the token and count (u16, larger counts in a side table); for the contexts with more than BIG next
//!      tokens their total, top KEEP and the TOP-th (count, token); a directory of H's top bits.
//! Lookup: per validation position and level the context's H found from the directory, its run's target count by
//! binary search, the rows as val_build.rs's reduce writes them.
use rayon::prelude::*;
use std::cmp::Reverse;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::sync::Mutex;
use std::time::Instant;

use crate::rows::{Lv, Out, Times, Val, BOS, HEADER_BYTES, KEEP, LEVELS, TOP};
const SUB_BITS: u32 = 10; // a partition's sub-buckets (2^SUB_BITS) for its sort
const HASH_FROM: usize = 50; // level 2 over at least this many shards counts its partitions in hash tables
const BIG: usize = 16; // contexts with more next tokens get their top KEEP (and TOP threshold) precomputed
const PIECE: usize = 1 << 20;
const COUNTS: [usize; 6] = [2, 3, 5, 9, 17, 33];
const PURITIES: [f64; 5] = [0.3, 0.5, 0.7, 0.85, 0.95];

/// An integer of up to 128 bits: a context key (16k bits) or an occurrence (16k + 16 bits).
pub trait Word: Copy + Ord + Eq + Default + Send + Sync + std::fmt::Debug + 'static {
    fn from(x: u128) -> Self;
    fn wide(self) -> u128;
    /// A hash of the value for the aggregation tables.
    fn slot(self, bits: u32) -> usize;
}

impl Word for u64 {
    #[inline]
    fn from(x: u128) -> Self {
        x as u64
    }
    #[inline]
    fn wide(self) -> u128 {
        self as u128
    }
    #[inline]
    fn slot(self, bits: u32) -> usize {
        (self.wrapping_mul(0x9e3779b97f4a7c15) >> (64 - bits)) as usize
    }
}

impl Word for u128 {
    #[inline]
    fn from(x: u128) -> Self {
        x
    }
    #[inline]
    fn wide(self) -> u128 {
        self
    }
    #[inline]
    fn slot(self, bits: u32) -> usize {
        let f = (self as u64) ^ ((self >> 64) as u64).wrapping_mul(0xd6e8feb86659fd93);
        (f.wrapping_mul(0x9e3779b97f4a7c15) >> (64 - bits)) as usize
    }
}

/// The bijection of b-bit integers (b = 16k <= 112): odd multiplications and xor-shifts modulo 2^b.
#[inline]
pub fn mix(c: u128, b: u32) -> u128 {
    if b <= 64 {
        mix64(c as u64, b) as u128
    } else {
        let m = (1u128 << b) - 1;
        let mut x = c;
        x = x.wrapping_mul(0x9e3779b97f4a7c15_d6e8feb86659fd93) & m;
        x ^= x >> (b / 2);
        x = x.wrapping_mul(0xbf58476d1ce4e5b9_94d049bb133111eb) & m;
        x ^= x >> (b / 2 - 5);
        x = x.wrapping_mul(0x2545f4914f6cdd1d_9e3779b185ebca87) & m;
        x ^= x >> (b / 2 + 3);
        x
    }
}

/// mix for b <= 64.
#[inline]
pub fn mix64(c: u64, b: u32) -> u64 {
    let m = if b == 64 { u64::MAX } else { (1u64 << b) - 1 };
    let mut x = c;
    x = x.wrapping_mul(0x9e3779b97f4a7c15) & m;
    x ^= x >> (b / 2);
    x = x.wrapping_mul(0xbf58476d1ce4e5b9) & m;
    x ^= x >> (b / 2 - 3);
    x = x.wrapping_mul(0x94d049bb133111eb) & m;
    x ^= x >> (b / 2 + 1);
    x
}

/// The packed context: tokens + 1, 16 bits each, the first token highest.
#[inline]
fn pack(tokens: &[u16]) -> u128 {
    tokens.iter().fold(0u128, |c, &t| c << 16 | (t as u128 + 1))
}

const CHUNK_BYTES: usize = 64 << 20;
const BLOCK_BYTES: usize = 4096; // a list block

/// The NUMA node of the calling thread's CPU.
fn this_node() -> usize {
    let (mut cpu, mut node) = (0u32, 0u32);
    let r = unsafe { libc::syscall(libc::SYS_getcpu, &mut cpu as *mut u32, &mut node as *mut u32, std::ptr::null_mut::<libc::c_void>()) };
    if r == 0 && node < 64 {
        node as usize
    } else {
        0
    }
}

/// A chunk of `bytes`, transparent huge pages, bound to NUMA node `node`. Bound, its pages
/// come from that node even when the node's free memory is page cache (the kernel reclaims it); unbound they would
/// come from the other node, and half the threads would work remotely.
fn map_chunk(bytes: usize, node: usize) -> *mut u8 {
    let flags = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE;
    let p = unsafe { libc::mmap(std::ptr::null_mut(), bytes, libc::PROT_READ | libc::PROT_WRITE, flags, -1, 0) };
    assert!(p != libc::MAP_FAILED, "arena mmap failed");
    unsafe { libc::madvise(p, bytes, libc::MADV_HUGEPAGE) };
    let mask: u64 = 1 << node;
    const MPOL_BIND: i32 = 2;
    unsafe { libc::syscall(libc::SYS_mbind, p, bytes, MPOL_BIND, &mask as *const u64, 64u64, 0u32) };
    p.cast()
}

/// A chunk: its memory, size and node.
#[derive(Clone, Copy)]
struct Chunk(*mut u8, usize, usize);
unsafe impl Send for Chunk {}

/// Chunks per NUMA node, pre-faulted before the clock and shared by the node's threads (a thread takes from its own
/// node's, then any node's, then maps a new one); every chunk unmapped when the pool drops.
pub struct Pool {
    free: Vec<Mutex<Vec<Chunk>>>,
    all: Mutex<Vec<Chunk>>,
}

impl Pool {
    pub fn new() -> Self {
        Self { free: (0..64).map(|_| Mutex::new(Vec::new())).collect(), all: Mutex::new(Vec::new()) }
    }

    fn take(&self, bytes: usize) -> Chunk {
        let node = this_node();
        for n in std::iter::once(node).chain((0..self.free.len()).filter(|&n| n != node)) {
            let mut f = self.free[n].lock().unwrap();
            if let Some(i) = f.iter().rposition(|c| c.1 >= bytes) {
                return f.swap_remove(i);
            }
        }
        let size = bytes.max(CHUNK_BYTES);
        let c = Chunk(map_chunk(size, node), size, node);
        self.all.lock().unwrap().push(c);
        c
    }

    fn give(&self, c: Chunk) {
        self.free[c.2].lock().unwrap().push(c);
    }

    /// `bytes` more in chunks on the calling thread's node, every page touched.
    pub fn prefault(&self, bytes: usize) {
        let node = this_node();
        let mut done = 0;
        while done < bytes {
            let c = Chunk(map_chunk(CHUNK_BYTES, node), CHUNK_BYTES, node);
            for off in (0..CHUNK_BYTES).step_by(4096) {
                unsafe { std::ptr::write_volatile(c.0.add(off), 0) };
            }
            self.all.lock().unwrap().push(c);
            self.give(c);
            done += CHUNK_BYTES;
        }
    }

    pub fn bytes(&self) -> usize {
        self.all.lock().unwrap().iter().map(|c| c.1).sum()
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        for c in self.all.lock().unwrap().iter() {
            unsafe { libc::munmap(c.0.cast(), c.1) };
        }
    }
}

/// A thread's arena: 64-byte-aligned bump allocation from its pool's chunks; reset gives them back.
pub struct Arena {
    pool: std::sync::Arc<Pool>,
    chunks: Vec<Chunk>,
    at: *mut u8,
    left: usize,
}
unsafe impl Send for Arena {}

impl Arena {
    pub fn new(pool: std::sync::Arc<Pool>) -> Self {
        Self { pool, chunks: Vec::new(), at: std::ptr::null_mut(), left: 0 }
    }

    #[inline]
    fn alloc(&mut self, bytes: usize) -> *mut u8 {
        if self.left < bytes {
            let c = self.pool.take(bytes);
            self.chunks.push(c);
            self.at = c.0;
            self.left = c.1;
        }
        let r = self.at;
        self.at = self.at.wrapping_add(bytes);
        self.left -= bytes;
        r
    }

    pub fn reset(&mut self) {
        for c in self.chunks.drain(..) {
            self.pool.give(c);
        }
        self.at = std::ptr::null_mut();
        self.left = 0;
    }

    pub fn bytes(&self) -> usize {
        self.chunks.iter().map(|c| c.1).sum()
    }
}

/// Per thread, per partition: the round's occurrences in BLOCK_BYTES blocks from the thread's arena: the full ones and
/// the current one (its length).
struct Lists<E> {
    hot: Vec<(*mut E, u32)>,
    full: Vec<Vec<*mut E>>,
}

unsafe impl<E> Send for Lists<E> {}

impl<E: Copy> Lists<E> {
    const B: u32 = (BLOCK_BYTES / std::mem::size_of::<E>()) as u32;

    fn new(parts: usize) -> Self {
        Self { hot: vec![(std::ptr::null_mut(), Self::B); parts], full: vec![Vec::new(); parts] }
    }

    #[cold]
    #[inline(never)]
    fn grow(&mut self, p: usize, arena: &mut Arena) {
        let (cur, _) = self.hot[p];
        if !cur.is_null() {
            self.full[p].push(cur);
        }
        self.hot[p] = (arena.alloc(BLOCK_BYTES).cast(), 0);
    }

    #[inline]
    fn push(&mut self, p: usize, v: E, arena: &mut Arena) {
        if self.hot[p].1 == Self::B {
            self.grow(p, arena);
        }
        let h = &mut self.hot[p];
        unsafe { *h.0.add(h.1 as usize) = v };
        h.1 += 1;
    }

    /// Appends a write-combining line (64 bytes) with non-temporal stores; the blocks hold whole lines.
    #[inline]
    fn push_line(&mut self, p: usize, line: *const u8, arena: &mut Arena) {
        let h = unsafe { *self.hot.get_unchecked(p) };
        if h.1 == Self::B {
            self.grow(p, arena);
        }
        let h = unsafe { self.hot.get_unchecked_mut(p) };
        unsafe {
            use std::arch::x86_64::*;
            let d = h.0.add(h.1 as usize).cast::<__m128i>();
            let s = line.cast::<__m128i>();
            _mm_stream_si128(d, _mm_loadu_si128(s));
            _mm_stream_si128(d.add(1), _mm_loadu_si128(s.add(1)));
            _mm_stream_si128(d.add(2), _mm_loadu_si128(s.add(2)));
            _mm_stream_si128(d.add(3), _mm_loadu_si128(s.add(3)));
        }
        h.1 += (64 / std::mem::size_of::<E>()) as u32;
    }

    fn blocks(&self, p: usize) -> impl Iterator<Item = (*mut E, usize)> + '_ {
        let (cur, len) = self.hot[p];
        self.full[p].iter().map(|&b| (b, Self::B as usize)).chain((!cur.is_null() && len > 0).then_some((cur, len as usize)))
    }

    fn reset(&mut self) {
        for h in self.hot.iter_mut() {
            *h = (std::ptr::null_mut(), Self::B);
        }
        for f in self.full.iter_mut() {
            f.clear();
        }
    }
}

/// The engine's per-thread scratch, kept across levels (one per pool thread).
pub struct Scratch {
    pub arenas: Vec<Mutex<Arena>>,
    /// the finished tables' arrays (not reset)
    pub keep: Vec<Mutex<Arena>>,
    pub work_pool: std::sync::Arc<Pool>,
    pub keep_pool: std::sync::Arc<Pool>,
}

impl Scratch {
    /// For the current pool; `bytes` per thread pre-faulted on the thread itself.
    pub fn new(bytes: usize, keep_bytes: usize) -> Self {
        let n = rayon::current_num_threads();
        let (wp, kp) = (std::sync::Arc::new(Pool::new()), std::sync::Arc::new(Pool::new()));
        let arenas: Vec<Mutex<Arena>> = (0..n).map(|_| Mutex::new(Arena::new(wp.clone()))).collect();
        let keep: Vec<Mutex<Arena>> = (0..n).map(|_| Mutex::new(Arena::new(kp.clone()))).collect();
        rayon::broadcast(|_| {
            wp.prefault(bytes);
            kp.prefault(keep_bytes);
        });
        Self { arenas, keep, work_pool: wp, keep_pool: kp }
    }
}

/// A big context's precomputed total, top KEEP (count descending, token ascending) and, past TOP next tokens, the
/// TOP-th (count, token).
#[derive(Clone, Copy, Default)]
struct Big {
    tot: u32,
    top: [(u32, u16); KEEP],
    thr: Option<(u32, u16)>,
}

/// An array in an arena (freed with the arena).
pub struct Slab<T> {
    ptr: *mut T,
    len: usize,
}
unsafe impl<T> Send for Slab<T> {}
unsafe impl<T> Sync for Slab<T> {}

impl<T: Copy> Slab<T> {
    fn new(len: usize, arena: &mut Arena) -> Self {
        let bytes = (len * std::mem::size_of::<T>()).next_multiple_of(64).max(64);
        Self { ptr: arena.alloc(bytes).cast(), len }
    }
    #[inline]
    fn set(&mut self, i: usize, v: T) {
        debug_assert!(i < self.len);
        unsafe { *self.ptr.add(i) = v };
    }
}

impl<T> std::ops::Deref for Slab<T> {
    type Target = [T];
    #[inline]
    fn deref(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

/// A partition of a level's table, its contexts in H order: per context its key (H's `rbits` bits below the
/// partition's and the directory's, in `ks` bytes) and the start of its run of (token, count) pairs (tokens
/// ascending; counts of 65535 or more in `over`); the directory (per value of H's next `dbits` bits below the
/// partition's, its first context); the big contexts (more than BIG next tokens: total, top KEEP, TOP threshold).
struct Part {
    keys: Slab<u8>, // contexts * ks + 16
    ks: usize,
    rbits: u32,
    start: Slab<u32>, // contexts + 1
    tokens: Slab<u16>,
    counts: Slab<u16>,
    over: Slab<(u32, u32)>, // (pair, count) of the counts >= 65535, by pair
    dbits: u32,
    dir: Slab<u32>,
    big_ids: Slab<u32>,
    big: Slab<Big>,
}

impl Part {
    fn contexts(&self) -> usize {
        self.start.len - 1
    }
    fn bytes(&self) -> usize {
        self.keys.len + (self.start.len + self.dir.len + self.big_ids.len) * 4 + (self.tokens.len + self.counts.len) * 2 + self.over.len * 8 + self.big.len * std::mem::size_of::<Big>()
    }
    /// The interpolated position of context ci among the big ones.
    #[inline]
    fn big_guess(&self, ci: usize) -> usize {
        let nb = self.big_ids.len;
        if nb == 0 {
            return 0;
        }
        (ci * nb / self.contexts().max(1)).min(nb - 1)
    }

    /// Context ci's index among the big ones (it is one): from the guess, galloping, then binary search.
    #[inline]
    fn big_index(&self, ci: usize) -> usize {
        let ids = &self.big_ids;
        let c = ci as u32;
        let g = self.big_guess(ci);
        let (mut lo, mut hi);
        if ids[g] <= c {
            let mut step = 1;
            lo = g;
            hi = g + 1;
            while hi < ids.len && ids[hi] <= c {
                lo = hi;
                hi = (hi + step).min(ids.len);
                step *= 2;
            }
            hi = hi.min(ids.len);
        } else {
            let mut step = 1;
            hi = g;
            lo = g.saturating_sub(1);
            while lo > 0 && ids[lo] > c {
                hi = lo;
                lo = lo.saturating_sub(step);
                step *= 2;
            }
        }
        lo + ids[lo..hi.max(lo + 1)].binary_search(&c).expect("a big context")
    }

    #[inline]
    fn count(&self, i: usize) -> u32 {
        let c = self.counts[i];
        if c < u16::MAX {
            c as u32
        } else {
            let j = self.over.binary_search_by_key(&(i as u32), |o| o.0).expect("an overflowed count");
            self.over[j].1
        }
    }
}

/// A Part from its (occurrence, count) pairs in ascending order (`pairs` is iterated twice: sizes, then contents), in
/// `arena`; `below` = the bits of H below the partition's.
#[inline(never)]
fn build_part<I: Iterator<Item = (u128, u32)>>(pairs: impl Fn() -> I, below: u32, arena: &mut Arena, ranked: &mut Vec<(Reverse<u32>, u16)>) -> Part {
    let (mut np, mut nc, mut nb, mut no, mut d) = (0usize, 0usize, 0usize, 0usize, 0usize);
    let mut cur = u128::MAX;
    for (e, c) in pairs() {
        if e >> 16 != cur {
            nb += (d > BIG) as usize;
            cur = e >> 16;
            nc += 1;
            d = 0;
        }
        d += 1;
        np += 1;
        no += (c >= u16::MAX as u32) as usize;
    }
    nb += (d > BIG) as usize;
    let dbits = ((nc.max(2) as f64 / 4.0).log2().ceil() as u32).clamp(1, below.min(30));
    let dn = 1usize << dbits;
    let rbits = below - dbits;
    let ks = (rbits as usize).div_ceil(8).max(1);
    let mut p = Part {
        keys: Slab::new(nc * ks + 16, arena),
        ks,
        rbits,
        start: Slab::new(nc + 1, arena),
        tokens: Slab::new(np, arena),
        counts: Slab::new(np, arena),
        over: Slab::new(no, arena),
        dbits,
        dir: Slab::new(dn + 1, arena),
        big_ids: Slab::new(nb, arena),
        big: Slab::new(nb, arena),
    };
    let mask = (1u128 << rbits) - 1;
    let (mut ci, mut pi, mut bi, mut di, mut oi) = (0usize, 0usize, 0usize, 0usize, 0usize);
    let mut total = 0u64;
    // closes context ci - 1 (its pairs end at pi)
    let mut close = |p: &mut Part, ci: usize, pi: usize, total: u64, bi: &mut usize, oi: usize| {
        let c = ci - 1;
        let s = p.start[c] as usize;
        let d = pi - s;
        if d > BIG {
            ranked.clear();
            // the counts (the overflowed ones among the first oi of `over`, filled so far)
            let over = &p.over[..oi];
            ranked.extend((s..pi).map(|i| {
                let c = p.counts[i];
                let c = if c < u16::MAX { c as u32 } else { over[over.binary_search_by_key(&(i as u32), |o| o.0).expect("an overflowed count")].1 };
                (Reverse(c), p.tokens[i])
            }));
            let thr = (d > TOP).then(|| {
                ranked.select_nth_unstable(TOP - 1);
                (ranked[TOP - 1].0 .0, ranked[TOP - 1].1)
            });
            let head = d.min(TOP);
            ranked[..head].select_nth_unstable(KEEP - 1);
            ranked[..KEEP].sort_unstable();
            let mut big = Big { tot: u32::try_from(total).unwrap_or(u32::MAX), thr, ..Default::default() };
            for e in 0..KEEP {
                big.top[e] = (ranked[e].0 .0, ranked[e].1);
            }
            p.big_ids.set(*bi, c as u32);
            p.big.set(*bi, big);
            *bi += 1;
        }
    };
    let mut cur = u128::MAX;
    for (e, c) in pairs() {
        let h = e >> 16;
        if h != cur {
            if ci > 0 {
                close(&mut p, ci, pi, total, &mut bi, oi);
            }
            cur = h;
            // 16 bytes written, the next key's overwrite the excess (the slab has 16 bytes of slack)
            unsafe { std::ptr::write_unaligned(p.keys.ptr.add(ci * ks).cast::<u128>(), h & mask) };
            p.start.set(ci, pi as u32);
            let top = ((h >> rbits) & ((1u128 << dbits) - 1)) as usize;
            while di <= top {
                p.dir.set(di, ci as u32);
                di += 1;
            }
            ci += 1;
            total = 0;
        }
        p.tokens.set(pi, e as u16);
        if c >= u16::MAX as u32 {
            p.counts.set(pi, u16::MAX);
            p.over.set(oi, (pi as u32, c));
            oi += 1;
        } else {
            p.counts.set(pi, c as u16);
        }
        total += c as u64;
        pi += 1;
    }
    if ci > 0 {
        close(&mut p, ci, pi, total, &mut bi, oi);
    }
    p.start.set(nc, np as u32);
    while di <= dn {
        p.dir.set(di, nc as u32);
        di += 1;
    }
    p
}

/// A level's finished table: the partitions (H's top `pbits` bits).
pub struct LevelTable {
    pub k: usize,
    pub shards: usize,
    b: u32,
    pbits: u32,
    parts: Vec<Part>,
    hdrs: Vec<Hdr>,
    pub times: Vec<(String, f64)>,
}

/// A partition's fields the lookup's stages read, in one cache line.
#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct Hdr {
    dir: *const u32,
    keys: *const u8,
    start: *const u32,
    tokens: *const u16,
    counts: *const u16,
    ks: u32,
    rbits: u32,
    dbits: u32,
}
unsafe impl Send for Hdr {}
unsafe impl Sync for Hdr {}

impl Hdr {
    fn of(p: &Part) -> Self {
        Self { dir: p.dir.ptr, keys: p.keys.ptr, start: p.start.ptr, tokens: p.tokens.ptr, counts: p.counts.ptr, ks: p.ks as u32, rbits: p.rbits, dbits: p.dbits }
    }
    #[inline]
    fn key(&self, i: usize) -> u128 {
        let v = unsafe { std::ptr::read_unaligned(self.keys.add(i * self.ks as usize).cast::<u128>()) };
        v & ((1u128 << self.rbits) - 1)
    }
}

/// A level of any k in 2..=7.
pub type Level = LevelTable;

impl LevelTable {
    pub fn build(files: &[File], sizes: &[usize], k: usize, round_files: usize, part_bits: u32, scratch: &Scratch) -> Self {
        match k {
            2 | 3 => build_level::<u64, 8>(files, sizes, k, round_files, part_bits, scratch),
            4..=7 => build_level::<u128, 4>(files, sizes, k, round_files, part_bits, scratch),
            _ => panic!("short levels are 2..=7"),
        }
    }
    pub fn k(&self) -> usize {
        self.k
    }
    pub fn times(&self) -> &[(String, f64)] {
        &self.times
    }
    pub fn contexts(&self) -> usize {
        self.parts.iter().map(|p| p.contexts()).sum()
    }
    pub fn pairs(&self) -> usize {
        self.parts.iter().map(|p| p.tokens.len).sum()
    }
    pub fn bytes(&self) -> usize {
        self.parts.iter().map(|p| p.bytes()).sum()
    }
}

/// The shard pieces (file, start, end) of `sizes` from position k on.
fn pieces(sizes: &[usize], k: usize) -> Vec<(usize, usize, usize)> {
    let mut v = Vec::new();
    for (f, &n) in sizes.iter().enumerate() {
        let mut s = k;
        while s < n {
            let e = n.min(s + PIECE);
            v.push((f, s, e));
            s = e;
        }
    }
    v
}

/// A partition's round: its blocks' occurrences sorted into `out`: 2^sb buckets by the sb bits of H below the
/// partition's (bit `sub` of the occurrence up), each sorted. `hist` scratch.
#[inline(never)]
fn sort_partition<E: Word>(blocks: &[(*mut E, usize)], sub: u32, sb: u32, out: &mut Vec<E>, hist: &mut Vec<u32>) {
    let n: usize = blocks.iter().map(|b| b.1).sum();
    let nb = 1usize << sb;
    let mask = (nb - 1) as u128;
    hist.clear();
    hist.resize(nb + 1, 0);
    for &(bp, m) in blocks {
        for &e in unsafe { std::slice::from_raw_parts(bp, m) } {
            unsafe { *hist.get_unchecked_mut(((e.wide() >> sub) & mask) as usize + 1) += 1 };
        }
    }
    for i in 0..nb {
        hist[i + 1] += hist[i];
    }
    out.clear();
    out.reserve(n);
    unsafe { out.set_len(n) };
    let mut at = [0u32; 1 << 12];
    assert!(nb <= at.len());
    at[..nb].copy_from_slice(&hist[..nb]);
    for &(bp, m) in blocks {
        for &e in unsafe { std::slice::from_raw_parts(bp, m) } {
            let j = ((e.wide() >> sub) & mask) as usize;
            unsafe {
                let a = at.get_unchecked_mut(j);
                *out.get_unchecked_mut(*a as usize) = e;
                *a += 1;
            }
        }
    }
    for i in 0..nb {
        let (s, e) = (hist[i] as usize, hist[i + 1] as usize);
        if e - s > 1 {
            out[s..e].sort_unstable();
        }
    }
}

/// A thread's table for counting a partition's occurrences in cache (open addressing, 0 = empty: occurrences are
/// nonzero), grown at load 1/2.
struct HashScratch<E> {
    keys: Vec<E>,
    counts: Vec<u32>,
    bits: u32,
    used: usize,
}

impl<E: Word> HashScratch<E> {
    fn new() -> Self {
        Self { keys: vec![E::default(); 1 << 12], counts: vec![0; 1 << 12], bits: 12, used: 0 }
    }

    #[cold]
    fn grow(&mut self) {
        let bits = self.bits + 1;
        let mut keys = vec![E::default(); 1 << bits];
        let mut counts = vec![0u32; 1 << bits];
        let mask = (1usize << bits) - 1;
        for (i, &k) in self.keys.iter().enumerate() {
            if k != E::default() {
                let mut s = k.slot(bits);
                while keys[s] != E::default() {
                    s = (s + 1) & mask;
                }
                keys[s] = k;
                counts[s] = self.counts[i];
            }
        }
        (self.keys, self.counts, self.bits) = (keys, counts, bits);
    }

    /// The blocks' distinct occurrences with their counts appended to `out` (unordered); the table left empty.
    fn count(&mut self, blocks: &[(*mut E, usize)], out: &mut Vec<(E, u32)>) {
        const AHEAD: usize = 8;
        for &(bp, m) in blocks {
            let xs = unsafe { std::slice::from_raw_parts(bp, m) };
            for (i, &e) in xs.iter().enumerate() {
                if 2 * (self.used + 1) > self.keys.len() {
                    self.grow();
                }
                if i + AHEAD < m {
                    pf(self.keys.as_ptr().wrapping_add(xs[i + AHEAD].slot(self.bits)));
                }
                let mask = self.keys.len() - 1;
                let mut s = e.slot(self.bits);
                loop {
                    let k = unsafe { *self.keys.get_unchecked(s) };
                    if k == e {
                        unsafe { *self.counts.get_unchecked_mut(s) += 1 };
                        break;
                    }
                    if k == E::default() {
                        unsafe {
                            *self.keys.get_unchecked_mut(s) = e;
                            *self.counts.get_unchecked_mut(s) = 1;
                        }
                        self.used += 1;
                        break;
                    }
                    s = (s + 1) & mask;
                }
            }
        }
        for (k, c) in self.keys.iter_mut().zip(self.counts.iter_mut()) {
            if *k != E::default() {
                out.push((*k, *c));
                *k = E::default();
            }
        }
        self.used = 0;
    }
}

/// Pairs sorted by occurrence into `out`: 2^sb buckets by bits `sub`.., each sorted.
fn sort_pairs<E: Word>(src: &[(E, u32)], sub: u32, sb: u32, out: &mut Vec<(E, u32)>) {
    let nb = 1usize << sb;
    let mask = (nb - 1) as u128;
    let mut hist = [0u32; (1 << 12) + 1];
    for x in src {
        hist[((x.0.wide() >> sub) & mask) as usize + 1] += 1;
    }
    for i in 0..nb {
        hist[i + 1] += hist[i];
    }
    out.clear();
    out.reserve(src.len());
    unsafe { out.set_len(src.len()) };
    let mut at = [0u32; 1 << 12];
    at[..nb].copy_from_slice(&hist[..nb]);
    for &x in src {
        let j = ((x.0.wide() >> sub) & mask) as usize;
        unsafe { *out.get_unchecked_mut(at[j] as usize) = x };
        at[j] += 1;
    }
    for i in 0..nb {
        let (a, b) = (hist[i] as usize, hist[i + 1] as usize);
        if b - a > 1 {
            out[a..b].sort_unstable_by_key(|x| x.0);
        }
    }
}

/// Per thread: the shard reader's buffer, the write-combining lines (L occurrences per partition) and the lists.
struct ThreadState<E, const L: usize> {
    lists: Lists<E>,
    wc: Vec<[E; L]>,
    fill: Vec<u8>,
    buf: Vec<u16>,
    sorted: Vec<E>,
}
unsafe impl<E, const L: usize> Send for ThreadState<E, L> {}

/// Builds level k's table over `files` (all of them), in the current rayon pool. E: the occurrence type (16k + 16
/// bits), C: the context type (16k bits), L: occurrences per 64-byte line.
pub fn build_level<E: Word, const L: usize>(files: &[File], sizes: &[usize], k: usize, round_files: usize, part_bits: u32, scratch: &Scratch) -> LevelTable {
    let b = 16 * k as u32;
    assert!(b + 16 <= 8 * std::mem::size_of::<E>() as u32);
    assert!(L * std::mem::size_of::<E>() == 64);
    let mut times: Vec<(String, f64)> = Vec::new();
    let t0 = Instant::now();
    let parts = 1usize << part_bits;
    let below = b - part_bits;
    let sb: u32 = SUB_BITS.min(below);
    let sub = 16 + below - sb; // the sub-bucket bits in an occurrence
    // level 2 over many shards (most occurrences repeat a pair) counts each partition in a cache-resident hash table first
    let hashed = k == 2 && files.len() >= HASH_FROM;
    let threads = rayon::current_num_threads();
    assert!(scratch.arenas.len() >= threads, "the scratch is for a smaller pool");
    let states: Vec<Mutex<ThreadState<E, L>>> = (0..threads)
        .map(|_| Mutex::new(ThreadState { lists: Lists::new(parts), wc: vec![[E::default(); L]; parts], fill: vec![0; parts], buf: Vec::new(), sorted: Vec::new() }))
        .collect();
    // rounds: (files, partitions); past round_files shards rounds of partition ranges over all the shards (each
    // partition aggregated once; the scatter's hashing repeated per round)
    let nrounds = files.len().div_ceil(round_files.max(1)).max(1);
    let rounds: Vec<(usize, usize, usize, usize)> = (0..nrounds).map(|r| (0, files.len(), r * parts / nrounds, (r + 1) * parts / nrounds)).collect();
    let hscr: Vec<Mutex<(HashScratch<E>, Vec<(E, u32)>, Vec<(E, u32)>)>> = (0..threads).map(|_| Mutex::new((HashScratch::new(), Vec::new(), Vec::new()))).collect();
    let mut finals: Vec<Option<Part>> = (0..parts).map(|_| None).collect();
    let t_setup = t0.elapsed().as_secs_f64();
    let (mut t_scatter, mut t_agg) = (0.0, 0.0);
    let mut arena_peak = 0usize;
    for &(r0, r1, plo, phi) in &rounds {
        let t = Instant::now();
        let ps = pieces(&sizes[r0..r1], k);
        ps.par_iter().for_each(|&(f, start, end)| {
            let f = f + r0;
            let ti = rayon::current_thread_index().unwrap_or(0);
            let mut guard = states[ti].lock().unwrap();
            let st = &mut *guard;
            let back = start - k; // start >= k
            st.buf.resize(end - back, 0);
            let bytes = unsafe { std::slice::from_raw_parts_mut(st.buf.as_mut_ptr().cast::<u8>(), 2 * (end - back)) };
            files[f].read_exact_at(bytes, (HEADER_BYTES + 2 * back) as u64).expect("a shard read failed");
            let mut arena = scratch.arenas[ti].lock().unwrap();
            let arena = &mut *arena;
            let ThreadState { lists, wc, fill, buf, .. } = st;
            let tok: &[u16] = buf;
            // local index i = shard position back + i; positions q = k .. len: context tok[q-k..q], next tok[q]
            // the last BOS at a local index < q
            let last0: isize = tok[..k].iter().rposition(|&t| t == BOS).map_or(-1, |p| p as isize);
            // the rolling context in a u64 (k <= 4) or a u128
            macro_rules! scan {
                ($t:ty, $mix:expr) => {{
                    let mask: $t = if b as usize == 8 * std::mem::size_of::<$t>() { <$t>::MAX } else { ((1 as $t) << b) - 1 };
                    let mut c: $t = pack(&tok[0..k]) as $t;
                    let mut last_bos = last0;
                    for q in k..tok.len() {
                        // context tok[q-k..q]: skip if a BOS at q-k+1 .. q-1
                        let t = unsafe { *tok.get_unchecked(q) };
                        if last_bos <= (q - k) as isize {
                            let h: $t = $mix(c);
                            let e = E::from((h as u128) << 16 | t as u128);
                            let p = (h >> below) as usize;
                            if p < plo || p >= phi {
                                if t == BOS {
                                    last_bos = q as isize;
                                }
                                c = (c << 16 | (t as $t + 1)) & mask;
                                continue;
                            }
                            let fl = unsafe { fill.get_unchecked_mut(p) };
                            unsafe { *wc.get_unchecked_mut(p).get_unchecked_mut(*fl as usize) = e };
                            *fl += 1;
                            if *fl as usize == L {
                                *fl = 0;
                                lists.push_line(p, unsafe { wc.get_unchecked(p) }.as_ptr().cast(), arena);
                            }
                        }
                        if t == BOS {
                            last_bos = q as isize;
                        }
                        c = (c << 16 | (t as $t + 1)) & mask;
                    }
                }};
            }
            if b <= 64 {
                scan!(u64, |c: u64| mix64(c, b));
            } else {
                scan!(u128, |c: u128| mix(c, b));
            }
        });
        // the partial lines (each thread its own)
        rayon::broadcast(|ctx| {
            unsafe { std::arch::x86_64::_mm_sfence() }; // the thread's non-temporal stores
            let mut g = states[ctx.index()].lock().unwrap();
            let mut arena = scratch.arenas[ctx.index()].lock().unwrap();
            let ThreadState { lists, wc, fill, .. } = &mut *g;
            for p in plo..phi {
                for j in 0..fill[p] as usize {
                    lists.push(p, wc[p][j], &mut arena);
                }
                fill[p] = 0;
            }
        });
        t_scatter += t.elapsed().as_secs_f64();
        let t = Instant::now();
        let mut guards: Vec<_> = states.iter().map(|a| a.lock().unwrap()).collect();
        arena_peak = arena_peak.max(scratch.work_pool.bytes());
        let mut sorted_bufs: Vec<Vec<E>> = guards.iter_mut().map(|g| std::mem::take(&mut g.sorted)).collect();
        {
            struct Shared<T>(T);
            unsafe impl<T> Sync for Shared<T> {}
            unsafe impl<T> Send for Shared<T> {}
            let lists: Shared<Vec<&Lists<E>>> = Shared(guards.iter().map(|g| &g.lists).collect());
            let lists = &lists;
            let bufs: Vec<Mutex<Vec<E>>> = sorted_bufs.drain(..).map(Mutex::new).collect();
            let fin = Shared(finals.as_mut_ptr());
            let fin = &fin;
            let hscr = &hscr;
            (plo..phi).into_par_iter().for_each(|p| {
                let blocks: Vec<(*mut E, usize)> = lists.0.iter().flat_map(|a| a.blocks(p)).collect();
                let ti = rayon::current_thread_index().unwrap_or(0);
                if hashed {
                    // count in cache, then sort the distinct pairs
                    let mut hsc = hscr[ti].lock().unwrap();
                    let (hs, pairs, sorted) = &mut *hsc;
                    pairs.clear();
                    hs.count(&blocks, pairs);
                    sort_pairs(pairs, sub, sb, sorted);
                    let mut keep = scratch.keep[ti].lock().unwrap();
                    let mut ranked = Vec::new();
                    let o: &[(E, u32)] = sorted;
                    let part = build_part(|| o.iter().map(|&(e, c)| (e.wide(), c)), below, &mut keep, &mut ranked);
                    unsafe { *fin.0.add(p) = Some(part) };
                    return;
                }
                let mut out = bufs[ti].lock().unwrap();
                let mut hist = Vec::new();
                sort_partition(&blocks, sub, sb, &mut out, &mut hist);
                let mut keep = scratch.keep[ti].lock().unwrap();
                let mut ranked = Vec::new();
                let o: &[E] = &out;
                let part = build_part(|| o.chunk_by(|a, b| a == b).map(|same| (same[0].wide(), same.len() as u32)), below, &mut keep, &mut ranked);
                unsafe { *fin.0.add(p) = Some(part) };
            });
            sorted_bufs = bufs.into_iter().map(|m| m.into_inner().unwrap()).collect();
        }
        for (g, s) in guards.iter_mut().zip(sorted_bufs) {
            g.sorted = s;
            g.lists.reset();
        }
        for a in &scratch.arenas {
            a.lock().unwrap().reset();
        }
        t_agg += t.elapsed().as_secs_f64();
    }
    // the threads' buffers freed now: freed in the background, they held the memory map's lock while the next level
    // allocated its own (2-4 s more on level 4 in some runs)
    let t = Instant::now();
    drop((states, hscr));
    times.push(("free_s".into(), t.elapsed().as_secs_f64()));
    times.push(("setup_s".into(), t_setup));
    times.push(("scatter_s".into(), t_scatter));
    times.push(("aggregate_s".into(), t_agg));
    times.push(("arena_gb".into(), arena_peak as f64 / 1e9));
    times.push(("rounds".into(), rounds.len() as f64));
    times.push(("build_s".into(), t0.elapsed().as_secs_f64()));
    let parts: Vec<Part> = finals.into_iter().map(|p| p.unwrap()).collect();
    let hdrs = parts.iter().map(Hdr::of).collect();
    let mut lt = LevelTable { k, shards: files.len(), b, pbits: part_bits, parts, hdrs, times };
    let (nc, np) = (lt.contexts(), lt.pairs());
    let nbig: usize = lt.parts.iter().map(|p| p.big.len).sum();
    let ntop: usize = lt.parts.iter().map(|p| p.big.iter().filter(|b| b.thr.is_some()).count()).sum();
    let gb = lt.bytes() as f64 / 1e9;
    lt.times.extend([("contexts".into(), nc as f64), ("pairs".into(), np as f64), ("big".into(), nbig as f64), ("past_top".into(), ntop as f64), ("final_gb".into(), gb)]);
    lt
}

#[inline(always)]
fn pf<T>(p: *const T) {
    unsafe { std::arch::x86_64::_mm_prefetch(p.cast::<i8>(), std::arch::x86_64::_MM_HINT_T0) };
}

/// Level t's rows at the validation positions, in batches of LB positions whose dependent loads (directory, keys, run
/// bounds, run) are prefetched stage by stage.
fn lookup_level(t: &LevelTable, x: &[u16], target: &[u16], avail: &[u32], col: usize, bucket: Option<usize>, out: &Out) {
    lookup_level_b::<32>(t, x, target, avail, col, bucket, out)
}

fn lookup_level_b<const LB: usize>(t: &LevelTable, x: &[u16], target: &[u16], avail: &[u32], col: usize, bucket: Option<usize>, out: &Out) {
    const CH: usize = 4096;
    let nl = LEVELS.len();
    let k = t.k;
    let n = x.len();
    let below = t.b - t.pbits;
    (0..n.div_ceil(CH)).into_par_iter().for_each(|c| {
        let (c0, c1) = (c * CH, n.min((c + 1) * CH));
        let mut qs = [0usize; LB];
        let mut keys = [0u128; LB];
        let mut pis = [0usize; LB];
        let mut tops = [0usize; LB];
        let mut cis = [usize::MAX; LB];
        let mut b0 = c0;
        while b0 < c1 {
            let b1 = c1.min(b0 + LB);
            let mut m = 0;
            for q in b0..b1 {
                if avail[q] as usize >= k {
                    let w = mix(pack(&x[q + 1 - k..=q]), t.b);
                    let pi = (w >> below) as usize;
                    pf(t.hdrs.as_ptr().wrapping_add(pi));
                    (qs[m], keys[m], pis[m]) = (q, w, pi);
                    m += 1;
                }
            }
            b0 = b1;
            for j in 0..m {
                let h = unsafe { t.hdrs.get_unchecked(pis[j]) };
                let w = keys[j];
                let top = ((w >> h.rbits) & ((1u128 << h.dbits) - 1)) as usize;
                pf(h.dir.wrapping_add(top));
                (keys[j], tops[j]) = (w & ((1u128 << h.rbits) - 1), top);
            }
            for j in 0..m {
                let h = unsafe { t.hdrs.get_unchecked(pis[j]) };
                let (lo, hi) = unsafe { (*h.dir.add(tops[j]) as usize, *h.dir.add(tops[j] + 1) as usize) };
                if lo < hi {
                    pf(h.keys.wrapping_add(lo * h.ks as usize));
                }
                tops[j] = lo;
                cis[j] = hi; // the bucket's end, for now
            }
            for j in 0..m {
                let h = unsafe { t.hdrs.get_unchecked(pis[j]) };
                let (mut lo, hi) = (tops[j], cis[j]);
                cis[j] = usize::MAX;
                while lo < hi {
                    let v = h.key(lo);
                    if v >= keys[j] {
                        if v == keys[j] {
                            cis[j] = lo;
                            pf(h.start.wrapping_add(lo));
                        }
                        break;
                    }
                    lo += 1;
                }
            }
            for j in 0..m {
                if cis[j] != usize::MAX {
                    let h = unsafe { t.hdrs.get_unchecked(pis[j]) };
                    let (s, e) = unsafe { (*h.start.add(cis[j]) as usize, *h.start.add(cis[j] + 1) as usize) };
                    pf(h.tokens.wrapping_add(s));
                    pf(h.counts.wrapping_add(s));
                    if e - s > BIG {
                        let p = &t.parts[pis[j]];
                        pf(p.big_ids.as_ptr().wrapping_add(p.big_guess(cis[j])));
                        pf(h.tokens.wrapping_add((s + e) / 2));
                    }
                }
            }
            for j in 0..m {
                let ci = cis[j];
                if ci == usize::MAX {
                    continue;
                }
                let (q, p) = (qs[j], &t.parts[pis[j]]);
                let (s, e) = (p.start[ci] as usize, p.start[ci + 1] as usize);
                let d = e - s;
                let toks = &p.tokens[s..e];
                let tg = target[q];
                let big = if d > BIG { Some(&p.big[p.big_index(ci)]) } else { None };
                let total = match big {
                    Some(bg) => bg.tot,
                    None => u32::try_from((s..e).map(|i| p.count(i) as u64).sum::<u64>()).unwrap_or(u32::MAX),
                };
                let c = match toks.binary_search(&tg) {
                    Ok(j) => {
                        let cc = p.count(s + j);
                        match big.and_then(|bg| bg.thr) {
                            Some((tc, tt)) if (Reverse(cc), tg) > (Reverse(tc), tt) => 0,
                            _ => cc,
                        }
                    }
                    Err(_) => 0,
                };
                unsafe {
                    out.tot.set(q * nl + col, total);
                    out.dist.set(q * nl + col, d.min(u16::MAX as usize) as u16);
                    out.cnt.set(q * nl + col, c);
                    if let Some(bk) = bucket {
                        if *out.done.0.add(q) == 0 {
                            out.done.set(q, 1);
                            let mut top = [(0u32, 0u16); KEEP];
                            let m = d.min(KEEP);
                            match big {
                                Some(bg) => top = bg.top,
                                None => {
                                    let mut r: [(Reverse<u32>, u16); BIG] = [(Reverse(0), 0); BIG];
                                    for j in 0..d {
                                        r[j] = (Reverse(p.count(s + j)), toks[j]);
                                    }
                                    r[..d].sort_unstable();
                                    for j in 0..m {
                                        top[j] = (r[j].0 .0, r[j].1);
                                    }
                                }
                            }
                            let total = total as usize;
                            let tc = top[0].0 as f64;
                            let count = COUNTS.iter().filter(|&&e| e <= total).count();
                            let purity = PURITIES.iter().filter(|&&p| p <= tc / total as f64).count();
                            let cell = (((1 + bk) * (COUNTS.len() + 1) + count) * (PURITIES.len() + 1) + purity) as i16;
                            let wsum: f64 = top[..m].iter().map(|x| x.0 as f64).sum();
                            out.cell.set(q, cell);
                            for (j, &(cc, tok)) in top[..m].iter().enumerate() {
                                out.tokens.set(q * KEEP + j, tok);
                                out.weights.set(q * KEEP + j, (cc as f64 / wsum.max(1.0) * 65535.0).round_ties_even() as u16);
                            }
                        }
                    }
                }
            }
        }
    });
}

/// The rows of the validation positions `val` at the built `levels` (`lvs[i]` = levels[i]'s column and hint bucket),
/// deepest first (the hint row from the deepest hint level whose context occurred, where `done` is 0).
pub fn lookup(levels: &[Level], lvs: &[Lv], val: &Val, out: &Out) -> Times {
    let mut times = Vec::new();
    let t0 = Instant::now();
    let mut order: Vec<usize> = (0..levels.len()).collect();
    order.sort_by_key(|&i| Reverse(levels[i].k()));
    for i in order {
        let t = Instant::now();
        let (col, bucket) = (lvs[i].col, lvs[i].bucket);
        lookup_level(&levels[i], val.x, val.target, &val.avail, col, bucket, out);
        times.push((format!("lookup_k{}_s", levels[i].k()), t.elapsed().as_secs_f64()));
    }
    times.push(("lookup_all_s".into(), t0.elapsed().as_secs_f64()));
    times
}
