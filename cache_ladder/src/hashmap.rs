//! HashMapIndex: the Python dict design of the ladder's first rung, written in Rust with a library hash map.
//!
//! Three maps, for k = 6, 10 and 20: for each shard and each position i with k <= i < len (i indexes the token that
//! follows the context), the context tokens[i-k..i] is keyed by a 64-bit hash of the k-gram and maps to a counter of
//! next tokens. A counter holds at most 288 distinct next tokens: a new one beyond that is dropped (and counted),
//! counts of present ones still increase. Every count is kept. No BOS special-casing; contexts never cross shards.
//!
//! Parallelism over T threads, sharded by key, as the 48 Python processes are:
//!   1. thread t takes a contiguous 1/T of the concatenated token range and, for each position and each k, hands
//!      (key, next) to the key's owner, appending it to its outgoing list for (owner, k, group of the key's sub-map);
//!   2. after all of phase 1, thread d drains every source's lists for (d, k, group), in source order, into its own
//!      three maps.
//! A key's (key, next) pairs therefore arrive in corpus order, so its counter keeps the first 288 distinct next tokens
//! of the corpus. The structure is the T x 3 maps; they live until the object is dropped (not timed).
//!
//! That is the Python design step for step (the same keys, entries and drops as the version with std's HashMap and
//! SipHash), and every map is made of library hash maps, hashbrown's `HashMap` (the SwissTable that std's HashMap
//! wraps), from the context's hash to its counter. A first version with `HashMap<u64, HashMap<u16, u32>>` (SipHash, no
//! presizing) was no faster than the 48 Python processes: one process's 48 threads spent the time allocating hundreds
//! of millions of small maps, growing tables and faulting in 4 KiB pages, contending for the allocator and the one
//! address space that separate processes do not share. What changed is how the maps are allocated, sized and
//! filled, and what a value holds:
//! - a map's value is its counter while that is one next token (a u32: the token and a 15-bit count); a context
//!   followed by a second distinct token (about 1% of them at 1B tokens), or one counted beyond 32767 times, moves to
//!   a `Vec` of (next, count) in the map's spill list, and the value becomes its index. The key is the 64-bit hash
//!   stored 4-byte aligned, so a bucket is 12 bytes; most keys allocate nothing of their own;
//! - the keys are already hashes (as Python's dict hashes an int as itself), so the hasher is one multiply (not the
//!   identity: hashbrown's 7-bit tags come from the hash's top bits, which for the raw keys are the sub-map's bits);
//! - each owner's map is a set of library maps chosen by the key's high bits (a sharded map, as concurrent hash maps
//!   are), of about SUB pairs each: 2^14 buckets, 208 KiB, which stay in the core's L2 while they fill (one big table
//!   filled at random is bound by memory latency). Phase 1 already routes a pair to one of GROUPS groups of its map's
//!   sub-maps; phase 2 splits a group's pairs by sub-map, keeping their order, then builds the group's sub-maps one at
//!   a time, each presized for the pairs it receives so that it never rehashes. (Splitting a whole map at once wrote
//!   to thousands of lists at 3B tokens, more buffers than the L2 holds, and needed scratch for the whole map.);
//! - all memory comes from two mappings made by one thread before each phase (phase 1's lists; phase 2's arenas and
//!   scratch), advised to huge pages and faulted in by writes of the thread that uses each part (on its NUMA node).
//!   The maps and the spilled counters allocate from their owner's part of phase 2's mapping through a bump allocator
//!   (the `Allocator` API of allocator-api2, which hashbrown and its `Vec` take; freeing is a no-op, and the mapping
//!   is unmapped with the index). No thread maps memory while others run: an mmap waits for the address space's lock
//!   behind every other thread's page faults (this alone took 1B tokens from 21 s to 4 s), and a first touch by a
//!   read would map the zero page in 4 KiB pages;
//! - phase 1's lists are chunks of 10-byte (key, next) records; a list collects WC records in cache and writes them
//!   out with non-temporal stores (3T x GROUPS lists, or a group's sub-maps, are too many write streams for the
//!   caches);
//! - the window hash is a wyhash-style multiply-mix (7 multiplies for a position's three keys) instead of SipHash
//!   (its 40-bit projections collide as often as SipHash's, i.e. as a random function's), and the owner comes from the
//!   key's low 32 bits by a multiply instead of a division (the group and sub-map from its high bits).
use crate::{pinned_pool, shards, Shard};
use allocator_api2::alloc::{AllocError, Allocator, Layout};
use allocator_api2::vec::Vec as AVec;
use hashbrown::hash_map::Entry;
use pyo3::{exceptions::PyValueError, prelude::*};
use std::collections::HashMap;
use std::hash::{BuildHasher, Hash, Hasher};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

const KS: [usize; 3] = [6, 10, 20];
const CAP: usize = 288; // distinct next tokens per counter
const GROUPS: usize = 8; // phase 1 splits each owner's map's pairs into this many groups of sub-maps
const SUB: usize = 13_500; // pairs per sub-map, on average at most this (a 2^14-bucket table, 208 KiB)
const CHUNK: usize = 4096; // records per chunk of a phase-1 list
const CHUNK2: usize = 512; // records per chunk of a phase-2 (sub-map) list
const WC: usize = 64; // records a list collects before writing them out (640 bytes, 10 cache lines)

// ---------------------------------------------------------------------------------------------------------------
// Memory.

const HUGE: usize = 2 << 20;

/// An anonymous mapping advised to huge pages, reserved (MAP_NORESERVE: only touched pages take memory) and carved
/// into 2 MiB-aligned parts. Unmapped on drop.
struct Region {
    ptr: *mut u8,
    bytes: usize,
}
unsafe impl Send for Region {}
unsafe impl Sync for Region {}

impl Region {
    /// A region for parts of these byte sizes, and each part's offset.
    fn new(sizes: &[usize]) -> (Self, Vec<usize>) {
        let mut offs = Vec::with_capacity(sizes.len());
        let mut bytes = 0;
        for &s in sizes {
            offs.push(bytes);
            bytes += s.div_ceil(HUGE).max(1) * HUGE;
        }
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                bytes + HUGE,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                -1,
                0,
            )
        };
        assert!(ptr != libc::MAP_FAILED, "mmap of {bytes} bytes: {}", std::io::Error::last_os_error());
        // Align to 2 MiB so every part starts on a huge page; unmap the slack.
        let (raw, start) = (ptr as usize, (ptr as usize).next_multiple_of(HUGE));
        unsafe {
            if start > raw {
                libc::munmap(ptr, start - raw);
            }
            libc::munmap((start + bytes) as *mut libc::c_void, raw + HUGE - start);
            libc::madvise(start as *mut libc::c_void, bytes, libc::MADV_HUGEPAGE);
        }
        (Self { ptr: start as *mut u8, bytes }, offs)
    }

    fn part<T>(&self, off: usize, len: usize) -> Part<T> {
        Part { ptr: unsafe { self.ptr.add(off) }.cast(), len }
    }
}

impl Drop for Region {
    fn drop(&mut self) {
        unsafe { libc::munmap(self.ptr.cast(), self.bytes) };
    }
}

/// `len` T in a region (zeroed until written; T must be valid all-zero).
struct Part<T> {
    ptr: *mut T,
    len: usize,
}
unsafe impl<T> Send for Part<T> {}
unsafe impl<T> Sync for Part<T> {}

impl<T> std::ops::Deref for Part<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}
impl<T> std::ops::DerefMut for Part<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Keys.

const P0: u64 = 0xa076_1d64_78bd_642f;
const P1: u64 = 0xe703_7ed1_a0b4_28db;
const P2: u64 = 0x8ebc_6af0_9c88_c6e3;
const P3: u64 = 0x5899_65cc_7537_4cc3;

/// The 128-bit product of a and b, folded to 64 bits (wyhash's mum).
#[inline(always)]
fn mum(a: u64, b: u64) -> u64 {
    let r = a as u128 * b as u128;
    r as u64 ^ (r >> 64) as u64
}

/// The 64-bit keys of the 6-, 10- and 20-token windows ending just before t[i] (only those of k <= kmax are
/// computed; kmax <= i). wyhash-style: the window's tokens are read as 64- and 32-bit words from its end backwards,
/// mixed into an accumulator with folded 128-bit multiplies, and each key is a final mix of the accumulator with k:
/// the 10-token window's accumulator continues the 6-token one, the 20-token one the 10-token one, so a position
/// costs 7 multiplies for its three keys.
#[inline(always)]
fn keys_at(t: &[u16], i: usize, kmax: usize) -> [u64; 3] {
    assert!(kmax <= i && i <= t.len());
    let w = t.as_ptr();
    // SAFETY: every read is of tokens in t[i - kmax..i].
    let r64 = |p: usize| unsafe { (w.add(p) as *const u64).read_unaligned() };
    let r32 = |p: usize| unsafe { (w.add(p) as *const u32).read_unaligned() as u64 };
    let fin = |h: u64, k: u64| mum(h ^ P2, k ^ P3);
    let mut keys = [0; 3];
    let h6 = mum(r64(i - 4) ^ P1, r32(i - 6) ^ P0);
    keys[0] = fin(h6, 6);
    if kmax >= 10 {
        let h10 = mum(r64(i - 10) ^ P1, h6 ^ P0);
        keys[1] = fin(h10, 10);
        if kmax >= 20 {
            let h = mum(r64(i - 14) ^ P1, r64(i - 18) ^ h10);
            keys[2] = fin(mum(r32(i - 20) ^ P1, h ^ P0), 20);
        }
    }
    keys
}

/// The key of a k-token window (k = 6, 10 or 20).
fn key_dyn(k: usize, w: &[u16]) -> u64 {
    let j = KS.iter().position(|&x| x == k).unwrap();
    keys_at(w, w.len(), k)[j]
}

/// The thread that owns `key`: from its low 32 bits (a table position comes from its high bits).
#[inline(always)]
fn owner(key: u64, threads: usize) -> usize {
    ((key as u32 as u64 * threads as u64) >> 32) as usize
}

// ---------------------------------------------------------------------------------------------------------------
// Lists of (key, next) records.

/// A (key, next token) record.
#[derive(Clone, Copy)]
#[repr(C, packed)]
struct Pair {
    key: u64,
    next: u16,
}

/// WC records, a whole number of cache lines.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct Buf([Pair; WC]);

/// Copy `buf` to `dst` (64-byte aligned) with non-temporal stores, which skip reading the lines first.
#[inline(always)]
unsafe fn stream(buf: &Buf, dst: *mut Pair) {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::{__m128i, _mm_load_si128, _mm_stream_si128};
        let (src, dst) = (buf as *const Buf as *const __m128i, dst as *mut __m128i);
        for i in 0..size_of::<Buf>() / 16 {
            _mm_stream_si128(dst.add(i), _mm_load_si128(src.add(i)));
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    std::ptr::copy_nonoverlapping(buf.0.as_ptr(), dst, WC);
}

/// Appends records to n lists, each a sequence of `chunk`-record chunks of an arena (64-byte aligned, `chunk` a
/// multiple of WC); a list's next WC records collect in a buffer and go out with non-temporal stores.
struct Writer {
    arena: *mut Pair,
    cap: usize,
    chunk: usize,
    top: usize,
    bufs: Vec<Buf>,
    fill: Vec<usize>,
    cur: Vec<usize>,
    end: Vec<usize>,
    chunks: Vec<Vec<usize>>,
}

impl Writer {
    /// Lists in arena[..cap], which must hold the records plus one partly filled chunk per list.
    fn new(arena: *mut Pair, cap: usize, lists: usize, chunk: usize) -> Self {
        let per = cap / (lists * chunk) + 2;
        Writer {
            arena,
            cap,
            chunk,
            top: 0,
            bufs: vec![Buf([Pair { key: 0, next: 0 }; WC]); lists],
            fill: vec![0; lists],
            cur: vec![0; lists],
            end: vec![0; lists],
            chunks: vec![Vec::with_capacity(per); lists],
        }
    }

    /// Room in list b's chunk for at least one record (they come WC at a time, or once at the end).
    #[inline]
    fn room(&mut self, b: usize) {
        if self.cur[b] == self.end[b] {
            assert!(self.top + self.chunk <= self.cap, "list arena overflow");
            self.chunks[b].push(self.top);
            self.cur[b] = self.top;
            self.end[b] = self.top + self.chunk;
            self.top += self.chunk;
        }
    }

    #[inline(always)]
    fn push(&mut self, b: usize, p: Pair) {
        assert!(b < self.fill.len());
        // SAFETY: b is in range (checked above; bufs and fill have the same length) and fill[b] < WC.
        let f = unsafe { self.fill.get_unchecked_mut(b) };
        unsafe { *self.bufs.get_unchecked_mut(b).0.get_unchecked_mut(*f) = p };
        if *f + 1 < WC {
            *f += 1;
            return;
        }
        *f = 0;
        self.room(b);
        unsafe { stream(&self.bufs[b], self.arena.add(self.cur[b])) };
        self.cur[b] += WC;
    }

    fn finish(mut self) -> Lists {
        for b in 0..self.fill.len() {
            if self.fill[b] > 0 {
                self.room(b);
                unsafe { std::ptr::copy_nonoverlapping(self.bufs[b].0.as_ptr(), self.arena.add(self.cur[b]), self.fill[b]) };
                self.cur[b] += self.fill[b];
            }
        }
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_mm_sfence()
        };
        let last = (0..self.fill.len()).map(|b| self.chunks[b].last().map_or(0, |&s| self.cur[b] - s)).collect();
        Lists { arena: Part { ptr: self.arena, len: self.cap }, chunk: self.chunk, chunks: self.chunks, last }
    }
}

/// Finished lists (in an arena owned elsewhere).
struct Lists {
    arena: Part<Pair>,
    chunk: usize,
    chunks: Vec<Vec<usize>>, // list b's chunks, by start in `arena`
    last: Vec<usize>,        // records in list b's last chunk
}

impl Lists {
    /// List b's records, chunk by chunk.
    fn list(&self, b: usize) -> impl Iterator<Item = &[Pair]> + '_ {
        let n = self.chunks[b].len();
        self.chunks[b]
            .iter()
            .enumerate()
            .map(move |(c, &s)| &self.arena[s..s + if c + 1 == n { self.last[b] } else { self.chunk }])
    }

    fn len(&self, b: usize) -> usize {
        self.chunks[b].len().saturating_sub(1) * self.chunk + self.last[b]
    }
}

// ---------------------------------------------------------------------------------------------------------------
// The maps: hashbrown hash maps allocating from a per-owner bump arena.

/// A bump allocator over one owner's part of phase 2's region, used only by that owner's thread while it builds:
/// an allocation moves the top up, growing the latest allocation extends it in place, freeing does nothing (the maps
/// live until the index is dropped, and then the region is unmapped).
struct Arena {
    top: AtomicUsize, // only the owner's thread moves it (Relaxed loads and stores)
    start: usize,
    end: usize,
}

/// The `Allocator` a map holds: its owner's arena.
#[derive(Clone, Copy)]
struct Bump(*const Arena);
unsafe impl Send for Bump {}
unsafe impl Sync for Bump {}

impl Bump {
    #[inline(always)]
    fn arena(&self) -> &Arena {
        unsafe { &*self.0 }
    }
}

unsafe impl Allocator for Bump {
    #[inline]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let a = self.arena();
        let start = a.top.load(Relaxed).next_multiple_of(layout.align());
        let end = start + layout.size();
        if end > a.end {
            return Err(AllocError);
        }
        a.top.store(end, Relaxed);
        Ok(NonNull::slice_from_raw_parts(unsafe { NonNull::new_unchecked(start as *mut u8) }, layout.size()))
    }

    #[inline]
    unsafe fn deallocate(&self, _: NonNull<u8>, _: Layout) {}

    unsafe fn grow(&self, ptr: NonNull<u8>, old: Layout, new: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let a = self.arena();
        let p = ptr.as_ptr() as usize;
        if p + old.size() == a.top.load(Relaxed) && p % new.align() == 0 {
            if p + new.size() > a.end {
                return Err(AllocError);
            }
            a.top.store(p + new.size(), Relaxed);
            return Ok(NonNull::slice_from_raw_parts(ptr, new.size()));
        }
        let q = self.allocate(new)?;
        unsafe { std::ptr::copy_nonoverlapping(ptr.as_ptr(), q.as_ptr().cast::<u8>(), old.size()) };
        Ok(q)
    }
}

/// A map key: the context's 64-bit hash, stored 4-byte aligned so that a bucket (key, value) is 12 bytes.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(C, packed(4))]
struct Key(u64);

impl Hash for Key {
    #[inline(always)]
    fn hash<H: Hasher>(&self, h: &mut H) {
        h.write_u64(self.0);
    }
}

/// The hash a sub-map files `key` under. The keys are already hashes, so it is one multiply by an odd constant, not
/// the identity: hashbrown takes a bucket from the hash's low bits and a 7-bit tag from its top bits, and a key's top
/// bits are the same for a whole sub-map (they choose it), while the product's top bits depend on every bit of the key.
#[inline(always)]
fn hash_of(key: u64) -> u64 {
    key.wrapping_mul(0x9e37_79b9_7f4a_7c15)
}

/// The sub-maps' `BuildHasher`: `hash_of` the key.
#[derive(Clone, Copy, Default)]
struct MulHash;

struct MulHasher(u64);

impl BuildHasher for MulHash {
    type Hasher = MulHasher;
    #[inline(always)]
    fn build_hasher(&self) -> MulHasher {
        MulHasher(0)
    }
}

impl Hasher for MulHasher {
    #[inline(always)]
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, _: &[u8]) {
        unreachable!("only u64 keys")
    }
    #[inline(always)]
    fn write_u64(&mut self, x: u64) {
        self.0 = hash_of(x);
    }
}

/// A map's value is the counter of next tokens: without the SPILL bit, one next token `v & 0xffff` counted `v >> 16`
/// times (up to INLINE_MAX); with it, the index of the counter in the map's `spill` list.
const SPILL: u32 = 1 << 31;
const INLINE_MAX: u32 = (1 << 15) - 1;

type Sub = hashbrown::HashMap<Key, u32, MulHash, Bump>;
/// A counter of more than one next token (or of one counted more than INLINE_MAX times): its (next, count) pairs.
type Counter = AVec<(u16, u32), Bump>;
const _: () = assert!(size_of::<(Key, u32)>() == 12);

#[derive(Default, Clone, Copy)]
struct Stats {
    keys: [u64; 3],
    entries: u64,
    dropped: u64,
    spilled: u64,
}

/// Sub-maps for a map receiving `pairs` pairs: a multiple of GROUPS, of at most SUB pairs on average.
fn subtables(pairs: usize) -> usize {
    pairs.div_ceil(GROUPS * SUB).max(1) * GROUPS
}

/// The sub-map of `key` among `parts`: from its high bits (its owner comes from its low 32 bits).
#[inline(always)]
fn sub_of(key: u64, parts: usize) -> usize {
    ((key as u128 * parts as u128) >> 64) as usize
}

/// Arena bytes a map receiving `pairs` pairs can need, at most: its sub-maps (one for n pairs has at most
/// 2 * 8/7 * n + 16 buckets of 13 bytes, plus 32), its spill list (at most pairs / 2 counters of 32 bytes, doubled by
/// the list's growth) and counters (one of n tokens has allocated at most 2 * 2n slots of 8 bytes).
fn arena_bound(pairs: usize) -> usize {
    96 * pairs + 320 * subtables(pairs) + 4096
}

/// One map from a context's hash to its counter: library hash maps chosen by the key's high bits (sub-map p holds
/// the keys whose `sub_of` is p), and the counters that outgrew the value.
struct Map {
    subs: AVec<Sub, Bump>,
    spill: AVec<Counter, Bump>,
}

impl Map {
    /// Make the next sub-map from list p of `list` (its pairs, in order).
    #[inline(never)]
    fn fill(&mut self, list: &Lists, p: usize, bump: Bump, j: usize, st: &mut Stats) {
        // At most one key per pair it receives: sized for all of them, it never grows.
        let mut sub = Sub::with_capacity_and_hasher_in(list.len(p), MulHash, bump);
        let spill = &mut self.spill;
        let mut t = Stats::default(); // local, so its counters stay in registers
        for chunk in list.list(p) {
            for &Pair { key, next } in chunk {
                match sub.entry(Key(key)) {
                    Entry::Vacant(e) => {
                        e.insert(next as u32 | 1 << 16);
                        t.keys[0] += 1;
                        t.entries += 1;
                    }
                    Entry::Occupied(mut e) => {
                        let v = e.get_mut();
                        if *v & SPILL == 0 {
                            if *v as u16 == next && *v >> 16 < INLINE_MAX {
                                *v += 1 << 16;
                            } else {
                                // A second next token, or a count beyond INLINE_MAX: the counter moves to `spill`.
                                let mut c: Counter = AVec::with_capacity_in(4, bump);
                                c.push((*v as u16, *v >> 16));
                                if *v as u16 == next {
                                    c[0].1 += 1;
                                } else {
                                    c.push((next, 1));
                                    t.entries += 1;
                                }
                                assert!(spill.len() < SPILL as usize);
                                *v = SPILL | spill.len() as u32;
                                spill.push(c);
                                t.spilled += 1;
                            }
                        } else {
                            let c = &mut spill[(*v & !SPILL) as usize];
                            if let Some(item) = c.iter_mut().find(|item| item.0 == next) {
                                item.1 += 1;
                            } else if c.len() == CAP {
                                t.dropped += 1;
                            } else {
                                if c.len() == c.capacity() {
                                    c.reserve_exact(c.len().min(CAP - c.len())); // double, up to CAP
                                }
                                c.push((next, 1));
                                t.entries += 1;
                            }
                        }
                    }
                }
            }
        }
        self.subs.push(sub);
        st.keys[j] += t.keys[0];
        st.entries += t.entries;
        st.dropped += t.dropped;
        st.spilled += t.spilled;
    }

    fn items(&self, v: u32) -> Vec<(u16, u32)> {
        if v & SPILL == 0 {
            vec![(v as u16, v >> 16)]
        } else {
            self.spill[(v & !SPILL) as usize].iter().copied().collect()
        }
    }

    /// The counter of `key` as (next, count) pairs.
    fn counter(&self, key: u64) -> Vec<(u16, u32)> {
        self.subs[sub_of(key, self.subs.len())].get(&Key(key)).map(|&v| self.items(v)).unwrap_or_default()
    }
}

/// List b of every source's `outgoing`, in order, split by sub-map into lists in `scratch`: the list holds the
/// keys of sub-maps first..first + n (of `parts`).
#[inline(never)]
fn split_pass(outgoing: &[Lists], b: usize, parts: usize, first: usize, n: usize, scratch: &Part<Pair>) -> Lists {
    let mut w = Writer::new(scratch.ptr, scratch.len, n, CHUNK2);
    for src in outgoing {
        for chunk in src.list(b) {
            for &p in chunk {
                w.push(sub_of(p.key, parts) - first, p);
            }
        }
    }
    w.finish()
}

type Maps = [Map; 3];

#[pyclass]
pub struct HashMapIndex {
    shards: Vec<Shard>,
    pool: rayon::ThreadPool,
    threads: usize,
    maps: Vec<Maps>,      // allocated from `arenas`, in `region` (declared before them: dropped first)
    arenas: Box<[Arena]>, // one per owner
    region: Option<Region>,
}

#[pymethods]
impl HashMapIndex {
    #[staticmethod]
    fn create(files: Vec<String>, tokens: usize, threads: usize, cpus: Vec<usize>) -> PyResult<Self> {
        if threads == 0 {
            return Err(PyValueError::new_err("threads must be positive"));
        }
        let shards = shards(&files, tokens)?;
        let pool = pinned_pool(threads, &cpus, "hashmap")?;
        Ok(Self { shards, pool, threads, maps: Vec::new(), arenas: Box::new([]), region: None })
    }

    fn build(&mut self, py: Python<'_>) -> PyResult<HashMap<String, f64>> {
        if !self.maps.is_empty() {
            return Err(PyValueError::new_err("already built"));
        }
        let (shards, pool, t_count) = (&self.shards, &self.pool, self.threads);
        let (maps, arenas, region, st, phase1_s, phase2_s) = py.detach(|| {
            let t0 = std::time::Instant::now();
            // Shard s covers [starts[s], starts[s + 1]) of the concatenated range.
            let mut starts = vec![0usize];
            for s in shards {
                starts.push(starts.last().unwrap() + s.len);
            }
            let n = *starts.last().unwrap();
            let lists = t_count * 3 * GROUPS;
            // Each source's pairs, exactly, and its arena: those plus one partly filled chunk per list.
            let range = |t: usize| (n * t / t_count, n * (t + 1) / t_count);
            let pairs: Vec<usize> = (0..t_count)
                .map(|t| {
                    let (lo, hi) = range(t);
                    (0..shards.len())
                        .map(|s| {
                            let (a, b) = (lo.max(starts[s]), hi.min(starts[s + 1]));
                            let (a, b) = (a.saturating_sub(starts[s]), b.saturating_sub(starts[s]));
                            KS.iter().map(|&k| b.saturating_sub(a.max(k))).sum::<usize>()
                        })
                        .sum()
                })
                .collect();
            let arena_len = |t: usize| pairs[t] + lists * CHUNK;
            let sizes: Vec<usize> = (0..t_count).map(|t| arena_len(t) * size_of::<Pair>()).collect();
            let (lists_region, offs1) = Region::new(&sizes);
            // Phase 1: hash every (position, k) and route it to list b = (owner * 3 + j) * GROUPS + group, the group
            // of its sub-map.
            let outgoing: Vec<Lists> = pool.broadcast(|ctx| {
                let t = ctx.index();
                let (lo, hi) = range(t);
                let arena: Part<Pair> = lists_region.part(offs1[t], arena_len(t));
                let mut w = Writer::new(arena.ptr, arena.len, lists, CHUNK);
                let mut push = |j: usize, key: u64, next: u16| {
                    w.push((owner(key, t_count) * 3 + j) * GROUPS + sub_of(key, GROUPS), Pair { key, next })
                };
                for (s, shard) in shards.iter().enumerate() {
                    let (a, b) = (lo.max(starts[s]), hi.min(starts[s + 1]));
                    if a >= b {
                        continue;
                    }
                    let (a, b) = (a - starts[s], b - starts[s]);
                    let toks = shard.tokens();
                    let full = a.max(KS[2]).min(b); // from here on every k has a window
                    for i in a..full {
                        let next = toks[i];
                        for (j, &k) in KS.iter().enumerate() {
                            if i >= k {
                                push(j, key_dyn(k, &toks[i - k..i]), next);
                            }
                        }
                    }
                    for i in full..b {
                        let next = toks[i];
                        let keys = keys_at(toks, i, 20);
                        push(0, keys[0], next);
                        push(1, keys[1], next);
                        push(2, keys[2], next);
                    }
                }
                w.finish()
            });
            let phase1_s = t0.elapsed().as_secs_f64();
            // Phase 2's memory, per owner d: an arena for its three maps (bounded by the pairs they receive) and
            // scratch for splitting the largest group's pairs by sub-map.
            let received: Vec<usize> = (0..lists).map(|b| outgoing.iter().map(|o| o.len(b)).sum()).collect();
            let map_received = |m: usize| received[m * GROUPS..(m + 1) * GROUPS].iter().sum::<usize>();
            let scratch_len = |d: usize| {
                (0..3 * GROUPS)
                    .map(|i| received[d * 3 * GROUPS + i] + subtables(map_received(d * 3 + i / GROUPS)) / GROUPS * CHUNK2)
                    .max()
                    .unwrap()
            };
            let mut sizes = Vec::with_capacity(2 * t_count);
            for d in 0..t_count {
                sizes.push((0..3).map(|j| arena_bound(map_received(d * 3 + j))).sum());
                sizes.push(scratch_len(d) * size_of::<Pair>());
            }
            let (region, offs) = Region::new(&sizes);
            let arenas: Box<[Arena]> = (0..t_count)
                .map(|d| {
                    let start = region.ptr as usize + offs[2 * d];
                    Arena { top: AtomicUsize::new(start), start, end: start + sizes[2 * d] }
                })
                .collect();
            // Phase 2: owner d drains what was sent to it, in source order, into its own three maps.
            let results: Vec<(Maps, Stats)> = pool.broadcast(|ctx| {
                let d = ctx.index();
                let bump = Bump(&arenas[d]);
                let mut st = Stats::default();
                let scratch: Part<Pair> = region.part(offs[2 * d + 1], scratch_len(d));
                let maps: Maps = std::array::from_fn(|j| {
                    let pairs = map_received(d * 3 + j);
                    let parts = subtables(pairs);
                    let per = parts / GROUPS;
                    // Most counters stay in the value: about 1% of a map's pairs make one spill.
                    let mut map =
                        Map { subs: AVec::with_capacity_in(parts, bump), spill: AVec::with_capacity_in(pairs / 64, bump) };
                    for g in 0..GROUPS {
                        // Group g's pairs, split by sub-map; then its sub-maps, one at a time.
                        let split = split_pass(&outgoing, (d * 3 + j) * GROUPS + g, parts, g * per, per, &scratch);
                        for q in 0..per {
                            map.fill(&split, q, bump, j, &mut st);
                        }
                    }
                    map
                });
                // Release the scratch's pages.
                let bytes = (scratch_len(d) * size_of::<Pair>()).next_multiple_of(4096);
                unsafe { libc::madvise(scratch.ptr.cast(), bytes, libc::MADV_DONTNEED) };
                (maps, st)
            });
            // Free phase 1's lists: each thread releases its own arena's pages, then one munmap.
            {
                let lists_region = &lists_region;
                pool.broadcast(|ctx| {
                    let t = ctx.index();
                    let bytes = (arena_len(t) * size_of::<Pair>()).next_multiple_of(4096);
                    unsafe { libc::madvise(lists_region.ptr.add(offs1[t]).cast(), bytes, libc::MADV_DONTNEED) };
                });
            }
            drop(outgoing);
            drop(lists_region);
            let mut all = Vec::with_capacity(t_count);
            let mut total = Stats::default();
            for (m, s) in results {
                all.push(m);
                for j in 0..3 {
                    total.keys[j] += s.keys[j];
                }
                total.entries += s.entries;
                total.dropped += s.dropped;
                total.spilled += s.spilled;
            }
            (all, arenas, region, total, phase1_s, t0.elapsed().as_secs_f64() - phase1_s)
        });
        self.maps = maps;
        self.arenas = arenas;
        self.region = Some(region);
        let mut stats = self.stats(st);
        stats.insert("phase1_s".into(), phase1_s);
        stats.insert("phase2_s".into(), phase2_s);
        Ok(stats)
    }

    /// The counter stored for `context` (its last k tokens), or an empty dict (for tests).
    fn lookup(&self, k: usize, context: Vec<u16>) -> PyResult<HashMap<u16, u32>> {
        let j = KS.iter().position(|&x| x == k).ok_or_else(|| PyValueError::new_err("k must be 6, 10 or 20"))?;
        if context.len() < k {
            return Err(PyValueError::new_err("context shorter than k"));
        }
        let key = key_dyn(k, &context[context.len() - k..]);
        Ok(self.maps.get(owner(key, self.threads)).map(|m| m[j].counter(key).into_iter().collect()).unwrap_or_default())
    }

    /// Sum of all counts per k (for tests: every (position, k) is counted unless dropped).
    fn totals(&self) -> Vec<u64> {
        (0..3)
            .map(|j| {
                self.maps
                    .iter()
                    .map(|m| {
                        let m = &m[j];
                        m.subs.iter().flat_map(|s| s.values()).flat_map(|&v| m.items(v)).map(|(_, c)| c as u64).sum::<u64>()
                    })
                    .sum()
            })
            .collect()
    }
}

impl HashMapIndex {
    /// "entries": distinct (key, next) pairs stored (one counter item each); "keys", "keysK": distinct keys; "spilled":
    /// counters that outgrew the value (moved to `spill`); "bytes": all the maps' memory (the arenas' used bytes).
    fn stats(&self, st: Stats) -> HashMap<String, f64> {
        let mut out = HashMap::new();
        for (j, &k) in KS.iter().enumerate() {
            out.insert(format!("keys{k}"), st.keys[j] as f64);
        }
        let bytes: usize = self.arenas.iter().map(|a| a.top.load(Relaxed) - a.start).sum();
        out.insert("spilled".into(), st.spilled as f64);
        out.insert("bytes".into(), bytes as f64);
        out.insert("keys".into(), st.keys.iter().sum::<u64>() as f64);
        out.insert("entries".into(), st.entries as f64);
        out.insert("dropped".into(), st.dropped as f64);
        out.insert("threads".into(), self.threads as f64);
        out.insert("tokens".into(), self.shards.iter().map(|s| s.len).sum::<usize>() as f64);
        out
    }
}
