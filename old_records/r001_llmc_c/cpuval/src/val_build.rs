//! The `scan` engine: the rows of its levels found by scanning the training shards for the validation contexts, all of
//! it in the lookup (it needs the validation tokens from the start). Level k counted over the first K_k training
//! shards only: old_records' ApproxValCountRows / ApproxValHintRows with `level_shards` (with every K_k all the shards,
//! exactly ValCountRows / ValHintRows). kgram's definitions: contexts within a validation segment, occurrences within
//! one shard; next tokens ranked by count descending, then token ascending.
//!
//!   1. index: per level the distinct valid validation contexts (64-bit hashes, ids, a table of fingerprints), a
//!      blocked Bloom filter of the contexts for levels 3-6 (tested before the table), and per anchored level (the
//!      shallowest level of a shard range, if at least 5) a filter of its contexts' w-grams and one of the contexts;
//!   2. scan: the shards in ranges between the distinct K's, each with the levels active there; per position the
//!      levels ascending (a position whose k-context is no validation k-context has no longer one either), probes
//!      batched with prefetches. In an anchored range only the positions near a w-gram that passes the first filter
//!      (anchors at shard offsets k0 - w + 1 apart: every k0-context contains one) whose k0-context passes the second
//!      probe. Each hit appends (context id << 16 | next token) to its level's partition in the thread's lists;
//!   3. reduce, the levels deepest first: per partition the occurrences counted (sorted, or hashed if many), each
//!      context's rows written at its validation positions (rows.rs's write_context).
use crate::rows::{Engine, Lv, Out, Output, Scratch, Shards, Times, Val, HEADER_BYTES};
use rayon::prelude::*;
use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

const PIECE: usize = 1 << 20; // shard positions per parallel piece
const BACK: usize = 32; // tokens read before a piece (at least the longest level)
const BATCH: usize = 32; // probes in flight per thread
const WINDOW: usize = 64; // anchors whose filter words are prefetched at once
const PARTS_LOG2: u32 = 12; // ~4096 reduce partitions per level
const TABLE_LOAD: f64 = 0.5; // the context tables' maximum load
const PF_LEVELS: (usize, usize) = (3, 6); // the levels with a probe filter, PF_BITS bits per context
const PF_BITS: f64 = 8.0;
const ANCHORED_FROM: usize = 5; // a shard range whose shallowest level is at least this scans through anchors
const FILTER_BITS: u32 = 28; // the anchored levels' two filters
const HASH_MIN: usize = 1 << 19; // a partition with more occurrences counts them in a hash table instead of sorting
const P: u64 = 0x9e3779b185ebca87; // kgram's context hash
const R: u64 = 0x2545f4914f6cdd1d; // the rolling hash's multiplier
const M: u64 = 0x9e3779b97f4a7c15; // the filters' mixer
const FP: u64 = (1 << 40) - 1; // the tables' fingerprints

/// kgram's context hash (nonzero).
#[inline]
fn hash(tokens: &[u16]) -> u64 {
    let mut h = tokens.iter().fold(0u64, |h, &t| h.wrapping_mul(P).wrapping_add(t as u64 + 1));
    h ^= h >> 30;
    h = h.wrapping_mul(0xbf58476d1ce4e5b9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94d049bb133111eb);
    (h ^ (h >> 31)) | 1
}

/// The rolling hash of `tokens` (polynomial in R over token + 1).
#[inline]
fn roll(tokens: &[u16]) -> u64 {
    tokens.iter().fold(0u64, |h, &t| h.wrapping_mul(R).wrapping_add(t as u64 + 1))
}

/// The filters' key of `tokens`: for 4 to 8 tokens two unaligned 8-byte loads (the first and the last 4 tokens) mixed
/// by two multiplications, else the rolling hash.
#[inline]
fn fkey(tokens: &[u16]) -> u64 {
    let n = tokens.len();
    if (4..=8).contains(&n) {
        let p = tokens.as_ptr();
        let a = unsafe { std::ptr::read_unaligned(p.cast::<u64>()) };
        let b = unsafe { std::ptr::read_unaligned(p.add(n - 4).cast::<u64>()) };
        (a.wrapping_mul(0x9e3779b97f4a7c15) ^ b ^ n as u64).wrapping_mul(0xd6e8feb86659fd93)
    } else {
        roll(tokens)
    }
}

/// `n` zeros (zeroed in parallel) in a Vec whose pages are faulted as transparent huge pages: the random probes of big
/// tables then hit the TLB.
fn huge_zeroed<T: Copy + Default + Send + Sync>(n: usize) -> Vec<T> {
    let mut v: Vec<T> = Vec::with_capacity(n);
    let bytes = n * std::mem::size_of::<T>();
    if bytes >= 1 << 21 {
        let start = (v.as_mut_ptr() as usize).next_multiple_of(4096);
        let end = v.as_mut_ptr() as usize + bytes;
        if end > start {
            unsafe { libc::madvise(start as *mut libc::c_void, end - start, libc::MADV_HUGEPAGE) };
        }
    }
    unsafe { v.set_len(n) };
    v.par_chunks_mut(1 << 16).for_each(|c| c.fill(T::default()));
    v
}

/// A level's context table in one array of 8-byte slots, fingerprint << 24 | id (the hash's low 40 bits, nonzero as its
/// lowest bit is set; ids below 2^24), homes at the hash's top bits, linear probing: one cache line per probe. Two
/// contexts differ undetected only if their hashes agree in the low 40 bits and their homes in the same run.
struct Table {
    bits: u32,
    slots: Vec<u64>,
}

impl Table {
    /// The table of the contexts with hashes `hashes` (id = index), filled in parallel.
    fn new(hashes: &[u64]) -> Result<Self, String> {
        if hashes.len() >= 1 << 24 {
            return Err("more than 2^24 contexts at a level".into());
        }
        let bits = ((hashes.len() as f64 / TABLE_LOAD).ceil() as usize).next_power_of_two().max(1024).trailing_zeros();
        let slots: Vec<u64> = huge_zeroed(1 << bits);
        let mask = (1usize << bits) - 1;
        {
            let atomics: &[AtomicU64] = unsafe { &*(slots.as_slice() as *const [u64] as *const [AtomicU64]) };
            hashes.par_iter().enumerate().for_each(|(id, &h)| {
                let v = (h & FP) << 24 | id as u64;
                let mut at = (h >> (64 - bits)) as usize;
                while atomics[at].compare_exchange(0, v, Ordering::Relaxed, Ordering::Relaxed).is_err() {
                    at = (at + 1) & mask;
                }
            });
        }
        Ok(Self { bits, slots })
    }

    #[inline]
    fn prefetch(&self, h: u64) {
        unsafe { _mm_prefetch(self.slots.as_ptr().wrapping_add((h >> (64 - self.bits)) as usize).cast(), _MM_HINT_T0) };
    }

    #[inline]
    fn find(&self, h: u64) -> Option<usize> {
        let mask = self.slots.len() - 1;
        let fp = h & FP;
        let mut s = (h >> (64 - self.bits)) as usize;
        loop {
            let v = unsafe { *self.slots.get_unchecked(s) };
            if v == 0 {
                return None;
            }
            if v >> 24 == fp {
                return Some((v & 0xFF_FFFF) as usize);
            }
            s = (s + 1) & mask;
        }
    }
}

/// A blocked Bloom filter: two bits in one 64-bit word per key.
struct Filter {
    words: Vec<u64>,
    shift: u32,
}

impl Filter {
    fn new(bits: u32, keys: impl ParallelIterator<Item = u64>) -> Self {
        let words: Vec<u64> = huge_zeroed(1usize << (bits - 6));
        let shift = 64 - (bits - 6);
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
    fn prefetch(&self, h: u64) {
        let (w, _) = Self::at(h, self.shift);
        unsafe { _mm_prefetch(self.words.as_ptr().wrapping_add(w).cast(), _MM_HINT_T0) };
    }

    #[inline]
    fn test(&self, h: u64) -> bool {
        let (w, b) = Self::at(h, self.shift);
        unsafe { *self.words.get_unchecked(w) & b == b }
    }
}

/// Per thread, per level, per partition: the occurrences found, in chains of blocks (64 entries, doubling up to 2^16)
/// carved from the thread's arena of 64 MB transparent-huge-page chunks: no reallocation or copying, page faults at
/// 2 MB.
struct Lists {
    hot: Vec<Vec<Hot>>,
    full: Vec<Vec<Vec<(*mut u64, u32)>>>,
    arena: Arena,
}

#[derive(Clone, Copy)]
struct Hot {
    cur: *mut u64,
    len: u32,
    cap: u32,
}

unsafe impl Send for Lists {}

const CHUNK: usize = 8 << 20; // u64s per arena chunk (64 MB)

struct Arena {
    chunks: Vec<(*mut u64, usize)>,
    at: *mut u64,
    left: usize,
}

impl Arena {
    fn alloc(&mut self, n: usize) -> *mut u64 {
        if self.left < n {
            let size = n.max(CHUNK);
            let flags = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE;
            let p = unsafe { libc::mmap(std::ptr::null_mut(), size * 8, libc::PROT_READ | libc::PROT_WRITE, flags, -1, 0) };
            assert!(p != libc::MAP_FAILED, "arena mmap failed");
            unsafe { libc::madvise(p, size * 8, libc::MADV_HUGEPAGE) };
            self.chunks.push((p.cast(), size));
            self.at = p.cast();
            self.left = size;
        }
        let r = self.at;
        self.at = self.at.wrapping_add(n);
        self.left -= n;
        r
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        for &(p, n) in &self.chunks {
            unsafe { libc::munmap(p.cast(), n * 8) };
        }
    }
}

impl Lists {
    fn new(parts: impl Iterator<Item = usize>) -> Self {
        let hot: Vec<Vec<Hot>> = parts.map(|n| vec![Hot { cur: std::ptr::null_mut(), len: 0, cap: 0 }; n]).collect();
        let full = hot.iter().map(|h| vec![Vec::new(); h.len()]).collect();
        Self { hot, full, arena: Arena { chunks: Vec::new(), at: std::ptr::null_mut(), left: 0 } }
    }

    #[inline]
    fn push(&mut self, i: usize, p: usize, v: u64) {
        let h = unsafe { *self.hot.get_unchecked(i).get_unchecked(p) };
        if h.len == h.cap {
            self.grow(i, p);
        }
        let h = unsafe { self.hot.get_unchecked_mut(i).get_unchecked_mut(p) };
        unsafe { *h.cur.add(h.len as usize) = v };
        h.len += 1;
    }

    #[cold]
    #[inline(never)]
    fn grow(&mut self, i: usize, p: usize) {
        let h = self.hot[i][p];
        if h.cap > 0 {
            self.full[i][p].push((h.cur, h.len));
        }
        let cap = if h.cap == 0 { 64 } else { (h.cap * 2).min(1 << 16) };
        self.hot[i][p] = Hot { cur: self.arena.alloc(cap as usize), len: 0, cap };
    }

    /// Level i, partition p's blocks (pointer, length).
    fn blocks(&self, i: usize, p: usize) -> impl Iterator<Item = (*mut u64, usize)> + '_ {
        let h = self.hot[i][p];
        self.full[i][p].iter().map(|&(b, n)| (b, n as usize)).chain((h.len > 0).then_some((h.cur, h.len as usize)))
    }
}

/// A partition's blocks across the threads, sent to the reduce's workers.
struct Blocks(Vec<(*mut u64, usize)>);
unsafe impl Send for Blocks {}

/// A level: its distinct valid validation contexts (a position of each id; the positions grouped by id:
/// order[first[id]..first[id + 1]]), their table, its partitions (id >> shift) and its probe filter.
struct Level {
    k: usize,
    reps: Vec<u32>,
    first: Vec<u64>,
    order: Vec<u32>,
    table: Table,
    shift: u32,
    parts: usize,
    pf: Option<Filter>,
}

impl Level {
    /// The contexts of level k: the positions' hashes deduplicated in a scratch table (ids by slot order), the table
    /// built at its size, each position's id looked up (u32::MAX where its k-context is invalid), the positions
    /// grouped by id, and each position's context checked token by token against its id's first position.
    fn new(x: &[u16], avail: &[u32], k: usize) -> Result<Self, String> {
        let hs: Vec<u64> = (0..x.len()).into_par_iter().map(|i| if avail[i] as usize >= k { hash(&x[i + 1 - k..=i]) } else { 0 }).collect();
        let valid = hs.par_iter().filter(|&&h| h != 0).count();
        let bits = ((valid as f64 / 0.75).ceil() as usize).next_power_of_two().max(1024).trailing_zeros();
        let scratch: Vec<u64> = huge_zeroed(1 << bits);
        {
            let atomics: &[AtomicU64] = unsafe { &*(scratch.as_slice() as *const [u64] as *const [AtomicU64]) };
            let mask = (1usize << bits) - 1;
            hs.par_iter().filter(|&&h| h != 0).for_each(|&h| {
                let mut at = (h >> (64 - bits)) as usize;
                loop {
                    match atomics[at].compare_exchange(0, h, Ordering::Relaxed, Ordering::Relaxed) {
                        Ok(_) => break,
                        Err(v) if v == h => break,
                        Err(_) => at = (at + 1) & mask,
                    }
                }
            });
        }
        let hashes: Vec<u64> = scratch.par_iter().copied().filter(|&h| h != 0).collect();
        drop(scratch);
        let table = Table::new(&hashes)?;
        let key_of: Vec<u32> = hs.par_iter().map(|&h| if h == 0 { u32::MAX } else { table.find(h).expect("an indexed hash") as u32 }).collect();
        drop(hs);
        let keys = hashes.len();
        // the positions sorted by id, and each id's first index from the boundaries (every id has a position)
        let mut pairs: Vec<u64> = key_of.par_iter().enumerate().filter(|&(_, &id)| id != u32::MAX).map(|(i, &id)| (id as u64) << 32 | i as u64).collect();
        pairs.par_sort_unstable();
        let mut first = vec![0u64; keys + 1];
        let out = Output(first.as_mut_ptr());
        pairs.par_iter().enumerate().for_each(|(j, &p)| {
            if j == 0 || pairs[j - 1] >> 32 != p >> 32 {
                unsafe { out.set((p >> 32) as usize, j as u64) };
            }
        });
        first[keys] = pairs.len() as u64;
        let order: Vec<u32> = pairs.par_iter().map(|&p| p as u32).collect();
        drop(pairs);
        let reps: Vec<u32> = first[..keys].par_iter().map(|&f| order[f as usize]).collect();
        let bad = key_of.par_iter().enumerate().find_any(|&(i, &id)| {
            id != u32::MAX && {
                let r = reps[id as usize] as usize;
                x[i + 1 - k..=i] != x[r + 1 - k..=r]
            }
        });
        if let Some((i, &id)) = bad {
            return Err(format!("hash collision between validation {k}-contexts at {} and {i}", reps[id as usize]));
        }
        let bits = (keys.max(2) - 1).ilog2() + 1;
        let shift = bits.saturating_sub(PARTS_LOG2);
        let parts = ((keys.max(2) - 1) >> shift) + 1;
        let pf = (PF_LEVELS.0..=PF_LEVELS.1).contains(&k).then(|| {
            let bits = ((reps.len().max(1) as f64 * PF_BITS).log2().ceil() as u32).clamp(16, 34);
            Filter::new(bits, reps.par_iter().map(|&r| hash(&x[r as usize + 1 - k..=r as usize])))
        });
        Ok(Self { k, reps, first, order, table, shift, parts, pf })
    }
}

/// The scan of the shards `files` (`sizes` tokens each) for the levels `act` (indices into `levels`, ascending): step 2.
/// Each piece's tokens (and up to BACK before it) are read with pread into the thread's buffer. With `anchor` (w and
/// the shallowest level's two filters) only the candidates of the anchors that pass the first filter and whose
/// context passes the second probe.
fn scan(files: &[File], sizes: &[usize], levels: &[Level], act: &[usize], anchor: Option<(usize, &Filter, &Filter)>, accs: &[Mutex<Lists>]) {
    let k0 = levels[act[0]].k;
    let mut pieces = Vec::new(); // (file, start, end), shard positions
    for (f, &n) in sizes.iter().enumerate() {
        let mut start = k0;
        while start < n {
            let end = n.min(start + PIECE);
            pieces.push((f, start, end));
            start = end;
        }
    }
    thread_local! {
        static BUF: std::cell::RefCell<Vec<u16>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    pieces.par_iter().for_each(|&(f, start, end)| {
        let t = rayon::current_thread_index().unwrap_or(0);
        let mut guard = accs[t].lock().unwrap();
        let acc = &mut *guard;
        BUF.with_borrow_mut(|buf| {
            // the piece's tokens and up to BACK before it: local index i is shard position back + i
            let back = start - BACK.min(start);
            buf.resize(end - back, 0);
            let bytes = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr().cast::<u8>(), 2 * (end - back)) };
            files[f].read_exact_at(bytes, (HEADER_BYTES + 2 * back) as u64).expect("a shard read failed");
            let tokens: &[u16] = buf;
            let (s0, e0) = (start - back, end - back);
            let first0 = s0.max(k0.saturating_sub(back));
            let mut cand: Vec<(usize, u64)> = Vec::with_capacity(BATCH);
            let mut next: Vec<(usize, u64)> = Vec::with_capacity(BATCH);
            let Some((w, f1, f2)) = anchor else {
                // every position, WINDOW at a time
                let mut p = first0;
                while p < e0 {
                    let stop = e0.min(p + WINDOW);
                    cand.extend((p..stop).map(|q| (q, 0)));
                    probe(&mut cand, &mut next, tokens, levels, act, acc, back);
                    p = stop;
                }
                return;
            };
            // anchors: the w-grams starting at shard offsets that are multiples of S = k0 - w + 1, each the only one
            // of its S candidate positions' k0-contexts (q .. q + S for the anchor ending at q); per window of anchors
            // their filter words prefetched, then the hits' k0-contexts' words in the second filter
            let step = k0 - w + 1;
            let mut q = (s0 + 1).saturating_sub(step).max(w);
            while (q + back - w) % step != 0 {
                q += 1;
            }
            let mut ha = [(0usize, 0u64); WINDOW];
            let mut c2: Vec<(usize, u64)> = Vec::with_capacity(WINDOW * step);
            while q < e0 {
                let mut m = 0;
                while m < WINDOW && q < e0 {
                    let hq = fkey(&tokens[q - w..q]);
                    f1.prefetch(hq);
                    ha[m] = (q, hq);
                    m += 1;
                    q += step;
                }
                c2.clear();
                for &(qa, hq) in &ha[..m] {
                    if f1.test(hq) {
                        for pp in qa.max(s0).max(first0)..(qa + step).min(e0) {
                            let h2 = fkey(&tokens[pp - k0..pp]);
                            f2.prefetch(h2);
                            c2.push((pp, h2));
                        }
                    }
                }
                cand.extend(c2.iter().filter(|&&(_, h2)| f2.test(h2)).map(|&(pp, _)| (pp, 0)));
                if cand.len() >= BATCH || (q >= e0 && !cand.is_empty()) {
                    probe(&mut cand, &mut next, tokens, levels, act, acc, back);
                }
            }
        })
    });
}

/// The probes of the candidate positions `cand` (local indices into `tokens`, whose index 0 has `back` shard tokens
/// before it), level by level of `act` while they hit; `cand` is left empty.
#[inline]
fn probe(cand: &mut Vec<(usize, u64)>, next: &mut Vec<(usize, u64)>, tokens: &[u16], levels: &[Level], act: &[usize], acc: &mut Lists, back: usize) {
    for &i in act {
        if cand.is_empty() {
            break;
        }
        let level = &levels[i];
        let k = level.k;
        next.clear();
        for x in cand.iter_mut() {
            x.1 = hash(&tokens[x.0 - k..x.0]);
            match &level.pf {
                Some(pf) => pf.prefetch(x.1),
                None => level.table.prefetch(x.1),
            }
        }
        if let Some(pf) = &level.pf {
            cand.retain(|x| {
                let pass = pf.test(x.1);
                if pass {
                    level.table.prefetch(x.1);
                }
                pass
            });
        }
        for &(q, hq) in cand.iter() {
            if let Some(id) = level.table.find(hq) {
                acc.push(i, id >> level.shift, (id as u64) << 16 | tokens[q] as u64);
                next.push((q, 0));
            }
        }
        // the next level needs its k tokens in the shard
        if let Some(&j) = act.iter().find(|&&j| j > i) {
            let kk = levels[j].k;
            next.retain(|&(q, _)| q + back >= kk);
        }
        std::mem::swap(cand, next);
    }
    cand.clear();
}

/// The distinct values of the blocks (each `base` + a key below 2^32) with their counts, sorted by value: an
/// open-addressing table of key << 32 | count (0: empty), doubled at load 1/2.
fn hash_count(blocks: &[(*mut u64, usize)], base: u64) -> Vec<(u64, u32)> {
    let mut bits = 12u32;
    let mut t: Vec<u64> = vec![0; 1 << bits];
    let mut used = 0usize;
    let slot = |k: u64, bits: u32| (k.wrapping_mul(0x9e3779b97f4a7c15) >> (64 - bits)) as usize;
    for &(b, m) in blocks {
        for &x in unsafe { std::slice::from_raw_parts(b, m) } {
            let k = x - base;
            let mask = (1usize << bits) - 1;
            let mut s = slot(k, bits);
            loop {
                let v = unsafe { *t.get_unchecked(s) };
                if v == 0 {
                    unsafe { *t.get_unchecked_mut(s) = k << 32 | 1 };
                    used += 1;
                    break;
                }
                if v >> 32 == k {
                    unsafe { *t.get_unchecked_mut(s) = v + 1 };
                    break;
                }
                s = (s + 1) & mask;
            }
            if 2 * used > 1usize << bits {
                bits += 1;
                let mut t2: Vec<u64> = vec![0; 1 << bits];
                let mask = (1usize << bits) - 1;
                for &v in &t {
                    if v != 0 {
                        let mut s = slot(v >> 32, bits);
                        while t2[s] != 0 {
                            s = (s + 1) & mask;
                        }
                        t2[s] = v;
                    }
                }
                t = t2;
            }
        }
    }
    let mut pairs: Vec<(u64, u32)> = t.iter().filter(|&&v| v != 0).map(|&v| ((v >> 32) + base, v as u32)).collect();
    pairs.sort_unstable_by_key(|p| p.0);
    pairs
}

/// Step 3 for level `l` (its partitions' blocks `lists`, its rows at `lv`'s column): its rows at its validation
/// positions.
fn reduce_level(lv: &Lv, lists: Vec<Blocks>, l: &Level, target: &[u16], out: &Out) {
    lists.into_par_iter().enumerate().for_each_init(Scratch::default, |scratch, (p, v)| {
        let nocc: usize = v.0.iter().map(|b| b.1).sum();
        // the distinct (context, next token) values with their counts, in order
        let pairs: Vec<(u64, u32)> = if nocc > HASH_MIN {
            hash_count(&v.0, ((p << l.shift) as u64) << 16)
        } else {
            let mut all: Vec<u64> = Vec::with_capacity(nocc);
            for &(b, n) in &v.0 {
                all.extend_from_slice(unsafe { std::slice::from_raw_parts(b, n) });
            }
            all.sort_unstable();
            all.chunk_by(|a, b| a == b).map(|same| (same[0], same.len() as u32)).collect()
        };
        let mut by_token: Vec<(u16, u32)> = Vec::new();
        for run in pairs.chunk_by(|a, b| a.0 >> 16 == b.0 >> 16) {
            let id = (run[0].0 >> 16) as usize;
            by_token.clear();
            by_token.extend(run.iter().map(|&(x, c)| ((x & 0xFFFF) as u16, c)));
            let at = l.order[l.first[id] as usize..l.first[id + 1] as usize].iter().map(|&q| q as usize);
            unsafe { out.write_context(lv.col, lv.bucket, &by_token, at, target, scratch) };
        }
    });
}

/// The engine: its levels (ascending k), nothing before the last step.
pub struct Scan {
    levels: Vec<Lv>,
}

impl Scan {
    pub fn new(levels: &[Lv]) -> Result<Self, String> {
        if levels.iter().any(|l| l.k < 2 || l.k > BACK) {
            return Err(format!("scan: levels 2 to {BACK}"));
        }
        let mut levels = levels.to_vec();
        levels.sort_by_key(|l| l.k);
        Ok(Self { levels })
    }
}

impl Engine for Scan {
    fn build(&mut self, _shards: &Shards) -> Result<Times, String> {
        Ok(Vec::new())
    }

    fn scans(&self) -> bool {
        true
    }

    fn deepest(&self) -> usize {
        self.levels.iter().map(|l| l.k).max().unwrap_or(0)
    }

    /// Steps 1-3 for its levels, in the current rayon pool.
    fn lookup(&mut self, shards: &Shards, val: &Val, out: &Out) -> Result<Times, String> {
        let (files, sizes, levels) = (&shards.files, &shards.sizes, &self.levels);
        let mut times: Times = Vec::new();
        let t0 = Instant::now();
        // the shard ranges [bounds[j], bounds[j + 1]), each with its active levels (indices)
        let mut bounds: Vec<usize> = std::iter::once(0).chain(levels.iter().map(|l| l.files)).collect();
        bounds.sort_unstable();
        bounds.dedup();
        let segs: Vec<(usize, usize, Vec<usize>)> = bounds.windows(2).map(|b| (b[0], b[1], (0..levels.len()).filter(|&i| levels[i].files > b[0]).collect())).collect();
        let mut anchored: Vec<usize> = segs.iter().map(|s| s.2[0]).filter(|&i| levels[i].k >= ANCHORED_FROM).collect();
        anchored.sort_unstable();
        anchored.dedup();
        // 1. index
        let (x, target, avail) = (val.x, val.target, &val.avail);
        let lv: Vec<Level> = levels.par_iter().map(|l| Level::new(x, avail, l.k)).collect::<Result<_, _>>()?;
        // per anchored level (w = k - 3 from level 8 on, else k - 1): the filter of its contexts' w-grams, and of the
        // contexts
        let filters: Vec<(usize, usize, Filter, Filter)> = anchored
            .iter()
            .map(|&i| {
                let k = lv[i].k;
                let w = if k >= 8 { k - 3 } else { k - 1 };
                let reps = &lv[i].reps;
                let f1 = Filter::new(
                    FILTER_BITS,
                    reps.par_iter().flat_map_iter(move |&r| {
                        let ctx = &x[r as usize + 1 - k..=r as usize];
                        (0..=k - w).map(move |o| fkey(&ctx[o..o + w]))
                    }),
                );
                let f2 = Filter::new(FILTER_BITS, reps.par_iter().map(|&r| fkey(&x[r as usize + 1 - k..=r as usize])));
                (i, w, f1, f2)
            })
            .collect();
        times.push(("index_s".into(), t0.elapsed().as_secs_f64()));
        // 2. scan
        let accs: Vec<Mutex<Lists>> = (0..rayon::current_num_threads()).map(|_| Mutex::new(Lists::new(lv.iter().map(|l| l.parts)))).collect();
        for (a, b, act) in &segs {
            let t = Instant::now();
            let fl = filters.iter().find(|f| f.0 == act[0]);
            scan(&files[*a..*b], &sizes[*a..*b], &lv, act, fl.map(|(_, w, f1, f2)| (*w, f1, f2)), &accs);
            times.push((format!("scan_k{}_files{a}-{b}_s", lv[act[0]].k), t.elapsed().as_secs_f64()));
        }
        // 3. reduce, the levels deepest first
        let t = Instant::now();
        let accs: Vec<Lists> = accs.into_iter().map(|a| a.into_inner().unwrap()).collect();
        let mut lists: Vec<Vec<Blocks>> = lv.iter().enumerate().map(|(i, l)| (0..l.parts).map(|p| Blocks(accs.iter().flat_map(|a| a.blocks(i, p)).collect())).collect()).collect();
        for i in (0..lv.len()).rev() {
            let tl = Instant::now();
            reduce_level(&levels[i], std::mem::take(&mut lists[i]), &lv[i], target, out);
            times.push((format!("reduce_k{}_s", lv[i].k), tl.elapsed().as_secs_f64()));
        }
        times.push(("reduce_s".into(), t.elapsed().as_secs_f64()));
        times.push(("total_s".into(), t0.elapsed().as_secs_f64()));
        // the big structures freed in the background (munmap of GBs took 1.5-4 s)
        std::thread::spawn(move || drop((lists, accs, lv, filters)));
        Ok(times)
    }
}
