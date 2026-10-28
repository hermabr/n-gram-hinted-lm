//! The validation-driven CPU scan of the cpuval experiment (retrieval_blog_post/cpu_val.py): the validation count rows
//! (old_records.ValCountRows: tot, dist, cnt per position and level) and per level each validation context's top next
//! tokens (for the hint rows), counted on CPUs in one pass over the corpus, with the levels up to `short` counted over
//! the first `short_files` shards only (capping: short contexts converge with few tokens, long ones need the
//! whole corpus but are rare). kgram.rs's definitions (contexts within a validation segment, occurrences within one
//! shard; next tokens ranked by count descending, then token ascending); with short_files = all files, exactly
//! kgram_counts's counts at every level.
//!
//!   1. index: per level, the distinct valid validation contexts (kgram's Contexts: 64-bit hashes, ids by hash order),
//!      and a blocked Bloom filter of the shallowest long level's contexts under a rolling hash;
//!   2. short scan: the short levels over the first `short_files` shards: per position the levels ascending (a
//!      position whose k-context is no validation k-context has no longer one either), probes batched with prefetches;
//!      each hit appends (context id << 16 | next token) to its level's partition (by context id) in the thread's lists;
//!   3. short reduce: per level and partition, the occurrences sorted; per context its total, distinct next tokens and
//!      top `keep` next tokens, and per validation position of it tot, dist and cnt (the target's count if among the
//!      `top` most frequent). With `scale` f > 0 these are extrapolated to the whole corpus (iid): tot and cnt times f,
//!      dist times f ** growth[k] (at most the scaled tot and 65535);
//!   4. the hybrid (`rare_below` T > 0): the short levels' contexts that occurred fewer than T times in the short scan
//!      (unseen ones too) are "rare", each level's in its own Bloom filter;
//!   5. long scan, over all shards: per position the rolling hash of its shallowest long context tests the long
//!      filter, and only the positions that pass probe the long levels' tables, ascending; with the hybrid, the
//!      position's k-context at each short level k also tests that level's rare filter, and a rare hit is counted;
//!   6. long reduce as 3 (unscaled), and the rare occurrences' reduce, whose counts (exact, all shards) replace the
//!      rare contexts' rows and top next tokens.
use crate::kgram::{hash, segments, Corpus};
use crate::Output;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::{exceptions::PyValueError, prelude::*};
use rayon::prelude::*;
use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

const PIECE: usize = 1 << 20; // corpus positions per parallel piece
const BACK: usize = 32; // tokens read before a piece (at least the longest level)
const BATCH: usize = 32; // probes in flight per thread
const WINDOW: usize = 64; // positions whose filter words are prefetched at once
const R: u64 = 0x2545f4914f6cdd1d; // the rolling hash's multiplier
const M: u64 = 0x9e3779b97f4a7c15; // the filter's mixer

/// `n` zeros in a Vec whose pages are faulted as transparent huge pages (MADV_HUGEPAGE before the first touch): the
/// random probes of big tables then hit the TLB.
pub(crate) fn huge_zeroed<T: Copy + Default>(n: usize) -> Vec<T> {
    let mut v: Vec<T> = Vec::with_capacity(n);
    let bytes = n * std::mem::size_of::<T>();
    if bytes >= 1 << 21 {
        let start = (v.as_mut_ptr() as usize).next_multiple_of(4096);
        let end = v.as_mut_ptr() as usize + bytes;
        if end > start {
            unsafe { libc::madvise(start as *mut libc::c_void, end - start, libc::MADV_HUGEPAGE) };
        }
    }
    v.resize(n, T::default());
    v
}

/// The rolling hash of `tokens` (polynomial in R over token + 1), which the scan updates position by position.
#[inline]
fn roll(tokens: &[u16]) -> u64 {
    tokens.iter().fold(0u64, |h, &t| h.wrapping_mul(R).wrapping_add(t as u64 + 1))
}

/// The validation positions grouped by context id (kgram's by_context, in parallel): positions
/// order[first[id]..first[id + 1]] have context id `id`, ascending.
fn by_context(key_of: &[u32], keys: usize) -> (Vec<u64>, Vec<u32>) {
    let mut pairs: Vec<u64> = key_of.par_iter().enumerate().filter(|&(_, &id)| id != u32::MAX).map(|(i, &id)| (id as u64) << 32 | i as u64).collect();
    pairs.par_sort_unstable();
    let first: Vec<u64> = (0..=keys).into_par_iter().map(|id| pairs.partition_point(|&p| p >> 32 < id as u64) as u64).collect();
    let order: Vec<u32> = pairs.par_iter().map(|&p| p as u32).collect();
    (first, order)
}

/// A level's context table in one array of 8-byte slots, fingerprint << 24 | id (the hash's low 40 bits, nonzero
/// as its lowest bit is set; ids below 2^24), homes at the hash's top bits, linear probing: one cache line per probe
/// (kgram's Contexts has two arrays). A hit compares the home's run of slots by fingerprint, so two contexts
/// differ undetected only if their hashes agree in the low 40 bits and their homes in the same run (~1e-12 per probe).
struct Table {
    bits: u32,
    slots: Vec<u64>,
}

impl Table {
    /// The table of the contexts with hashes `hashes` (id = index), filled in parallel (compare-and-swap into the
    /// first empty slot of the home's run: the ids' slots differ from a sequential fill, the lookups agree).
    fn new(hashes: &[u64]) -> PyResult<Self> {
        if hashes.len() >= 1 << 24 {
            return Err(PyValueError::new_err("more than 2^24 contexts at a level"));
        }
        let bits = (2 * hashes.len()).next_power_of_two().max(1024).trailing_zeros();
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

const FP: u64 = (1 << 40) - 1;

/// A blocked Bloom filter: two bits in one 64-bit word per key.
struct Filter {
    words: Vec<u64>,
    shift: u32,
}

impl Filter {
    fn new(bits: u32, keys: impl ParallelIterator<Item = u64>) -> Self {
        let n = 1usize << (bits - 6);
        let words: Vec<u64> = huge_zeroed(n);
        let shift = 64 - (bits - 6);
        {
            // the same layout: AtomicU64 is a u64
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

/// Per thread, per level, per partition: the occurrences found.
type Lists = Vec<Vec<Vec<u64>>>;

/// A level's distinct valid validation contexts (kgram's Contexts, built in parallel): each position's context id
/// (u32::MAX where its k-context is invalid; ids by hash order, as kgram's) and a position of each id.
struct VContexts {
    key_of: Vec<u32>,
    reps: Vec<u32>,
}

impl VContexts {
    /// The contexts of level k and their table, without sorting: the positions' hashes deduplicated in a scratch
    /// table (compare-and-swap), ids by slot order (about hash order), the table rebuilt at its size, then each
    /// position's id looked up and checked token by token against its id's first position.
    fn new(x: &[u16], avail: &[u32], k: usize) -> PyResult<(Self, Table)> {
        let hs: Vec<u64> = (0..x.len()).into_par_iter().map(|i| if avail[i] as usize >= k { hash(&x[i + 1 - k..=i]) } else { 0 }).collect();
        let valid = hs.par_iter().filter(|&&h| h != 0).count();
        let bits = (2 * valid).next_power_of_two().max(1024).trailing_zeros();
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
        let first: Vec<std::sync::atomic::AtomicU32> = (0..hashes.len()).map(|_| std::sync::atomic::AtomicU32::new(u32::MAX)).collect();
        key_of.par_iter().enumerate().for_each(|(i, &id)| {
            if id != u32::MAX {
                first[id as usize].fetch_min(i as u32, Ordering::Relaxed);
            }
        });
        let reps: Vec<u32> = first.into_iter().map(|a| a.into_inner()).collect();
        let bad = key_of.par_iter().enumerate().find_any(|&(i, &id)| {
            id != u32::MAX && {
                let r = reps[id as usize] as usize;
                x[i + 1 - k..=i] != x[r + 1 - k..=r]
            }
        });
        if let Some((i, &id)) = bad {
            return Err(PyValueError::new_err(format!("hash collision between validation {k}-contexts at {} and {i}", reps[id as usize])));
        }
        Ok((Self { key_of, reps }, table))
    }
}

struct Level {
    k: usize,
    contexts: VContexts,
    table: Table,
    shift: u32, // partition = id >> shift
    parts: usize,
}

/// A short level's rare contexts (the hybrid): their filter (rolling hashes) and their ids' bitmap.
struct Rare {
    i: usize, // the level's index
    k: usize,
    pow: u64, // R^(k - 1)
    filter: Filter,
    ids: Vec<u64>,
}

impl Rare {
    #[inline]
    fn has(&self, id: usize) -> bool {
        self.ids[id / 64] >> (id % 64) & 1 == 1
    }
}

#[derive(Default, Clone, Copy)]
struct Counters {
    anchors: u64,
    passed: u64,
    probed: u64,
    rare_passed: u64,
    rare_found: u64,
}

impl std::ops::Add for Counters {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Self { anchors: self.anchors + o.anchors, passed: self.passed + o.passed, probed: self.probed + o.probed, rare_passed: self.rare_passed + o.rare_passed, rare_found: self.rare_found + o.rare_found }
    }
}

/// The scan of the shards `files` (token ranges `ranges`, as `ranges[i]` spans file i's tokens in the Corpus layout)
/// for the levels `lv` (indices into `levels`, ascending): steps 2 and 5. Each piece's tokens (and up to BACK before
/// it) are read with pread into the thread's buffer (no page faults on a shared mapping). With `filter` (the long
/// scan), a position's first level is only probed if its rolling hash passes it; each of `rare` tests its filter at
/// every position and counts its rare contexts' hits into `rare_accs`.
#[allow(clippy::too_many_arguments)]
fn scan(
    files: &[std::fs::File],
    ranges: &[(usize, usize)],
    levels: &[Level],
    lv: &[usize],
    filter: Option<&Filter>,
    accs: &[Mutex<Lists>],
    rare: &[Rare],
    rare_accs: &[Mutex<Lists>],
    anchor: usize,
    filter2: Option<&Filter>,
) -> Counters {
    use std::os::unix::fs::FileExt;
    if lv.is_empty() && rare.is_empty() {
        return Counters::default();
    }
    let k0 = lv.first().map(|&i| levels[i].k).unwrap_or(usize::MAX);
    let kmin = rare.iter().map(|r| r.k).min().unwrap_or(usize::MAX).min(k0);
    let mut pieces = Vec::new(); // (file, lo, start, end), positions in the Corpus layout
    for (f, &(lo, hi)) in ranges.iter().enumerate() {
        let mut start = lo + kmin;
        while start < hi {
            let end = hi.min(start + PIECE);
            pieces.push((f, lo, start, end));
            start = end;
        }
    }
    let pow = if k0 == usize::MAX { 0 } else { (0..k0 - 1).fold(1u64, |p, _| p.wrapping_mul(R)) }; // R^(k0 - 1)
    thread_local! {
        static BUF: std::cell::RefCell<Vec<u16>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    pieces
        .par_iter()
        .map(|&(f, lo, start, end)| {
            let t = rayon::current_thread_index().unwrap_or(0);
            let mut guard = accs[t].lock().unwrap();
            let acc = &mut *guard;
            let mut rare_guard = rare_accs.get(t).map(|m| m.lock().unwrap());
            BUF.with_borrow_mut(|buf| {
                // the piece's tokens and up to BACK before it: local index i is position rs + i
                let rs = lo.max(start - BACK.min(start));
                let back = rs - lo; // shard tokens before local index 0
                buf.resize(end - rs, 0);
                let bytes = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr().cast::<u8>(), 2 * (end - rs)) };
                files[f].read_exact_at(bytes, (crate::HEADER_BYTES + 2 * (rs - lo)) as u64).expect("a shard read failed");
                let tokens: &[u16] = buf;
                let (s0, e0) = (start - rs, end - rs);
                let mut c = Counters::default();
                let mut cand: Vec<(usize, u64)> = Vec::with_capacity(BATCH);
                let mut next: Vec<(usize, u64)> = Vec::with_capacity(BATCH);
                let mut rcand: Vec<(usize, usize, u64)> = Vec::with_capacity(BATCH * rare.len().max(1));
                // the rolling hashes: of the long level's context ending at q and of each rare level's
                let mut h = 0u64;
                let mut rh: Vec<u64> = vec![0; rare.len()];
                let first0 = s0.max(k0.saturating_sub(back));
                let firsts: Vec<usize> = rare.iter().map(|r| s0.max(r.k.saturating_sub(back))).collect();
                // positions in windows of WINDOW: their rolling hashes computed and their filter words prefetched,
                // then tested; the candidates accumulate until BATCH, then probe (prefetched) level by level
                if let (Some(fl), true, true) = (filter, anchor > 0, rare.is_empty()) {
                    // anchors: the w-grams starting at shard offsets that are multiples of S = k0 - w + 1, each the
                    // only one of its S candidate positions' k0-contexts (q .. q + S for the anchor ending at q)
                    let (w, step) = (anchor, k0 - anchor + 1);
                    let mut q = (s0 + 1).saturating_sub(step).max(w);
                    while (q + back - w) % step != 0 {
                        q += 1;
                    }
                    let mut ha = [(0usize, 0u64); WINDOW];
                    let mut c2: Vec<(usize, u64)> = Vec::with_capacity(WINDOW * step);
                    while q < e0 {
                        let mut m = 0;
                        while m < WINDOW && q < e0 {
                            let hq = roll(&tokens[q - w..q]);
                            fl.prefetch(hq);
                            ha[m] = (q, hq);
                            m += 1;
                            q += step;
                        }
                        c.anchors += m as u64;
                        match filter2 {
                            None => {
                                for &(qa, hq) in &ha[..m] {
                                    if fl.test(hq) {
                                        cand.extend((qa.max(s0).max(first0)..(qa + step).min(e0)).map(|pp| (pp, 0)));
                                    }
                                }
                            }
                            Some(f2) => {
                                // the second filter, of the k0-contexts themselves, prefetched for the window's hits
                                c2.clear();
                                for &(qa, hq) in &ha[..m] {
                                    if fl.test(hq) {
                                        for pp in qa.max(s0).max(first0)..(qa + step).min(e0) {
                                            let h2 = roll(&tokens[pp - k0..pp]);
                                            f2.prefetch(h2);
                                            c2.push((pp, h2));
                                        }
                                    }
                                }
                                cand.extend(c2.iter().filter(|&&(_, h2)| f2.test(h2)).map(|&(pp, _)| (pp, 0)));
                            }
                        }
                        if cand.len() >= BATCH || (q >= e0 && !cand.is_empty()) {
                            probe(&mut cand, &mut next, tokens, levels, lv, acc, back, &mut c);
                        }
                    }
                    return c;
                }
                let mut hl = [0u64; WINDOW];
                let mut hr: Vec<[u64; WINDOW]> = vec![[0u64; WINDOW]; rare.len()];
                let mut p = s0;
                while p < e0 {
                    let stop = e0.min(p + WINDOW);
                    for q in p..stop {
                        if k0 != usize::MAX && q >= first0 {
                            if let Some(fl) = filter {
                                h = if q == first0 {
                                    roll(&tokens[q - k0..q])
                                } else {
                                    h.wrapping_sub((tokens[q - 1 - k0] as u64 + 1).wrapping_mul(pow)).wrapping_mul(R).wrapping_add(tokens[q - 1] as u64 + 1)
                                };
                                hl[q - p] = h;
                                fl.prefetch(h);
                            }
                        }
                        for (j, r) in rare.iter().enumerate() {
                            if q >= firsts[j] {
                                rh[j] = if q == firsts[j] {
                                    roll(&tokens[q - r.k..q])
                                } else {
                                    rh[j].wrapping_sub((tokens[q - 1 - r.k] as u64 + 1).wrapping_mul(r.pow)).wrapping_mul(R).wrapping_add(tokens[q - 1] as u64 + 1)
                                };
                                hr[j][q - p] = rh[j];
                                r.filter.prefetch(rh[j]);
                            }
                        }
                    }
                    for q in p..stop {
                        if k0 != usize::MAX && q >= first0 && filter.is_none_or(|fl| fl.test(hl[q - p])) {
                            cand.push((q, 0));
                        }
                        for (j, r) in rare.iter().enumerate() {
                            if q >= firsts[j] && r.filter.test(hr[j][q - p]) {
                                rcand.push((q, j, 0));
                            }
                        }
                    }
                    p = stop;
                    let last = p >= e0;
                    if rcand.len() >= BATCH || (last && !rcand.is_empty()) {
                        c.rare_passed += rcand.len() as u64;
                        for x in rcand.iter_mut() {
                            let r = &rare[x.1];
                            x.2 = hash(&tokens[x.0 - r.k..x.0]);
                            levels[r.i].table.prefetch(x.2);
                        }
                        let racc = rare_guard.as_mut().unwrap();
                        for &(q, j, hq) in rcand.iter() {
                            let r = &rare[j];
                            let level = &levels[r.i];
                            if let Some(id) = level.table.find(hq) {
                                if r.has(id) {
                                    racc[r.i][id >> level.shift].push((id as u64) << 16 | tokens[q] as u64);
                                    c.rare_found += 1;
                                }
                            }
                        }
                        rcand.clear();
                    }
                    if cand.len() >= BATCH || (last && !cand.is_empty()) {
                        probe(&mut cand, &mut next, tokens, levels, lv, acc, back, &mut c);
                    }
                }
                c
            })
        })
        .reduce(Counters::default, |a, b| a + b)
}

/// The long levels' probes of the candidate positions `cand` (local indices into `tokens`, whose index 0 has `back`
/// shard tokens before it), level by level of `lv` while they hit; `cand` is left empty.
#[allow(clippy::too_many_arguments)]
#[inline]
fn probe(cand: &mut Vec<(usize, u64)>, next: &mut Vec<(usize, u64)>, tokens: &[u16], levels: &[Level], lv: &[usize], acc: &mut Lists, back: usize, c: &mut Counters) {
    c.passed += cand.len() as u64;
    for &i in lv {
        if cand.is_empty() {
            break;
        }
        let level = &levels[i];
        let k = level.k;
        next.clear();
        for x in cand.iter_mut() {
            x.1 = hash(&tokens[x.0 - k..x.0]);
            level.table.prefetch(x.1);
        }
        c.probed += cand.len() as u64;
        let lists = &mut acc[i];
        for &(q, hq) in cand.iter() {
            if let Some(id) = level.table.find(hq) {
                lists[id >> level.shift].push((id as u64) << 16 | tokens[q] as u64);
                next.push((q, 0));
            }
        }
        // the next level needs k' tokens in the shard
        if let Some(&j) = lv.iter().find(|&&j| j > i) {
            let kk = levels[j].k;
            next.retain(|&(q, _)| q + back >= kk);
        }
        std::mem::swap(cand, next);
    }
    cand.clear();
}

/// One partition's results: per context id from `first`, its total, distinct and kept next tokens.
struct Part {
    first: usize,
    totals: Vec<u64>,
    distinct: Vec<u32>,
    kept: Vec<u32>,
    next: Vec<u16>,
    next_counts: Vec<u32>,
}

/// How a reduce writes its rows: the outputs ([n * nl], position-major), and per level the scale (f, growth; f = 0:
/// none).
struct Rows {
    nl: usize,
    tot: Output<u32>,
    dist: Output<u16>,
    cnt: Output<u32>,
}

/// Steps 3 and 6: per level of `todo` (index, scale f, growth) and partition, the threads' lists of `accs` sorted and
/// read: the Parts (in order) and the rows of the levels' validation positions. Contexts without occurrences write
/// nothing.
#[allow(clippy::too_many_arguments)]
fn reduce(
    accs: Vec<Mutex<Lists>>,
    lv: &[Level],
    todo: &[(usize, f64, f64)],
    groups: &[(Vec<u64>, Vec<u32>)],
    target: &[u16],
    top: usize,
    keep: usize,
    rows: &Rows,
) -> Vec<(usize, Vec<Part>)> {
    let mut lists: Vec<Vec<Vec<Vec<u64>>>> = lv.iter().map(|l| (0..l.parts).map(|_| Vec::new()).collect()).collect();
    for acc in accs {
        for (i, parts) in acc.into_inner().unwrap().into_iter().enumerate() {
            for (p, v) in parts.into_iter().enumerate() {
                if !v.is_empty() {
                    lists[i][p].push(v);
                }
            }
        }
    }
    let nl = rows.nl;
    let mut out = Vec::new();
    for &(i, f, growth) in todo {
        let l = &lv[i];
        let (first_pos, order) = &groups[i];
        let fg = if f > 0.0 { f.powf(growth) } else { 1.0 };
        let parts: Vec<Part> = std::mem::take(&mut lists[i])
            .into_par_iter()
            .enumerate()
            .map(|(p, v)| {
                let mut all: Vec<u64> = Vec::with_capacity(v.iter().map(Vec::len).sum());
                for x in v {
                    all.extend_from_slice(&x);
                }
                all.sort_unstable();
                let first = p << l.shift;
                let last = l.contexts.reps.len().min(first + (1 << l.shift));
                let m = last - first;
                let mut part = Part { first, totals: vec![0; m], distinct: vec![0; m], kept: vec![0; m], next: Vec::new(), next_counts: Vec::new() };
                let mut ranked: Vec<(std::cmp::Reverse<usize>, u16)> = Vec::new();
                let mut lookup: Vec<(u16, u32)> = Vec::new();
                for run in all.chunk_by(|a, b| a >> 16 == b >> 16) {
                    let gid = (run[0] >> 16) as usize;
                    let id = gid - first;
                    part.totals[id] = run.len() as u64;
                    ranked.clear();
                    ranked.extend(run.chunk_by(|a, b| a == b).map(|same| (std::cmp::Reverse(same.len()), (same[0] & 0xFFFF) as u16)));
                    ranked.sort_unstable();
                    part.distinct[id] = ranked.len() as u32;
                    part.kept[id] = ranked.len().min(keep) as u32;
                    for &(std::cmp::Reverse(c), token) in &ranked[..ranked.len().min(keep)] {
                        part.next.push(token);
                        part.next_counts.push(c as u32);
                    }
                    lookup.clear();
                    lookup.extend(ranked[..ranked.len().min(top)].iter().map(|&(std::cmp::Reverse(c), t)| (t, c as u32)));
                    lookup.sort_unstable();
                    let tt = u32::try_from(run.len()).unwrap_or(u32::MAX);
                    let dd = ranked.len().min(u16::MAX as usize) as u16;
                    let (tt, dd) = if f > 0.0 {
                        let t = tt as f64 * f;
                        ((t.round_ties_even()) as u32, ((dd as f64 * fg).round_ties_even().min(t.min(65535.0))) as u16)
                    } else {
                        (tt, dd)
                    };
                    for &q in &order[first_pos[gid] as usize..first_pos[gid + 1] as usize] {
                        let q = q as usize;
                        let c = lookup.binary_search_by_key(&target[q], |&(t, _)| t).map(|at| lookup[at].1).unwrap_or(0);
                        let c = if f > 0.0 { (c as f64 * f).round_ties_even() as u32 } else { c };
                        unsafe {
                            rows.tot.set(q * nl + i, tt);
                            rows.dist.set(q * nl + i, dd);
                            rows.cnt.set(q * nl + i, c);
                        }
                    }
                }
                part
            })
            .collect();
        out.push((i, parts));
    }
    out
}

/// A level's per-context arrays from its Parts (in order): totals (times `f` if > 0, rounded), distinct, offsets,
/// next tokens and their counts.
fn assemble(keys: usize, parts: &[Part], f: f64) -> (Vec<u64>, Vec<u32>, Vec<u64>, Vec<u16>, Vec<u32>) {
    let (mut totals, mut distinct, mut kept) = (vec![0u64; keys], vec![0u32; keys], vec![0u32; keys]);
    let (mut next, mut next_counts) = (Vec::new(), Vec::new());
    for part in parts {
        let m = part.totals.len();
        totals[part.first..part.first + m].copy_from_slice(&part.totals);
        distinct[part.first..part.first + m].copy_from_slice(&part.distinct);
        kept[part.first..part.first + m].copy_from_slice(&part.kept);
        next.extend_from_slice(&part.next);
        next_counts.extend_from_slice(&part.next_counts);
    }
    if f > 0.0 {
        totals.par_iter_mut().for_each(|t| *t = (*t as f64 * f).round_ties_even() as u64);
    }
    let offsets: Vec<u64> = std::iter::once(0)
        .chain(kept.iter().scan(0u64, |at, &c| {
            *at += c as u64;
            Some(*at)
        }))
        .collect();
    (totals, distinct, offsets, next, next_counts)
}

type LevelArrays = (Vec<u32>, Vec<u64>, Vec<u32>, Vec<u64>, Vec<u16>, Vec<u32>);

/// The validation count rows and per-level top next tokens of the validation `tokens` (inputs, then the last one's next
/// token) at `levels` (ascending) against the training shards `files`, the levels <= `short` over the first
/// `short_files` files only (scaled by `scale` f > 0 with `growth` per level, and with `rare_below` T > 0 the hybrid:
/// see the module's steps); `threads` threads; the long filter has 2^`filter_bits` bits, the rare ones about
/// `rare_bits` bits per rare context. Returns tot (u32), dist (u16) and cnt (u32), each [n * len(levels)]
/// (position-major); per level (context, total, distinct, offsets, next, next_counts) as kgram_counts's with the `keep`
/// most frequent next tokens; and the timings and counters.
#[pyfunction]
#[pyo3(signature = (files, tokens, levels, short, short_files, threads, top=1024, keep=8, filter_bits=28, scale=0.0, growth=Vec::new(), rare_below=0, rare_bits=16, anchor=0, filter2_bits=0))]
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn val_scan<'py>(
    py: Python<'py>,
    files: Vec<String>,
    tokens: PyReadonlyArray1<u16>,
    levels: Vec<usize>,
    short: usize,
    short_files: usize,
    threads: usize,
    top: usize,
    keep: usize,
    filter_bits: u32,
    scale: f64,
    growth: Vec<f64>,
    rare_below: u64,
    rare_bits: u32,
    anchor: usize,
    filter2_bits: u32,
) -> PyResult<(
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<u16>>,
    Bound<'py, PyArray1<u32>>,
    Vec<(
        Bound<'py, PyArray1<u32>>,
        Bound<'py, PyArray1<u64>>,
        Bound<'py, PyArray1<u32>>,
        Bound<'py, PyArray1<u64>>,
        Bound<'py, PyArray1<u16>>,
        Bound<'py, PyArray1<u32>>,
    )>,
    Vec<(String, f64)>,
)> {
    let tokens = tokens.as_slice()?;
    if files.is_empty() || tokens.len() < 2 || levels.is_empty() || levels[0] == 0 || !levels.windows(2).all(|w| w[0] < w[1]) {
        return Err(PyValueError::new_err("need files, 2+ tokens and ascending levels >= 1"));
    }
    if short_files > files.len() || top == 0 || keep == 0 || !(10..=40).contains(&filter_bits) || !(4..=40).contains(&rare_bits) {
        return Err(PyValueError::new_err("bad short_files, top, keep, filter_bits or rare_bits"));
    }
    let nshort = levels.iter().filter(|&&k| k <= short).count();
    if scale > 0.0 && growth.len() != nshort {
        return Err(PyValueError::new_err("scale needs a growth exponent per short level"));
    }
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let first_long = levels.iter().copied().find(|&k| k > short).unwrap_or(0);
    if anchor > 0 && (anchor > first_long || first_long == 0) {
        return Err(PyValueError::new_err("the anchor w-grams must fit the shallowest long level"));
    }
    if filter2_bits != 0 && !(10..=40).contains(&filter2_bits) {
        return Err(PyValueError::new_err("bad filter2_bits"));
    }
    let cfg = Config { short, short_files, top, keep, filter_bits, scale, growth, rare_below, rare_bits, anchor, filter2_bits };
    let out = py.detach(|| pool.install(|| run(&files, tokens, &levels, &cfg)))?;
    let (tot, dist, cnt, per_level, times) = out;
    let per_level = per_level
        .into_iter()
        .map(|(a, b, c, d, e, f)| (a.into_pyarray(py), b.into_pyarray(py), c.into_pyarray(py), d.into_pyarray(py), e.into_pyarray(py), f.into_pyarray(py)))
        .collect();
    Ok((tot.into_pyarray(py), dist.into_pyarray(py), cnt.into_pyarray(py), per_level, times))
}

struct Config {
    short: usize,
    short_files: usize,
    top: usize,
    keep: usize,
    filter_bits: u32,
    scale: f64,
    growth: Vec<f64>,
    rare_below: u64,
    rare_bits: u32,
    anchor: usize,
    filter2_bits: u32,
}

#[allow(clippy::type_complexity)]
fn run(files: &[String], tokens: &[u16], levels: &[usize], cfg: &Config) -> PyResult<(Vec<u32>, Vec<u16>, Vec<u32>, Vec<LevelArrays>, Vec<(String, f64)>)> {
    let mut times: Vec<(String, f64)> = Vec::new();
    let t0 = Instant::now();
    let corpus = Corpus::map(files, None)?; // its ranges only: the scans read the shards with pread
    let shards: Vec<std::fs::File> = files.iter().map(std::fs::File::open).collect::<std::io::Result<_>>()?;
    if levels.iter().any(|&k| k > BACK) {
        return Err(PyValueError::new_err(format!("levels up to {BACK}")));
    }
    times.push(("map_s".into(), t0.elapsed().as_secs_f64()));
    // 1. index
    let t = Instant::now();
    let n = tokens.len() - 1;
    let (x, target) = (&tokens[..n], &tokens[1..]);
    let avail = segments(x);
    let lv: Vec<Level> = levels
        .par_iter()
        .map(|&k| -> PyResult<Level> {
            let (contexts, table) = VContexts::new(x, &avail, k)?;
            let keys = contexts.reps.len().max(2);
            let bits = (keys - 1).ilog2() + 1;
            let want = if k <= 6 { 12 } else { 8 }; // ~4096 partitions for the busy levels, ~256 for the rare ones
            let shift = bits.saturating_sub(want);
            let parts = ((keys - 1) >> shift) + 1;
            Ok(Level { k, contexts, table, shift, parts })
        })
        .collect::<PyResult<_>>()?;
    times.push(("index_contexts_s".into(), t.elapsed().as_secs_f64()));
    let short_lv: Vec<usize> = (0..lv.len()).filter(|&i| lv[i].k <= cfg.short).collect();
    let long_lv: Vec<usize> = (0..lv.len()).filter(|&i| lv[i].k > cfg.short).collect();
    let filter = long_lv.first().map(|&i| {
        let k = lv[i].k;
        let reps = &lv[i].contexts.reps;
        if cfg.anchor > 0 && cfg.rare_below == 0 {
            // every w-gram of the shallowest long level's contexts: an occurrence contains one at an anchor
            let w = cfg.anchor;
            Filter::new(cfg.filter_bits, reps.par_iter().flat_map_iter(move |&r| {
                let ctx = &x[r as usize + 1 - k..=r as usize];
                (0..=k - w).map(move |o| roll(&ctx[o..o + w]))
            }))
        } else {
            Filter::new(cfg.filter_bits, reps.par_iter().map(|&r| roll(&x[r as usize + 1 - k..=r as usize])))
        }
    });
    // with anchors, a second filter of the shallowest long level's contexts themselves: an anchor hit's candidate
    // positions probe the tables only if their context passes it too
    let filter2 = match (long_lv.first(), cfg.anchor > 0 && cfg.rare_below == 0 && cfg.filter2_bits > 0) {
        (Some(&i), true) => {
            let k = lv[i].k;
            let reps = &lv[i].contexts.reps;
            Some(Filter::new(cfg.filter2_bits, reps.par_iter().map(|&r| roll(&x[r as usize + 1 - k..=r as usize]))))
        }
        _ => None,
    };
    times.push(("index_filters_s".into(), t.elapsed().as_secs_f64()));
    let groups: Vec<(Vec<u64>, Vec<u32>)> = lv.par_iter().map(|l| by_context(&l.contexts.key_of, l.contexts.reps.len())).collect();
    times.push(("index_s".into(), t.elapsed().as_secs_f64()));
    let nthreads = rayon::current_num_threads();
    let new_accs = || -> Vec<Mutex<Lists>> { (0..nthreads).map(|_| Mutex::new(lv.iter().map(|l| vec![Vec::new(); l.parts]).collect())).collect() };
    let nl = lv.len();
    let mut tot = vec![0u32; n * nl];
    let mut dist = vec![0u16; n * nl];
    let mut cnt = vec![0u32; n * nl];
    let rows = Rows { nl, tot: Output(tot.as_mut_ptr()), dist: Output(dist.as_mut_ptr()), cnt: Output(cnt.as_mut_ptr()) };
    // 2. short scan, 3. short reduce
    let t = Instant::now();
    let accs = new_accs();
    let cs = scan(&shards, &corpus.ranges[..cfg.short_files], &lv, &short_lv, None, &accs, &[], &[], 0, None);
    times.push(("scan_short_s".into(), t.elapsed().as_secs_f64()));
    let t = Instant::now();
    for &i in &short_lv {
        let occ: usize = accs.iter().map(|a| a.lock().unwrap()[i].iter().map(Vec::len).sum::<usize>()).sum();
        times.push((format!("occurrences_short_k{}", lv[i].k), occ as f64));
    }
    let todo: Vec<(usize, f64, f64)> = short_lv.iter().enumerate().map(|(j, &i)| (i, cfg.scale, cfg.growth.get(j).copied().unwrap_or(0.0))).collect();
    let short_parts = reduce(accs, &lv, &todo, &groups, target, cfg.top, cfg.keep, &rows);
    times.push(("reduce_short_s".into(), t.elapsed().as_secs_f64()));
    // 4. the rare contexts
    let t = Instant::now();
    let mut rare: Vec<Rare> = Vec::new();
    if cfg.rare_below > 0 {
        for (i, parts) in &short_parts {
            let l = &lv[*i];
            let keys = l.contexts.reps.len();
            let mut ids = vec![0u64; keys.div_ceil(64)];
            let mut count = 0usize;
            for part in parts {
                for (j, &total) in part.totals.iter().enumerate() {
                    if total < cfg.rare_below {
                        let id = part.first + j;
                        ids[id / 64] |= 1 << (id % 64);
                        count += 1;
                    }
                }
            }
            let k = l.k;
            let bits = ((count.max(1) as f64 * cfg.rare_bits as f64).log2().ceil() as u32).clamp(16, 32);
            let reps = &l.contexts.reps;
            let ids_ref = &ids;
            let filter = Filter::new(
                bits,
                (0..keys).into_par_iter().filter(move |&id| ids_ref[id / 64] >> (id % 64) & 1 == 1).map(|id| roll(&x[reps[id] as usize + 1 - k..=reps[id] as usize])),
            );
            times.push((format!("rare_contexts_k{k}"), count as f64));
            times.push((format!("rare_filter_bits_k{k}"), bits as f64));
            let pow = (0..k - 1).fold(1u64, |p, _| p.wrapping_mul(R));
            rare.push(Rare { i: *i, k, pow, filter, ids });
        }
    }
    times.push(("rare_index_s".into(), t.elapsed().as_secs_f64()));
    // 5. long scan (and the rare occurrences), 6. their reduces
    let t = Instant::now();
    let accs = new_accs();
    let rare_accs = if rare.is_empty() { Vec::new() } else { new_accs() };
    let anchor = if rare.is_empty() { cfg.anchor } else { 0 };
    let cl = scan(&shards, &corpus.ranges, &lv, &long_lv, filter.as_ref(), &accs, &rare, &rare_accs, anchor, filter2.as_ref());
    times.push(("scan_long_s".into(), t.elapsed().as_secs_f64()));
    let t = Instant::now();
    for &i in &long_lv {
        let occ: usize = accs.iter().map(|a| a.lock().unwrap()[i].iter().map(Vec::len).sum::<usize>()).sum();
        times.push((format!("occurrences_long_k{}", lv[i].k), occ as f64));
    }
    let todo: Vec<(usize, f64, f64)> = long_lv.iter().map(|&i| (i, 0.0, 0.0)).collect();
    let long_parts = reduce(accs, &lv, &todo, &groups, target, cfg.top, cfg.keep, &rows);
    let rare_parts = if rare.is_empty() {
        Vec::new()
    } else {
        let todo: Vec<(usize, f64, f64)> = rare.iter().map(|r| (r.i, 0.0, 0.0)).collect();
        reduce(rare_accs, &lv, &todo, &groups, target, cfg.top, cfg.keep, &rows)
    };
    times.push(("reduce_long_s".into(), t.elapsed().as_secs_f64()));
    let positions: usize = corpus.ranges.iter().map(|&(lo, hi)| hi - lo).sum();
    for (name, v) in [
        ("corpus_positions", positions as u64),
        ("short_probes", cs.probed),
        ("long_anchors", cl.anchors),
        ("long_filter_passed", cl.passed),
        ("long_probes", cl.probed),
        ("rare_filter_passed", cl.rare_passed),
        ("rare_found", cl.rare_found),
    ] {
        times.push((name.into(), v as f64));
    }
    // the per-level arrays: the short levels' scaled, with the rare contexts' from the rare reduce
    let t = Instant::now();
    let mut per_level: Vec<Option<LevelArrays>> = (0..nl).map(|_| None).collect();
    let built: Vec<(usize, LevelArrays)> = short_parts
        .par_iter()
        .chain(long_parts.par_iter())
        .map(|(i, parts)| {
            let l = &lv[*i];
            let f = if l.k <= cfg.short { cfg.scale } else { 0.0 };
            let (totals, distinct, offsets, next, next_counts) = assemble(l.contexts.reps.len(), parts, f);
            (*i, (l.contexts.key_of.clone(), totals, distinct, offsets, next, next_counts))
        })
        .collect();
    for (i, arrays) in built {
        per_level[i] = Some(arrays);
    }
    for (r, (i, parts)) in rare.iter().zip(rare_parts.iter()) {
        let (rt, rd, ro, rn, rc) = assemble(lv[*i].contexts.reps.len(), parts, 0.0);
        let (key_of, totals, distinct, offsets, next, next_counts) = per_level[*i].take().unwrap();
        let keys = totals.len();
        let (mut t2, mut d2, mut o2, mut n2, mut c2) = (vec![0u64; keys], vec![0u32; keys], vec![0u64; keys + 1], Vec::new(), Vec::new());
        for id in 0..keys {
            let (src_t, src_d, src_o, src_n, src_c) = if r.has(id) { (&rt, &rd, &ro, &rn, &rc) } else { (&totals, &distinct, &offsets, &next, &next_counts) };
            t2[id] = src_t[id];
            d2[id] = src_d[id];
            let (a, b) = (src_o[id] as usize, src_o[id + 1] as usize);
            n2.extend_from_slice(&src_n[a..b]);
            c2.extend_from_slice(&src_c[a..b]);
            o2[id + 1] = n2.len() as u64;
        }
        per_level[*i] = Some((key_of, t2, d2, o2, n2, c2));
    }
    times.push(("assemble_s".into(), t.elapsed().as_secs_f64()));
    times.push(("total_s".into(), t0.elapsed().as_secs_f64()));
    Ok((tot, dist, cnt, per_level.into_iter().map(Option::unwrap).collect(), times))
}

/// The validation hint rows (old_records.ValHintRows, one slot) from val_scan's per-level arrays of the variant's
/// levels (`ks`, ascending; per level (context, total, offsets, next, next_counts)): per position the deepest level
/// whose context occurred, its cell with that level's index in `buckets` as the length bucket (val_hints.deepest),
/// and its context's top `entries` next tokens with their weights (counts over their sum, in units of 1 / 65535,
/// rounded half to even: old_records._weights). Returns cell (i16 [n]), tokens and weights (u16 [n * entries]).
#[pyfunction]
#[allow(clippy::type_complexity)]
pub fn val_hint_rows<'py>(
    py: Python<'py>,
    ks: Vec<usize>,
    levels: Vec<(PyReadonlyArray1<'py, u32>, PyReadonlyArray1<'py, u64>, PyReadonlyArray1<'py, u64>, PyReadonlyArray1<'py, u16>, PyReadonlyArray1<'py, u32>)>,
    buckets: Vec<usize>,
    entries: usize,
    threads: usize,
) -> PyResult<(Bound<'py, PyArray1<i16>>, Bound<'py, PyArray1<u16>>, Bound<'py, PyArray1<u16>>)> {
    if ks.len() != levels.len() || ks.is_empty() || entries == 0 || ks.iter().any(|k| !buckets.contains(k)) {
        return Err(PyValueError::new_err("need one array set per level, levels of buckets, entries > 0"));
    }
    let views: Vec<(&[u32], &[u64], &[u64], &[u16], &[u32])> = levels
        .iter()
        .map(|(a, b, c, d, e)| Ok((a.as_slice()?, b.as_slice()?, c.as_slice()?, d.as_slice()?, e.as_slice()?)))
        .collect::<PyResult<_>>()?;
    let n = views[0].0.len();
    if views.iter().any(|v| v.0.len() != n) {
        return Err(PyValueError::new_err("the levels' context arrays differ in length"));
    }
    let bucket: Vec<usize> = ks.iter().map(|k| buckets.iter().position(|b| b == k).unwrap()).collect();
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut cell = vec![0i16; n];
    let mut tokens = vec![0u16; n * entries];
    let mut weights = vec![0u16; n * entries];
    py.detach(|| {
        pool.install(|| {
            cell.par_chunks_mut(1 << 14)
                .zip(tokens.par_chunks_mut(entries << 14))
                .zip(weights.par_chunks_mut(entries << 14))
                .enumerate()
                .for_each(|(c, ((cell, tokens), weights))| {
                    for (i, out) in cell.iter_mut().enumerate() {
                        let q = (c << 14) + i;
                        for (j, &(context, totals, offsets, next, next_counts)) in views.iter().enumerate().rev() {
                            let id = context[q];
                            if id == u32::MAX || totals[id as usize] == 0 {
                                continue;
                            }
                            let total = totals[id as usize];
                            let (lo, hi) = (offsets[id as usize] as usize, offsets[id as usize + 1] as usize);
                            let m = (hi - lo).min(entries);
                            let top = next_counts[lo] as f64;
                            let count = crate::COUNTS.iter().filter(|&&e| e as u64 <= total).count();
                            let purity = crate::PURITIES.iter().filter(|&&p| p <= top / total as f64).count();
                            *out = (((1 + bucket[j]) * (crate::COUNTS.len() + 1) + count) * (crate::PURITIES.len() + 1) + purity) as i16;
                            let sum: f64 = next_counts[lo..lo + m].iter().map(|&v| v as f64).sum();
                            for e in 0..m {
                                tokens[i * entries + e] = next[lo + e];
                                weights[i * entries + e] = (next_counts[lo + e] as f64 / sum.max(1.0) * 65535.0).round_ties_even() as u16;
                            }
                            break;
                        }
                    }
                });
        })
    });
    Ok((cell.into_pyarray(py), tokens.into_pyarray(py), weights.into_pyarray(py)))
}
