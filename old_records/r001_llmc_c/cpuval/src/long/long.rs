//! The long levels (8, 10, 12, 16, 20 over all the shards): the validation side, after the clock. The rows
//! equal val_build's for those levels (count rows; hint rows at 20, else 10, where the context occurred).
//!
//! `lookup` (after the clock, no allocation in it):
//!   1. tables: per level one table of the validation contexts (fingerprint << 24 | position: one CAS per position
//!      dedups, checks hash collisions token by token against the slot's position, and chains the positions of a
//!      context through `next`, newest in the slot); ids are slot numbers. The 8-contexts' filter.
//!   2. candidates (index.rs): the corpus positions of the validation 8-contexts from the anchor index, probed level
//!      by level (or through the direct path), each occurrence pushed into its thread's lists;
//!   3. reduce: per level, deepest first, per partition the occurrences counted, per context its rows written at the
//!      positions of its chain (val_build's reduce_level).
use super::index::AIndex;
use super::util::{fkey, hash, huge_zeroed, Filter, Lists, Output, Pool, COUNTS, FILTER_BITS, FP, KEEP, PARTS_LOG2, PURITIES, TABLE_LOAD, TOP};
use rayon::prelude::*;
use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
use std::cmp::Reverse;
use std::fs::File;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

pub const LONG: [usize; 5] = [8, 10, 12, 16, 20];
const NIL: u32 = u32::MAX;
/// the lists' arena chunks faulted before the clock (~1.1B occurrences of 8 bytes at full scale, + a chunk per thread)
const ARENA_BYTES: usize = 24 << 30;

/// Where the rows go: `nl` columns per position, level i's at `cols[i]`, its hint bucket `buckets[i]`.
pub struct RowsOut {
    pub nl: usize,
    pub cols: [usize; 5],
    pub buckets: [Option<usize>; 5],
    pub tot: *mut u32,
    pub dist: *mut u16,
    pub cnt: *mut u32,
    pub cell: *mut i16,
    pub tokens: *mut u16,
    pub weights: *mut u16,
    /// per position: 1 once its hint row is written
    pub done: *mut u8,
}
unsafe impl Send for RowsOut {}
unsafe impl Sync for RowsOut {}

/// One lookup thread's lists and buffers (allocated before the clock, reused).
pub(crate) struct Work {
    pub(crate) lists: Lists,
    pub(crate) pend: Vec<(u32, usize, u8)>, // (shard, anchor offset, mask): tokens prefetched
    pub(crate) got: Vec<(u32, usize, u8)>,
    pub(crate) c2: Vec<(u32, u32, u64)>, // filter words prefetched
    pub(crate) cand: Vec<(u32, u32)>,
    pub(crate) hs: Vec<u64>,
    pub(crate) hits: Vec<(u32, u32, u32, u32)>,
    pub(crate) fallback: Vec<(u32, u32)>,
    pub(crate) next: Vec<(u32, u32)>,
    /// the direct path's matches: (shard, position, length, validation position)
    pub(crate) deep: Vec<(u32, u32, u32, u32)>,
    red: Red,
}

/// The reduce's per-thread buffers.
#[derive(Default)]
struct Red {
    first: Vec<u32>,
    tok: Vec<u16>,
    pairs: Vec<(u64, u32)>,
    hist: Vec<u32>,
    ranked: Vec<(Reverse<usize>, u16)>,
    by_token: Vec<(u16, u32)>,
}

impl Work {
    fn new(parts: &[usize], pool: Arc<Pool>, nslots: usize) -> Self {
        let mut red = Red::default();
        red.first.reserve(nslots + 1);
        red.tok.reserve(1 << 20);
        red.pairs.reserve(1 << 19);
        super::util::interleave(&mut red.tok);
        super::util::interleave(&mut red.pairs);
        red.hist.resize(1 << 16, 0);
        red.ranked.reserve(1 << 16);
        red.by_token.reserve(1 << 16);
        // the reserved buffers faulted now
        red.tok.resize(1 << 20, 0);
        red.tok.clear();
        red.pairs.resize(1 << 19, (0, 0));
        red.pairs.clear();
        let c = 1024;
        Self {
            lists: Lists::new(parts, pool),
            pend: Vec::with_capacity(c),
            got: Vec::with_capacity(c),
            c2: Vec::with_capacity(4 * c),
            cand: Vec::with_capacity(c),
            hs: Vec::with_capacity(c),
            hits: Vec::with_capacity(c),
            fallback: Vec::with_capacity(c),
            next: Vec::with_capacity(c),
            deep: Vec::with_capacity(c),
            red,
        }
    }
}

/// A level's validation contexts: slots of fingerprint << 24 | a position, homes at the hash's top bits, linear
/// probing; `next` chains a context's positions (NIL ends).
pub(crate) struct VLevel {
    pub(crate) k: usize,
    bits: u32,
    pub(crate) slots: Vec<u64>,
    pub(crate) next: Vec<u32>,
    /// per position its slot (NIL: no valid context)
    pub(crate) keys: Vec<u32>,
    pub(crate) shift: u32,
    pub(crate) parts: usize,
}

impl VLevel {
    fn alloc(k: usize, n: usize) -> Self {
        let bits = ((n as f64 / TABLE_LOAD).ceil() as usize).next_power_of_two().max(1024).trailing_zeros();
        let shift = bits.saturating_sub(PARTS_LOG2);
        Self { k, bits, slots: huge_zeroed(1 << bits), next: huge_zeroed(n), keys: huge_zeroed(n), shift, parts: 1 << (bits - shift) }
    }

    /// Inserts every valid position of `x` (one CAS each).
    fn fill(&mut self, x: &[u16], avail: &[u32]) -> Result<(), String> {
        let (k, bits) = (self.k, self.bits);
        let mask = (1usize << bits) - 1;
        let atomics: &[AtomicU64] = unsafe { &*(self.slots.as_slice() as *const [u64] as *const [AtomicU64]) };
        let next = Output(self.next.as_mut_ptr());
        let keys = Output(self.keys.as_mut_ptr());
        let n = x.len();
        let slots = self.slots.as_ptr() as usize;
        (0..n.div_ceil(256)).into_par_iter().try_for_each(|c| {
            // 256 positions: hashes computed and their homes prefetched, then the inserts
            let mut hs = [0u64; 256];
            let (a, b) = (c * 256, n.min(c * 256 + 256));
            for i in a..b {
                if avail[i] as usize >= k {
                    let h = hash(&x[i + 1 - k..=i]);
                    unsafe { _mm_prefetch((slots as *const u64).wrapping_add((h >> (64 - bits)) as usize).cast(), _MM_HINT_T0) };
                    hs[i - a] = h;
                }
            }
            for i in a..b {
                let h = hs[i - a];
                if h == 0 {
                    unsafe { keys.set(i, NIL) };
                    continue;
                }
                let ctx = &x[i + 1 - k..=i];
                let fp = h & FP;
                let v = fp << 24 | i as u64;
                let mut s = (h >> (64 - bits)) as usize;
                let mut cur = atomics[s].load(Ordering::Relaxed);
                loop {
                    if cur == 0 {
                        unsafe { next.set(i, NIL) };
                        match atomics[s].compare_exchange(0, v, Ordering::AcqRel, Ordering::Relaxed) {
                            Ok(_) => {
                                unsafe { keys.set(i, s as u32) };
                                break;
                            }
                            Err(c) => {
                                cur = c;
                                continue;
                            }
                        }
                    }
                    if cur >> 24 == fp {
                        let r = (cur & 0xFF_FFFF) as usize;
                        if x[r + 1 - k..=r] != *ctx {
                            return Err(format!("hash collision between validation {k}-contexts at {r} and {i}"));
                        }
                        unsafe { next.set(i, r as u32) };
                        match atomics[s].compare_exchange(cur, v, Ordering::AcqRel, Ordering::Relaxed) {
                            Ok(_) => {
                                unsafe { keys.set(i, s as u32) };
                                break;
                            }
                            Err(c) => {
                                cur = c;
                                continue;
                            }
                        }
                    }
                    s = (s + 1) & mask;
                    cur = atomics[s].load(Ordering::Relaxed);
                }
            }
            Ok(())
        })
    }

    #[inline]
    pub(crate) fn prefetch(&self, h: u64) {
        unsafe { _mm_prefetch(self.slots.as_ptr().wrapping_add((h >> (64 - self.bits)) as usize).cast(), _MM_HINT_T0) };
    }

    /// The slot of the context with hash `h`, if a validation context.
    #[inline]
    pub(crate) fn find(&self, h: u64) -> Option<usize> {
        let mask = self.slots.len() - 1;
        let fp = h & FP;
        let mut s = (h >> (64 - self.bits)) as usize;
        loop {
            let v = unsafe { *self.slots.get_unchecked(s) };
            if v == 0 {
                return None;
            }
            if v >> 24 == fp {
                return Some(s);
            }
            s = (s + 1) & mask;
        }
    }
}

/// The direct path's view of the validation side.
pub(crate) struct Ext<'a> {
    pub x: &'a [u16],
    /// per position how far its contexts reach (its segment's tokens, at most 20)
    pub cap: &'a [u8],
    pub next: &'a [u32],
    pub rec: &'a [[u32; 4]],
    /// a bit per level-8 slot whose chain is longer than EXT_LIM, and those slots' (slot, offset << 32 | length of
    /// its positions in xarr), sorted
    pub xbit: &'a [u64],
    pub xo: &'a [(u32, u64)],
    /// those chains' positions as (the two tokens before the 8-context, position), sorted per chain
    pub xarr: &'a [(u32, u32)],
}

/// The two tokens before position i's 8-context as a key (u32::MAX: its 10-context is invalid).
#[inline]
pub(crate) fn ext_key(x: &[u16], cap: &[u8], i: usize) -> u32 {
    if cap[i] as usize > LONG[0] + 1 {
        (x[i - 9] as u32) << 16 | x[i - 8] as u32
    } else {
        u32::MAX
    }
}


/// The engine: buffers allocated (and faulted) before the clock for up to `n_max` validation positions, and the
/// anchor index.
pub struct Long {
    n_max: usize,
    lv: Vec<VLevel>,
    pub(crate) index: AIndex,
    /// per position the slots of its 10-, 12-, 16- and 20-contexts (the probes' direct path)
    rec: Vec<[u32; 4]>,
    cap: Vec<u8>,
    xbit: Vec<u64>,
    /// the 8-contexts' filter's words
    f2w: Option<Vec<u64>>,
    pool: Arc<Pool>,
    /// per lookup thread (by index in its pool)
    work: Vec<Mutex<Work>>,
    used: bool,
}

impl Long {
    /// Before the clock: the shards mapped, the index and the validation side's buffers allocated and faulted.
    pub fn new(files: &[File], sizes: &[usize], w: usize, n_max: usize) -> Self {
        assert!(n_max < 1 << 24, "positions must fit 24 bits");
        let lv: Vec<VLevel> = LONG.par_iter().map(|&k| VLevel::alloc(k, n_max)).collect();
        let xbit = huge_zeroed(lv[0].slots.len().div_ceil(64));
        let pool = Arc::new(Pool::new(ARENA_BYTES));
        let parts: Vec<usize> = lv.iter().map(|l| l.parts).collect();
        let nslots = lv.iter().map(|l| 1usize << l.shift).max().unwrap();
        // one per CPU of the machine (the lookup pool's threads are at most that many)
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).max(unsafe { libc::sysconf(libc::_SC_NPROCESSORS_ONLN) } as usize);
        let work = (0..threads).into_par_iter().map(|_| Mutex::new(Work::new(&parts, pool.clone(), nslots))).collect();
        let index = AIndex::alloc(files, sizes, w, n_max);
        Self { n_max, xbit, lv, index, rec: huge_zeroed(n_max), cap: huge_zeroed(n_max), f2w: Some(huge_zeroed(1 << (FILTER_BITS - 6))), pool, work, used: false }
    }

    /// The memory in bytes: index entries, index directories, build scratch, validation side (tables, lists' arena,
    /// per-thread buffers).
    pub fn bytes(&self) -> [usize; 4] {
        let (e, d, s) = self.index.bytes();
        let tables: usize = self.lv.iter().map(|l| l.slots.len() * 8 + l.next.len() * 4 + l.keys.len() * 4).sum();
        let val = tables + self.rec.len() * 16 + self.cap.len() + self.xbit.len() * 8 + (1 << (FILTER_BITS - 3)) + self.pool.chunks() * super::util::CHUNK * 8;
        [e, d, s, val]
    }

    /// Ready for another lookup (benchmarks): the tables, the filter and the lists emptied.
    fn reset(&mut self) {
        self.lv.par_iter_mut().for_each(|l| l.slots.par_chunks_mut(1 << 16).for_each(|c| c.fill(0)));
        if let Some(w) = self.f2w.as_mut() {
            w.par_chunks_mut(1 << 16).for_each(|c| c.fill(0));
        }
        self.index.reset();
        self.xbit.par_chunks_mut(1 << 16).for_each(|c| c.fill(0));
        for w in self.work.iter_mut() {
            w.get_mut().unwrap().lists.reset();
        }
    }

    /// The rows of the validation positions (tokens `x`, next tokens `target`, segments `avail`), in the current rayon
    /// pool. Returns the timings.
    pub fn lookup(&mut self, x: &[u16], target: &[u16], avail: &[u32], out: &RowsOut) -> Result<Vec<(String, f64)>, String> {
        let mut times: Vec<(String, f64)> = Vec::new();
        let t0 = Instant::now();
        let n = x.len();
        if n > self.n_max {
            return Err(format!("{n} validation positions, at most {}", self.n_max));
        }
        if rayon::current_num_threads() > self.work.len() {
            return Err(format!("{} lookup threads, at most {}", rayon::current_num_threads(), self.work.len()));
        }
        if self.used {
            self.reset();
        }
        self.used = true;
        self.lv.par_iter_mut().try_for_each(|l| l.fill(x, avail))?;
        let lv = &self.lv;
        let kmax = LONG[LONG.len() - 1];
        self.rec[..n].par_iter_mut().enumerate().for_each(|(i, r)| *r = [lv[1].keys[i], lv[2].keys[i], lv[3].keys[i], lv[4].keys[i]]);
        self.cap[..n].par_iter_mut().enumerate().for_each(|(i, c)| *c = (avail[i] as usize).min(kmax) as u8);
        times.push(("tables_s".into(), t0.elapsed().as_secs_f64()));
        let k0 = LONG[0];
        let words = self.f2w.take().ok_or("the filter's words are gone")?;
        let f2 = Filter::with_words(words, (0..n).into_par_iter().filter(|&i| avail[i] as usize >= k0).map(|i| fkey(&x[i + 1 - k0..=i])));
        times.push(("index_s".into(), t0.elapsed().as_secs_f64()));
        // the level-8 contexts with more than EXT_LIM positions: their positions sorted by the two tokens before
        let t = Instant::now();
        let (cap, next) = (&self.cap[..n], &self.lv[0].next[..n]);
        let longs: Vec<(usize, Vec<(u32, u32)>)> = self.lv[0]
            .slots
            .par_iter()
            .enumerate()
            .with_min_len(1 << 12)
            .filter_map(|(sl, &v)| {
                if v == 0 {
                    return None;
                }
                let head = (v & 0xFF_FFFF) as u32;
                let (mut m, mut q) = (0, head);
                while q != NIL && m <= super::index::EXT_LIM {
                    m += 1;
                    q = next[q as usize];
                }
                if m <= super::index::EXT_LIM {
                    return None;
                }
                let mut a = Vec::new();
                let mut q = head;
                while q != NIL {
                    a.push((ext_key(x, cap, q as usize), q));
                    q = next[q as usize];
                }
                a.sort_unstable();
                Some((sl, a))
            })
            .collect();
        let mut xarr: Vec<(u32, u32)> = Vec::with_capacity(longs.iter().map(|l| l.1.len()).sum());
        let mut xo: Vec<(u32, u64)> = Vec::with_capacity(longs.len());
        for (sl, a) in &longs {
            self.xbit[sl / 64] |= 1 << (sl % 64);
            xo.push((*sl as u32, (xarr.len() as u64) << 32 | a.len() as u64));
            xarr.extend_from_slice(a);
        }
        drop(longs);
        times.push(("xmap_s".into(), t.elapsed().as_secs_f64()));
        let t = Instant::now();
        let ext = Ext { x, cap, next, rec: &self.rec[..n], xbit: &self.xbit, xo: &xo, xarr: &xarr };
        let mut tt = self.index.candidates(x, avail, &self.lv, &f2, &self.work, &ext)?;
        times.append(&mut tt);
        times.push(("cand_all_s".into(), t.elapsed().as_secs_f64()));
        // reduce, deepest first
        let t = Instant::now();
        let mut lists: Vec<&Lists> = Vec::with_capacity(self.work.len());
        let mut reds: Vec<Mutex<&mut Red>> = Vec::with_capacity(self.work.len());
        for w in self.work.iter_mut() {
            let Work { lists: l, red, .. } = w.get_mut().unwrap();
            lists.push(l);
            reds.push(Mutex::new(red));
        }
        for i in (0..LONG.len()).rev() {
            let tl = Instant::now();
            reduce_level(i, &lists, &reds, &self.lv[i], target, out);
            times.push((format!("reduce_k{}_s", self.lv[i].k), tl.elapsed().as_secs_f64()));
        }
        times.push(("reduce_s".into(), t.elapsed().as_secs_f64()));
        times.push(("arena_fresh_chunks".into(), self.pool.fresh.load(Ordering::Relaxed) as f64));
        times.push(("total_s".into(), t0.elapsed().as_secs_f64()));
        drop(reds);
        self.f2w = Some(f2.words);
        Ok(times)
    }
}

/// A partition's distinct (slot << 16 | token) values with their counts, in order, into `r.pairs`: the occurrences
/// (the threads' blocks of level i, partition p) counting-sorted by slot (the partition's `nslots` from `base`), then
/// each slot's tokens sorted (or counted in a histogram if many).
fn slot_count(lists: &[&Lists], i: usize, p: usize, base: usize, nslots: usize, r: &mut Red) -> usize {
    let first = &mut r.first;
    first.clear();
    first.resize(nslots + 1, 0);
    let mut nocc = 0usize;
    for l in lists {
        l.for_blocks(i, p, |b| {
            nocc += b.len();
            for &x in b {
                first[(x >> 16) as usize - base + 1] += 1;
            }
        });
    }
    r.pairs.clear();
    if nocc == 0 {
        return 0;
    }
    for j in 0..nslots {
        first[j + 1] += first[j];
    }
    let tok = &mut r.tok;
    tok.clear();
    tok.resize(nocc, 0);
    // first[j] doubles as slot j's cursor, then is restored
    for l in lists {
        l.for_blocks(i, p, |b| {
            for &x in b {
                let j = (x >> 16) as usize - base;
                tok[first[j] as usize] = x as u16;
                first[j] += 1;
            }
        });
    }
    for j in (1..=nslots).rev() {
        first[j] = first[j - 1];
    }
    first[0] = 0;
    for j in 0..nslots {
        let (a, b) = (first[j] as usize, first[j + 1] as usize);
        if a == b {
            continue;
        }
        let key = ((base + j) as u64) << 16;
        let t = &mut tok[a..b];
        if t.len() > 1 << 15 {
            // a token histogram
            let hist = &mut r.hist;
            for &x in t.iter() {
                hist[x as usize] += 1;
            }
            for x in 0..1usize << 16 {
                if hist[x] > 0 {
                    r.pairs.push((key | x as u64, hist[x]));
                    hist[x] = 0;
                }
            }
        } else {
            t.sort_unstable();
            r.pairs.extend(t.chunk_by(|a, b| a == b).map(|same| (key | same[0] as u64, same.len() as u32)));
        }
    }
    nocc
}

/// Level i's rows from its partitions: val_build's reduce_level, the positions from the slot's chain.
fn reduce_level(i: usize, lists: &[&Lists], reds: &[Mutex<&mut Red>], l: &VLevel, target: &[u16], out: &RowsOut) {
    let nl = out.nl;
    let col = out.cols[i];
    let bucket = out.buckets[i];
    let (tot_o, dist_o, cnt_o) = (Output(out.tot), Output(out.dist), Output(out.cnt));
    let (cell_o, tok_o, w_o) = (Output(out.cell), Output(out.tokens), Output(out.weights));
    let done = Output(out.done);
    (0..l.parts).into_par_iter().for_each(|p| {
        let th = rayon::current_thread_index().unwrap_or(0);
        let mut guard = reds[th].lock().unwrap();
        let r: &mut Red = &mut guard;
        if slot_count(lists, i, p, p << l.shift, 1 << l.shift, r) == 0 {
            return;
        }
        let Red { pairs, ranked, by_token, .. } = r;
        for run in pairs.chunk_by(|a, b| a.0 >> 16 == b.0 >> 16) {

            let slot = (run[0].0 >> 16) as usize;
            let total: usize = run.iter().map(|x| x.1 as usize).sum();
            by_token.clear();
            by_token.extend(run.iter().map(|&(x, c)| ((x & 0xFFFF) as u16, c)));
            ranked.clear();
            ranked.extend(run.iter().map(|&(x, c)| (Reverse(c as usize), (x & 0xFFFF) as u16)));
            let d = ranked.len();
            let thr = if d > TOP {
                ranked.select_nth_unstable(TOP - 1);
                Some(ranked[TOP - 1])
            } else {
                None
            };
            let head = d.min(TOP);
            if head > KEEP {
                ranked[..head].select_nth_unstable(KEEP - 1);
            }
            ranked[..head.min(KEEP)].sort_unstable();
            let tt = u32::try_from(total).unwrap_or(u32::MAX);
            let dd = d.min(u16::MAX as usize) as u16;
            let hint = bucket.map(|b| {
                let top = ranked[0].0 .0 as f64;
                let count = COUNTS.iter().filter(|&&e| e <= total).count();
                let purity = PURITIES.iter().filter(|&&p| p <= top / total as f64).count();
                let cell = (((1 + b) * (COUNTS.len() + 1) + count) * (PURITIES.len() + 1) + purity) as i16;
                let m = d.min(KEEP);
                let wsum: f64 = ranked[..m].iter().map(|&(Reverse(c), _)| c as f64).sum();
                (cell, m, wsum)
            });
            let mut q = (l.slots[slot] & 0xFF_FFFF) as u32;
            while q != NIL {
                let qi = q as usize;
                let t = target[qi];
                let c = match by_token.binary_search_by_key(&t, |&(x, _)| x) {
                    Ok(at) if thr.is_none_or(|th| (Reverse(by_token[at].1 as usize), t) <= th) => by_token[at].1,
                    _ => 0,
                };
                unsafe {
                    tot_o.set(qi * nl + col, tt);
                    dist_o.set(qi * nl + col, dd);
                    cnt_o.set(qi * nl + col, c);
                    if let Some((cell, m, wsum)) = hint {
                        if *done.0.add(qi) == 0 {
                            done.set(qi, 1);
                            cell_o.set(qi, cell);
                            for (e, &(Reverse(cc), token)) in ranked[..m].iter().enumerate() {
                                tok_o.set(qi * KEEP + e, token);
                                w_o.set(qi * KEEP + e, (cc as f64 / wsum.max(1.0) * 65535.0).round_ties_even() as u16);
                            }
                        }
                    }
                }
                q = l.next[qi];
            }
        }
    });
}

