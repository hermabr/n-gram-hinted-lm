//! Exact, causal next-token hints of every training position, the training rows of the experiment that adds
//! exact-match retrieval hints to old speedrun records (retrieval_blog_post/old_records.py's TrainHintRows). The
//! validation rows come from kgram.rs's counts (val_hints.py); these use the same levels, cells and definitions, with
//! causality added.
//!
//! The stream is the shards in order, a position's global index its offset plus the tokens of the shards before it. A
//! segment starts at offset 0 of each shard and at every BOS (kgram.rs's segments). A position's k-context is the k
//! tokens ending at it, valid if they lie in its segment. An occurrence is any k tokens within one shard and the token
//! after them (it may cross a BOS, as kgram.rs's corpus side), and counts for a position iff it ends, next token
//! included, before the position's segment starts: never from the position's own document or anything later. A
//! position's hint has `slots` slots, its deepest levels with a counted occurrence, deepest first. Per level k of the
//! variant, deepest first:
//!   1. key each position by its k-context: its tokens packed into a u128 for k <= 8 (exact), else a 128-bit hash, and
//!      every group that is used is then compared token by token (a collision is an error: the counts stay exact);
//!   2. partition the positions that are an occurrence, or a valid query with a slot still empty, into buckets by key,
//!      in stream order, and sort each bucket by key and position;
//!   3. walk each key group in stream order with one pointer over its occurrences: before answering a query, count
//!      every occurrence that ends before the query's segment starts (segment starts only grow along the stream).
//!      Counts per next token are dense, and the top E by (count descending, token ascending) are kept exact as
//!      counts only grow by one: a token outside the top E enters only by passing the E-th;
//!   4. a query with a counted occurrence gets its first empty slot's row at this level, and once its slots are all
//!      filled, shallower levels skip it.
use crate::{open_shard, Map, Output, BOS, COUNTS, PURITIES};
use numpy::ndarray::{Array2, Array3};
use numpy::IntoPyArray;
use pyo3::{exceptions::PyValueError, prelude::*};
use rayon::prelude::*;
use std::io::Read;
use std::time::Instant;

/// val_hints.py's LEVELS, the default buckets: the levels whose index in them is a cell's length bucket.
const LEVELS: [usize; 7] = [2, 3, 4, 5, 6, 10, 20];
const WEIGHT_ONE: f64 = 65535.0; // hint_runtime.py's
pub(crate) const PACKED: usize = 8; // the longest context whose tokens pack into a key
pub(crate) const BUCKET: usize = 1 << 16; // the positions per bucket aimed at
pub(crate) const NONE: u32 = u32::MAX;
const P: u64 = 0x9e3779b185ebca87;

/// A position of one level: its context's key, its global index, its segment's global start if it is a valid query
/// with a slot still empty (else NONE), its next token if it ends an occurrence (else NONE), and its first empty slot
/// (still 32 bytes: the key's alignment pads the rest).
#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct Elem {
    pub(crate) key: u128,
    pub(crate) at: u32,
    pub(crate) start: u32,
    pub(crate) next: u32,
    pub(crate) slot: u32,
}

/// Per position of the shards `files` (in order), its hint over the variant `levels` (ascending, of `buckets`, by
/// default LEVELS), in `slots` slots: the deepest levels of `levels` whose context occurred before its segment, deepest
/// first. Per slot its cell (int16; 0 for an empty slot: val_hints.py's, with the level's index in `buckets` as the
/// length bucket), the `entries` most frequent next tokens at its level (uint16, count descending, then token
/// ascending, padded with 0), their counts as weights summing to about 65535 (uint16, round(count / their sum *
/// 65535), 0 for padding), and its level's index in `buckets` (int8; -1 for an empty slot). Cell and level are [N]
/// and tokens and weights [N, entries] with one slot, else [N, slots] and [N, slots, entries]. About
/// 39 + slots * (3 + 4 * entries) bytes per position. Exact for contexts of up to 8 tokens; longer ones are keyed by a
/// 128-bit hash, and a collision within a group that is used is an error.
#[pyfunction]
#[pyo3(signature = (files, levels, entries, buckets=None, slots=1))]
#[allow(clippy::type_complexity)]
pub fn train_hints<'py>(
    py: Python<'py>,
    files: Vec<String>,
    levels: Vec<usize>,
    entries: usize,
    buckets: Option<Vec<usize>>,
    slots: usize,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    if files.is_empty() || entries == 0 || slots == 0 || slots > u8::MAX as usize {
        return Err(PyValueError::new_err(format!(
            "need files, entries >= 1 and 1 <= slots <= 255: {} files, {entries} entries, {slots} slots",
            files.len()
        )));
    }
    let buckets = buckets.unwrap_or(LEVELS.to_vec());
    // bucket indices fit int8, and cells int16
    if buckets.is_empty() || buckets.len() > 127 || buckets[0] == 0 || !buckets.windows(2).all(|w| w[0] < w[1]) {
        return Err(PyValueError::new_err(format!("buckets must ascend from 1, at most 127 of them: {buckets:?}")));
    }
    if levels.is_empty() || !levels.iter().all(|k| buckets.contains(k)) || !levels.windows(2).all(|w| w[0] < w[1]) {
        return Err(PyValueError::new_err(format!("levels must ascend and be of {buckets:?}: {levels:?}")));
    }
    let (cell, tokens, weights, level) = py.detach(|| run(&files, &levels, &buckets, entries, slots))?;
    let n = cell.len() / slots;
    if slots == 1 {
        let shape = |v: Vec<u16>| Array2::from_shape_vec((n, entries), v).expect("rows of `entries`");
        return Ok((
            cell.into_pyarray(py).into_any(),
            shape(tokens).into_pyarray(py).into_any(),
            shape(weights).into_pyarray(py).into_any(),
            level.into_pyarray(py).into_any(),
        ));
    }
    let shape = |v: Vec<u16>| Array3::from_shape_vec((n, slots, entries), v).expect("rows of `entries`");
    Ok((
        Array2::from_shape_vec((n, slots), cell).expect("rows of `slots`").into_pyarray(py).into_any(),
        shape(tokens).into_pyarray(py).into_any(),
        shape(weights).into_pyarray(py).into_any(),
        Array2::from_shape_vec((n, slots), level).expect("rows of `slots`").into_pyarray(py).into_any(),
    ))
}

#[allow(clippy::type_complexity)]
fn run(files: &[String], levels: &[usize], lengths: &[usize], entries: usize, slots: usize) -> PyResult<(Vec<i16>, Vec<u16>, Vec<u16>, Vec<i8>)> {
    let t0 = Instant::now();
    let (tokens, bounds) = read(files)?;
    let n = tokens.len();
    let start = segment_starts(&tokens, &bounds);
    // a row per slot of each position, which holds its slots filled so far
    let rows = n * slots;
    let (mut cell, mut next, mut weights, mut level) = (vec![0i16; rows], vec![0u16; rows * entries], vec![0u16; rows * entries], vec![-1i8; rows]);
    let mut filled = vec![0u8; n];
    eprintln!("train_hints: {n} positions in {} shards read, {:.1} s", files.len(), t0.elapsed().as_secs_f64());
    // pieces of one shard each, several per thread
    let piece = (n / (8 * rayon::current_num_threads())).clamp(1 << 14, 1 << 20);
    let mut pieces = Vec::new();
    for w in bounds.windows(2) {
        for lo in (w[0]..w[1]).step_by(piece) {
            pieces.push((lo, w[1].min(lo + piece), w[0], w[1]));
        }
    }
    let buckets = n.div_ceil(BUCKET).next_power_of_two();
    let mut elems: Map<Elem> = Map::new(n)?;
    for &k in levels.iter().rev() {
        let t = Instant::now();
        let li = lengths.iter().position(|&l| l == k).unwrap(); // its length bucket: its index in `buckets`
        // the position's element at this level, if it is an occurrence or a query with a slot still empty
        let element = |g: usize, shard: (usize, usize)| -> Option<Elem> {
            if g + 1 < shard.0 + k {
                return None; // the context leaves the shard, so it is no valid query either
            }
            let next = if g + 1 < shard.1 { tokens[g + 1] as u32 } else { NONE };
            let s = start[g];
            let query = if g + 1 - s as usize >= k && (filled[g] as usize) < slots { s } else { NONE };
            if next == NONE && query == NONE {
                return None;
            }
            Some(Elem { key: key(&tokens[g + 1 - k..=g]), at: g as u32, start: query, next, slot: filled[g] as u32 })
        };
        // step 2: count each piece's elements per bucket, then scatter them, pieces in stream order
        let mut offsets: Vec<Vec<u32>> = pieces
            .par_iter()
            .map(|&(lo, hi, a, b)| {
                let mut c = vec![0u32; buckets];
                for g in lo..hi {
                    if let Some(e) = element(g, (a, b)) {
                        c[bucket(e.key, buckets)] += 1;
                    }
                }
                c
            })
            .collect();
        let mut first = vec![0usize; buckets + 1];
        let mut at = 0usize;
        for b in 0..buckets {
            first[b] = at;
            for c in offsets.iter_mut() {
                let m = c[b] as usize;
                c[b] = at as u32;
                at += m;
            }
        }
        first[buckets] = at;
        let total = at;
        let out = Output(elems.as_mut_ptr());
        pieces.par_iter().zip(offsets.par_iter_mut()).for_each(|(&(lo, hi, a, b), cursor)| {
            for g in lo..hi {
                if let Some(e) = element(g, (a, b)) {
                    let c = &mut cursor[bucket(e.key, buckets)];
                    unsafe { out.set(*c as usize, e) };
                    *c += 1;
                }
            }
        });
        drop(offsets);
        let scattered = t.elapsed().as_secs_f64();
        // step 3: per bucket, largest first, sort and walk its groups
        let mut slices = Vec::with_capacity(buckets);
        let mut rest = &mut elems[..total];
        for b in 0..buckets {
            let (s, r) = rest.split_at_mut(first[b + 1] - first[b]);
            slices.push(s);
            rest = r;
        }
        slices.sort_unstable_by_key(|s| std::cmp::Reverse(s.len()));
        let outs = Outs { slots, cell: Output(cell.as_mut_ptr()), next: Output(next.as_mut_ptr()), weights: Output(weights.as_mut_ptr()), level: Output(level.as_mut_ptr()) };
        let (tokens, outs) = (&tokens, &outs);
        slices.into_par_iter().with_max_len(1).try_for_each_init(
            || Walker::new(entries),
            |w, slice| -> PyResult<()> {
                slice.sort_unstable_by_key(|e| (e.key, e.at));
                for group in slice.chunk_by(|a, b| a.key == b.key) {
                    w.walk(group, tokens, k, li, outs)?;
                }
                Ok(())
            },
        )?;
        // a query answered at this level wrote its first empty slot, which is filled now
        let hinted: usize = filled
            .par_iter_mut()
            .enumerate()
            .map(|(g, f)| {
                let answered = (*f as usize) < slots && level[g * slots + *f as usize] == li as i8;
                *f += answered as u8;
                answered as usize
            })
            .sum();
        eprintln!(
            "train_hints: k={k}: {total} positions in {buckets} buckets, {hinted} hinted, {:.1} s ({scattered:.1} s partitioning)",
            t.elapsed().as_secs_f64()
        );
    }
    Ok((cell, next, weights, level))
}

/// The tokens of `files`, concatenated, and the global index of each one's first token (then the total).
pub(crate) fn read(files: &[String]) -> PyResult<(Vec<u16>, Vec<usize>)> {
    let mut bounds = vec![0];
    for path in files {
        let (_, n) = open_shard(path)?;
        bounds.push(bounds.last().unwrap() + n);
    }
    let n = *bounds.last().unwrap();
    if n >= NONE as usize {
        return Err(PyValueError::new_err(format!("{n} tokens: positions are u32")));
    }
    let mut tokens = vec![0u16; n];
    let mut parts = Vec::with_capacity(files.len());
    let mut rest = &mut tokens[..];
    for w in bounds.windows(2) {
        let (part, r) = rest.split_at_mut(w[1] - w[0]);
        parts.push(part);
        rest = r;
    }
    files.par_iter().zip(parts).try_for_each(|(path, part)| -> PyResult<()> {
        let (mut file, n) = open_shard(path)?; // past the header
        if n != part.len() {
            return Err(PyValueError::new_err(format!("{path} changed while read")));
        }
        // little-endian u16, as on this crate's x86-64
        file.read_exact(unsafe { std::slice::from_raw_parts_mut(part.as_mut_ptr().cast::<u8>(), 2 * n) })?;
        Ok(())
    })?;
    Ok((tokens, bounds))
}

/// Per position, the global index of its segment's start: its shard's first token or its last BOS at or before it.
pub(crate) fn segment_starts(tokens: &[u16], bounds: &[usize]) -> Vec<u32> {
    let mut start = vec![0u32; tokens.len()];
    let mut parts = Vec::with_capacity(bounds.len() - 1);
    let mut rest = &mut start[..];
    for w in bounds.windows(2) {
        let (part, r) = rest.split_at_mut(w[1] - w[0]);
        parts.push((w[0], part));
        rest = r;
    }
    parts.into_par_iter().for_each(|(lo, part)| {
        let mut s = lo;
        for (i, out) in part.iter_mut().enumerate() {
            if tokens[lo + i] == BOS {
                s = lo + i;
            }
            *out = s as u32;
        }
    });
    start
}

/// A context's key: its tokens packed (16 bits each) up to PACKED tokens, else a 128-bit hash, two chains of
/// splitmix64's finalizer over its tokens packed 4 to a word.
#[inline]
pub(crate) fn key(context: &[u16]) -> u128 {
    if context.len() <= PACKED {
        return context.iter().fold(0u128, |h, &t| h << 16 | t as u128);
    }
    let (mut a, mut b) = (0x243f6a8885a308d3u64, 0x13198a2e03707344u64);
    for word in context.chunks(4) {
        let w = word.iter().fold(0u64, |w, &t| w << 16 | t as u64);
        a = mix(a ^ w);
        b = mix(b.rotate_left(29) ^ w.wrapping_mul(P));
    }
    (a as u128) << 64 | b as u128
}

/// splitmix64's finalizer.
#[inline]
fn mix(mut h: u64) -> u64 {
    h ^= h >> 30;
    h = h.wrapping_mul(0xbf58476d1ce4e5b9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94d049bb133111eb);
    h ^ (h >> 31)
}

/// A key's bucket of `buckets`.
#[inline]
pub(crate) fn bucket(key: u128, buckets: usize) -> usize {
    let h = mix(key as u64 ^ ((key >> 64) as u64).wrapping_mul(P));
    ((h as u128 * buckets as u128) >> 64) as usize
}

/// The output arrays, a row per slot of each position, which the walkers write disjoint rows of.
struct Outs {
    slots: usize,
    cell: Output<i16>,
    next: Output<u16>,
    weights: Output<u16>,
    level: Output<i8>,
}

/// One thread's state for walking key groups: the current group's counted occurrences.
struct Walker {
    entries: usize,
    counts: Vec<u32>,     // per next token
    touched: Vec<u16>,    // the next tokens with a count
    top: Vec<(u32, u16)>, // the `entries` most frequent next tokens (count, token): count descending, then token ascending
    total: u64,
    row: Vec<u16>,
    weights: Vec<u16>,
}

impl Walker {
    fn new(entries: usize) -> Self {
        Self {
            entries,
            counts: vec![0; 1 << 16],
            touched: Vec::new(),
            top: Vec::with_capacity(entries + 1),
            total: 0,
            row: vec![0; entries],
            weights: vec![0; entries],
        }
    }

    /// Step 3 for one key group of level k (length bucket li), in stream order.
    fn walk(&mut self, group: &[Elem], tokens: &[u16], k: usize, li: usize, outs: &Outs) -> PyResult<()> {
        // a lone position cannot count itself, and a group without queries answers none
        if group.len() < 2 || group.iter().all(|e| e.start == NONE) {
            return Ok(());
        }
        if k > PACKED {
            let context = |e: &Elem| &tokens[e.at as usize + 1 - k..=e.at as usize];
            if let Some(e) = group[1..].iter().find(|e| context(e) != context(&group[0])) {
                return Err(PyValueError::new_err(format!("128-bit hash collision between {k}-contexts at {} and {}", group[0].at, e.at)));
            }
        }
        for &t in &self.touched {
            self.counts[t as usize] = 0;
        }
        self.touched.clear();
        self.top.clear();
        self.total = 0;
        let mut next = 0; // the first occurrence not yet counted
        for e in group {
            if e.start == NONE {
                continue;
            }
            // stops before e itself, which starts at or after its segment's start
            while (group[next].at as u64) + 1 < e.start as u64 {
                if group[next].next != NONE {
                    self.add(group[next].next as u16);
                }
                next += 1;
            }
            if self.total > 0 {
                self.answer(e.at as usize * outs.slots + e.slot as usize, li, outs);
            }
        }
        Ok(())
    }

    /// Count an occurrence followed by `t`.
    #[inline]
    fn add(&mut self, t: u16) {
        let c = &mut self.counts[t as usize];
        if *c == 0 {
            self.touched.push(t);
        }
        *c += 1;
        let c = *c;
        self.total += 1;
        let better = |a: (u32, u16), b: (u32, u16)| a.0 > b.0 || (a.0 == b.0 && a.1 < b.1);
        let mut i = match self.top.iter().position(|&(_, u)| u == t) {
            Some(i) => i,
            None if self.top.len() < self.entries => {
                self.top.push((c, t));
                self.top.len() - 1
            }
            // only t's count grew: it enters the top only by passing the last of it
            None if better((c, t), self.top[self.entries - 1]) => self.entries - 1,
            None => return,
        };
        self.top[i] = (c, t);
        while i > 0 && better(self.top[i], self.top[i - 1]) {
            self.top.swap(i, i - 1);
            i -= 1;
        }
    }

    /// Write row `at` (a slot of a position) from the counted occurrences (total > 0), in length bucket li:
    /// val_hints.py's deepest() cell (lib.rs's row with a length bucket per level) and old_records.py's weights.
    fn answer(&mut self, at: usize, li: usize, outs: &Outs) {
        let top1 = self.top[0].0;
        let count = COUNTS.iter().filter(|&&c| c as u64 <= self.total).count();
        let purity = PURITIES.iter().filter(|&&p| p <= top1 as f64 / self.total as f64).count();
        let cell = ((1 + li) * (COUNTS.len() + 1) + count) * (PURITIES.len() + 1) + purity;
        let sum = self.top.iter().map(|&(c, _)| c as u64).sum::<u64>() as f64;
        self.row.fill(0);
        self.weights.fill(0);
        for (j, &(c, t)) in self.top.iter().enumerate() {
            self.row[j] = t;
            self.weights[j] = (c as f64 / sum * WEIGHT_ONE).round_ties_even() as u16; // np.round
        }
        unsafe {
            outs.cell.set(at, cell as i16);
            outs.level.set(at, li as i8);
            outs.next.write(at * self.entries, &self.row);
            outs.weights.write(at * self.entries, &self.weights);
        }
    }
}
