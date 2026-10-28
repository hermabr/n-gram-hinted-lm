//! CountTable: exact next-token counts of every context of a few lengths in the training shards, the validation side
//! of the old-records experiment's online hints (old_records/README.md; retrieval_blog_post/old_records.py). Built on
//! the clock while training runs, from the training shards alone (it never sees validation tokens), and queried with
//! the validation tokens after training.
//!
//! Definitions are kgram.rs's (and so val_hints.py's and old_records.val_count_features's): an occurrence of a
//! k-context is any k consecutive tokens within one shard and the token after them (it may cross a BOS); a query
//! position's k-context is the k tokens ending at it, valid iff they lie in its segment (segments start at the first
//! query token and at every BOS).
//!
//! Per level k, a context's key is a 64-bit value: for k <= 4 the tokens packed (exact), for longer contexts a
//! polynomial hash of 64-bit random token codes, both put through splitmix64's (bijective) finalizer. Two distinct
//! longer contexts share a key with probability ~2^-64 per pair: ~0.1 expected merged pairs among 2B distinct contexts,
//! and ~1e-10 false matches per query position. The build, per level (all of the pool's threads):
//!   1. count: each thread hashes its static share of the corpus (pieces of one shard each), counting entries per top
//!      bucket (the key's top TBITS bits);
//!   2. scatter: the same again, writing each entry's key and next token to its top bucket's place (few streams);
//!   3. per top bucket (largest first): scatter its entries (key's low 48 bits << 16 | next token) into its partitions
//!      (the key's top PBITS bits), sort each (by its top bits into sub-buckets, each sorted; a hot context's huge run by
//!      radix), and write its runs to the level's arrays from the partition's first slot (every array has a slot per
//!      corpus position, faulted in before the clock, 22 bytes per position and level): per context its key, total
//!      and first pair; per pair its next token and count (by token); per context with more than TOPE next tokens its
//!      TOPE most frequent, per context with more than CNT_TOP its CNT_TOP-th most frequent (count descending, then
//!      token ascending), for the top-CNT_TOP rule of the target's count.
//! Memory: 22 bytes per corpus token and level, plus 20 per corpus token of scratch (freed after the build).
use crate::{open_shard, pinned_pool, Map, Output, BOS, COUNTS, HEADER_BYTES, PURITIES};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::{exceptions::PyValueError, prelude::*};
use rayon::prelude::*;
use std::os::unix::fs::FileExt;
use std::time::Instant;

const PBITS: u32 = 16; // partitions: 2^16
const PARTS: usize = 1 << PBITS;
const TBITS: u32 = 10; // the first scatter's buckets: 2^8, by the key's top bits
const TOPS: usize = 1 << TBITS;
const SUB: u32 = 11; // sub-buckets of a partition's sort: 2^11, by the entries' top bits
const TOPE: usize = 8; // the next tokens kept per rich context (query_rows' most entries)
const CNT_TOP: usize = 1024; // query_counts' top rule (val_hints.TOP)
const PIECE: usize = 1 << 20; // corpus positions per piece of the count and scatter passes
const PACK: usize = 4; // the longest context whose tokens pack into a key
const B: u64 = 0x100000001b3; // the polynomial hash's base (odd)
const WEIGHT_ONE: f64 = 65535.0;

/// splitmix64's finalizer: a bijection of u64.
#[inline(always)]
fn mix(mut h: u64) -> u64 {
    h ^= h >> 30;
    h = h.wrapping_mul(0xbf58476d1ce4e5b9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94d049bb133111eb);
    h ^ (h >> 31)
}

/// A token's random code for the polynomial hash.
fn codes() -> Vec<u64> {
    (0..1u64 << 16).map(|t| mix(t.wrapping_add(0x9e3779b97f4a7c15).wrapping_mul(0x2545f4914f6cdd1d)) | 1).collect()
}

/// The key of the k-context `c` (k tokens), directly (the queries): the build's rolling hash gives the same.
#[inline]
fn key_of(c: &[u16], codes: &[u64]) -> u64 {
    if c.len() <= PACK {
        mix(c.iter().fold(0u64, |h, &t| h << 16 | t as u64))
    } else {
        mix(c.iter().fold(0u64, |h, &t| h.wrapping_mul(B).wrapping_add(codes[t as usize])))
    }
}

/// A level's arrays, one slot per corpus position (faulted in before the clock): partition p's contexts and pairs
/// take the first of its entries' slots.
struct Arrays {
    keys: Map<u64>,  // per context, sorted within a partition
    tot: Map<u32>,   // per context, its occurrences
    first: Map<u32>, // per context, its first pair, relative to its partition's first slot
    next: Map<u16>,  // per pair (by token within a context)
    count: Map<u32>,
}

/// A mapping of `len` T's, in huge pages (Map::new) or else in small pages: faulting in hundreds of GB of huge pages
/// on a host whose memory is fragmented compacts memory at every fault (minutes, before the clock).
fn map<T>(len: usize, huge: bool) -> std::io::Result<Map<T>> {
    if huge {
        return Map::new(len);
    }
    let bytes = (len * std::mem::size_of::<T>()).max(1).next_multiple_of(4096);
    let flags = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE;
    let ptr = unsafe { libc::mmap(std::ptr::null_mut(), bytes, libc::PROT_READ | libc::PROT_WRITE, flags, -1, 0) };
    if ptr == libc::MAP_FAILED {
        return Err(std::io::Error::last_os_error());
    }
    Ok(Map { ptr: ptr.cast(), len, bytes })
}

impl Arrays {
    fn new(n: usize, huge: bool) -> std::io::Result<Self> {
        Ok(Self { keys: map(n, huge)?, tot: map(n, huge)?, first: map(n, huge)?, next: map(n, huge)?, count: map(n, huge)? })
    }

    fn prefault(&mut self) {
        crate::prefault(&mut self.keys);
        crate::prefault(&mut self.tot);
        crate::prefault(&mut self.first);
        crate::prefault(&mut self.next);
        crate::prefault(&mut self.count);
    }
}

/// One level's table.
struct Level {
    k: usize,
    part: Vec<u64>,   // [PARTS + 1]: partition p's first slot
    nctx: Vec<u32>,   // [PARTS]: its contexts
    npairs: Vec<u32>, // [PARTS]: its pairs
    a: Arrays,
    rich: Vec<u64>,       // the contexts (slot, ascending) with more than TOPE next tokens
    rich_top: Vec<u16>,   // their TOPE most frequent next tokens, [rich, TOPE]
    rich_count: Vec<u32>, // and counts
    deep: Vec<u64>,       // the contexts with more than CNT_TOP next tokens
    deep_last: Vec<u64>,  // their CNT_TOP-th most frequent next token, as count << 16 | token
    contexts: usize,
    pairs: usize,
}

/// A context found: its global index, total, pairs [lo, hi).
#[derive(Clone, Copy)]
struct Found {
    ctx: u64,
    tot: u32,
    lo: usize,
    hi: usize,
}

impl Level {
    fn find(&self, key: u64) -> Option<Found> {
        let p = (key >> (64 - PBITS)) as usize;
        let a = self.part[p] as usize;
        let b = a + self.nctx[p] as usize;
        let i = a + self.a.keys[a..b].binary_search(&key).ok()?;
        let hi = a + if i + 1 < b { self.a.first[i + 1] } else { self.npairs[p] } as usize;
        Some(Found { ctx: i as u64, tot: self.a.tot[i], lo: a + self.a.first[i] as usize, hi })
    }

    /// The count of `t` after the context, if among its CNT_TOP most frequent next tokens (else 0).
    fn cnt(&self, f: Found, t: u16) -> u32 {
        let Ok(j) = self.a.next[f.lo..f.hi].binary_search(&t) else { return 0 };
        let c = self.a.count[f.lo + j];
        if f.hi - f.lo > CNT_TOP {
            let d = self.deep[..].binary_search(&f.ctx).expect("a deep context's threshold");
            let last = self.deep_last[d];
            let (lc, lt) = ((last >> 16) as u32, (last & 0xffff) as u16);
            if c < lc || (c == lc && t > lt) {
                return 0;
            }
        }
        c
    }

    /// The context's TOPE most frequent next tokens (count descending, token ascending) into `out`.
    fn top(&self, f: Found, out: &mut Vec<(u32, u16)>) {
        out.clear();
        if f.hi - f.lo <= TOPE {
            out.extend((f.lo..f.hi).map(|j| (self.a.count[j], self.a.next[j])));
            out.sort_unstable_by_key(|&(c, t)| (std::cmp::Reverse(c), t));
        } else {
            let r = self.rich[..].binary_search(&f.ctx).expect("a rich context's top");
            out.extend((0..TOPE).map(|j| (self.rich_count[r * TOPE + j], self.rich_top[r * TOPE + j])));
        }
    }
}

#[pyclass(frozen)]
pub struct CountTable {
    bounds: Vec<usize>, // each shard's first token in `tokens`, then the total
    levels: Vec<usize>,
    pool: rayon::ThreadPool,
    codes: Vec<u64>,
    state: std::sync::Mutex<State>,
}

/// The build's scratch: the top buckets' keys and next tokens, then the partitions' entries.
struct Scratch {
    tokens: Map<u16>, // the corpus, the shards concatenated
    keys: Map<u64>,
    next: Map<u16>,
    entries: Map<u64>,
}

struct State {
    scratch: Option<Scratch>,
    arrays: Vec<Arrays>, // per level, until built
    tables: Vec<Level>,
    built: bool,
    released: bool,
}

#[pymethods]
impl CountTable {
    /// Before the clock: read the shards `files` into memory, make the pool of `threads` threads (pinned round-robin to
    /// `cpus`, if any) and fault in the scratch the build sorts in and every level's arrays (22 bytes per corpus token
    /// and level; in huge pages if `huge_pages`, else small ones). Nothing is indexed.
    #[staticmethod]
    #[pyo3(signature = (files, levels, threads, cpus, huge_pages=false))]
    fn create(py: Python<'_>, files: Vec<String>, levels: Vec<usize>, threads: usize, cpus: Vec<usize>, huge_pages: bool) -> PyResult<Self> {
        if files.is_empty() || levels.is_empty() || levels[0] == 0 || !levels.windows(2).all(|w| w[0] < w[1]) || threads == 0 {
            return Err(PyValueError::new_err(format!("need files, ascending levels >= 1 and threads: {levels:?}")));
        }
        let mut bounds = vec![0usize];
        let mut opened = Vec::with_capacity(files.len());
        for f in &files {
            let (file, n) = open_shard(f)?;
            bounds.push(bounds.last().unwrap() + n);
            opened.push(file);
        }
        let n = *bounds.last().unwrap();
        let pool = pinned_pool(threads, cpus, "counttable")?;
        let mut arrays = levels.iter().map(|_| Arrays::new(n, huge_pages)).collect::<std::io::Result<Vec<_>>>()?;
        let mut scratch = Scratch { keys: map(n, huge_pages)?, next: map(n, huge_pages)?, entries: map(n, huge_pages)?, tokens: map(n, huge_pages)? };
        py.detach(|| -> PyResult<()> {
            pool.install(|| -> PyResult<()> {
                crate::prefault(&mut scratch.keys);
                crate::prefault(&mut scratch.next);
                crate::prefault(&mut scratch.entries);
                for a in arrays.iter_mut() {
                    a.prefault();
                }
                let out = Output(scratch.tokens.as_mut_ptr());
                // pieces of 64M tokens, read with pread
                let mut reads = Vec::new();
                for (i, w) in bounds.windows(2).enumerate() {
                    for a in (w[0]..w[1]).step_by(1 << 26) {
                        reads.push((i, a, w[1].min(a + (1 << 26))));
                    }
                }
                reads.par_iter().try_for_each(|&(i, a, b)| -> PyResult<()> {
                    let part = unsafe { out.slice(a, b - a) };
                    let bytes = unsafe { std::slice::from_raw_parts_mut(part.as_mut_ptr().cast::<u8>(), 2 * part.len()) };
                    opened[i].read_exact_at(bytes, (HEADER_BYTES + 2 * (a - bounds[i])) as u64)?;
                    Ok(())
                })
            })
        })?;
        if scratch.tokens.iter().any(|&t| t > BOS) {
            return Err(PyValueError::new_err("invalid GPT-2 token in the shards"));
        }
        Ok(Self {
            bounds,
            levels,
            pool,
            codes: codes(),
            state: std::sync::Mutex::new(State { scratch: Some(scratch), arrays, tables: Vec::new(), built: false, released: false }),
        })
    }

    /// On the clock: build every level (once).
    fn build(&self, py: Python<'_>) -> PyResult<()> {
        let mut state = self.state.lock().unwrap();
        if state.built || state.released {
            return Err(PyValueError::new_err("the count table is already built"));
        }
        state.built = true;
        let state = &mut *state;
        py.detach(|| -> PyResult<()> {
            let scratch = state.scratch.as_ref().unwrap();
            let arrays = std::mem::take(&mut state.arrays);
            for (&k, arrays) in self.levels.iter().zip(arrays) {
                let t = Instant::now();
                let level = self.pool.install(|| self.build_level(k, scratch, arrays))?;
                if std::env::var_os("COUNT_TABLE_VERBOSE").is_some() {
                    eprintln!(
                        "count_table: k={k}: {} contexts, {} pairs, {} rich, {} deep, {:.2} s",
                        level.contexts,
                        level.pairs,
                        level.rich.len(),
                        level.deep.len(),
                        t.elapsed().as_secs_f64()
                    );
                }
                state.tables.push(level);
            }
            state.scratch = None;
            Ok(())
        })
    }

    /// After the build: per position of `tokens` (the inputs, then the last one's next token) and level, tot (uint32),
    /// dist (uint16, at most 65535) and cnt (uint32: the count of the position's next token if among its context's
    /// `top` most frequent next tokens, else 0), [n, levels]; 0 where the context is invalid or never occurred. `top`
    /// is CNT_TOP or 65536 and more (no limit).
    #[pyo3(signature = (tokens, top=CNT_TOP))]
    #[allow(clippy::type_complexity)]
    fn query_counts<'py>(&self, py: Python<'py>, tokens: PyReadonlyArray1<u16>, top: usize) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        if top != CNT_TOP && top < 1 << 16 {
            return Err(PyValueError::new_err(format!("top must be {CNT_TOP} or at least 65536")));
        }
        let x = tokens.as_slice()?;
        if x.len() < 2 || x.iter().any(|&t| t > BOS) {
            return Err(PyValueError::new_err("need 2+ valid tokens"));
        }
        let state = self.state.lock().unwrap();
        let tables = self.tables(&state)?;
        let (n, nl) = (x.len() - 1, tables.len());
        let (mut tot, mut dist, mut cnt) = (vec![0u32; n * nl], vec![0u16; n * nl], vec![0u32; n * nl]);
        let avail = segments(&x[..n]);
        let (o_tot, o_dist, o_cnt) = (Output(tot.as_mut_ptr()), Output(dist.as_mut_ptr()), Output(cnt.as_mut_ptr()));
        py.detach(|| {
            self.pool.install(|| {
                (0..n).into_par_iter().with_min_len(4096).for_each(|i| {
                    for (j, l) in tables.iter().enumerate() {
                        if (avail[i] as usize) < l.k {
                            break; // levels ascend
                        }
                        let Some(f) = l.find(key_of(&x[i + 1 - l.k..=i], &self.codes)) else { continue };
                        let c = if top == CNT_TOP {
                            l.cnt(f, x[i + 1])
                        } else {
                            l.a.next[f.lo..f.hi].binary_search(&x[i + 1]).map(|j| l.a.count[f.lo + j]).unwrap_or(0)
                        };
                        unsafe {
                            o_tot.set(i * nl + j, f.tot);
                            o_dist.set(i * nl + j, (f.hi - f.lo).min(u16::MAX as usize) as u16);
                            o_cnt.set(i * nl + j, c);
                        }
                    }
                })
            })
        });
        let shape = |v| Array2::from_shape_vec((n, nl), v).expect("rows of levels");
        Ok((shape(tot).into_pyarray(py).into_any(), Array2::from_shape_vec((n, nl), dist).unwrap().into_pyarray(py).into_any(), shape(cnt).into_pyarray(py).into_any()))
    }

    /// After the build: per position of `tokens` (as query_counts's), the hint row of the deepest level of
    /// `row_levels` (of the table's levels) whose context occurred: its cell (int16, with the level's index in
    /// `buckets` as the length bucket; 0 without a hint), its `entries` (<= TOPE) most frequent next tokens (uint16,
    /// count descending, then token ascending, padded with 0), their weights (uint16, round(count / their sum *
    /// 65535)), and the level's index in `buckets` (int8, -1 without a hint).
    #[pyo3(signature = (tokens, row_levels, buckets, entries=TOPE))]
    #[allow(clippy::type_complexity)]
    fn query_rows<'py>(
        &self,
        py: Python<'py>,
        tokens: PyReadonlyArray1<u16>,
        row_levels: Vec<usize>,
        buckets: Vec<usize>,
        entries: usize,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        let x = tokens.as_slice()?;
        if x.len() < 2 || x.iter().any(|&t| t > BOS) || entries == 0 || entries > TOPE {
            return Err(PyValueError::new_err(format!("need 2+ valid tokens and 1 <= entries <= {TOPE}")));
        }
        let state = self.state.lock().unwrap();
        let tables = self.tables(&state)?;
        // (table index, bucket index) of each row level, deepest first
        let mut use_: Vec<(usize, usize)> = Vec::new();
        for &k in &row_levels {
            let t = self.levels.iter().position(|&l| l == k).ok_or_else(|| PyValueError::new_err(format!("level {k} is not in the table")))?;
            let b = buckets.iter().position(|&l| l == k).ok_or_else(|| PyValueError::new_err(format!("level {k} is not in buckets")))?;
            use_.push((t, b));
        }
        use_.sort_unstable_by_key(|&(t, _)| std::cmp::Reverse(self.levels[t]));
        let n = x.len() - 1;
        let (mut cell, mut next, mut weights, mut level) = (vec![0i16; n], vec![0u16; n * entries], vec![0u16; n * entries], vec![-1i8; n]);
        let avail = segments(&x[..n]);
        let (o_cell, o_next, o_w, o_level) = (Output(cell.as_mut_ptr()), Output(next.as_mut_ptr()), Output(weights.as_mut_ptr()), Output(level.as_mut_ptr()));
        py.detach(|| {
            self.pool.install(|| {
                (0..n).into_par_iter().with_min_len(4096).for_each_init(
                    || (Vec::with_capacity(TOPE), vec![0u16; entries], vec![0u16; entries]),
                    |(top, row, w), i| {
                        for &(t, b) in &use_ {
                            let l = &tables[t];
                            if (avail[i] as usize) < l.k {
                                continue;
                            }
                            let Some(f) = l.find(key_of(&x[i + 1 - l.k..=i], &self.codes)) else { continue };
                            l.top(f, top);
                            let total = f.tot as u64;
                            let count = COUNTS.iter().filter(|&&c| c as u64 <= total).count();
                            let purity = PURITIES.iter().filter(|&&p| p <= top[0].0 as f64 / total as f64).count();
                            let c = ((1 + b) * (COUNTS.len() + 1) + count) * (PURITIES.len() + 1) + purity;
                            let kept = &top[..entries.min(top.len())];
                            let sum = kept.iter().map(|&(c, _)| c as u64).sum::<u64>() as f64;
                            row.fill(0);
                            w.fill(0);
                            for (j, &(c, tok)) in kept.iter().enumerate() {
                                row[j] = tok;
                                w[j] = (c as f64 / sum * WEIGHT_ONE).round_ties_even() as u16;
                            }
                            unsafe {
                                o_cell.set(i, c as i16);
                                o_level.set(i, b as i8);
                                o_next.write(i * entries, row);
                                o_w.write(i * entries, w);
                            }
                            break;
                        }
                    },
                )
            })
        });
        let shape = |v| Array2::from_shape_vec((n, entries), v).expect("rows of entries");
        Ok((cell.into_pyarray(py).into_any(), shape(next).into_pyarray(py).into_any(), shape(weights).into_pyarray(py).into_any(), level.into_pyarray(py).into_any()))
    }

    /// The table's levels.
    #[getter]
    fn levels(&self) -> Vec<usize> {
        self.levels.clone()
    }

    /// The corpus tokens indexed.
    #[getter]
    fn tokens(&self) -> usize {
        *self.bounds.last().unwrap()
    }

    /// Per level (as built): its contexts, pairs, and the bytes they use (the level holds 22 per corpus token).
    fn sizes(&self) -> Vec<(usize, usize, usize, usize)> {
        let state = self.state.lock().unwrap();
        state
            .tables
            .iter()
            .map(|l| {
                let bytes = l.contexts * 16 + l.pairs * 6 + l.rich.len() * (8 + 6 * TOPE) + l.deep.len() * 16;
                (l.k, l.contexts, l.pairs, bytes)
            })
            .collect()
    }

    /// Off the clock: free the tables and the corpus copy.
    fn release(&self, py: Python<'_>) {
        let mut state = self.state.lock().unwrap();
        state.released = true;
        let (tables, scratch, arrays) = (std::mem::take(&mut state.tables), state.scratch.take(), std::mem::take(&mut state.arrays));
        py.detach(|| drop((tables, scratch, arrays)));
    }
}

/// Per position of `x`, the tokens of its segment up to and including it (kgram.rs's segments).
fn segments(x: &[u16]) -> Vec<u32> {
    let mut avail = vec![0u32; x.len()];
    let mut start = 0;
    for i in 0..x.len() {
        if x[i] == BOS {
            start = i;
        }
        avail[i] = (i - start + 1) as u32;
    }
    avail
}

impl CountTable {
    fn tables<'a>(&self, state: &'a State) -> PyResult<&'a [Level]> {
        if !state.built || state.tables.len() != self.levels.len() || state.released {
            return Err(PyValueError::new_err("the count table is not built (or released)"));
        }
        Ok(&state.tables)
    }

    /// Call `f(key, next)` for every occurrence of a k-context in positions [lo, hi) of shard [s, e) (context ends).
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn scan(&self, t: &[u16], k: usize, lo: usize, hi: usize, s: usize, e: usize, mut f: impl FnMut(u64, u16)) {
        let lo = lo.max(s + k - 1);
        let hi = hi.min(e - 1); // the last context end with a next token in the shard
        if lo >= hi {
            return;
        }
        if k <= PACK {
            let mask = if k == 4 { u64::MAX } else { (1u64 << (16 * k)) - 1 };
            let mut h = t[lo + 1 - k..lo].iter().fold(0u64, |h, &x| h << 16 | x as u64);
            for g in lo..hi {
                h = (h << 16 | t[g] as u64) & mask;
                f(mix(h), t[g + 1]);
            }
        } else {
            let c = &self.codes;
            let bk = (0..k).fold(1u64, |a, _| a.wrapping_mul(B)); // B^k
            let mut h = t[lo + 1 - k..lo].iter().fold(0u64, |h, &x| h.wrapping_mul(B).wrapping_add(c[x as usize]));
            // h holds the k - 1 tokens before lo; each step adds t[g] and drops t[g - k]
            h = h.wrapping_mul(B).wrapping_add(c[t[lo] as usize]);
            f(mix(h), t[lo + 1]);
            for g in lo + 1..hi {
                h = h.wrapping_mul(B).wrapping_add(c[t[g] as usize]).wrapping_sub(c[t[g - k] as usize].wrapping_mul(bk));
                f(mix(h), t[g + 1]);
            }
        }
    }

    fn build_level(&self, k: usize, scratch: &Scratch, mut arrays: Arrays) -> PyResult<Level> {
        let threads = rayon::current_num_threads();
        let toks = &scratch.tokens[..];
        let t0 = Instant::now();
        let mut phases = Vec::new();
        // static shares: pieces of one shard each, contiguous runs of them per thread
        let mut pieces = Vec::new();
        for w in self.bounds.windows(2) {
            for a in (w[0]..w[1]).step_by(PIECE) {
                pieces.push((a, w[1].min(a + PIECE), w[0], w[1]));
            }
        }
        let share = |t: usize| &pieces[pieces.len() * t / threads..pieces.len() * (t + 1) / threads];
        // 1. count per top bucket (the key's top TBITS bits)
        let mut counts: Vec<Vec<u64>> = (0..threads)
            .into_par_iter()
            .with_max_len(1)
            .map(|t| {
                let mut c = vec![0u64; TOPS];
                for &(lo, hi, s, e) in share(t) {
                    self.scan(toks, k, lo, hi, s, e, |key, _| c[(key >> (64 - TBITS)) as usize] += 1);
                }
                c
            })
            .collect();
        let mut top_start = vec![0u64; TOPS + 1];
        let mut at = 0u64;
        for b in 0..TOPS {
            top_start[b] = at;
            for c in counts.iter_mut() {
                let m = c[b];
                c[b] = at;
                at += m;
            }
        }
        top_start[TOPS] = at;
        phases.push(t0.elapsed().as_secs_f64());
        // 2. scatter (key, next) to the top buckets, sequential streams
        let (ok, on) = (Output(scratch.keys.as_ptr() as *mut u64), Output(scratch.next.as_ptr() as *mut u16));
        counts.par_iter_mut().with_max_len(1).enumerate().for_each(|(t, cursor)| {
            for &(lo, hi, s, e) in share(t) {
                self.scan(toks, k, lo, hi, s, e, |key, next| {
                    let c = &mut cursor[(key >> (64 - TBITS)) as usize];
                    unsafe {
                        ok.set(*c as usize, key);
                        on.set(*c as usize, next);
                    }
                    *c += 1;
                });
            }
        });
        drop(counts);
        phases.push(t0.elapsed().as_secs_f64());
        // 3. per top bucket (largest first): scatter its entries (key's low 48 bits << 16 | next) into its partitions,
        // then sort each partition and write its runs to the level's arrays, from the partition's first slot
        let out = Output(scratch.entries.as_ptr() as *mut u64);
        let o = Outs {
            keys: Output(arrays.keys.as_mut_ptr()),
            tot: Output(arrays.tot.as_mut_ptr()),
            first: Output(arrays.first.as_mut_ptr()),
            next: Output(arrays.next.as_mut_ptr()),
            count: Output(arrays.count.as_mut_ptr()),
        };
        let o = &o;
        let mut tops: Vec<usize> = (0..TOPS).collect();
        tops.sort_unstable_by_key(|&b| std::cmp::Reverse(top_start[b + 1] - top_start[b]));
        const SUBS: usize = PARTS / TOPS;
        let done: Vec<(usize, Vec<u64>, Vec<(u32, u32)>, Side)> = tops
            .par_iter()
            .with_max_len(1)
            .map_init(
                || (Vec::new(), vec![0u32; (1 << SUB) + 1], Vec::new()),
                |(tmp, hist, buf), &b| {
                    let (a, z) = (top_start[b] as usize, top_start[b + 1] as usize);
                    let keys = unsafe { ok.slice(a, z - a) };
                    let nexts = unsafe { on.slice(a, z - a) };
                    let sub = |key: u64| ((key >> (64 - PBITS)) as usize) & (SUBS - 1);
                    let mut starts = vec![0u64; SUBS + 1];
                    for &key in keys.iter() {
                        starts[sub(key) + 1] += 1;
                    }
                    for j in 0..SUBS {
                        starts[j + 1] += starts[j];
                    }
                    let mut cursor: Vec<u64> = starts[..SUBS].to_vec();
                    let e = unsafe { out.slice(a, z - a) };
                    for (&key, &next) in keys.iter().zip(nexts.iter()) {
                        let c = &mut cursor[sub(key)];
                        e[*c as usize] = key << PBITS | next as u64;
                        *c += 1;
                    }
                    let mut side = Side::default();
                    let sizes = (0..SUBS)
                        .map(|j| {
                            let part = &mut e[starts[j] as usize..starts[j + 1] as usize];
                            sort(part, tmp, hist);
                            runs(part, b * SUBS + j, a + starts[j] as usize, o, &mut side, buf)
                        })
                        .collect();
                    (b, starts.iter().map(|&x| x + a as u64).collect(), sizes, side)
                },
            )
            .collect();
        let mut done = done;
        done.sort_unstable_by_key(|d| d.0);
        let mut level = Level {
            k,
            part: vec![0u64; PARTS + 1],
            nctx: vec![0u32; PARTS],
            npairs: vec![0u32; PARTS],
            a: arrays,
            rich: Vec::new(),
            rich_top: Vec::new(),
            rich_count: Vec::new(),
            deep: Vec::new(),
            deep_last: Vec::new(),
            contexts: 0,
            pairs: 0,
        };
        level.part[PARTS] = top_start[TOPS];
        for (b, starts, sizes, side) in done {
            level.part[b * SUBS..(b + 1) * SUBS].copy_from_slice(&starts[..SUBS]);
            for (j, &(c, q)) in sizes.iter().enumerate() {
                level.nctx[b * SUBS + j] = c;
                level.npairs[b * SUBS + j] = q;
                level.contexts += c as usize;
                level.pairs += q as usize;
            }
            // buckets in slot order: the side arrays stay sorted by slot
            level.rich.extend(side.rich);
            level.rich_top.extend(side.rich_top);
            level.rich_count.extend(side.rich_count);
            level.deep.extend(side.deep);
            level.deep_last.extend(side.deep_last);
        }
        phases.push(t0.elapsed().as_secs_f64());
        if std::env::var_os("COUNT_TABLE_VERBOSE").is_some() {
            eprintln!("count_table: k={k} phases (count, scatter, sort+write) ends {phases:.2?}");
        }
        Ok(level)
    }
}

/// Sort a partition's entries: by their top bits into about len / 8 sub-buckets (at most 2^SUB, through `tmp`), each
/// then sorted.
fn sort(e: &mut [u64], tmp: &mut Vec<u64>, hist: &mut [u32]) {
    if e.len() <= 64 {
        e.sort_unstable();
        return;
    }
    let bits = (e.len() / 8).ilog2().clamp(1, SUB);
    let subs = 1usize << bits;
    let sub = |v: u64| (v >> (64 - bits)) as usize; // the entries' top bits: they sort first
    let hist = &mut hist[..subs + 1];
    hist.fill(0);
    for &v in e.iter() {
        hist[sub(v) + 1] += 1;
    }
    for i in 0..subs {
        hist[i + 1] += hist[i];
    }
    tmp.clear();
    tmp.resize(e.len(), 0);
    let mut fill: Vec<u32> = hist[..subs].to_vec();
    for &v in e.iter() {
        let s = &mut fill[sub(v)];
        tmp[*s as usize] = v;
        *s += 1;
    }
    for i in 0..subs {
        let r = &mut tmp[hist[i] as usize..hist[i + 1] as usize];
        if r.len() > 1 << 15 {
            radix_sort(r); // a hot context's run (pdqsort takes ~50 ns per entry on millions of them)
        } else if r.len() > 1 {
            r.sort_unstable();
        }
    }
    e.copy_from_slice(tmp);
}

/// LSD radix sort by 16-bit digits, skipping the digits all entries share (a hot context's run sorts by its next
/// tokens only).
fn radix_sort(r: &mut [u64]) {
    let mut buf = vec![0u64; r.len()];
    let mut count = vec![0usize; 1 << 16];
    let (mut src, mut dst): (&mut [u64], &mut [u64]) = (r, &mut buf);
    let mut swapped = false;
    for d in 0..4 {
        let digit = |v: u64| ((v >> (16 * d)) & 0xffff) as usize;
        count.fill(0);
        for &v in src.iter() {
            count[digit(v)] += 1;
        }
        if count[digit(src[0])] == src.len() {
            continue;
        }
        let mut at = 0;
        for c in count.iter_mut() {
            let m = *c;
            *c = at;
            at += m;
        }
        for &v in src.iter() {
            let c = &mut count[digit(v)];
            dst[*c] = v;
            *c += 1;
        }
        std::mem::swap(&mut src, &mut dst);
        swapped = !swapped;
    }
    if swapped {
        dst.copy_from_slice(src); // src is buf: the sorted entries back into r
    }
}

/// A bucket's rich and deep contexts (Level's side arrays).
#[derive(Default)]
struct Side {
    rich: Vec<u64>,
    rich_top: Vec<u16>,
    rich_count: Vec<u32>,
    deep: Vec<u64>,
    deep_last: Vec<u64>,
}

struct Outs {
    keys: Output<u64>,
    tot: Output<u32>,
    first: Output<u32>,
    next: Output<u16>,
    count: Output<u32>,
}

/// Write sorted partition `p`'s contexts and pairs from slot `base`; its rich and deep contexts to `side`. Returns its
/// contexts and pairs.
fn runs(e: &[u64], p: usize, base: usize, o: &Outs, side: &mut Side, buf: &mut Vec<u64>) -> (u32, u32) {
    let (mut c, mut q) = (base, base);
    let top_bits = (p as u64) << (64 - PBITS);
    let mut i = 0;
    while i < e.len() {
        let ctx = e[i] >> 16;
        let q_first = q;
        let mut j = i;
        while j < e.len() && e[j] >> 16 == ctx {
            let v = e[j];
            let a = j;
            while j < e.len() && e[j] == v {
                j += 1;
            }
            unsafe {
                o.next.set(q, (v & 0xffff) as u16);
                o.count.set(q, (j - a) as u32);
            }
            q += 1;
        }
        let tot = j - i;
        assert!(tot <= u32::MAX as usize, "a context's total does not fit u32");
        unsafe {
            o.keys.set(c, top_bits | ctx);
            o.tot.set(c, tot as u32);
            o.first.set(c, (q_first - base) as u32);
        }
        let dist = q - q_first;
        if dist > TOPE {
            // the pairs as sort keys: (count descending, token ascending) ascending
            buf.clear();
            buf.extend((q_first..q).map(|x| unsafe { ((u32::MAX - *o.count.0.add(x)) as u64) << 16 | *o.next.0.add(x) as u64 }));
            if dist > CNT_TOP {
                buf.select_nth_unstable(CNT_TOP - 1);
                let last = buf[CNT_TOP - 1];
                side.deep.push(c as u64);
                side.deep_last.push(((u32::MAX - (last >> 16) as u32) as u64) << 16 | (last & 0xffff));
            }
            buf.select_nth_unstable(TOPE - 1);
            let top = &mut buf[..TOPE];
            top.sort_unstable();
            side.rich.push(c as u64);
            for &v in top.iter() {
                side.rich_top.push((v & 0xffff) as u16);
                side.rich_count.push(u32::MAX - (v >> 16) as u32);
            }
        }
        c += 1;
        i = j;
    }
    ((c - base) as u32, (q - base) as u32)
}
