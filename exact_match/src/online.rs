//! Online, causal training rows of the old-records experiment (old_records/hint_runtime.py's online hints): the hint
//! rows (train_hints.rs's) and the n-gram count rows (train_counts.rs's) of a run's training stream, computed on the
//! clock ahead of training, under the step rule: a step may only use tokens trained in earlier steps.
//!
//! The stream is the run's training tokens in parts (each a shard's consumed prefix): parts are hard boundaries, as
//! shards are in train_hints.rs (an occurrence, its k tokens and its next token, lies within one part; a segment starts
//! at each part's start and at every BOS). `spans` say when each position is trained: (lo, hi, step) makes positions
//! [lo, hi) inputs of `step`, and position hi (within the part) that step's last target. tau(p) is the first step that
//! trains p (as an input or a target).
//!
//! A query is a position g trained as an input at step s (tau_in(g) = s) whose k-context lies in its segment
//! (train_hints.rs's validity). An occurrence ending at g (tokens g-k+1..=g, next token g+1) is available from step
//! alpha(g) = 1 + max tau over its k + 1 tokens, and counts for a query at step s iff alpha(g) <= s. Per level, every
//! position gives up to two elements, an occurrence (time alpha) and a query (time s); each key group is sorted by
//! (time, kind: occurrences first, position) and walked once: an occurrence adds its next token, a query reads the
//! counts so far. rule="segment" instead gives every position a query at time segstart(g) and every occurrence the time
//! g + 2 (counted iff g + 1 < segstart): train_hints.rs's and train_counts.rs's rule, for the exactness tests.
use crate::train_hints::{segment_starts, BUCKET, NONE};
use crate::{pinned_pool, Map, Output, COUNTS, PURITIES};
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1, PyReadwriteArray2, PyUntypedArrayMethods};
use pyo3::{exceptions::PyValueError, prelude::*};
use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering::Relaxed};
use std::sync::Mutex;

const WEIGHT_ONE: f64 = 65535.0; // hint_runtime.py's

/// One position's element at one level: its context's key (key64), its time (t << 1 | kind, kind 0 an occurrence, 1 a
/// query) and its stream index. An occurrence's next token, and a query's own target (if within its part), is
/// tokens[at + 1].
#[derive(Clone, Copy)]
#[repr(C)]
struct Elem {
    key: u64,
    time: u32,
    at: u32,
}

/// The longest context whose tokens pack into a key64 exactly; longer ones are hashed, and the walks split a key group
/// whose contexts differ (a collision), so the rows stay exact.
const PACKED64: usize = 4;
const P: u64 = 0x9e3779b185ebca87;

/// A context's 64-bit key: its tokens packed (16 bits each) up to PACKED64 tokens, else a hash chain over its tokens
/// packed 4 to a word.
#[inline]
fn key64(context: &[u16]) -> u64 {
    if context.len() <= PACKED64 {
        return context.iter().fold(0u64, |h, &t| h << 16 | t as u64);
    }
    let mut a = 0x243f6a8885a308d3u64 ^ context.len() as u64;
    for word in context.chunks(4) {
        a = mix(a.rotate_left(23) ^ word.iter().fold(0u64, |w, &t| w << 16 | t as u64).wrapping_mul(P));
    }
    a & *HASH_MASK
}

/// ONLINE_ROWS_VERBOSE=1 prints each level's times.
static VERBOSE: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var("ONLINE_ROWS_VERBOSE").is_ok_and(|v| v == "1"));

/// ONLINE_ROWS_HASH_BITS (tests only) keeps that many bits of the hashed keys, so that collisions are common.
static HASH_MASK: std::sync::LazyLock<u64> = std::sync::LazyLock::new(|| {
    std::env::var("ONLINE_ROWS_HASH_BITS").ok().and_then(|b| b.parse::<u32>().ok()).map_or(u64::MAX, |b| (1u64 << b.min(63)) - 1)
});

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
fn bucket64(key: u64, buckets: usize) -> usize {
    ((mix(key ^ 0x5bd1e9955bd1e995) as u128 * buckets as u128) >> 64) as usize
}

#[derive(Clone, Copy)]
enum Rule {
    Step { lo: u32, hi: u32 },
    Segment,
}

fn rule_of(rule: &str, lo: usize, hi: usize) -> PyResult<Rule> {
    match rule {
        "step" => {
            if lo > hi || hi >= 1 << 30 {
                return Err(PyValueError::new_err(format!("bad step range [{lo}, {hi})")));
            }
            Ok(Rule::Step { lo: lo as u32, hi: hi as u32 })
        }
        "segment" => Ok(Rule::Segment),
        _ => Err(PyValueError::new_err(format!("rule must be 'step' or 'segment': {rule}"))),
    }
}

/// A run's training stream and when each of its positions is trained, for its online rows.
#[pyclass(frozen)]
pub struct OnlineRows {
    tokens: Vec<u16>,
    bounds: Vec<usize>,
    seg: Vec<u32>,
    tau: Vec<u32>,    // the first step that trains each position (input or target), NONE if none
    tau_in: Vec<u32>, // the step that trains each position as an input, NONE if none
    pieces: Vec<(usize, usize, usize, usize, u32)>, // (lo, hi, part start, part end, min tau over lo..=hi)
    pool: rayon::ThreadPool,
    elems: Mutex<Option<(Map<Elem>, Map<u64>)>>,
    incr: Mutex<Option<Incr>>, // advance()'s state
}

#[pymethods]
impl OnlineRows {
    #[new]
    fn new(
        py: Python<'_>,
        tokens: PyReadonlyArray1<u16>,
        bounds: Vec<usize>,
        spans: PyReadonlyArray2<i64>,
        threads: usize,
        cpus: Vec<usize>,
    ) -> PyResult<Self> {
        let tokens = tokens.as_slice()?.to_vec();
        let n = tokens.len();
        if n == 0 || n >= 1 << 30 {
            return Err(PyValueError::new_err(format!("{n} tokens: need 1 to 2^30")));
        }
        if bounds.len() < 2 || bounds[0] != 0 || *bounds.last().unwrap() != n || !bounds.windows(2).all(|w| w[0] < w[1]) {
            return Err(PyValueError::new_err("bounds must ascend strictly from 0 to the token count"));
        }
        let spans = spans.as_array();
        if spans.ncols() != 3 {
            return Err(PyValueError::new_err("spans must be [m, 3]: (lo, hi, step)"));
        }
        let spans: Vec<(usize, usize, u32)> = spans
            .rows()
            .into_iter()
            .map(|r| {
                let (lo, hi, step) = (r[0], r[1], r[2]);
                if lo < 0 || lo > hi || hi as usize > n || step < 0 || step >= NONE as i64 {
                    return Err(PyValueError::new_err(format!("bad span ({lo}, {hi}, {step})")));
                }
                Ok((lo as usize, hi as usize, step as u32))
            })
            .collect::<PyResult<_>>()?;
        let pool = pinned_pool(threads.max(1), cpus, "onlinerows")?;
        let (seg, tau, tau_in) = py.detach(|| {
            pool.install(|| {
                let seg = segment_starts(&tokens, &bounds);
                let tau: Vec<AtomicU32> = (0..n).into_par_iter().map(|_| AtomicU32::new(NONE)).collect();
                let tau_in: Vec<AtomicU32> = (0..n).into_par_iter().map(|_| AtomicU32::new(NONE)).collect();
                spans.par_iter().for_each(|&(lo, hi, step)| {
                    for p in lo..hi {
                        tau[p].fetch_min(step, Relaxed);
                        tau_in[p].fetch_min(step, Relaxed);
                    }
                    if hi > lo && hi < n {
                        // the last input's target, if it lies in the same part
                        let part = bounds.partition_point(|&b| b <= hi - 1);
                        if hi < bounds[part] {
                            tau[hi].fetch_min(step, Relaxed);
                        }
                    }
                });
                let unwrap = |v: Vec<AtomicU32>| v.into_par_iter().map(|a| a.into_inner()).collect::<Vec<u32>>();
                (seg, unwrap(tau), unwrap(tau_in))
            })
        });
        // pieces of one part each, small enough that a short step range still spreads over the threads
        let piece = (n / (32 * pool.current_num_threads())).clamp(1 << 15, 1 << 20);
        let mut pieces = Vec::new();
        for w in bounds.windows(2) {
            for lo in (w[0]..w[1]).step_by(piece) {
                let hi = w[1].min(lo + piece);
                let min = tau[lo..(hi + 1).min(n)].iter().copied().min().unwrap();
                pieces.push((lo, hi, w[0], w[1], min));
            }
        }
        Ok(Self { tokens, bounds, seg, tau, tau_in, pieces, pool, elems: Mutex::new(None), incr: Mutex::new(None) })
    }

    /// The hint rows (train_hints.rs's, one slot) of this call's queries, written into the stream-sized `cell` [n],
    /// `next` and `weights` [n, entries]: per query its deepest level of `levels` with a counted occurrence. Returns the
    /// queries hinted.
    #[pyo3(signature = (levels, buckets, entries, step_lo, step_hi, cell, next, weights, rule="step"))]
    #[allow(clippy::too_many_arguments)]
    fn hints(
        &self,
        py: Python<'_>,
        levels: Vec<usize>,
        buckets: Vec<usize>,
        entries: usize,
        step_lo: usize,
        step_hi: usize,
        mut cell: PyReadwriteArray1<i16>,
        mut next: PyReadwriteArray2<u16>,
        mut weights: PyReadwriteArray2<u16>,
        rule: &str,
    ) -> PyResult<usize> {
        let rule = rule_of(rule, step_lo, step_hi)?;
        let n = self.tokens.len();
        if entries == 0 || buckets.is_empty() || buckets.len() > 127 || buckets[0] == 0 || !buckets.windows(2).all(|w| w[0] < w[1]) {
            return Err(PyValueError::new_err(format!("need entries >= 1 and buckets ascending from 1: {entries}, {buckets:?}")));
        }
        if levels.is_empty() || !levels.iter().all(|k| buckets.contains(k)) || !levels.windows(2).all(|w| w[0] < w[1]) {
            return Err(PyValueError::new_err(format!("levels must ascend and be of {buckets:?}: {levels:?}")));
        }
        if cell.len() != n || next.shape() != [n, entries] || weights.shape() != [n, entries] {
            return Err(PyValueError::new_err(format!("outputs must be [{n}] and [{n}, {entries}]")));
        }
        let outs = Outs {
            cell: Output(cell.as_slice_mut()?.as_mut_ptr()),
            next: Output(next.as_slice_mut()?.as_mut_ptr()),
            weights: Output(weights.as_slice_mut()?.as_mut_ptr()),
        };
        py.detach(|| {
            let mut guard = self.elems.lock().unwrap();
            let buf = self.buffer(&mut guard)?;
            let mut filled = vec![0u8; n];
            let mut hinted = 0;
            for &k in levels.iter().rev() {
                let t0 = std::time::Instant::now();
                let li = buckets.iter().position(|&l| l == k).unwrap();
                let fill = Output(filled.as_mut_ptr());
                let f = &filled;
                let groups = self.partition(buf, k, rule, &|g| f[g] == 0)?;
                let outs = &outs;
                let tokens = &self.tokens;
                let answered: usize = self.pool.install(|| {
                    groups
                        .into_par_iter()
                        .with_max_len(1)
                        .map_init(
                            || Walker::new(entries),
                            |w, slice| {
                                let mut a = 0;
                                walk_bucket(slice, tokens, k, |group| a += w.walk(group, tokens, li, outs, fill));
                                a
                            },
                        )
                        .sum::<usize>()
                });
                hinted += answered;
                if *VERBOSE {
                    eprintln!("onlinerows hints k={k}: {:.3} s, {answered} answered", t0.elapsed().as_secs_f64());
                }
            }
            Ok(hinted)
        })
    }

    /// The count rows (train_counts.rs's) of this call's queries at each level of `levels`, written into the
    /// stream-sized `tot`, `dist` and `cnt` [n, L].
    #[pyo3(signature = (levels, step_lo, step_hi, tot, dist, cnt, rule="step"))]
    #[allow(clippy::too_many_arguments)]
    fn counts(
        &self,
        py: Python<'_>,
        levels: Vec<usize>,
        step_lo: usize,
        step_hi: usize,
        mut tot: PyReadwriteArray2<u32>,
        mut dist: PyReadwriteArray2<u16>,
        mut cnt: PyReadwriteArray2<u32>,
        rule: &str,
    ) -> PyResult<()> {
        let rule = rule_of(rule, step_lo, step_hi)?;
        let n = self.tokens.len();
        let nl = levels.len();
        if levels.is_empty() || levels.contains(&0) {
            return Err(PyValueError::new_err(format!("levels must be >= 1: {levels:?}")));
        }
        if tot.shape() != [n, nl] || dist.shape() != [n, nl] || cnt.shape() != [n, nl] {
            return Err(PyValueError::new_err(format!("outputs must be [{n}, {nl}]")));
        }
        let outs = CountOuts {
            levels: nl,
            tot: Output(tot.as_slice_mut()?.as_mut_ptr()),
            dist: Output(dist.as_slice_mut()?.as_mut_ptr()),
            cnt: Output(cnt.as_slice_mut()?.as_mut_ptr()),
        };
        py.detach(|| {
            let mut guard = self.elems.lock().unwrap();
            let buf = self.buffer(&mut guard)?;
            for (j, &k) in levels.iter().enumerate() {
                let groups = self.partition(buf, k, rule, &|_| true)?;
                let (outs, tokens) = (&outs, &self.tokens);
                self.pool.install(|| {
                    groups.into_par_iter().with_max_len(1).for_each_init(
                        || Counter { counts: vec![0; 1 << 16], touched: Vec::new() },
                        |w, slice| walk_bucket(slice, tokens, k, |group| w.walk(group, tokens, |g| self.has_next(g), j, outs)),
                    )
                });
            }
            Ok(())
        })
    }

    /// Before the clock: allocate the work buffers and fault them in (on a host with fragmented memory, faulting huge
    /// pages on the clock compacts memory and was ~10x slower).
    fn prefault(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| {
            let mut guard = self.elems.lock().unwrap();
            let buf = self.buffer(&mut guard)?;
            self.pool.install(|| {
                crate::prefault(&mut buf.0);
                crate::prefault(&mut buf.1);
            });
            Ok(())
        })
    }

    /// Before the clock: advance()'s state for its fixed configuration (hint `levels` with `buckets` and `entries`, count
    /// `count_levels`; either may be empty), allocated and faulted in, and each level's occurrences ordered by the step
    /// they become available in (from the spans alone). advance() calls it itself if needed.
    #[pyo3(signature = (levels, buckets, entries, count_levels))]
    fn prepare(&self, py: Python<'_>, levels: Vec<usize>, buckets: Vec<usize>, entries: usize, count_levels: Vec<usize>) -> PyResult<()> {
        let mut guard = self.incr.lock().unwrap();
        self.prepared(py, &mut guard, &levels, &buckets, entries, &count_levels)?;
        Ok(())
    }

    /// The incremental path: the rows of the queries trained in steps [done, step_hi), exactly what hints(levels, buckets,
    /// entries, done, step_hi, cell, next, weights) and counts(count_levels, done, step_hi, tot, dist, cnt) (rule "step")
    /// write, with the counted occurrences kept between calls, so a call only adds the occurrences that become available
    /// in its steps and answers its own queries. `done` starts at 0 and becomes step_hi; the configuration is fixed by
    /// the first call (or prepare). Returns the queries hinted.
    #[pyo3(signature = (step_hi, levels, buckets, entries, count_levels, cell=None, next=None, weights=None, tot=None, dist=None, cnt=None))]
    #[allow(clippy::too_many_arguments)]
    fn advance(
        &self,
        py: Python<'_>,
        step_hi: usize,
        levels: Vec<usize>,
        buckets: Vec<usize>,
        entries: usize,
        count_levels: Vec<usize>,
        cell: Option<PyReadwriteArray1<i16>>,
        next: Option<PyReadwriteArray2<u16>>,
        weights: Option<PyReadwriteArray2<u16>>,
        tot: Option<PyReadwriteArray2<u32>>,
        dist: Option<PyReadwriteArray2<u16>>,
        cnt: Option<PyReadwriteArray2<u32>>,
    ) -> PyResult<usize> {
        let n = self.tokens.len();
        let mut guard = self.incr.lock().unwrap();
        self.prepared(py, &mut guard, &levels, &buckets, entries, &count_levels)?;
        let inc = guard.as_mut().unwrap();
        if step_hi < inc.done as usize || step_hi >= 1 << 30 {
            return Err(PyValueError::new_err(format!("step_hi {step_hi} is before the steps done ({})", inc.done)));
        }
        let (mut cell, mut next, mut weights, mut tot, mut dist, mut cnt) = (cell, next, weights, tot, dist, cnt);
        let mut outs = None;
        if !levels.is_empty() {
            let (Some(c), Some(x), Some(w)) = (cell.as_mut(), next.as_mut(), weights.as_mut()) else {
                return Err(PyValueError::new_err("hint levels need cell, next and weights"));
            };
            if c.len() != n || x.shape() != [n, entries] || w.shape() != [n, entries] {
                return Err(PyValueError::new_err(format!("outputs must be [{n}] and [{n}, {entries}]")));
            }
            outs = Some(Outs { cell: Output(c.as_slice_mut()?.as_mut_ptr()), next: Output(x.as_slice_mut()?.as_mut_ptr()), weights: Output(w.as_slice_mut()?.as_mut_ptr()) });
        }
        let mut couts = None;
        if !count_levels.is_empty() {
            let nl = count_levels.len();
            let (Some(t), Some(d), Some(c)) = (tot.as_mut(), dist.as_mut(), cnt.as_mut()) else {
                return Err(PyValueError::new_err("count levels need tot, dist and cnt"));
            };
            if t.shape() != [n, nl] || d.shape() != [n, nl] || c.shape() != [n, nl] {
                return Err(PyValueError::new_err(format!("outputs must be [{n}, {nl}]")));
            }
            couts = Some(CountOuts { levels: nl, tot: Output(t.as_slice_mut()?.as_mut_ptr()), dist: Output(d.as_slice_mut()?.as_mut_ptr()), cnt: Output(c.as_slice_mut()?.as_mut_ptr()) });
        }
        py.detach(|| self.advance_steps(inc, step_hi as u32, outs.as_ref(), couts.as_ref()))
    }

    /// The steps advance() has done.
    #[getter]
    fn done(&self) -> u32 {
        self.incr.lock().unwrap().as_ref().map_or(0, |i| i.done)
    }

    /// The positions in the stream.
    fn __len__(&self) -> usize {
        self.tokens.len()
    }
}

impl OnlineRows {
    /// The element buffer (two per position) and the per-position keys, made once and reused.
    fn buffer<'a>(&self, guard: &'a mut Option<(Map<Elem>, Map<u64>)>) -> PyResult<&'a mut (Map<Elem>, Map<u64>)> {
        if guard.is_none() {
            *guard = Some((Map::new(2 * self.tokens.len())?, Map::new(self.tokens.len())?));
        }
        Ok(guard.as_mut().unwrap())
    }

    /// Level k's elements, in buckets by key (each bucket in stream order): an occurrence for every position that ends
    /// one available within the call's range, a query for every valid position of the call (`query(g)` also).
    fn partition<'a>(
        &self,
        buf: &'a mut (Map<Elem>, Map<u64>),
        k: usize,
        rule: Rule,
        query: &(dyn Fn(usize) -> bool + Sync),
    ) -> PyResult<Vec<&'a mut [Elem]>> {
        let (tokens, seg, tau, tau_in) = (&self.tokens, &self.seg, &self.tau, &self.tau_in);
        let n = tokens.len();
        let pieces = &self.pieces;
        // position g's occurrence and query times (t, not yet shifted), if any
        let times = |g: usize, a: usize, b: usize| -> (Option<u32>, Option<u32>) {
            if g + 1 < a + k {
                return (None, None); // the context leaves the part
            }
            let occ = if g + 1 < b {
                match rule {
                    Rule::Segment => Some(g as u32 + 2),
                    Rule::Step { hi, .. } => {
                        if tau[g + 1] >= hi {
                            None
                        } else {
                            let m = tau[g + 1 - k..=g + 1].iter().copied().max().unwrap();
                            (m < hi - 1).then_some(m + 1) // m < hi, so no overflow (NONE never passes)
                        }
                    }
                }
            } else {
                None
            };
            let q = if g + 1 - seg[g] as usize >= k {
                match rule {
                    Rule::Segment => query(g).then_some(seg[g]),
                    Rule::Step { lo, hi } => {
                        let s = tau_in[g];
                        (s >= lo && s < hi && query(g)).then_some(s)
                    }
                }
            } else {
                None
            };
            (occ, q)
        };
        // a piece without a position trained before the range's end has no element
        let live = |min: u32| match rule {
            Rule::Segment => true,
            Rule::Step { hi: end, .. } => min < end,
        };
        let t0 = std::time::Instant::now();
        let buckets = (2 * n).div_ceil(BUCKET).next_power_of_two();
        let (elems, keys) = (&mut buf.0, &mut buf.1);
        let kout = Output(keys.as_mut_ptr());
        self.pool.install(|| {
            // pass 1: keys and counts per live piece and bucket
            let live: Vec<_> = pieces.iter().copied().filter(|p| live(p.4)).collect();
            let mut offsets: Vec<Vec<u32>> = live
                .par_iter()
                .map(|&(lo, hi, a, b, _)| {
                    let mut c = vec![0u32; buckets];
                    for g in lo..hi {
                        let (o, q) = times(g, a, b);
                        if o.is_none() && q.is_none() {
                            continue;
                        }
                        let key = key64(&tokens[g + 1 - k..=g]);
                        unsafe { kout.set(g, key) };
                        c[bucket64(key, buckets)] += o.is_some() as u32 + q.is_some() as u32;
                    }
                    c
                })
                .collect();
            // each bucket's start, then each piece's cursor in it (per bucket in parallel)
            let columns: Vec<Output<u32>> = offsets.iter_mut().map(|c| Output(c.as_mut_ptr())).collect();
            let totals: Vec<usize> = (0..buckets).into_par_iter().map(|bk| offsets.iter().map(|c| c[bk] as usize).sum()).collect();
            let mut first = vec![0usize; buckets + 1];
            for bk in 0..buckets {
                first[bk + 1] = first[bk] + totals[bk];
            }
            let at = first[buckets];
            (0..buckets).into_par_iter().for_each(|bk| {
                let mut run = first[bk] as u32;
                for col in &columns {
                    unsafe {
                        let c = col.0.add(bk);
                        let m = *c;
                        *c = run;
                        run += m;
                    }
                }
            });
            // pass 2: scatter, pieces in stream order
            let out = Output(elems.as_mut_ptr());
            let keys = &*keys;
            live.par_iter().zip(offsets.par_iter_mut()).for_each(|(&(lo, hi, a, b, _), cursor)| {
                for g in lo..hi {
                    let (o, q) = times(g, a, b);
                    if o.is_none() && q.is_none() {
                        continue;
                    }
                    let key = keys[g];
                    let c = &mut cursor[bucket64(key, buckets)];
                    if let Some(t) = o {
                        unsafe { out.set(*c as usize, Elem { key, time: t << 1, at: g as u32 }) };
                        *c += 1;
                    }
                    if let Some(t) = q {
                        unsafe { out.set(*c as usize, Elem { key, time: t << 1 | 1, at: g as u32 }) };
                        *c += 1;
                    }
                }
            });
            let mut slices = Vec::with_capacity(buckets);
            let mut rest = &mut elems[..at];
            for bk in 0..buckets {
                let (s, r) = rest.split_at_mut(first[bk + 1] - first[bk]);
                if !s.is_empty() {
                    slices.push(s);
                }
                rest = r;
            }
            slices.sort_unstable_by_key(|s| std::cmp::Reverse(s.len()));
            if *VERBOSE {
                eprintln!("onlinerows partition k={k}: {at} elements, {:.3} s", t0.elapsed().as_secs_f64());
            }
            Ok(slices)
        })
    }

    /// Whether position g has a next token in its part.
    fn has_next(&self, g: usize) -> bool {
        let part = self.bounds.partition_point(|&b| b <= g);
        g + 1 < self.bounds[part]
    }
}

/// Sort a bucket and call `walk` on each group of one context, in (time, position) order: a key group whose contexts
/// differ (a hash collision, only for k > PACKED64) is split by context.
fn walk_bucket(slice: &mut [Elem], tokens: &[u16], k: usize, mut walk: impl FnMut(&[Elem])) {
    slice.sort_unstable_by_key(|e| e.key);
    let context = |e: &Elem| &tokens[e.at as usize + 1 - k..=e.at as usize];
    let order = |e: &Elem| (e.time as u64) << 32 | e.at as u64;
    for group in slice.chunk_by_mut(|x, y| x.key == y.key) {
        if !needs_walk(group) {
            continue;
        }
        if k > PACKED64 && group[1..].iter().any(|e| context(e) != context(&group[0])) {
            group.sort_by(|x, y| context(x).cmp(context(y)).then(order(x).cmp(&order(y))));
            for sub in group.chunk_by(|x, y| context(x) == context(y)) {
                if needs_walk(sub) {
                    walk(sub);
                }
            }
        } else {
            group.sort_unstable_by_key(order);
            walk(group);
        }
    }
}

/// Whether a key group needs a walk: two elements at least, one a query.
fn needs_walk(group: &[Elem]) -> bool {
    group.len() >= 2 && group.iter().any(|e| e.time & 1 == 1)
}

struct Outs {
    cell: Output<i16>,
    next: Output<u16>,
    weights: Output<u16>,
}

/// train_hints.rs's Walker, over time-ordered groups.
struct Walker {
    entries: usize,
    counts: Vec<u32>,
    touched: Vec<u16>,
    top: Vec<(u32, u16)>,
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

    /// One group of one context (walk_bucket's), at length bucket li; returns the queries answered.
    fn walk(&mut self, group: &[Elem], tokens: &[u16], li: usize, outs: &Outs, filled: Output<u8>) -> usize {
        for &t in &self.touched {
            self.counts[t as usize] = 0;
        }
        self.touched.clear();
        self.top.clear();
        self.total = 0;
        let mut answered = 0;
        for e in group {
            if e.time & 1 == 0 {
                self.add(tokens[e.at as usize + 1]);
            } else if self.total > 0 {
                self.answer(e.at as usize, li, outs);
                unsafe { filled.set(e.at as usize, 1) };
                answered += 1;
            }
        }
        answered
    }

    /// train_hints.rs's add: count an occurrence followed by `t`, keeping the top `entries` exact.
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
            None if better((c, t), self.top[self.entries - 1]) => self.entries - 1,
            None => return,
        };
        self.top[i] = (c, t);
        while i > 0 && better(self.top[i], self.top[i - 1]) {
            self.top.swap(i, i - 1);
            i -= 1;
        }
    }

    /// train_hints.rs's answer, for position `at`.
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
            self.weights[j] = (c as f64 / sum * WEIGHT_ONE).round_ties_even() as u16;
        }
        unsafe {
            outs.cell.set(at, cell as i16);
            outs.next.write(at * self.entries, &self.row);
            outs.weights.write(at * self.entries, &self.weights);
        }
    }
}

struct CountOuts {
    levels: usize,
    tot: Output<u32>,
    dist: Output<u16>,
    cnt: Output<u32>,
}

/// train_counts.rs's Counter, over time-ordered groups.
struct Counter {
    counts: Vec<u32>,
    touched: Vec<u16>,
}

impl Counter {
    /// One group of one context (walk_bucket's), at level index j.
    fn walk(&mut self, group: &[Elem], tokens: &[u16], has_next: impl Fn(usize) -> bool, j: usize, outs: &CountOuts) {
        for &t in &self.touched {
            self.counts[t as usize] = 0;
        }
        self.touched.clear();
        let mut total = 0u32;
        for e in group {
            if e.time & 1 == 0 {
                let t = tokens[e.at as usize + 1];
                let c = &mut self.counts[t as usize];
                if *c == 0 {
                    self.touched.push(t);
                }
                *c += 1;
                total += 1;
            } else if total > 0 {
                let at = e.at as usize * outs.levels + j;
                let g = e.at as usize;
                let own = if has_next(g) { self.counts[tokens[g + 1] as usize] } else { 0 };
                unsafe {
                    outs.tot.set(at, total);
                    outs.dist.set(at, self.touched.len().min(u16::MAX as usize) as u16);
                    outs.cnt.set(at, own);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// The incremental path (advance): the same rows as hints() + counts() under the step rule, with each level's counted
// occurrences kept across calls in per-bucket hash tables, so a call costs its own steps' positions, not the prefix's.
//
// Per level, a position's key goes to bucket bucket64(key, nb) (nb fixed), and each bucket's state is touched by one task
// at a time. A call's elements are made in time order (per step s of the call: the occurrences that become available at
// s, alpha = s, then the queries of step s) and scattered stably into the buckets, so each bucket walks its elements in
// time order: an occurrence adds its next token to its context's counts, a query reads them (all occurrences with
// alpha <= s, the batch path's rule). A context is identified by its 64-bit key and a 32-bit check hash of its tokens
// (keys of up to PACKED64 tokens are the tokens themselves): rows differ from the batch path's (which splits a key
// group by the actual tokens) only if two contexts share both, ~2^-96 per pair.

/// One element of a call at one level: key, check, and `w`: an occurrence's next token, or QUERY | the query's position
/// (with HAS_NEXT if its next token lies in its part).
#[derive(Clone, Copy)]
#[repr(C)]
struct IElem {
    key: u64,
    check: u32,
    w: u32,
}

const QUERY: u32 = 1 << 31;
const HAS_NEXT: u32 = 1 << 30;

/// A context's check hash (0 for packed keys, which are exact).
#[inline]
fn check32(context: &[u16]) -> u32 {
    if context.len() <= PACKED64 {
        return 0;
    }
    let mut a = 0x13198a2e03707344u64 ^ (context.len() as u64) << 32;
    for word in context.chunks(4) {
        a = mix(a.rotate_left(29) ^ word.iter().fold(0u64, |w, &t| w << 16 | t as u64).wrapping_add(0xa4093822299f31d0));
    }
    (a >> 32) as u32
}

/// An open-addressing (linear probing) table from u64 keys to u32 values (NONE marks an empty slot).
struct Table {
    keys: Vec<u64>,
    vals: Vec<u32>,
    len: usize,
    mask: usize,
}

impl Table {
    /// Room for `entries` at load 0.7, faulted in.
    fn new(entries: usize) -> Self {
        let cap = (entries.max(8) * 10 / 7).next_power_of_two();
        let mut keys = vec![0u64; cap];
        crate::touch(&mut keys);
        Self { keys, vals: vec![NONE; cap], len: 0, mask: cap - 1 }
    }

    #[inline]
    fn home(&self, key: u64) -> usize {
        mix(key.wrapping_add(0x632be59bd9b4e019)) as usize & self.mask
    }

    #[inline]
    fn get(&self, key: u64) -> Option<u32> {
        let mut i = self.home(key);
        loop {
            let v = self.vals[i];
            if v == NONE {
                return None;
            }
            if self.keys[i] == key {
                return Some(v);
            }
            i = (i + 1) & self.mask;
        }
    }

    /// The slot of `key`, inserted (value unset) if new; and whether it is new.
    #[inline]
    fn slot(&mut self, key: u64, fresh: u32) -> (usize, bool) {
        if (self.len + 1) * 10 > (self.mask + 1) * 7 {
            self.grow();
        }
        let mut i = self.home(key);
        loop {
            if self.vals[i] == NONE {
                self.keys[i] = key;
                self.vals[i] = fresh;
                self.len += 1;
                return (i, true);
            }
            if self.keys[i] == key {
                return (i, false);
            }
            i = (i + 1) & self.mask;
        }
    }

    fn grow(&mut self) {
        let cap = 2 * (self.mask + 1);
        let keys = std::mem::replace(&mut self.keys, vec![0; cap]);
        let vals = std::mem::replace(&mut self.vals, vec![NONE; cap]);
        self.mask = cap - 1;
        for (k, v) in keys.into_iter().zip(vals) {
            if v != NONE {
                let mut i = self.home(k);
                while self.vals[i] != NONE {
                    i = (i + 1) & self.mask;
                }
                self.keys[i] = k;
                self.vals[i] = v;
            }
        }
    }
}

/// One bucket's counted occurrences at one level: its contexts (by key, then check), per context (Ctx) its check, the
/// next context with the same key, its total and distinct next tokens and its first next token. A context with one
/// distinct next token so far needs nothing else (its counts are total at `first`); from its second, its (context, next
/// token) counts are in `pairs`, and (hint levels) its top `entries` next tokens in `pool` (at top * entries), packed
/// count << 16 | (65535 - token) (larger is better; 0 empty).
struct BucketState {
    ctx: Table,
    pairs: Table,
    ctxs: Vec<Ctx>,
    pool: Vec<u64>,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct Ctx {
    check: u32,
    chain: u32,
    total: u32,
    distinct: u32,
    top: u32,
    first: u16,
}

#[inline(always)]
fn prefetch<T>(p: *const T) {
    unsafe { std::arch::x86_64::_mm_prefetch(p as *const i8, std::arch::x86_64::_MM_HINT_T0) };
}

impl BucketState {
    fn new(contexts: usize, entries: usize) -> Self {
        let mut ctxs: Vec<Ctx> = Vec::with_capacity(contexts);
        crate::touch(ctxs.spare_capacity_mut());
        let mut pool: Vec<u64> = Vec::with_capacity(contexts / 4 * entries);
        crate::touch(pool.spare_capacity_mut());
        Self { ctx: Table::new(contexts), pairs: Table::new(contexts / 2), ctxs, pool }
    }

    #[inline]
    fn find(&self, key: u64, check: u32) -> Option<usize> {
        let mut i = self.ctx.get(key)?;
        loop {
            let c = &self.ctxs[i as usize];
            if c.check == check {
                return Some(i as usize);
            }
            i = c.chain;
            if i == NONE {
                return None;
            }
        }
    }

    /// Prefetches (hints: a stale guess is harmless) for an element of context `key`: its table slot (stage 1), its
    /// context (stage 2), its pair count and top entries if it has more than one next token (stage 3).
    #[inline]
    fn prefetch_slot(&self, key: u64) {
        let i = self.ctx.home(key);
        prefetch(self.ctx.vals.as_ptr().wrapping_add(i));
        prefetch(self.ctx.keys.as_ptr().wrapping_add(i));
    }

    #[inline]
    fn guess(&self, key: u64) -> Option<usize> {
        let i = self.ctx.home(key);
        let v = self.ctx.vals[i];
        (v != NONE && self.ctx.keys[i] == key).then_some(v as usize)
    }

    #[inline]
    fn prefetch_ctx(&self, key: u64) {
        if let Some(v) = self.guess(key) {
            prefetch(self.ctxs.as_ptr().wrapping_add(v));
        }
    }

    #[inline]
    fn prefetch_rest(&self, key: u64, next: u16, entries: usize) {
        if let Some(v) = self.guess(key) {
            let c = &self.ctxs[v];
            if c.distinct > 1 {
                let j = self.pairs.home((v as u64) << 16 | next as u64);
                prefetch(self.pairs.vals.as_ptr().wrapping_add(j));
                prefetch(self.pairs.keys.as_ptr().wrapping_add(j));
                if entries > 0 {
                    prefetch(self.pool.as_ptr().wrapping_add(c.top as usize * entries));
                }
            }
        }
    }

    #[inline]
    fn find_or_insert(&mut self, key: u64, check: u32) -> usize {
        let (slot, new) = self.ctx.slot(key, 0);
        let first = self.ctx.vals[slot];
        if !new {
            let mut i = first;
            loop {
                let c = &self.ctxs[i as usize];
                if c.check == check {
                    return i as usize;
                }
                i = c.chain;
                if i == NONE {
                    break;
                }
            }
        }
        let idx = self.ctxs.len();
        self.ctxs.push(Ctx { check, chain: if new { NONE } else { first }, total: 0, distinct: 0, top: NONE, first: 0 });
        self.ctx.vals[slot] = idx as u32;
        idx
    }

    /// The count of next token t after context idx.
    #[inline]
    fn count(&self, idx: usize, t: u16) -> u32 {
        let c = &self.ctxs[idx];
        if c.distinct == 1 {
            return if c.first == t { c.total } else { 0 };
        }
        self.pairs.get((idx as u64) << 16 | t as u64).unwrap_or(0)
    }

    /// Count an occurrence of context idx followed by t (Walker::add's top rule when `entries` > 0).
    #[inline]
    fn add(&mut self, idx: usize, t: u16, entries: usize) {
        let ctx = self.ctxs[idx];
        if ctx.total == 0 || (ctx.distinct == 1 && ctx.first == t) {
            let c = &mut self.ctxs[idx];
            c.total += 1;
            c.distinct = 1;
            c.first = t;
            return;
        }
        if ctx.distinct == 1 {
            // its second distinct next token: the first one's count into pairs, its top into the pool
            let (slot, _) = self.pairs.slot((idx as u64) << 16 | ctx.first as u64, 0);
            self.pairs.vals[slot] = ctx.total;
            if entries > 0 {
                let top = (self.pool.len() / entries) as u32;
                self.pool.extend(std::iter::repeat_n(0, entries));
                self.pool[top as usize * entries] = (ctx.total as u64) << 16 | (65535 - ctx.first as u64);
                self.ctxs[idx].top = top;
            }
        }
        let (slot, _) = self.pairs.slot((idx as u64) << 16 | t as u64, 0);
        self.pairs.vals[slot] += 1;
        let c = self.pairs.vals[slot] as u64;
        let ctx = &mut self.ctxs[idx];
        ctx.total += 1;
        if c == 1 {
            ctx.distinct += 1;
        }
        if entries == 0 {
            return;
        }
        let at = ctx.top as usize * entries;
        let top = &mut self.pool[at..at + entries];
        let low = 65535 - t as u64;
        let v = c << 16 | low;
        let len = top.iter().take_while(|&&x| x != 0).count();
        let mut i = match top[..len].iter().position(|&x| x & 0xffff == low) {
            Some(i) => i,
            None if len < entries => len,
            None if v > top[entries - 1] => entries - 1,
            None => return,
        };
        top[i] = v;
        while i > 0 && top[i] > top[i - 1] {
            top.swap(i, i - 1);
            i -= 1;
        }
    }

    /// Context idx's top entries (packed), in `buf` for a context with one next token.
    #[inline]
    fn top<'a>(&'a self, idx: usize, entries: usize, buf: &'a mut [u64; 256]) -> &'a [u64] {
        let c = &self.ctxs[idx];
        if c.distinct == 1 {
            buf[..entries].fill(0);
            buf[0] = (c.total as u64) << 16 | (65535 - c.first as u64);
            return &buf[..entries];
        }
        &self.pool[c.top as usize * entries..(c.top as usize + 1) * entries]
    }
}

/// advance()'s state.
struct Incr {
    hint_levels: Vec<usize>,
    buckets: Vec<usize>,
    entries: usize,
    count_levels: Vec<usize>,
    levels: Vec<(usize, Option<usize>, Option<usize>)>, // per level (descending): k, its length bucket (hints), its count column
    nb: usize,
    states: Vec<Vec<Mutex<BucketState>>>,
    occ: Vec<Vec<u32>>,      // per level, the positions with an occurrence, by alpha (then position)
    occ_off: Vec<Vec<usize>>, // per level, each alpha's start in occ (alpha 0 ..= amax + 1)
    queries: Vec<u32>,       // the input positions, by tau_in (then position)
    q_off: Vec<usize>,       // each step's start in queries
    filled: Vec<u8>,
    keys: Map<u64>,
    checks: Map<u32>,
    elems: Map<IElem>,
    done: u32,
}

/// Positions with a value of `of` (NONE: none) below `limit`, sorted by it (then position), and each value's start;
/// counted per piece, in parallel.
fn sort_by_value(pieces: &[(usize, usize)], limit: usize, of: &(dyn Fn(usize) -> u32 + Sync)) -> (Vec<u32>, Vec<usize>) {
    let mut counts: Vec<Vec<u32>> = pieces
        .par_iter()
        .map(|&(lo, hi)| {
            let mut c = vec![0u32; limit];
            for g in lo..hi {
                let v = of(g);
                if v != NONE {
                    c[v as usize] += 1;
                }
            }
            c
        })
        .collect();
    let mut off = vec![0usize; limit + 1];
    let mut at = 0usize;
    for v in 0..limit {
        off[v] = at;
        for c in counts.iter_mut() {
            let m = c[v] as usize;
            c[v] = at as u32;
            at += m;
        }
    }
    off[limit] = at;
    let mut out = vec![0u32; at];
    let o = Output(out.as_mut_ptr());
    pieces.par_iter().zip(counts.par_iter_mut()).for_each(|(&(lo, hi), cursor)| {
        for g in lo..hi {
            let v = of(g);
            if v != NONE {
                let c = &mut cursor[v as usize];
                unsafe { o.set(*c as usize, g as u32) };
                *c += 1;
            }
        }
    });
    (out, off)
}

impl OnlineRows {
    /// advance()'s state in `guard`, made (prepare's work) if missing; an error if made for another configuration.
    fn prepared(&self, py: Python<'_>, guard: &mut Option<Incr>, levels: &[usize], buckets: &[usize], entries: usize, count_levels: &[usize]) -> PyResult<()> {
        if let Some(inc) = guard.as_ref() {
            if inc.hint_levels != levels || inc.buckets != buckets || inc.entries != entries || inc.count_levels != count_levels {
                return Err(PyValueError::new_err("advance's levels, buckets, entries and count levels are fixed by its first call"));
            }
            return Ok(());
        }
        if !levels.is_empty() {
            if entries == 0 || entries > 256 || buckets.is_empty() || buckets.len() > 127 || buckets[0] == 0 || !buckets.windows(2).all(|w| w[0] < w[1]) {
                return Err(PyValueError::new_err(format!("need 1 <= entries <= 256 and buckets ascending from 1: {entries}, {buckets:?}")));
            }
            if !levels.iter().all(|k| buckets.contains(k)) || !levels.windows(2).all(|w| w[0] < w[1]) {
                return Err(PyValueError::new_err(format!("levels must ascend and be of {buckets:?}: {levels:?}")));
            }
        }
        let mut distinct = count_levels.to_vec();
        distinct.sort_unstable();
        distinct.dedup();
        if count_levels.contains(&0) || distinct.len() != count_levels.len() {
            return Err(PyValueError::new_err(format!("count levels must be distinct and >= 1: {count_levels:?}")));
        }
        if levels.is_empty() && count_levels.is_empty() {
            return Err(PyValueError::new_err("no levels"));
        }
        let inc = py.detach(|| self.pool.install(|| self.make_incr(levels, buckets, entries, count_levels)))?;
        *guard = Some(inc);
        Ok(())
    }

    fn make_incr(&self, levels: &[usize], buckets: &[usize], entries: usize, count_levels: &[usize]) -> PyResult<Incr> {
        let t0 = std::time::Instant::now();
        let (tokens, tau, tau_in, seg) = (&self.tokens, &self.tau, &self.tau_in, &self.seg);
        let _ = seg;
        let n = tokens.len();
        let mut all: Vec<usize> = levels.iter().chain(count_levels).copied().collect();
        all.sort_unstable_by(|a, b| b.cmp(a));
        all.dedup();
        let table: Vec<(usize, Option<usize>, Option<usize>)> = all
            .iter()
            .map(|&k| {
                let li = levels.contains(&k).then(|| buckets.iter().position(|&l| l == k).unwrap());
                (k, li, count_levels.iter().position(|&c| c == k))
            })
            .collect();
        let smax = tau.par_iter().copied().filter(|&t| t != NONE).max().unwrap_or(0) as usize;
        let limit = smax + 2; // alpha <= smax + 1
        let pieces: Vec<(usize, usize, usize, usize)> = self.pieces.iter().map(|p| (p.0, p.1, p.2, p.3)).collect();
        let plain: Vec<(usize, usize)> = pieces.iter().map(|p| (p.0, p.1)).collect();
        let (queries, q_off) = sort_by_value(&plain, limit, &|g| tau_in[g]);
        let part_of = |g: usize| pieces[pieces.partition_point(|p| p.1 <= g)];
        let mut occ = Vec::new();
        let mut occ_off = Vec::new();
        for &(k, _, _) in &table {
            let alpha = |g: usize| -> u32 {
                let (_, _, a, b) = part_of(g);
                if g + 1 < a + k || g + 1 >= b {
                    return NONE;
                }
                let m = tau[g + 1 - k..=g + 1].iter().copied().max().unwrap();
                if m == NONE {
                    NONE
                } else {
                    m + 1
                }
            };
            let (o, off) = sort_by_value(&plain, limit, &alpha);
            occ.push(o);
            occ_off.push(off);
        }
        let nb = (n / 65536).next_power_of_two().clamp(64, 1 << 14);
        let per = n / nb + n / nb / 20 + 64;
        let states: Vec<Vec<Mutex<BucketState>>> = table
            .iter()
            .map(|&(_, li, _)| {
                let e = if li.is_some() { entries } else { 0 };
                (0..nb).into_par_iter().map(|_| Mutex::new(BucketState::new(per, e))).collect()
            })
            .collect();
        let mut filled = vec![0u8; n];
        crate::touch(&mut filled);
        let (mut keys, mut checks, mut elems) = (Map::<u64>::new(n)?, Map::<u32>::new(n)?, Map::<IElem>::new(2 * n)?);
        crate::prefault(&mut keys);
        crate::prefault(&mut checks);
        crate::prefault(&mut elems);
        if *VERBOSE {
            eprintln!("onlinerows prepare: {} levels, {nb} buckets, {per} contexts each, {:.2} s", table.len(), t0.elapsed().as_secs_f64());
        }
        Ok(Incr {
            hint_levels: levels.to_vec(),
            buckets: buckets.to_vec(),
            entries,
            count_levels: count_levels.to_vec(),
            levels: table,
            nb,
            states,
            occ,
            occ_off,
            queries,
            q_off,
            filled,
            keys,
            checks,
            elems,
            done: 0,
        })
    }

    /// advance()'s work for steps [inc.done, hi).
    fn advance_steps(&self, inc: &mut Incr, hi: u32, outs: Option<&Outs>, couts: Option<&CountOuts>) -> PyResult<usize> {
        let lo = inc.done;
        if hi <= lo {
            return Ok(0);
        }
        let (tokens, seg) = (&self.tokens, &self.seg);
        let nb = inc.nb;
        let entries = inc.entries;
        let mut hinted = 0usize;
        let filled = Output(inc.filled.as_mut_ptr());
        let kout = Output(inc.keys.as_mut_ptr());
        let cout = Output(inc.checks.as_mut_ptr());
        let eout = Output(inc.elems.as_mut_ptr());
        for (lv, &(k, li, cj)) in inc.levels.iter().enumerate() {
            let t0 = std::time::Instant::now();
            // this call's sources in time order: per step, its occurrences (alpha = s), then its queries
            let (occ, off) = (&inc.occ[lv], &inc.occ_off[lv]);
            let mut segs: Vec<(&[u32], bool)> = Vec::new();
            for s in lo as usize..hi as usize {
                if s + 1 < off.len() {
                    segs.push((&occ[off[s]..off[s + 1]], false));
                }
                if s + 1 < inc.q_off.len() {
                    segs.push((&inc.queries[inc.q_off[s]..inc.q_off[s + 1]], true));
                }
            }
            let starts: Vec<usize> = segs.iter().scan(0, |a, s| { let x = *a; *a += s.0.len(); Some(x) }).collect();
            let total: usize = segs.iter().map(|s| s.0.len()).sum();
            if total == 0 {
                continue;
            }
            let chunk = (total / (4 * self.pool.current_num_threads())).max(1 << 14);
            let chunks: Vec<(usize, usize)> = (0..total).step_by(chunk).map(|a| (a, total.min(a + chunk))).collect();
            let f = &inc.filled;
            // the element of virtual index i, if any: (position, is query)
            let visit = |a: usize, b: usize, mut each: &mut dyn FnMut(usize, bool)| {
                let mut si = starts.partition_point(|&x| x <= a) - 1;
                let mut i = a;
                while i < b {
                    while starts[si] + segs[si].0.len() <= i {
                        si += 1;
                    }
                    let (list, query) = segs[si];
                    let end = b.min(starts[si] + list.len());
                    for &g in &list[i - starts[si]..end - starts[si]] {
                        each(g as usize, query);
                    }
                    i = end;
                }
                let _ = &mut each;
            };
            let keep = |g: usize, query: bool| -> bool {
                !query || (g + 1 - seg[g] as usize >= k && (cj.is_some() || (li.is_some() && f[g] == 0)))
            };
            let mut counts: Vec<Vec<u32>> = self.pool.install(|| {
                chunks
                    .par_iter()
                    .map(|&(a, b)| {
                        let mut c = vec![0u32; nb];
                        visit(a, b, &mut |g, query| {
                            if !keep(g, query) {
                                return;
                            }
                            let ctx = &tokens[g + 1 - k..=g];
                            let key = key64(ctx);
                            unsafe {
                                kout.set(g, key);
                                cout.set(g, check32(ctx));
                            }
                            c[bucket64(key, nb)] += 1;
                        });
                        c
                    })
                    .collect()
            });
            // each bucket's start, then each chunk's cursor in it (per bucket in parallel)
            let sizes: Vec<usize> = self.pool.install(|| (0..nb).into_par_iter().map(|bk| counts.iter().map(|c| c[bk] as usize).sum()).collect());
            let mut first = vec![0usize; nb + 1];
            for bk in 0..nb {
                first[bk + 1] = first[bk] + sizes[bk];
            }
            let at = first[nb];
            let columns: Vec<Output<u32>> = counts.iter_mut().map(|c| Output(c.as_mut_ptr())).collect();
            self.pool.install(|| {
                (0..nb).into_par_iter().for_each(|bk| {
                    let mut run = first[bk] as u32;
                    for col in &columns {
                        unsafe {
                            let c = col.0.add(bk);
                            let m = *c;
                            *c = run;
                            run += m;
                        }
                    }
                })
            });
            let t1 = t0.elapsed().as_secs_f64();
            let (keys, checks) = (&inc.keys, &inc.checks);
            self.pool.install(|| {
                chunks.par_iter().zip(counts.par_iter_mut()).for_each(|(&(a, b), cursor)| {
                    visit(a, b, &mut |g, query| {
                        if !keep(g, query) {
                            return;
                        }
                        let key = keys[g];
                        let c = &mut cursor[bucket64(key, nb)];
                        let w = if query { QUERY | g as u32 | if self.has_next(g) { HAS_NEXT } else { 0 } } else { tokens[g + 1] as u32 };
                        unsafe { eout.set(*c as usize, IElem { key, check: checks[g], w }) };
                        *c += 1;
                    });
                });
            });
            let t2 = t0.elapsed().as_secs_f64();
            let elems = &inc.elems[..at];
            let mut order: Vec<usize> = (0..nb).filter(|&bk| first[bk + 1] > first[bk]).collect();
            order.sort_unstable_by_key(|&bk| std::cmp::Reverse(first[bk + 1] - first[bk]));
            let states = &inc.states[lv];
            let e = if li.is_some() { entries } else { 0 };
            let answered: usize = self.pool.install(|| {
                order
                    .par_iter()
                    .with_max_len(1)
                    .map(|&bk| {
                        let mut st = states[bk].lock().unwrap();
                        let mut answered = 0;
                        let els = &elems[first[bk]..first[bk + 1]];
                        const D: usize = 4;
                        let mut buf = [0u64; 256];
                        let nexts = |el: &IElem| if el.w & QUERY == 0 { el.w as u16 } else { tokens[((el.w & !(QUERY | HAS_NEXT)) as usize + 1).min(tokens.len() - 1)] };
                        for (i, el) in els.iter().enumerate() {
                            if let Some(f) = els.get(i + 4 * D) {
                                st.prefetch_slot(f.key);
                                if f.w & QUERY != 0 {
                                    prefetch(tokens.as_ptr().wrapping_add((f.w & !(QUERY | HAS_NEXT)) as usize + 1));
                                }
                            }
                            if let Some(f) = els.get(i + 2 * D) {
                                st.prefetch_ctx(f.key);
                            }
                            if let Some(f) = els.get(i + D) {
                                st.prefetch_rest(f.key, nexts(f), e);
                            }
                            if el.w & QUERY == 0 {
                                let idx = st.find_or_insert(el.key, el.check);
                                st.add(idx, el.w as u16, e);
                                continue;
                            }
                            let g = (el.w & !(QUERY | HAS_NEXT)) as usize;
                            let Some(idx) = st.find(el.key, el.check) else { continue };
                            let ctx = st.ctxs[idx];
                            if let (Some(j), Some(co)) = (cj, couts) {
                                let own = if el.w & HAS_NEXT != 0 { st.count(idx, tokens[g + 1]) } else { 0 };
                                let at = g * co.levels + j;
                                unsafe {
                                    co.tot.set(at, ctx.total);
                                    co.dist.set(at, ctx.distinct.min(u16::MAX as u32) as u16);
                                    co.cnt.set(at, own);
                                }
                            }
                            if let (Some(li), Some(o)) = (li, outs) {
                                if unsafe { *filled.0.add(g) } == 0 {
                                    answer_top(st.top(idx, entries, &mut buf), ctx.total as u64, g, li, entries, o);
                                    unsafe { filled.set(g, 1) };
                                    answered += 1;
                                }
                            }
                        }
                        answered
                    })
                    .sum()
            });
            hinted += answered;
            if *VERBOSE {
                eprintln!("onlinerows advance k={k} steps {lo}-{hi}: {at} elements, {answered} hinted, {:.3} s (count {t1:.3}, scatter {:.3}, walk {:.3})", t0.elapsed().as_secs_f64(), t2 - t1, t0.elapsed().as_secs_f64() - t2);
            }
        }
        inc.done = hi;
        Ok(hinted)
    }
}

/// Walker::answer from a context's packed top entries: the row of position `at`.
fn answer_top(top: &[u64], total: u64, at: usize, li: usize, entries: usize, outs: &Outs) {
    let top1 = top[0] >> 16;
    let count = COUNTS.iter().filter(|&&c| c as u64 <= total).count();
    let purity = PURITIES.iter().filter(|&&p| p <= top1 as f64 / total as f64).count();
    let cell = ((1 + li) * (COUNTS.len() + 1) + count) * (PURITIES.len() + 1) + purity;
    let sum = top.iter().filter(|&&v| v != 0).map(|&v| v >> 16).sum::<u64>() as f64;
    let mut row = [0u16; 256];
    let mut weights = [0u16; 256];
    for (j, &v) in top.iter().enumerate().take_while(|&(_, &v)| v != 0) {
        row[j] = 65535 - (v & 0xffff) as u16;
        weights[j] = ((v >> 16) as f64 / sum * WEIGHT_ONE).round_ties_even() as u16;
    }
    unsafe {
        outs.cell.set(at, cell as i16);
        outs.next.write(at * entries, &row[..entries]);
        outs.weights.write(at * entries, &weights[..entries]);
    }
}
