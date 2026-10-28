//! Exact, causal n-gram counts of every training position at every level, the training count rows of the old-records
//! experiment's n-gram mixing (retrieval_blog_post/old_records.py's TrainCountRows; hint_runtime.py's NgramMix). The
//! validation counts come from kgram.rs's (val_hints.py's ValContextCounts); these have train_hints.rs's definitions
//! and causality: the stream is the shards in order, a segment starts at each shard's first token and at every BOS, a
//! position's k-context is the k tokens ending at it if they lie in its segment, and an occurrence (any k tokens within
//! one shard and the token after them) counts for a position iff it ends, next token included, before the position's
//! segment starts.
//!
//! Per level, train_hints.rs's partition into key groups, then one walk per group in stream order: before each valid
//! query, every occurrence that ends before its segment starts is counted, and the query gets the total, the distinct
//! next tokens and the count of its own next token (its target) among them.
use crate::train_hints::{bucket, key, read, segment_starts, Elem, BUCKET, NONE, PACKED};
use crate::{Map, Output};
use numpy::ndarray::Array2;
use numpy::IntoPyArray;
use pyo3::{exceptions::PyValueError, prelude::*};
use rayon::prelude::*;
use std::time::Instant;

/// Per position of the shards `files` (in order) and level k of `levels` (distinct, >= 1), over the occurrences of its
/// k-context that end before its segment starts: tot (uint32 [N, L], their number), dist (uint16 [N, L], their distinct
/// next tokens) and cnt (uint32 [N, L], those followed by the position's own next token; 0 for a shard's last
/// position, which has none). All 0 where the k-context does not lie in the position's segment. 10 bytes per position
/// and level. Exact for contexts of up to 8 tokens; longer ones are keyed by a 128-bit hash, and a collision within a
/// group with a query is an error.
#[pyfunction]
#[pyo3(signature = (files, levels))]
#[allow(clippy::type_complexity)]
pub fn train_counts<'py>(
    py: Python<'py>,
    files: Vec<String>,
    levels: Vec<usize>,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let mut sorted = levels.clone();
    sorted.sort_unstable();
    sorted.dedup();
    if files.is_empty() || levels.is_empty() || levels[0] == 0 || sorted.len() != levels.len() || sorted[0] == 0 {
        return Err(PyValueError::new_err(format!("need files and distinct levels >= 1: {} files, levels {levels:?}", files.len())));
    }
    let (tot, dist, cnt) = py.detach(|| run(&files, &levels))?;
    let n = tot.len() / levels.len();
    let l = levels.len();
    Ok((
        Array2::from_shape_vec((n, l), tot).expect("rows of levels").into_pyarray(py).into_any(),
        Array2::from_shape_vec((n, l), dist).expect("rows of levels").into_pyarray(py).into_any(),
        Array2::from_shape_vec((n, l), cnt).expect("rows of levels").into_pyarray(py).into_any(),
    ))
}

/// The output arrays, [N, L] row-major, which the walkers write disjoint cells of.
struct Outs {
    levels: usize,
    tot: Output<u32>,
    dist: Output<u16>,
    cnt: Output<u32>,
}

#[allow(clippy::type_complexity)]
fn run(files: &[String], levels: &[usize]) -> PyResult<(Vec<u32>, Vec<u16>, Vec<u32>)> {
    let t0 = Instant::now();
    let (tokens, bounds) = read(files)?;
    let n = tokens.len();
    let start = segment_starts(&tokens, &bounds);
    let nl = levels.len();
    let (mut tot, mut dist, mut cnt) = (vec![0u32; n * nl], vec![0u16; n * nl], vec![0u32; n * nl]);
    eprintln!("train_counts: {n} positions in {} shards read, {:.1} s", files.len(), t0.elapsed().as_secs_f64());
    let piece = (n / (8 * rayon::current_num_threads())).clamp(1 << 14, 1 << 20);
    let mut pieces = Vec::new();
    for w in bounds.windows(2) {
        for lo in (w[0]..w[1]).step_by(piece) {
            pieces.push((lo, w[1].min(lo + piece), w[0], w[1]));
        }
    }
    let buckets = n.div_ceil(BUCKET).next_power_of_two();
    let mut elems: Map<Elem> = Map::new(n)?;
    for (j, &k) in levels.iter().enumerate() {
        let t = Instant::now();
        // the position's element at this level, if it is an occurrence or a valid query
        let element = |g: usize, shard: (usize, usize)| -> Option<Elem> {
            if g + 1 < shard.0 + k {
                return None; // the context leaves the shard, so it is no valid query either
            }
            let next = if g + 1 < shard.1 { tokens[g + 1] as u32 } else { NONE };
            let s = start[g];
            let query = if g + 1 - s as usize >= k { s } else { NONE };
            if next == NONE && query == NONE {
                return None;
            }
            Some(Elem { key: key(&tokens[g + 1 - k..=g]), at: g as u32, start: query, next, slot: 0 })
        };
        // count each piece's elements per bucket, then scatter them, pieces in stream order
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
        let mut slices = Vec::with_capacity(buckets);
        let mut rest = &mut elems[..total];
        for b in 0..buckets {
            let (s, r) = rest.split_at_mut(first[b + 1] - first[b]);
            slices.push(s);
            rest = r;
        }
        slices.sort_unstable_by_key(|s| std::cmp::Reverse(s.len()));
        let outs = Outs { levels: nl, tot: Output(tot.as_mut_ptr()), dist: Output(dist.as_mut_ptr()), cnt: Output(cnt.as_mut_ptr()) };
        let (tokens, outs) = (&tokens, &outs);
        slices.into_par_iter().with_max_len(1).try_for_each_init(
            || Counter { counts: vec![0; 1 << 16], touched: Vec::new() },
            |w, slice| -> PyResult<()> {
                slice.sort_unstable_by_key(|e| (e.key, e.at));
                for group in slice.chunk_by(|a, b| a.key == b.key) {
                    w.walk(group, tokens, k, j, outs)?;
                }
                Ok(())
            },
        )?;
        eprintln!("train_counts: k={k}: {total} positions in {buckets} buckets, {:.1} s", t.elapsed().as_secs_f64());
    }
    Ok((tot, dist, cnt))
}

/// One thread's state for walking key groups: the current group's counted occurrences per next token.
struct Counter {
    counts: Vec<u32>,
    touched: Vec<u16>,
}

impl Counter {
    /// One key group of level k (index j of the levels), in stream order.
    fn walk(&mut self, group: &[Elem], tokens: &[u16], k: usize, j: usize, outs: &Outs) -> PyResult<()> {
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
        let mut total = 0u32;
        let mut next = 0; // the first occurrence not yet counted
        for e in group {
            if e.start == NONE {
                continue;
            }
            // stops before e itself, which starts at or after its segment's start
            while (group[next].at as u64) + 1 < e.start as u64 {
                let t = group[next].next;
                if t != NONE {
                    let c = &mut self.counts[t as usize];
                    if *c == 0 {
                        self.touched.push(t as u16);
                    }
                    *c += 1;
                    total += 1;
                }
                next += 1;
            }
            if total > 0 {
                let at = e.at as usize * outs.levels + j;
                let own = if e.next != NONE { self.counts[e.next as usize] } else { 0 };
                unsafe {
                    outs.tot.set(at, total);
                    outs.dist.set(at, self.touched.len().min(u16::MAX as usize) as u16);
                    outs.cnt.set(at, own);
                }
            }
        }
        Ok(())
    }
}
