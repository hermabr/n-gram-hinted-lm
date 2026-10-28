//! Exact k-gram statistics of validation positions against training shards, for the k-gram figure of the blog post
//! (retrieval_blog_post/kgram.py), and the exact next-token counts behind the precomputed validation hints
//! (retrieval_blog_post/val_hints.py); analysis only, nothing here runs in training. A port of ExactVal (modded-nanogpt,
//! branch usefulness-top4), which made the figure's 10.3B-token version, with the next token's rank up to any depth
//! and with bounded memory.
//!
//! A validation position's k-context is the k tokens ending at its input, valid if they lie in its segment (segments
//! start at the first token and at every BOS). A training occurrence is any k consecutive tokens within one shard and
//! the token after them. A context's next tokens are ranked by count, ties to the lower token id. For one k:
//!   1. hash the distinct valid validation contexts into a table (a 64-bit hash, every hit then compared token by
//!      token, so the counts are exact);
//!   2. stream the corpus in parallel and emit (context id, next token) for every hit. A corpus position can only
//!      match at k if it matched at every smaller k, so given the bitmap of a smaller k's hits only those are probed.
//!      The context ids are split into rounds whose occurrences fit `memory`; rounds after the first probe only the
//!      first round's hits;
//!   3. per range of context ids, sort the occurrences and read each context's count and top next tokens off them;
//!   4. find each validation position's next token among its context's top next tokens (kgram_stats), or return the
//!      counts (kgram_counts).
use crate::{open_shard, Output, BOS, HEADER_BYTES};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::{exceptions::PyValueError, prelude::*};
use rayon::prelude::*;
use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
use std::os::fd::AsRawFd;
use std::sync::Mutex;

const HEADER: usize = HEADER_BYTES / 2; // in tokens
const P: u64 = 0x9e3779b185ebca87;
const PIECE: usize = 1 << 22; // corpus positions per parallel piece (a multiple of 64: pieces own whole bitmap words)
const BATCH: usize = 32; // probes in flight per thread

/// Statistics of the validation positions' k-contexts against the training shards `files`: the positions whose
/// k-context is valid, those whose k-context occurred, per rank r < `depth` those whose next token is their context's
/// (r + 1)-th most frequent next token, and the bitmap of the corpus positions whose k-context matched a validation
/// one. `tokens` are the validation inputs followed by the last one's next token. `train_tokens`, if given, limits the
/// training data to the first that many tokens of `files`. `filter`, the bitmap of a smaller k over the same training
/// data, only saves probes. `memory` bounds the bytes of occurrences held at once.
#[pyfunction]
#[pyo3(signature = (files, tokens, k, depth, memory, train_tokens=None, filter=None))]
#[allow(clippy::type_complexity)]
pub fn kgram_stats<'py>(
    py: Python<'py>,
    files: Vec<String>,
    tokens: PyReadonlyArray1<u16>,
    k: usize,
    depth: usize,
    memory: usize,
    train_tokens: Option<usize>,
    filter: Option<PyReadonlyArray1<u64>>,
) -> PyResult<(u64, u64, Bound<'py, PyArray1<u64>>, Bound<'py, PyArray1<u64>>)> {
    let tokens = tokens.as_slice()?;
    validate(&files, tokens, k, depth, memory)?;
    let filter = filter.as_ref().map(|f| f.as_slice()).transpose()?;
    let (eligible, matched, ranks, hits) = py.detach(|| run(&files, train_tokens, tokens, k, depth, memory, filter))?;
    Ok((eligible, matched, ranks.into_pyarray(py), hits.into_pyarray(py)))
}

/// Exact next-token counts of the validation positions' k-contexts in the training shards `files` (the arguments as
/// kgram_stats's, with `top` for `depth`): each position's context id (u32::MAX where its k-context is invalid); per
/// context id its occurrence count, its number of distinct next tokens, and the offsets of its `top` most frequent
/// next tokens (count descending, then token ascending) into the next tokens and their counts; and the bitmap of the
/// corpus positions whose k-context matched.
#[pyfunction]
#[pyo3(signature = (files, tokens, k, top, memory, train_tokens=None, filter=None))]
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn kgram_counts<'py>(
    py: Python<'py>,
    files: Vec<String>,
    tokens: PyReadonlyArray1<u16>,
    k: usize,
    top: usize,
    memory: usize,
    train_tokens: Option<usize>,
    filter: Option<PyReadonlyArray1<u64>>,
) -> PyResult<(
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u16>>,
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<u64>>,
)> {
    let tokens = tokens.as_slice()?;
    validate(&files, tokens, k, top, memory)?;
    let filter = filter.as_ref().map(|f| f.as_slice()).transpose()?;
    let (key_of, c) = py.detach(|| -> PyResult<_> {
        let corpus = Corpus::map(&files, train_tokens)?;
        check_filter(&corpus, filter)?;
        let x = &tokens[..tokens.len() - 1];
        let contexts = Contexts::new(x, &segments(x), k)?;
        let c = count(&corpus, x, k, &contexts, filter, top, memory);
        Ok((contexts.key_of, c))
    })?;
    Ok((
        key_of.into_pyarray(py),
        c.totals.into_pyarray(py),
        c.distinct.into_pyarray(py),
        c.offsets.into_pyarray(py),
        c.next.into_pyarray(py),
        c.next_counts.into_pyarray(py),
        c.hits.into_pyarray(py),
    ))
}

/// kgram_counts for a query of other training shards (the training positions' counts of the old-records experiment,
/// retrieval_blog_post/old_records.py's LateCountRows): the query `tokens` are the query shards concatenated, followed
/// by one more token; their segments also start at each of `breaks` (each later query shard's first position), so no
/// context spans two query shards. Returns kgram_counts's arrays, then `cnt`: per query position i, the count of its
/// next token tokens[i + 1] among its context's `cnt_top` most frequent next tokens (count descending, then token
/// ascending; 0 outside them, and where the context is invalid or never occurred), the validation count rows' rule.
#[pyfunction]
#[pyo3(signature = (files, tokens, k, top, cnt_top, memory, breaks=Vec::new(), filter=None))]
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn kgram_query_counts<'py>(
    py: Python<'py>,
    files: Vec<String>,
    tokens: PyReadonlyArray1<u16>,
    k: usize,
    top: usize,
    cnt_top: usize,
    memory: usize,
    breaks: Vec<usize>,
    filter: Option<PyReadonlyArray1<u64>>,
) -> PyResult<(
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u16>>,
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u32>>,
)> {
    let tokens = tokens.as_slice()?;
    validate(&files, tokens, k, top, memory)?;
    if cnt_top == 0 || breaks.iter().any(|&b| b == 0 || b >= tokens.len() - 1) {
        return Err(PyValueError::new_err("need cnt_top > 0 and breaks within the query"));
    }
    let filter = filter.as_ref().map(|f| f.as_slice()).transpose()?;
    let (key_of, c, cnt) = py.detach(|| -> PyResult<_> {
        let corpus = Corpus::map(&files, None)?;
        check_filter(&corpus, filter)?;
        let x = &tokens[..tokens.len() - 1];
        let mut avail = segments(x);
        for &b in &breaks {
            let mut i = b;
            while i < x.len() && x[i] != BOS {
                avail[i] = (i - b + 1) as u32;
                i += 1;
            }
        }
        let contexts = Contexts::new(x, &avail, k)?;
        let (first, order) = by_context(&contexts.key_of, contexts.reps.len());
        let mut cnt = vec![0u32; x.len()];
        let targets = Targets { target: &tokens[1..], first, order, top: cnt_top, cnt: Output(cnt.as_mut_ptr()) };
        let c = count_with(&corpus, x, k, &contexts, filter, top, memory, Some(&targets));
        Ok((contexts.key_of, c, cnt))
    })?;
    Ok((
        key_of.into_pyarray(py),
        c.totals.into_pyarray(py),
        c.distinct.into_pyarray(py),
        c.offsets.into_pyarray(py),
        c.next.into_pyarray(py),
        c.next_counts.into_pyarray(py),
        c.hits.into_pyarray(py),
        cnt.into_pyarray(py),
    ))
}

/// The query positions grouped by context id: positions order[first[id]..first[id + 1]] have context id `id`.
pub(crate) fn by_context(key_of: &[u32], keys: usize) -> (Vec<u64>, Vec<u32>) {
    let mut first = vec![0u64; keys + 1];
    for &id in key_of {
        if id != u32::MAX {
            first[id as usize + 1] += 1;
        }
    }
    for i in 0..keys {
        first[i + 1] += first[i];
    }
    let mut fill = first.clone();
    let mut order = vec![0u32; first[keys] as usize];
    for (i, &id) in key_of.iter().enumerate() {
        if id != u32::MAX {
            order[fill[id as usize] as usize] = i as u32;
            fill[id as usize] += 1;
        }
    }
    (first, order)
}

/// The query's next tokens, for kgram_query_counts's `cnt` (written by the partitions, each its own contexts'
/// positions).
struct Targets<'a> {
    target: &'a [u16],
    first: Vec<u64>,
    order: Vec<u32>,
    top: usize,
    cnt: Output<u32>,
}
unsafe impl Sync for Targets<'_> {}

/// The arguments' checks shared by kgram_stats and kgram_counts.
fn validate(files: &[String], tokens: &[u16], k: usize, depth: usize, memory: usize) -> PyResult<()> {
    if files.is_empty() || tokens.len() < 2 || k == 0 || depth == 0 || memory == 0 {
        return Err(PyValueError::new_err("need files, 2+ tokens, and k, depth and memory > 0"));
    }
    if tokens.iter().any(|&t| t > BOS) {
        return Err(PyValueError::new_err("invalid GPT-2 token"));
    }
    Ok(())
}

fn check_filter(corpus: &Corpus, filter: Option<&[u64]>) -> PyResult<()> {
    if filter.is_some_and(|f| f.len() != corpus.len.div_ceil(64)) {
        return Err(PyValueError::new_err("filter is the bitmap of other files"));
    }
    Ok(())
}

/// Per position of `x`, the tokens of its segment up to and including it: its k-context is valid iff this is >= k.
pub(crate) fn segments(x: &[u16]) -> Vec<u32> {
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

#[allow(clippy::type_complexity)]
fn run(
    files: &[String],
    train_tokens: Option<usize>,
    tokens: &[u16],
    k: usize,
    depth: usize,
    memory: usize,
    filter: Option<&[u64]>,
) -> PyResult<(u64, u64, Vec<u64>, Vec<u64>)> {
    let corpus = Corpus::map(files, train_tokens)?;
    check_filter(&corpus, filter)?;
    let n = tokens.len() - 1;
    let (x, target) = (&tokens[..n], &tokens[1..]);
    let contexts = Contexts::new(x, &segments(x), k)?;
    let c = count(&corpus, x, k, &contexts, filter, depth, memory);
    let (matched, ranks) = contexts
        .key_of
        .par_iter()
        .zip(target.par_iter())
        .fold(
            || (0u64, vec![0u64; depth]),
            |(mut m, mut hist), (&id, &t)| {
                if id != u32::MAX && c.totals[id as usize] > 0 {
                    m += 1;
                    let tops = &c.next[c.offsets[id as usize] as usize..c.offsets[id as usize + 1] as usize];
                    if let Some(r) = tops.iter().position(|&v| v == t) {
                        hist[r] += 1;
                    }
                }
                (m, hist)
            },
        )
        .reduce(
            || (0, vec![0; depth]),
            |(a, mut hist), (b, other)| {
                hist.iter_mut().zip(other).for_each(|(h, o)| *h += o);
                (a + b, hist)
            },
        );
    let eligible = contexts.key_of.iter().filter(|&&id| id != u32::MAX).count() as u64;
    Ok((eligible, matched, ranks, c.hits))
}

/// Step 3's result for one k: per context id its occurrence count, its number of distinct next tokens, and its most
/// frequent next tokens with their counts (count descending, then token ascending) at offsets[id]..offsets[id + 1];
/// and the bitmap of the corpus positions whose k-context matched.
struct Counts {
    totals: Vec<u64>,
    distinct: Vec<u32>,
    offsets: Vec<u64>,
    next: Vec<u16>,
    next_counts: Vec<u32>,
    hits: Vec<u64>,
}

/// One partition's part of Counts: the ids from its first, the number of next tokens kept of each.
struct Part {
    totals: Vec<u64>,
    distinct: Vec<u32>,
    kept: Vec<u32>,
    next: Vec<u16>,
    next_counts: Vec<u32>,
}

/// Steps 2 and 3 for one k: Counts with the `top` most frequent next tokens of each context.
fn count(corpus: &Corpus, x: &[u16], k: usize, contexts: &Contexts, filter: Option<&[u64]>, top: usize, memory: usize) -> Counts {
    count_with(corpus, x, k, contexts, filter, top, memory, None)
}

/// count, and with `targets` each query position's count of its next token (Targets).
#[allow(clippy::too_many_arguments)]
fn count_with(
    corpus: &Corpus,
    x: &[u16],
    k: usize,
    contexts: &Contexts,
    filter: Option<&[u64]>,
    top: usize,
    memory: usize,
    targets: Option<&Targets>,
) -> Counts {
    let keys = contexts.reps.len();
    let (mut totals, mut distinct, mut kept) = (vec![0u64; keys], vec![0u32; keys], vec![0u32; keys]);
    let (mut next, mut next_counts) = (Vec::new(), Vec::new());
    let mut hits = vec![0u64; corpus.len.div_ceil(64)];
    if keys == 0 {
        return Counts { totals, distinct, offsets: vec![0], next, next_counts, hits };
    }
    // at most one occurrence per probe, of 8 bytes
    let probes: usize = match filter {
        Some(f) => f.par_iter().map(|w| w.count_ones() as usize).sum(),
        None => corpus.ranges.iter().map(|&(lo, hi)| (hi - lo).saturating_sub(k)).sum(),
    };
    let rounds = (8 * probes).div_ceil(memory).clamp(1, keys);
    for round in 0..rounds {
        let ids = (keys * round / rounds, keys * (round + 1) / rounds);
        let span = ids.1 - ids.0;
        let shift = ((span.max(2) - 1).ilog2() as usize + 1).saturating_sub(12); // about 4096 partitions
        let parts = ((span - 1) >> shift) + 1;
        let lists = if round == 0 {
            scan(corpus, x, k, contexts, filter, Some(Output(hits.as_mut_ptr())), ids, shift, parts)
        } else {
            scan(corpus, x, k, contexts, Some(&hits), None, ids, shift, parts)
        };
        // in order of the ids: the rounds' ranges ascend, and so do the partitions' within one
        let results: Vec<(usize, Part)> = lists
            .into_par_iter()
            .enumerate()
            .map(|(p, lists)| {
                let mut all: Vec<u64> = Vec::with_capacity(lists.iter().map(Vec::len).sum());
                for v in lists {
                    all.extend_from_slice(&v);
                }
                all.sort_unstable();
                let first = ids.0 + (p << shift);
                let last = ids.1.min(first + (1 << shift));
                let n = last - first;
                let mut part = Part { totals: vec![0; n], distinct: vec![0; n], kept: vec![0; n], next: Vec::new(), next_counts: Vec::new() };
                let mut next = Vec::new();
                let mut lookup: Vec<(u16, u32)> = Vec::new();
                for run in all.chunk_by(|a, b| a >> 16 == b >> 16) {
                    let id = (run[0] >> 16) as usize - first;
                    part.totals[id] = run.len() as u64;
                    next.clear();
                    next.extend(run.chunk_by(|a, b| a == b).map(|same| (std::cmp::Reverse(same.len()), (same[0] & 0xFFFF) as u16)));
                    next.sort_unstable(); // count descending, then token ascending
                    part.distinct[id] = next.len() as u32;
                    part.kept[id] = next.len().min(top) as u32;
                    for &(std::cmp::Reverse(n), token) in &next[..next.len().min(top)] {
                        part.next.push(token);
                        part.next_counts.push(u32::try_from(n).expect("a next token's count overflows u32"));
                    }
                    if let Some(q) = targets {
                        let gid = first + id;
                        lookup.clear();
                        lookup.extend(next[..next.len().min(q.top)].iter().map(|&(std::cmp::Reverse(n), t)| (t, n as u32)));
                        lookup.sort_unstable();
                        for &i in &q.order[q.first[gid] as usize..q.first[gid + 1] as usize] {
                            let t = q.target[i as usize];
                            if let Ok(at) = lookup.binary_search_by_key(&t, |&(t, _)| t) {
                                unsafe { q.cnt.set(i as usize, lookup[at].1) };
                            }
                        }
                    }
                }
                (first, part)
            })
            .collect();
        for (first, part) in results {
            let n = part.totals.len();
            totals[first..first + n].copy_from_slice(&part.totals);
            distinct[first..first + n].copy_from_slice(&part.distinct);
            kept[first..first + n].copy_from_slice(&part.kept);
            next.extend_from_slice(&part.next);
            next_counts.extend_from_slice(&part.next_counts);
        }
    }
    let offsets = std::iter::once(0).chain(kept.iter().scan(0u64, |at, &n| {
        *at += n as u64;
        Some(*at)
    }));
    Counts { totals, distinct, offsets: offsets.collect(), next, next_counts, hits }
}

/// Step 2 for the context ids in [ids.0, ids.1): every corpus position p (the next token's) whose k-context lies in
/// its shard and, with a `filter`, whose bit is set in it, is probed. A verified hit sets p's bit in `hits` (if given)
/// and, if its id is in range, adds `id << 16 | next token` to the id's partition. Returns per partition the lists
/// the threads added to.
#[allow(clippy::too_many_arguments)]
fn scan(
    corpus: &Corpus,
    x: &[u16],
    k: usize,
    contexts: &Contexts,
    filter: Option<&[u64]>,
    hits: Option<Output<u64>>,
    ids: (usize, usize),
    shift: usize,
    parts: usize,
) -> Vec<Vec<Vec<u64>>> {
    let tokens = corpus.get();
    let accs: Vec<Mutex<Vec<Vec<u64>>>> = (0..rayon::current_num_threads()).map(|_| Mutex::new(vec![Vec::new(); parts])).collect();
    let mut pieces = Vec::new();
    for &(lo, hi) in &corpus.ranges {
        let mut start = lo + k;
        while start < hi {
            let end = hi.min((start / PIECE + 1) * PIECE);
            pieces.push((start, end));
            start = end;
        }
    }
    // set the bits of a bitmap word, which only its piece writes
    let or = |out: Output<u64>, word: usize, bits: u64| {
        if bits != 0 {
            unsafe { *out.0.add(word) |= bits };
        }
    };
    pieces.par_iter().for_each(|&(start, end)| {
        let mut guard = accs[rayon::current_thread_index().unwrap_or(0)].lock().unwrap();
        let acc = &mut *guard;
        let (mut word, mut bits) = (start / 64, 0u64); // the bitmap word being set, written once the scan moves past it
        let probe = |batch: &[(usize, u64)], acc: &mut Vec<Vec<u64>>, word: &mut usize, bits: &mut u64| {
            for &(_, h) in batch {
                unsafe { _mm_prefetch(contexts.slot(h).cast(), _MM_HINT_T0) };
            }
            for &(p, h) in batch {
                let Some(id) = contexts.find(h) else { continue };
                let r = contexts.reps[id] as usize;
                if tokens[p - k..p] != x[r + 1 - k..=r] {
                    continue;
                }
                if let Some(out) = hits {
                    if p / 64 != *word {
                        or(out, *word, *bits);
                        (*word, *bits) = (p / 64, 0);
                    }
                    *bits |= 1 << (p % 64);
                }
                if (ids.0..ids.1).contains(&id) {
                    acc[(id - ids.0) >> shift].push((id as u64) << 16 | tokens[p] as u64);
                }
            }
        };
        let mut batch = [(0usize, 0u64); BATCH];
        let mut m = 0;
        for p in start..end {
            if let Some(f) = filter {
                if f[p / 64] >> (p % 64) & 1 == 0 {
                    continue;
                }
            }
            batch[m] = (p, hash(&tokens[p - k..p]));
            m += 1;
            if m == BATCH {
                probe(&batch, acc, &mut word, &mut bits);
                m = 0;
            }
        }
        probe(&batch[..m], acc, &mut word, &mut bits);
        if let Some(out) = hits {
            or(out, word, bits);
        }
    });
    let mut lists: Vec<Vec<Vec<u64>>> = (0..parts).map(|_| Vec::new()).collect();
    for acc in accs {
        for (p, v) in acc.into_inner().unwrap().into_iter().enumerate() {
            if !v.is_empty() {
                lists[p].push(v);
            }
        }
    }
    lists
}

/// A context's 64-bit hash: the polynomial over token + 1 with multiplier P, splitmix64's finalizer, the lowest bit
/// set (0 marks an empty table slot).
#[inline]
pub(crate) fn hash(tokens: &[u16]) -> u64 {
    let mut h = tokens.iter().fold(0u64, |h, &t| h.wrapping_mul(P).wrapping_add(t as u64 + 1));
    h ^= h >> 30;
    h = h.wrapping_mul(0xbf58476d1ce4e5b9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94d049bb133111eb);
    (h ^ (h >> 31)) | 1
}

/// The distinct valid validation k-contexts: each position's context id (u32::MAX where its k-context is invalid), a
/// position of each id, and an open-addressing table from a context's hash to its id.
pub(crate) struct Contexts {
    pub(crate) key_of: Vec<u32>,
    pub(crate) reps: Vec<u32>,
    pub(crate) bits: u32,
    pub(crate) slot_hash: Vec<u64>,
    pub(crate) slot_id: Vec<u32>,
}

impl Contexts {
    pub(crate) fn new(x: &[u16], avail: &[u32], k: usize) -> PyResult<Self> {
        let mut items: Vec<(u64, u32)> =
            (0..x.len()).into_par_iter().filter(|&i| avail[i] as usize >= k).map(|i| (hash(&x[i + 1 - k..=i]), i as u32)).collect();
        items.par_sort_unstable();
        let mut key_of = vec![u32::MAX; x.len()];
        let (mut reps, mut hashes) = (Vec::new(), Vec::new());
        for group in items.chunk_by(|a, b| a.0 == b.0) {
            let r = group[0].1 as usize;
            for &(_, i) in group {
                let i = i as usize;
                if x[i + 1 - k..=i] != x[r + 1 - k..=r] {
                    return Err(PyValueError::new_err(format!("64-bit hash collision between validation {k}-contexts at {r} and {i}")));
                }
                key_of[i] = reps.len() as u32;
            }
            reps.push(r as u32);
            hashes.push(group[0].0);
        }
        let bits = (2 * reps.len()).next_power_of_two().max(1024).trailing_zeros();
        let mask = (1usize << bits) - 1;
        // in huge pages: val_scan probes these tables at random from every corpus position
        let (mut slot_hash, mut slot_id): (Vec<u64>, Vec<u32>) = (crate::val_scan::huge_zeroed(1 << bits), crate::val_scan::huge_zeroed(1 << bits));
        for (id, &h) in hashes.iter().enumerate() {
            let mut s = (h >> (64 - bits)) as usize;
            while slot_hash[s] != 0 {
                s = (s + 1) & mask;
            }
            slot_hash[s] = h;
            slot_id[s] = id as u32;
        }
        Ok(Self { key_of, reps, bits, slot_hash, slot_id })
    }

    /// The home slot of hash `h`, to prefetch.
    pub(crate) fn slot(&self, h: u64) -> *const u64 {
        self.slot_hash.as_ptr().wrapping_add((h >> (64 - self.bits)) as usize)
    }

    /// The id of the context with hash `h`, if any: a candidate, whose tokens are still to compare.
    #[inline]
    pub(crate) fn find(&self, h: u64) -> Option<usize> {
        let mask = self.slot_hash.len() - 1;
        let mut s = (h >> (64 - self.bits)) as usize;
        loop {
            match self.slot_hash[s] {
                v if v == h => return Some(self.slot_id[s] as usize),
                0 => return None,
                _ => s = (s + 1) & mask,
            }
        }
    }
}

/// Read-only private mappings of shard files, `stride` tokens apart in one reserved range: token j of file i sits at
/// position i * stride + HEADER + j, and ranges[i] spans file i's tokens. The page cache is the corpus: nothing is
/// copied.
pub(crate) struct Corpus {
    pub(crate) ptr: *mut u16,
    pub(crate) len: usize,
    pub(crate) ranges: Vec<(usize, usize)>,
}
unsafe impl Send for Corpus {}
unsafe impl Sync for Corpus {}

impl Corpus {
    /// The first `limit` tokens of `files` if given (an error if they hold fewer), else all of them.
    pub(crate) fn map(files: &[String], limit: Option<usize>) -> PyResult<Self> {
        let mut longest = 0;
        for path in files {
            longest = longest.max(open_shard(path)?.1);
        }
        let stride = (HEADER + longest).next_multiple_of(1 << 16); // a multiple of 64: no bitmap word spans two shards
        let len = files.len() * stride;
        let flags = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE;
        let ptr = unsafe { libc::mmap(std::ptr::null_mut(), 2 * len, libc::PROT_READ, flags, -1, 0) };
        if ptr == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error().into());
        }
        let mut corpus = Self { ptr: ptr.cast(), len, ranges: Vec::new() };
        let mut left = limit.unwrap_or(usize::MAX);
        for (i, path) in files.iter().enumerate() {
            let (file, n) = open_shard(path)?; // opened one at a time: a corpus may have more shards than open files
            let used = n.min(left);
            left -= used;
            let at = corpus.ptr.wrapping_add(i * stride);
            let flags = libc::MAP_PRIVATE | libc::MAP_FIXED;
            let mapped = unsafe { libc::mmap(at.cast(), 2 * (HEADER + n), libc::PROT_READ, flags, file.as_raw_fd(), 0) };
            if mapped == libc::MAP_FAILED {
                return Err(std::io::Error::last_os_error().into());
            }
            corpus.ranges.push((i * stride + HEADER, i * stride + HEADER + used));
        }
        if limit.is_some() && left > 0 {
            return Err(PyValueError::new_err(format!("the files hold {left} tokens fewer than the {} asked for", limit.unwrap())));
        }
        Ok(corpus)
    }

    pub(crate) fn get(&self) -> &[u16] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl Drop for Corpus {
    fn drop(&mut self) {
        unsafe { libc::munmap(self.ptr.cast(), 2 * self.len) };
    }
}
