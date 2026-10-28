//! Which validation documents also occur in the training shards, as Gopher (Rae et al., 2021, MassiveText)
//! deduplicates: exactly (a training document with the same tokens) and fuzzily (the best Jaccard similarity between
//! the distinct word n-gram sets of the document and of any training document, with whitespace normalized and
//! punctuation ignored). Gopher estimates the Jaccard with MinHash; here it is exact. Analysis only, for the
//! deduplicated validation shards of retrieval_blog_post/fineweb.py; nothing here runs in training. Grew out of
//! valoverlap (modded-nanogpt, branch val-overlap), which used n-grams of GPT-2 tokens.
//!
//! A document is the tokens after a BOS up to the next BOS (a shard's leading partial document counts too). Its words
//! are its decoded text split at (Unicode) whitespace, each keeping only its letters and digits, and its n-grams are
//! the windows of n words. Pairs are found through the n-grams that occur in at most CAP validation documents, and
//! their Jaccard is computed exactly when the training document holds at least CAND of the validation document's such
//! n-grams; a pair below that is missed.
use crate::{open_shard, BOS};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
use std::collections::HashMap;
use std::io::Read;
use std::sync::atomic::{AtomicU32, Ordering::Relaxed};

const CAP: usize = 64;
const CAND: f64 = 0.25;

/// Per validation document of the shard at `val`: the start and length of its body (after its BOS), its exact copies
/// in the training shards `train`, and its best Jaccard similarity with a training document over distinct word
/// n-grams (1 for an exact copy, 0 when no pair was checked). Token t decodes to bytes[offsets[t]..offsets[t + 1]].
#[pyfunction]
#[allow(clippy::type_complexity)]
pub fn val_overlap<'py>(
    py: Python<'py>,
    val: String,
    train: Vec<String>,
    n: usize,
    offsets: PyReadonlyArray1<u64>,
    bytes: PyReadonlyArray1<u8>,
) -> PyResult<(Bound<'py, PyArray1<u64>>, Bound<'py, PyArray1<u64>>, Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<f32>>)> {
    let vocab = Vocab { offsets: offsets.as_slice()?, bytes: bytes.as_slice()? };
    let (starts, lens, exact, jaccard) = py.detach(|| overlap(&val, &train, n, &vocab))?;
    Ok((starts.into_pyarray(py), lens.into_pyarray(py), exact.into_pyarray(py), jaccard.into_pyarray(py)))
}

#[allow(clippy::type_complexity)]
fn overlap(val: &str, train: &[String], n: usize, vocab: &Vocab) -> PyResult<(Vec<u64>, Vec<u64>, Vec<u32>, Vec<f32>)> {
    // Validation documents: whole-document hashes, distinct n-grams, and each n-gram's documents.
    let vt = read(val)?;
    let vdocs = docs(&vt);
    let mut whole: HashMap<u64, Vec<u32>> = HashMap::new();
    for (d, &(a, b)) in vdocs.iter().enumerate() {
        whole.entry(hash(&vt[a..b])).or_default().push(d as u32);
    }
    let per: Vec<Vec<u64>> = vdocs.par_iter().map(|&(a, b)| distinct(grams(&words(&vocab.decode(&vt[a..b])), n))).collect();
    let mut pairs: Vec<(u64, u32)> = per.iter().enumerate().flat_map(|(d, g)| g.iter().map(move |&k| (k, d as u32))).collect();
    pairs.par_sort_unstable();
    let (mut keys, mut off, mut ids) = (Vec::new(), Vec::new(), Vec::with_capacity(pairs.len()));
    for &(k, d) in &pairs {
        if keys.last() != Some(&k) {
            keys.push(k);
            off.push(ids.len() as u32);
        }
        ids.push(d);
    }
    off.push(ids.len() as u32);
    drop(pairs);
    let list = |k: usize| &ids[off[k] as usize..off[k + 1] as usize];
    let mut usable = vec![0u32; vdocs.len()]; // n-grams that find pairs: in at most CAP validation documents
    for k in 0..keys.len() {
        if list(k).len() <= CAP {
            for &d in list(k) {
                usable[d as usize] += 1;
            }
        }
    }
    let table = Table::new(&keys);

    // Training documents, a shard per task.
    let exact: Vec<AtomicU32> = (0..vdocs.len()).map(|_| AtomicU32::new(0)).collect();
    let best: Vec<AtomicU32> = (0..vdocs.len()).map(|_| AtomicU32::new(0)).collect(); // f32 bits: ordered like the values
    train.par_iter().try_for_each(|path| -> PyResult<()> {
        let t = read(path)?;
        let (mut hits, mut shared) = (Vec::new(), HashMap::<u32, u32>::new());
        for (a, b) in docs(&t) {
            let body = &t[a..b];
            for &v in whole.get(&hash(body)).into_iter().flatten() {
                let (va, vb) = vdocs[v as usize];
                if vt[va..vb] == *body {
                    exact[v as usize].fetch_add(1, Relaxed);
                    best[v as usize].fetch_max(1f32.to_bits(), Relaxed);
                }
            }
            let g = grams(&words(&vocab.decode(body)), n);
            hits.clear();
            for i in 0..g.len() {
                if let Some(&k) = g.get(i + 16) {
                    unsafe { _mm_prefetch(table.slots.as_ptr().add(table.slot(k)).cast(), _MM_HINT_T0) };
                }
                if let Some(k) = table.get(g[i]) {
                    if list(k).len() <= CAP {
                        hits.push(k as u32);
                    }
                }
            }
            if hits.is_empty() {
                continue;
            }
            hits.sort_unstable();
            hits.dedup();
            shared.clear();
            for &k in &hits {
                for &v in list(k as usize) {
                    *shared.entry(v).or_insert(0) += 1;
                }
            }
            let mut tg: Option<Vec<u64>> = None;
            for (&v, &c) in &shared {
                if (c as f64) < CAND * usable[v as usize] as f64 {
                    continue;
                }
                let tg = tg.get_or_insert_with(|| distinct(g.clone()));
                let vg = &per[v as usize];
                let inter = intersect(vg, tg);
                let j = inter as f32 / (vg.len() + tg.len() - inter) as f32;
                best[v as usize].fetch_max(j.to_bits(), Relaxed);
            }
        }
        Ok(())
    })?;
    let starts = vdocs.iter().map(|&(a, _)| a as u64).collect();
    let lens = vdocs.iter().map(|&(a, b)| (b - a) as u64).collect();
    let exact = exact.into_iter().map(AtomicU32::into_inner).collect();
    let best = best.into_iter().map(|b| f32::from_bits(b.into_inner())).collect();
    Ok((starts, lens, exact, best))
}

/// A shard's tokens.
fn read(path: &str) -> PyResult<Vec<u16>> {
    let (mut file, n) = open_shard(path)?;
    let mut bytes = vec![0u8; 2 * n];
    file.read_exact(&mut bytes)?;
    Ok(bytes.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect())
}

/// The body (start, end) of each document: the tokens after a BOS up to the next BOS. A shard's leading partial
/// document counts too; lone BOS tokens are skipped.
fn docs(t: &[u16]) -> Vec<(usize, usize)> {
    let mut starts: Vec<usize> = t.iter().enumerate().filter(|&(_, &v)| v == BOS).map(|(i, _)| i).collect();
    if starts.first() != Some(&0) {
        starts.insert(0, 0);
    }
    starts.push(t.len());
    starts.windows(2).map(|w| (w[0] + (t[w[0]] == BOS) as usize, w[1])).filter(|&(a, b)| b > a).collect()
}

/// A whole document's hash (FNV-1a with an extra shift, over its length and tokens).
fn hash(t: &[u16]) -> u64 {
    let mut h = 0xcbf29ce484222325u64 ^ t.len() as u64;
    for &v in t {
        h = (h ^ v as u64).wrapping_mul(0x100000001b3);
        h ^= h >> 29;
    }
    h.wrapping_mul(0x9e3779b97f4a7c15)
}

/// MurmurHash3's finalizer, never 0 (an empty table slot).
fn mix(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
    (x ^ (x >> 33)).max(1)
}

/// GPT-2's token bytes.
struct Vocab<'a> {
    offsets: &'a [u64],
    bytes: &'a [u8],
}

impl Vocab<'_> {
    /// The bytes of tokens `t`.
    fn decode(&self, t: &[u16]) -> Vec<u8> {
        let mut out = Vec::with_capacity(4 * t.len());
        for &v in t {
            out.extend_from_slice(&self.bytes[self.offsets[v as usize] as usize..self.offsets[v as usize + 1] as usize]);
        }
        out
    }
}

/// The hash of each word of `text`: split at whitespace, each keeping only its letters and digits (a word with none is
/// dropped).
fn words(text: &[u8]) -> Vec<u64> {
    String::from_utf8_lossy(text)
        .split_whitespace()
        .filter_map(|word| {
            let (mut h, mut any) = (0xcbf29ce484222325u64, false);
            for c in word.chars().filter(|c| c.is_alphanumeric()) {
                h = (h ^ c as u64).wrapping_mul(0x100000001b3);
                any = true;
            }
            any.then(|| mix(h))
        })
        .collect()
}

/// The key of every window of n words, in order (none if there are fewer than n): a rolling polynomial hash, mixed.
fn grams(words: &[u64], n: usize) -> Vec<u64> {
    const B: u64 = 0x9e3779b97f4a7c15;
    let bn = B.wrapping_pow(n as u32);
    let mut out = Vec::with_capacity(words.len().saturating_sub(n - 1));
    let mut h = 0u64;
    for (i, &w) in words.iter().enumerate() {
        h = h.wrapping_mul(B).wrapping_add(w);
        if i >= n {
            h = h.wrapping_sub(words[i - n].wrapping_mul(bn));
        }
        if i + 1 >= n {
            out.push(mix(h));
        }
    }
    out
}

fn distinct(mut g: Vec<u64>) -> Vec<u64> {
    g.sort_unstable();
    g.dedup();
    g
}

/// The size of the intersection of two sorted sets.
fn intersect(a: &[u64], b: &[u64]) -> usize {
    let (mut i, mut j, mut n) = (0, 0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => (i, j, n) = (i + 1, j + 1, n + 1),
        }
    }
    n
}

/// An open-addressing table from a (nonzero) key to its index.
struct Table {
    slots: Vec<(u64, u32)>,
    shift: u32,
}

impl Table {
    fn new(keys: &[u64]) -> Self {
        let bits = (2 * keys.len()).next_power_of_two().trailing_zeros().max(4);
        let mut t = Table { slots: vec![(0, 0); 1 << bits], shift: 64 - bits };
        for (i, &k) in keys.iter().enumerate() {
            let mut s = t.slot(k);
            while t.slots[s].0 != 0 {
                s = (s + 1) & (t.slots.len() - 1);
            }
            t.slots[s] = (k, i as u32);
        }
        t
    }

    fn slot(&self, k: u64) -> usize {
        (k >> self.shift) as usize
    }

    fn get(&self, k: u64) -> Option<usize> {
        let mut s = self.slot(k);
        loop {
            match self.slots[s] {
                (key, i) if key == k => return Some(i as usize),
                (0, _) => return None,
                _ => s = (s + 1) & (self.slots.len() - 1),
            }
        }
    }
}
