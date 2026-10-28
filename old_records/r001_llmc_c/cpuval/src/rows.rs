//! What every engine shares: the training shards, the validation tokens (given only after the clock), the output
//! rows and how one context's next-token counts become its validation positions' rows (old_records' ValCountRows /
//! ValHintRows, as exact_match's val_scan and val_hint_rows define them).
use rayon::prelude::*;
use std::cmp::Reverse;
use std::fs::File;
use std::os::unix::fs::FileExt;

pub const HEADER_BYTES: usize = 1024; // a token shard's header
pub const BOS: u16 = 50256;
pub const LEVELS: [usize; 10] = [2, 3, 4, 5, 6, 8, 10, 12, 16, 20]; // the count rows' (llm.c's hint_stats::LEVELS)
pub const HINT_LEVELS: [usize; 7] = [2, 3, 4, 5, 6, 10, 20]; // the hint rows' (hint_stats::HINT_LEVELS), their buckets
pub const TOP: usize = 1024; // cnt: the target's count if among its context's TOP most frequent next tokens
pub const KEEP: usize = 8; // the hint rows' entries
const COUNTS: [usize; 6] = [2, 3, 5, 9, 17, 33]; // the hint cell's bin edges of a context's total
const PURITIES: [f64; 5] = [0.3, 0.5, 0.7, 0.85, 0.95]; // and of its top next token's share of it

pub type Times = Vec<(String, f64)>;

/// The training shards, opened (headers checked) before the clock.
pub struct Shards {
    pub files: Vec<File>,
    pub sizes: Vec<usize>, // tokens
}

/// A shard's token count (its header checked against its length).
pub fn shard_tokens(file: &File, path: &str) -> Result<usize, String> {
    let mut header = [0u8; HEADER_BYTES];
    file.read_exact_at(&mut header, 0).map_err(|e| format!("{path}: {e}"))?;
    let word = |at: usize| i32::from_le_bytes(header[at..at + 4].try_into().unwrap());
    let n = word(8);
    let len = file.metadata().map_err(|e| format!("{path}: {e}"))?.len();
    if word(0) != 20240520 || word(4) != 1 || n < 0 || len != (HEADER_BYTES + 2 * n as usize) as u64 {
        return Err(format!("{path}: not a token shard"));
    }
    Ok(n as usize)
}

impl Shards {
    pub fn open(paths: &[&str]) -> Result<Self, String> {
        let mut files = Vec::new();
        let mut sizes = Vec::new();
        for p in paths {
            let f = File::open(p).map_err(|e| format!("{p}: {e}"))?;
            sizes.push(shard_tokens(&f, p)?);
            files.push(f);
        }
        Ok(Self { files, sizes })
    }
}

/// One level as an engine gets it: its context length, the shards it counts over (the first `files`), its column in
/// the count rows and its hint bucket (None: no hint level).
#[derive(Clone, Copy, Debug)]
pub struct Lv {
    pub k: usize,
    pub files: usize,
    pub col: usize,
    pub bucket: Option<usize>,
}

/// The validation positions, after the clock: x = the first n tokens (the contexts end at them), target = the
/// next ones, avail = per position the tokens of its segment up to and including it (kgram's segments: a BOS starts
/// one), so position i's k-context x[i + 1 - k ..= i] is valid iff avail[i] >= k.
pub struct Val<'a> {
    pub x: &'a [u16],
    pub target: &'a [u16],
    pub avail: &'a [u32],
}

impl<'a> Val<'a> {
    /// From the n + 1 validation tokens (in parallel, in the current rayon pool: chunks, each from the last BOS before it),
    /// `avail` written into `buf` (at least n long).
    pub fn new(tokens: &'a [u16], buf: &'a mut [u32]) -> Self {
        let n = tokens.len() - 1;
        let x = &tokens[..n];
        let avail = &mut buf[..n];
        const CHUNK: usize = 1 << 16;
        avail.par_chunks_mut(CHUNK).enumerate().for_each(|(c, out)| {
            let a = c * CHUNK;
            let mut start = x[..a].iter().rposition(|&t| t == BOS).unwrap_or(0);
            for (j, v) in out.iter_mut().enumerate() {
                if x[a + j] == BOS {
                    start = a + j;
                }
                *v = (a + j - start + 1) as u32;
            }
        });
        Self { x, target: &tokens[1..], avail }
    }
}

/// A raw pointer that threads write disjoint parts of.
#[derive(Clone, Copy)]
pub struct Output<T>(pub *mut T);
unsafe impl<T> Sync for Output<T> {}
unsafe impl<T> Send for Output<T> {}
impl<T: Copy> Output<T> {
    /// # Safety
    /// In bounds, and no other thread writes `offset` meanwhile.
    pub unsafe fn set(self, offset: usize, value: T) {
        *self.0.add(offset) = value;
    }
}

/// The output rows (llm.c's hint_stats::Rows, host side; zeroed): tot, dist, cnt [n, 10] position-major, cell [n],
/// tokens and weights [n, KEEP]; `done` marks the positions whose hint row is written. Engines write their levels'
/// columns; the hint row of a position is the deepest hint level's whose context occurred, so the engines' lookups
/// run deepest level first and write a hint row only where none is yet.
pub struct Out {
    pub tot: Output<u32>,
    pub dist: Output<u16>,
    pub cnt: Output<u32>,
    pub cell: Output<i16>,
    pub tokens: Output<u16>,
    pub weights: Output<u16>,
    pub done: Output<u8>,
}

/// Scratch for write_context, one per thread.
#[derive(Default)]
pub struct Scratch {
    ranked: Vec<(Reverse<usize>, u16)>,
}

impl Out {
    /// One context's rows: `by_token` its (next token, count) pairs in token order (counts > 0), at the validation
    /// positions `at` (whose context it is): tot, dist and cnt (the target's count if among the context's TOP most
    /// frequent next tokens, ranked by count descending, then token ascending) in column `col`, and with a `bucket`,
    /// the hint row where none is written yet (cell from the bucket and the total's and the top token's share's
    /// bins; the top KEEP tokens and their weights, count / their sum * 65535).
    ///
    /// # Safety
    /// The positions are in bounds, and no other thread writes them meanwhile.
    #[inline]
    pub unsafe fn write_context(&self, col: usize, bucket: Option<usize>, by_token: &[(u16, u32)], at: impl IntoIterator<Item = usize>, target: &[u16], s: &mut Scratch) {
        let nl = LEVELS.len();
        let ranked = &mut s.ranked;
        ranked.clear();
        ranked.extend(by_token.iter().map(|&(t, c)| (Reverse(c as usize), t)));
        let total: usize = by_token.iter().map(|x| x.1 as usize).sum();
        let d = ranked.len();
        // the top TOP (its threshold at rank TOP - 1) and the top KEEP sorted by selections (a total order)
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
        for q in at {
            let t = target[q];
            let c = match by_token.binary_search_by_key(&t, |&(x, _)| x) {
                // among the top TOP: at or before the threshold in the order
                Ok(i) if thr.is_none_or(|th| (Reverse(by_token[i].1 as usize), t) <= th) => by_token[i].1,
                _ => 0,
            };
            self.tot.set(q * nl + col, tt);
            self.dist.set(q * nl + col, dd);
            self.cnt.set(q * nl + col, c);
            if let Some((cell, m, wsum)) = hint {
                if *self.done.0.add(q) == 0 {
                    self.done.set(q, 1);
                    self.cell.set(q, cell);
                    for (e, &(Reverse(cc), token)) in ranked[..m].iter().enumerate() {
                        self.tokens.set(q * KEEP + e, token);
                        self.weights.set(q * KEEP + e, (cc as f64 / wsum.max(1.0) * 65535.0).round_ties_even() as u16);
                    }
                }
            }
        }
    }
}

/// An engine: a level group's rows, its corpus built from the training shards beside the training and its lookups
/// of the validation contexts after the clock. Constructed before the clock (no validation tokens exist then).
pub trait Engine: Send {
    /// Before the clock, in the build pool: val-independent setup (allocations, pre-faulting).
    fn prepare(&mut self, _shards: &Shards) -> Result<Times, String> {
        Ok(Vec::new())
    }
    /// Beside the training, in the build pool: whatever the engine counts from the training shards alone.
    fn build(&mut self, shards: &Shards) -> Result<Times, String>;
    /// After the clock, in the lookup pool: its levels' rows at every validation position (count rows in their
    /// columns; hint rows deepest level first, where `done` is 0).
    fn lookup(&mut self, shards: &Shards, val: &Val, out: &Out) -> Result<Times, String>;
    /// Its deepest level (the engines' lookups run deepest first).
    fn deepest(&self) -> usize;
    /// Whether its lookup passes over the shards (the scan engine's does all of its work there, so it is a reference
    /// for cpuval_rows and checks, not for a timed run, whose lookups run after the clock).
    fn scans(&self) -> bool {
        false
    }
}
