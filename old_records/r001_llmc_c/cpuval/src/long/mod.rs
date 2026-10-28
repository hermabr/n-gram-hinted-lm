//! The engine `long`: the long levels 8, 10, 12, 16 and 20, each over all the shards. Beside the training it builds
//! a validation-independent anchor index of the shards (`index.rs`: per NUMA node, the w-gram anchors every
//! 8 - w + 1 positions bucketed by hash); after the clock it looks the validation contexts up in it (`long.rs`:
//! the validation side, the probes of the candidates' corpus positions and the rows; `util.rs`: the scan engine's
//! helpers, `numa.rs`: node-bound memory and per-node work queues). Before the clock (prepare, on the build pool) it
//! maps the shards and allocates and faults all its memory (w = 5: index 206 GB + directories 17 GB + build scratch
//! ~5 GB + lookup side ~28 GB), every part of it bound to a node or interleaved over the nodes, so neither the build
//! nor the lookup allocates, and the kernel's automatic NUMA balancing leaves it alone.
//!
//! Its settings (env):
//!   HINT_CPU_VAL_LONG_W   the anchors' width: 5 (default; an anchor every 4 positions, 25.7B entries of 8 bytes over
//!                         the 1028 shards), 6 (every 3: 34.3B; a 25% longer build, a lookup ~1 s shorter) or 7
//!                         (every 2: 51.4B; twice the build, a lookup 0.1-0.2 s shorter than 6's)
mod index;
mod long;
mod numa;
mod util;

use crate::rows::{Engine, Lv, Out, Shards, Times, Val};
use long::{Long, RowsOut, LONG};
use std::time::Instant;

pub const KNOWN: &[&str] = &["HINT_CPU_VAL_LONG_W"];

pub struct LongEngine {
    w: usize,
    n_max: usize,
    long: Option<Long>,
    cols: [usize; 5],
    buckets: [Option<usize>; 5],
    pub describe: String,
}

impl LongEngine {
    /// Before the clock: the settings checked (prepare() allocates, for up to `n_max` validation positions).
    pub fn new(shards: &Shards, levels: &[Lv], n_max: usize) -> Result<Self, String> {
        let ks: Vec<usize> = levels.iter().map(|l| l.k).collect();
        if ks != LONG {
            return Err(format!("the long engine serves levels {LONG:?}, got {ks:?}"));
        }
        if let Some(l) = levels.iter().find(|l| l.files < shards.files.len()) {
            return Err(format!("the long engine counts every level over all the shards (level {} over {})", l.k, l.files));
        }
        let w = match std::env::var("HINT_CPU_VAL_LONG_W").ok().filter(|v| !v.is_empty()) {
            None => 5,
            Some(v) => match v.parse() {
                Ok(w @ 5..=7) => w,
                _ => return Err(format!("HINT_CPU_VAL_LONG_W={v}: 5, 6 or 7")),
            },
        };
        let describe = format!("long: levels 8-20 over {} shards, an anchor index (w={w}) built beside the training", shards.files.len());
        Ok(Self { w, n_max, long: None, cols: std::array::from_fn(|i| levels[i].col), buckets: std::array::from_fn(|i| levels[i].bucket), describe })
    }
}

impl Engine for LongEngine {
    /// Before the clock, on the build pool: the shards mapped, the index and the lookup's buffers allocated and faulted.
    fn prepare(&mut self, shards: &Shards) -> Result<Times, String> {
        let t = Instant::now();
        let mut long = Long::new(&shards.files, &shards.sizes, self.w, self.n_max);
        long.index.prepare();
        let [e, d, s, v] = long.bytes();
        self.long = Some(long);
        Ok(vec![
            ("prepare_s".into(), t.elapsed().as_secs_f64()),
            ("index_gb".into(), e as f64 / 1e9),
            ("dir_gb".into(), d as f64 / 1e9),
            ("scratch_gb".into(), s as f64 / 1e9),
            ("val_gb".into(), v as f64 / 1e9),
        ])
    }

    fn build(&mut self, _shards: &Shards) -> Result<Times, String> {
        Ok(self.long.as_mut().ok_or("the long engine was not prepared")?.index.build())
    }

    fn lookup(&mut self, _shards: &Shards, val: &Val, out: &Out) -> Result<Times, String> {
        let ro = RowsOut {
            nl: crate::rows::LEVELS.len(),
            cols: self.cols,
            buckets: self.buckets,
            tot: out.tot.0,
            dist: out.dist.0,
            cnt: out.cnt.0,
            cell: out.cell.0,
            tokens: out.tokens.0,
            weights: out.weights.0,
            done: out.done.0,
        };
        self.long.as_mut().ok_or("the long engine was not prepared")?.lookup(val.x, val.target, &val.avail, &ro)
    }

    fn deepest(&self) -> usize {
        LONG[LONG.len() - 1]
    }
}
