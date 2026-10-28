//! The `table` engine: levels 2-7 from short.rs's val-independent (context, next token) count tables,
//! built from the training shards beside the training and looked up after the clock.
//!
//! Env (HINT_CPU_VAL_TABLE_*):
//!   ROUND_GB     a scatter round's occurrences at most this many GB (default 200): the shards per round
//!   PBITS        "k:b,..." level k's table in 2^b partitions (default 13 for levels 2-3, 14 above)
//!   PREFAULT_GB  the build's scratch pre-faulted before the clock (default: one round's)
//!   KEEP_GB      the tables' memory pre-faulted before the clock (default: the estimate below, +15%)
use crate::rows::{Engine, Lv, Out, Shards, Times, Val};
use crate::short::{Level, Scratch};
use std::time::Instant;

pub const KNOWN: &[&str] = &["HINT_CPU_VAL_TABLE_ROUND_GB", "HINT_CPU_VAL_TABLE_PBITS", "HINT_CPU_VAL_TABLE_PREFAULT_GB", "HINT_CPU_VAL_TABLE_KEEP_GB"];

const TOKENS_PER_SHARD: f64 = 1e8;

pub struct Table {
    levels: Vec<Lv>,
    round_gb: f64,
    pbits: Vec<(usize, u32)>,
    prefault_gb: Option<f64>,
    keep_gb: f64,
    scratch: Option<Scratch>,
    built: Vec<Level>,
}

fn env_f64(name: &str) -> Result<Option<f64>, String> {
    match std::env::var(name).ok().filter(|v| !v.is_empty()) {
        Some(v) => v.trim().parse().map(Some).map_err(|_| format!("{name}={v}: not a number")),
        None => Ok(None),
    }
}

impl Table {
    pub fn new(levels: &[Lv]) -> Result<Self, String> {
        if levels.iter().any(|l| l.k < 2 || l.k > 7) {
            return Err("table: levels 2 to 7".into());
        }
        let mut pbits = Vec::new();
        if let Ok(v) = std::env::var("HINT_CPU_VAL_TABLE_PBITS") {
            for item in v.split(',').filter(|s| !s.trim().is_empty()) {
                let bad = || format!("HINT_CPU_VAL_TABLE_PBITS: bad item {item}");
                let (k, b) = item.split_once(':').ok_or_else(bad)?;
                pbits.push((k.trim().parse().map_err(|_| bad())?, b.trim().parse().map_err(|_| bad())?));
            }
        }
        Ok(Self {
            levels: levels.to_vec(),
            round_gb: env_f64("HINT_CPU_VAL_TABLE_ROUND_GB")?.unwrap_or(200.0),
            pbits,
            prefault_gb: env_f64("HINT_CPU_VAL_TABLE_PREFAULT_GB")?,
            keep_gb: match env_f64("HINT_CPU_VAL_TABLE_KEEP_GB")? {
                Some(g) => g,
                None => 1.15 * levels.iter().map(|l| Self::table_gb(l.k, l.files)).sum::<f64>(),
            },
            scratch: None,
            built: Vec::new(),
        })
    }

    /// A level's table GB over K shards (power laws fitted to the measured K = 10..300, within ~10%).
    fn table_gb(k: usize, files: usize) -> f64 {
        let (c, e) = match k {
            2 => (0.343, 0.68),
            3 => (0.855, 0.78),
            4 => (1.36, 0.86),
            5 => (1.73, 0.90),
            6 => (1.97, 0.93),
            _ => (2.2, 0.95),
        };
        c * (files.max(1) as f64).powf(e)
    }

    fn entry_bytes(k: usize) -> f64 {
        if k <= 3 {
            8.0
        } else {
            16.0
        }
    }

    /// Level l's shards per scatter round.
    fn round_files(&self, l: &Lv) -> usize {
        ((self.round_gb * 1e9 / (TOKENS_PER_SHARD * Self::entry_bytes(l.k))) as usize).clamp(1, l.files.max(1))
    }

    fn part_bits(&self, k: usize) -> u32 {
        self.pbits.iter().find(|p| p.0 == k).map(|p| p.1).unwrap_or(if k <= 3 { 13 } else { 14 })
    }
}

impl Engine for Table {
    /// Before the clock, in the build pool: the scratch (one round's occurrences, ~10% slack) and KEEP_GB pre-faulted.
    fn prepare(&mut self, _shards: &Shards) -> Result<Times, String> {
        let t = Instant::now();
        let round_bytes = self.levels.iter().map(|l| self.round_files(l) as f64 * TOKENS_PER_SHARD * Self::entry_bytes(l.k)).fold(0.0, f64::max);
        let prefault = self.prefault_gb.map_or(round_bytes * 1.1, |g| g * 1e9);
        let threads = rayon::current_num_threads();
        self.scratch = Some(Scratch::new(prefault as usize / threads, (self.keep_gb * 1e9) as usize / threads));
        Ok(vec![("prefault_gb".into(), prefault / 1e9), ("keep_gb".into(), self.keep_gb), ("prepare_s".into(), t.elapsed().as_secs_f64())])
    }

    /// The levels' tables, one after the other, each over its first K_k shards.
    fn build(&mut self, shards: &Shards) -> Result<Times, String> {
        let mut times = Vec::new();
        if self.scratch.is_none() {
            self.scratch = Some(Scratch::new(0, 0));
        }
        let scratch = self.scratch.as_ref().unwrap();
        for l in &self.levels {
            let t = Instant::now();
            let n = l.files.min(shards.files.len());
            let round = self.round_files(l).min(n.max(1));
            let pb = self.part_bits(l.k);
            let level = Level::build(&shards.files[..n], &shards.sizes[..n], l.k, round, pb, scratch);
            times.extend(level.times().iter().map(|(name, v)| (format!("k{}_{name}", l.k), *v)));
            times.push((format!("k{}_s", l.k), t.elapsed().as_secs_f64()));
            self.built.push(level);
        }
        Ok(times)
    }

    fn lookup(&mut self, _shards: &Shards, val: &Val, out: &Out) -> Result<Times, String> {
        if self.built.len() != self.levels.len() {
            return Err("table: not built".into());
        }
        Ok(crate::short::lookup(&self.built, &self.levels, val, out))
    }

    fn deepest(&self) -> usize {
        self.levels.iter().map(|l| l.k).max().unwrap_or(0)
    }
}
