//! llm.c's validation hint rows on CPUs (llmc/cpu_val.h is the C side), in phases so that nothing reads the
//! validation tokens before the clock stops:
//!
//!   cpuval_new(shard list file)          before the clock: the settings (env, below), the shards opened and their
//!                                        headers checked, the engines set up, the two thread pools started (idle)
//!   cpuval_start(handle)                 on the clock, beside the training (llm.c: at step 0, after the training
//!                                        rows): the engines' corpus builds begin in a background thread, on the build
//!                                        pool, from the training shards alone
//!   cpuval_wait(handle)                  on the clock, after the last step: the end of the build waited for
//!                                        (training-side work)
//!   cpuval_lookup(handle, val, n, out)   after the clock: the engines' lookups of the validation tokens on the lookup
//!                                        pool, writing the rows into `out` (lookups only: no pass over the shards, no
//!                                        allocation)
//!   cpuval_free(handle)
//!
//! An engine (rows.rs's Engine) computes a group of levels' rows:
//!   table  (table.rs, short.rs) levels up to 7: per level the (context, next token) counts of its shards, built
//!          beside the training; the lookup reads the validation contexts' rows from them
//!   long   (long/) levels 8-20 over all the shards: an anchor index of the shards built beside the training (the
//!          shards mapped before the clock); the lookup finds the validation contexts' occurrences through it
//!   scan   (val_build.rs) any levels: the shards scanned for the validation contexts, all of it in the lookup (the
//!          reference the others are checked against; llm.c takes it only for HINT_CPU_VAL_CHECK runs)
//!
//! Env (HINT_CPU_VAL_*; an unknown name is an error):
//!   LEVEL_SHARDS  "k:K,..." level k over the first K shards of the list (default 2:110,3:110,4:10,5:10,6:10; unlisted
//!                 levels, and K beyond the list: all of them)
//!   ENGINES       "engine:levels,..." which engine computes which levels ("a-b" or single levels; default
//!                 table:2-6,long:8-20)
//!   CPUS          the build pool's CPUs, a list "a-b,c,..." (default: unpinned); thread i on the i-th of them
//!   THREADS       the build pool's threads (default: the CPUs of CPUS, else all)
//!   LOOKUP_CPUS   the lookup pool's (default: those of CPUS); LOOKUP_THREADS its threads (default: the CPUs, else all)
//! The engines' own settings: HINT_CPU_VAL_TABLE_* (table.rs), HINT_CPU_VAL_LONG_* (long/mod.rs). HINT_CPU_VAL itself
//! and HINT_CPU_VAL_SHM / _CHECK are the C side's.
pub mod long;
pub mod rows;
pub mod short;
pub mod table;
mod val_build;

use rows::{Engine, Lv, Out, Output, Shards, Times, Val, HINT_LEVELS, LEVELS};
use std::ffi::{c_char, CStr};
use std::sync::{mpsc, Arc};
use std::thread::JoinHandle;
use std::time::Instant;

const VAL_TOKENS: usize = 10_485_760; // the most validation positions a lookup takes (the long engine's buffers)
const LEVEL_SHARDS: &str = "2:110,3:110,4:10,5:10,6:10";
const ENGINES: &str = "table:2-6,long:8-20";
const KNOWN: [&str; 9] = [
    "HINT_CPU_VAL",
    "HINT_CPU_VAL_SHM",
    "HINT_CPU_VAL_CHECK",
    "HINT_CPU_VAL_LEVEL_SHARDS",
    "HINT_CPU_VAL_ENGINES",
    "HINT_CPU_VAL_CPUS",
    "HINT_CPU_VAL_THREADS",
    "HINT_CPU_VAL_LOOKUP_CPUS",
    "HINT_CPU_VAL_LOOKUP_THREADS",
];

/// The output rows (llm.c's hint_stats::Rows, host side; zeroed): tot, dist, cnt [n, 10], cell [n], tokens and
/// weights [n, 8].
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CpuValOut {
    pub cell: *mut i16,
    pub tokens: *mut u16,
    pub weights: *mut u16,
    pub tot: *mut u32,
    pub dist: *mut u16,
    pub cnt: *mut u32,
}

/// The shards and the engines (deepest level first), moved into the build thread and back.
struct Corpus {
    shards: Shards,
    engines: Vec<(String, Box<dyn Engine>)>,
}

pub struct Handle {
    corpus: Option<Corpus>,
    /// kept until cpuval_free: the pool's shutdown (its threads' exits) is no part of the build
    build_pool: Arc<rayon::ThreadPool>,
    lookup_pool: rayon::ThreadPool,
    /// the build's start and its result, sent by the build thread the moment the build ends (the thread is joined
    /// in cpuval_free)
    job: Option<(Instant, mpsc::Receiver<(Corpus, Result<Times, String>)>)>,
    thread: Option<JoinHandle<()>>,
    /// after cpuval_wait: the build's start, the built corpus and the build's timings
    built: Option<(Instant, Corpus, Times)>,
    /// the lookup's per-position buffers (hint row written, segments), allocated and faulted before the clock
    done: Vec<u8>,
    avail: Vec<u32>,
    /// some engine's lookup passes over the shards (rows.rs's Engine::scans)
    scans: bool,
    /// after the lookup: the corpus and those buffers, freed by cpuval_free (after the clock)
    spent: Option<(Corpus, Vec<u8>, Vec<u32>)>,
    describe: String,
}

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

/// A list "a-b,c,..." of numbers.
fn ranges(what: &str, v: &str) -> Result<Vec<usize>, String> {
    let mut out = Vec::new();
    for item in v.split(',').filter(|s| !s.trim().is_empty()) {
        let bad = || format!("{what}: bad item {item}");
        match item.split_once('-') {
            Some((a, b)) => {
                let (a, b): (usize, usize) = (a.trim().parse().map_err(|_| bad())?, b.trim().parse().map_err(|_| bad())?);
                out.extend(a..=b);
            }
            None => out.push(item.trim().parse().map_err(|_| bad())?),
        }
    }
    Ok(out)
}

fn pin(cpu: usize) {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_SET(cpu, &mut set);
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
    }
}

/// A pool of `threads` threads (default: one per CPU of `cpus`, else all), thread i pinned to the i-th of `cpus`.
fn pool(name: &'static str, cpus: Vec<usize>, threads: Option<usize>) -> Result<rayon::ThreadPool, String> {
    let all = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let threads = threads.unwrap_or(if cpus.is_empty() { all } else { cpus.len() });
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(move |i| format!("{name}-{i}"))
        .start_handler(move |i| {
            if !cpus.is_empty() {
                pin(cpus[i % cpus.len()]);
            }
        })
        .build()
        .map_err(|e| e.to_string())
}

fn threads_env(name: &str) -> Result<Option<usize>, String> {
    env(name).map(|v| v.trim().parse().map_err(|_| format!("{name}={v}: not a number"))).transpose()
}

fn new(shard_list: &str) -> Result<Handle, String> {
    for (name, _) in std::env::vars() {
        let known = [&KNOWN[..], table::KNOWN, long::KNOWN].concat();
        if name.starts_with("HINT_CPU_VAL") && !known.contains(&name.as_str()) {
            return Err(format!("unknown setting {name}"));
        }
    }
    let list = std::fs::read_to_string(shard_list).map_err(|e| format!("{shard_list}: {e}"))?;
    let paths: Vec<&str> = list.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    if paths.is_empty() {
        return Err(format!("{shard_list} lists no shards"));
    }
    let shards = Shards::open(&paths)?;
    let nfiles = shards.files.len();
    // per level its shards
    let mut level_files = vec![nfiles; LEVELS.len()];
    let spec = env("HINT_CPU_VAL_LEVEL_SHARDS").unwrap_or(LEVEL_SHARDS.into());
    for item in spec.split(',').filter(|s| !s.trim().is_empty()) {
        let bad = || format!("HINT_CPU_VAL_LEVEL_SHARDS: bad item {item} (k:K, k one of {LEVELS:?}, K > 0)");
        let (k, s) = item.split_once(':').ok_or_else(bad)?;
        let (k, s): (usize, usize) = (k.trim().parse().map_err(|_| bad())?, s.trim().parse().map_err(|_| bad())?);
        let i = LEVELS.iter().position(|&l| l == k).ok_or_else(bad)?;
        if s == 0 {
            return Err(bad());
        }
        level_files[i] = s.min(nfiles);
    }
    // per engine its levels, every level in exactly one
    let mut owner: Vec<Option<String>> = vec![None; LEVELS.len()];
    let spec = env("HINT_CPU_VAL_ENGINES").unwrap_or(ENGINES.into());
    for item in spec.split(',').filter(|s| !s.trim().is_empty()) {
        let bad = || format!("HINT_CPU_VAL_ENGINES: bad item {item} (engine:a-b or engine:k)");
        let (name, ks) = item.split_once(':').ok_or_else(bad)?;
        let (a, b) = match ks.split_once('-') {
            Some((a, b)) => (a.trim().parse().map_err(|_| bad())?, b.trim().parse().map_err(|_| bad())?),
            None => {
                let k: usize = ks.trim().parse().map_err(|_| bad())?;
                (k, k)
            }
        };
        for (i, &k) in LEVELS.iter().enumerate() {
            if (a..=b).contains(&k) {
                if owner[i].is_some() {
                    return Err(format!("HINT_CPU_VAL_ENGINES: level {k} in two engines"));
                }
                owner[i] = Some(name.trim().to_string());
            }
        }
    }
    if let Some(i) = owner.iter().position(Option::is_none) {
        return Err(format!("HINT_CPU_VAL_ENGINES: no engine for level {}", LEVELS[i]));
    }
    let mut names: Vec<String> = owner.iter().flatten().cloned().collect();
    names.sort();
    names.dedup();
    let mut engines: Vec<(String, Box<dyn Engine>)> = Vec::new();
    for name in &names {
        let lvs: Vec<Lv> = (0..LEVELS.len())
            .filter(|&i| owner[i].as_ref() == Some(name))
            .map(|i| Lv { k: LEVELS[i], files: level_files[i], col: i, bucket: HINT_LEVELS.iter().position(|&h| h == LEVELS[i]) })
            .collect();
        let engine: Box<dyn Engine> = match name.as_str() {
            "scan" => Box::new(val_build::Scan::new(&lvs)?),
            "table" => Box::new(table::Table::new(&lvs)?),
            "long" => Box::new(long::LongEngine::new(&shards, &lvs, VAL_TOKENS)?),
            _ => return Err(format!("HINT_CPU_VAL_ENGINES: unknown engine {name}")),
        };
        engines.push((name.clone(), engine));
    }
    engines.sort_by_key(|e| std::cmp::Reverse(e.1.deepest()));
    let cpus = env("HINT_CPU_VAL_CPUS").map(|v| ranges("HINT_CPU_VAL_CPUS", &v)).transpose()?.unwrap_or_default();
    let lookup_cpus = env("HINT_CPU_VAL_LOOKUP_CPUS").map(|v| ranges("HINT_CPU_VAL_LOOKUP_CPUS", &v)).transpose()?.unwrap_or(cpus.clone());
    let build_pool = pool("cpuval-build", cpus, threads_env("HINT_CPU_VAL_THREADS")?)?;
    let lookup_pool = pool("cpuval-lookup", lookup_cpus, threads_env("HINT_CPU_VAL_LOOKUP_THREADS")?)?;
    // the engines' val-independent setup, before the clock, on the build pool (its threads fault the pages they use)
    let mut prepared: Vec<String> = Vec::new();
    for (name, e) in engines.iter_mut() {
        let t = build_pool.install(|| e.prepare(&shards))?;
        prepared.extend(t.into_iter().map(|(k, v)| format!("{name}_{k}={v:.3}")));
    }
    let caps: Vec<String> = LEVELS.iter().zip(&level_files).map(|(k, s)| format!("{k}:{s}")).collect();
    let describe = format!(
        "levels:shards {} of {nfiles}; engines {}; build {} threads on {}, lookup {} threads on {}; prepared {}",
        caps.join(","),
        spec,
        build_pool.current_num_threads(),
        env("HINT_CPU_VAL_CPUS").unwrap_or("any CPU".into()),
        lookup_pool.current_num_threads(),
        env("HINT_CPU_VAL_LOOKUP_CPUS").or(env("HINT_CPU_VAL_CPUS")).unwrap_or("any CPU".into()),
        prepared.join(" "),
    );
    // the lookup allocates nothing: its page faults could wait behind the build's frees
    let (mut done, mut avail) = (vec![0u8; VAL_TOKENS], vec![0u32; VAL_TOKENS]);
    fault(&mut done);
    fault(&mut avail);
    let scans = engines.iter().any(|(_, e)| e.scans());
    Ok(Handle { corpus: Some(Corpus { shards, engines }), build_pool: Arc::new(build_pool), lookup_pool, job: None, thread: None, built: None, done, avail, scans, spent: None, describe })
}

/// Writes a zero into every page of `v` (zeroed memory from the allocator is not yet faulted).
fn fault<T: Copy + Default>(v: &mut [T]) {
    let step = (4096 / std::mem::size_of::<T>()).max(1);
    for i in (0..v.len()).step_by(step) {
        unsafe { std::ptr::write_volatile(&mut v[i], T::default()) };
    }
}

fn start(h: &mut Handle) -> Result<(), String> {
    let Some(mut corpus) = h.corpus.take() else { return Err("already started".into()) };
    let pool = h.build_pool.clone();
    let (send, receive) = mpsc::channel();
    let t0 = Instant::now();
    h.job = Some((t0, receive));
    h.thread = Some(std::thread::spawn(move || {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pool.install(|| -> Result<Times, String> {
                let mut times = Vec::new();
                let Corpus { shards, engines } = &mut corpus;
                for (name, e) in engines.iter_mut() {
                    let t = Instant::now();
                    times.extend(e.build(shards)?.into_iter().map(|(k, v)| (format!("{name}_{k}"), v)));
                    times.push((format!("{name}_build_s"), t.elapsed().as_secs_f64()));
                }
                Ok(times)
            })
        }));
        let mut result = result.unwrap_or_else(|p| Err(format!("the build panicked: {}", panic_message(&*p))));
        if let Ok(t) = &mut result {
            t.push(("build_s".into(), t0.elapsed().as_secs_f64()));
        }
        let _ = send.send((corpus, result));
    }));
    Ok(())
}

fn wait(h: &mut Handle) -> Result<(), String> {
    let t0 = Instant::now();
    let Some((t_start, receive)) = h.job.take() else { return Err("not started".into()) };
    let (corpus, built) = receive.recv().map_err(|_| "the build thread ended without its result".to_string())?;
    let mut times = built?;
    times.push(("wait_s".into(), t0.elapsed().as_secs_f64()));
    h.built = Some((t_start, corpus, times));
    Ok(())
}

fn lookup(h: &mut Handle, tokens: &[u16], out: CpuValOut) -> Result<Times, String> {
    let Some((t_start, mut corpus, mut times)) = h.built.take() else { return Err("the build was not waited for".into()) };
    let t0 = Instant::now();
    let n = tokens.len() - 1;
    if n > VAL_TOKENS {
        return Err(format!("{n} validation positions, at most {VAL_TOKENS}"));
    }
    let (mut done, mut avail) = (std::mem::take(&mut h.done), std::mem::take(&mut h.avail));
    let rows = Out {
        tot: Output(out.tot),
        dist: Output(out.dist),
        cnt: Output(out.cnt),
        cell: Output(out.cell),
        tokens: Output(out.tokens),
        weights: Output(out.weights),
        done: Output(done.as_mut_ptr()),
    };
    let pool = &h.lookup_pool;
    let found = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pool.install(|| -> Result<(), String> {
        let val = Val::new(tokens, &mut avail);
        times.push(("lookup_segments_s".into(), t0.elapsed().as_secs_f64()));
        let Corpus { shards, engines } = &mut corpus;
        for (name, e) in engines.iter_mut() {
            let t = Instant::now();
            times.extend(e.lookup(shards, &val, &rows)?.into_iter().map(|(k, v)| (format!("{name}_{k}"), v)));
            times.push((format!("{name}_lookup_s"), t.elapsed().as_secs_f64()));
        }
        Ok(())
        })
    }));
    found.map_err(|p| format!("the lookup panicked: {}", panic_message(&*p)))??;
    times.push(("lookup_s".into(), t0.elapsed().as_secs_f64()));
    times.push(("since_start_s".into(), t_start.elapsed().as_secs_f64()));
    // freed by cpuval_free, after the clock (a free now would hold the memory map's lock during the copies)
    h.spent = Some((corpus, done, avail));
    Ok(times)
}

fn write_c(s: &str, buf: *mut c_char, len: usize) {
    if buf.is_null() || len == 0 {
        return;
    }
    let b = s.as_bytes();
    let m = b.len().min(len - 1);
    unsafe {
        std::ptr::copy_nonoverlapping(b.as_ptr().cast::<c_char>(), buf, m);
        *buf.add(m) = 0;
    }
}

fn panic_message(panic: &(dyn std::any::Any + Send)) -> String {
    panic.downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| panic.downcast_ref::<String>().cloned()).unwrap_or("(no message)".into())
}

/// Before the clock. Returns null on an error (its message in `err`).
///
/// # Safety
/// `shard_list` is NUL-terminated; `err` holds `err_len` bytes.
#[no_mangle]
pub unsafe extern "C" fn cpuval_new(shard_list: *const c_char, err: *mut c_char, err_len: usize) -> *mut Handle {
    let list = CStr::from_ptr(shard_list).to_string_lossy();
    match std::panic::catch_unwind(|| new(&list)) {
        Ok(Ok(h)) => Box::into_raw(Box::new(h)),
        Ok(Err(e)) => {
            write_c(&e, err, err_len);
            std::ptr::null_mut()
        }
        Err(p) => {
            write_c(&panic_message(&*p), err, err_len);
            std::ptr::null_mut()
        }
    }
}

/// The configuration, one line.
///
/// # Safety
/// `h` from cpuval_new; `buf` holds `len` bytes.
#[no_mangle]
pub unsafe extern "C" fn cpuval_describe(h: *const Handle, buf: *mut c_char, len: usize) {
    write_c(&(*h).describe, buf, len);
}

/// 1 if some engine's lookup passes over the shards (the scan engine), else 0.
///
/// # Safety
/// `h` from cpuval_new.
#[no_mangle]
pub unsafe extern "C" fn cpuval_scans(h: *const Handle) -> i32 {
    (*h).scans as i32
}

/// Beside the training: the corpus builds begin in a background thread (training shards only). 0, or 1 on an error.
///
/// # Safety
/// `h` from cpuval_new.
#[no_mangle]
pub unsafe extern "C" fn cpuval_start(h: *mut Handle) -> i32 {
    start(&mut *h).map_or(1, |_| 0)
}

/// On the clock, after the last step: waits for the builds. 0, or 1 and the error in `report`.
///
/// # Safety
/// `h` from cpuval_new, started; `report` holds `len` bytes.
#[no_mangle]
pub unsafe extern "C" fn cpuval_wait(h: *mut Handle, report: *mut c_char, len: usize) -> i32 {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| wait(&mut *h))) {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => {
            write_c(&e, report, len);
            1
        }
        Err(p) => {
            write_c(&panic_message(&*p), report, len);
            1
        }
    }
}

/// After cpuval_wait: `val` holds the first n + 1 validation tokens. Writes the rows into `out` (zeroed, n rows). 0
/// and the timings ("name=value ...": the build's and the lookup's), or 1 and the error, in `report`.
///
/// # Safety
/// `h` from cpuval_new, waited for; `val` holds n + 1 tokens, `out`'s buffers n rows; `report` holds `len` bytes.
#[no_mangle]
pub unsafe extern "C" fn cpuval_lookup(h: *mut Handle, val: *const u16, n: usize, out: CpuValOut, report: *mut c_char, len: usize) -> i32 {
    let tokens = std::slice::from_raw_parts(val, n + 1);
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| lookup(&mut *h, tokens, out))) {
        Ok(Ok(times)) => {
            let s: Vec<String> = times.iter().map(|(k, v)| format!("{k}={v:.3}")).collect();
            write_c(&s.join(" "), report, len);
            0
        }
        Ok(Err(e)) => {
            write_c(&e, report, len);
            1
        }
        Err(p) => {
            write_c(&panic_message(&*p), report, len);
            1
        }
    }
}

/// Frees the handle (joining a build still running).
///
/// # Safety
/// `h` from cpuval_new.
#[no_mangle]
pub unsafe extern "C" fn cpuval_free(h: *mut Handle) {
    if !h.is_null() {
        let mut h = Box::from_raw(h);
        if let Some(thread) = h.thread.take() {
            let _ = thread.join();
        }
    }
}
