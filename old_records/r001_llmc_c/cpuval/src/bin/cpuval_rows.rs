//! The validation rows without llm.c, for checks and timings: the library's phases as llm.c's rank 0 runs them, with
//! the settings from the same env (HINT_CPU_VAL_*).
//!
//!   cpuval_rows <shard list> <validation shard> <out dir> [--n N] [--lookup-at S]
//!
//! new, start, then (at S seconds after start with --lookup-at, as if training ran that long, else at once) the wait
//! for the build, the validation tokens read and the lookup. Writes tot / dist / cnt / cell / tokens / weights .npy into <out dir> (the
//! layout of old_records' ValCountRows / ValHintRows data dirs) and prints the timings, "name=value" per line.
use cpuval::rows::{shard_tokens, HEADER_BYTES, KEEP, LEVELS};
use cpuval::{cpuval_describe, cpuval_free, cpuval_lookup, cpuval_new, cpuval_start, cpuval_wait, CpuValOut};
use std::ffi::CString;
use std::fs::File;
use std::io::Write;
use std::os::unix::fs::FileExt;
use std::time::Instant;

const VAL_TOKENS: usize = 10_485_760;

fn npy<T: Copy>(path: &str, descr: &str, shape: &[usize], data: &[T]) {
    let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
    let dims = if shape.len() == 1 { format!("{},", dims[0]) } else { dims.join(", ") };
    let mut head = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': ({dims}), }}");
    while (10 + head.len() + 1) % 64 != 0 {
        head.push(' ');
    }
    head.push('\n');
    let mut f = File::create(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    f.write_all(b"\x93NUMPY\x01\x00").unwrap();
    f.write_all(&(head.len() as u16).to_le_bytes()).unwrap();
    f.write_all(head.as_bytes()).unwrap();
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), std::mem::size_of_val(data)) };
    f.write_all(bytes).unwrap();
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: cpuval_rows <shard list> <validation shard> <out dir> [--n N] [--lookup-at S]");
        std::process::exit(2);
    }
    let (list, val_path, out_dir) = (&args[1], &args[2], &args[3]);
    let (mut n, mut at) = (VAL_TOKENS, None::<f64>);
    let mut i = 4;
    while i + 1 < args.len() {
        match args[i].as_str() {
            "--n" => n = args[i + 1].parse().expect("--n N"),
            "--lookup-at" => at = Some(args[i + 1].parse().expect("--lookup-at S")),
            a => panic!("unknown argument {a}"),
        }
        i += 2;
    }
    std::fs::create_dir_all(out_dir).unwrap();
    let mut err = vec![0u8; 4096];
    let t = Instant::now();
    let h = unsafe { cpuval_new(CString::new(list.as_str()).unwrap().as_ptr(), err.as_mut_ptr().cast(), err.len()) };
    if h.is_null() {
        eprintln!("cpuval_new: {}", String::from_utf8_lossy(&err).trim_end_matches('\0'));
        std::process::exit(1);
    }
    println!("new_s={:.3}", t.elapsed().as_secs_f64());
    let mut buf = vec![0u8; 4096];
    unsafe { cpuval_describe(h, buf.as_mut_ptr().cast(), buf.len()) };
    println!("describe={}", String::from_utf8_lossy(&buf).trim_end_matches('\0'));
    let nl = LEVELS.len();
    let (mut tot, mut dist, mut cnt) = (vec![0u32; n * nl], vec![0u16; n * nl], vec![0u32; n * nl]);
    let (mut cell, mut tokens, mut weights) = (vec![0i16; n], vec![0u16; n * KEEP], vec![0u16; n * KEEP]);
    let out = CpuValOut { cell: cell.as_mut_ptr(), tokens: tokens.as_mut_ptr(), weights: weights.as_mut_ptr(), tot: tot.as_mut_ptr(), dist: dist.as_mut_ptr(), cnt: cnt.as_mut_ptr() };
    let t0 = Instant::now();
    assert_eq!(unsafe { cpuval_start(h) }, 0, "cpuval_start");
    if let Some(s) = at {
        std::thread::sleep(std::time::Duration::from_secs_f64(s));
    }
    let mut report = vec![0u8; 1 << 16];
    let tw = Instant::now();
    if unsafe { cpuval_wait(h, report.as_mut_ptr().cast(), report.len()) } != 0 {
        eprintln!("cpuval_wait: {}", String::from_utf8_lossy(&report).trim_end_matches('\0'));
        std::process::exit(1);
    }
    let waited = tw.elapsed().as_secs_f64();
    // the validation tokens, read only now (as llm.c does after the clock stops)
    let t1 = Instant::now();
    let f = File::open(val_path).unwrap_or_else(|e| panic!("{val_path}: {e}"));
    assert!(shard_tokens(&f, val_path).unwrap() > n, "the validation shard is too short");
    let mut val = vec![0u16; n + 1];
    let bytes = unsafe { std::slice::from_raw_parts_mut(val.as_mut_ptr().cast::<u8>(), 2 * (n + 1)) };
    f.read_exact_at(bytes, HEADER_BYTES as u64).unwrap();
    let read = t1.elapsed().as_secs_f64();
    let rc = unsafe { cpuval_lookup(h, val.as_ptr(), n, out, report.as_mut_ptr().cast(), report.len()) };
    let after = t1.elapsed().as_secs_f64();
    let total = t0.elapsed().as_secs_f64();
    let report = String::from_utf8_lossy(&report).trim_end_matches('\0').to_string();
    if rc != 0 {
        eprintln!("cpuval_lookup: {report}");
        std::process::exit(1);
    }
    for kv in report.split_whitespace() {
        println!("{kv}");
    }
    println!("waited_s={waited:.3}\nread_val_s={read:.3}\nafter_clock_s={after:.3}\nwall_s={total:.3}");
    unsafe { cpuval_free(h) };
    npy(&format!("{out_dir}/tot.npy"), "<u4", &[n, nl], &tot);
    npy(&format!("{out_dir}/dist.npy"), "<u2", &[n, nl], &dist);
    npy(&format!("{out_dir}/cnt.npy"), "<u4", &[n, nl], &cnt);
    npy(&format!("{out_dir}/cell.npy"), "<i2", &[n], &cell);
    npy(&format!("{out_dir}/tokens.npy"), "<u2", &[n, KEEP], &tokens);
    npy(&format!("{out_dir}/weights.npy"), "<u2", &[n, KEEP], &weights);
}
