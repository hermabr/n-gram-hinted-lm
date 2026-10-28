//! The cache-build ladder: earlier designs of the validation index, each building about the same data as the record's
//! SlotIndex (exact_match/src/slot.rs) from the first `tokens` tokens of the training shards, for the blog post's
//! build-speed comparison (src/retrieval_blog_post/ladder.py times them).
//!
//! Every index here has the same interface: `create(files, tokens, threads, cpus)` does what the design did before
//! the clock (opening the shards, allocating and faulting in memory where it did), and `build()` is the timed part. It
//! reads the tokens itself (they are in the page cache), runs on `threads` threads pinned round-robin to `cpus`
//! without the GIL, and returns statistics of what it built as a dict of numbers.
use pyo3::{exceptions::PyValueError, prelude::*};
use std::fs::File;

mod ckfast;
mod hashmap;
mod per_length;
mod slot_branch_free;
mod slot_ifma;
mod slot_one_pass;
mod slot_v1;

/// A shard file's header: 256 int32 (magic, version, token count, ...), then the tokens as u16.
pub(crate) const HEADER_BYTES: usize = 1024;
pub(crate) const BOS: u16 = 50256;

/// A training shard, mapped read-only: `tokens()` is its first `len` tokens.
pub(crate) struct Shard {
    pub path: String,
    pub file: File,
    pub len: usize,
    ptr: *const u8,
    bytes: usize,
}
unsafe impl Send for Shard {}
unsafe impl Sync for Shard {}

impl Shard {
    pub fn tokens(&self) -> &[u16] {
        unsafe { std::slice::from_raw_parts(self.ptr.add(HEADER_BYTES).cast(), self.len) }
    }
}

impl Drop for Shard {
    fn drop(&mut self) {
        unsafe { libc::munmap(self.ptr as *mut libc::c_void, self.bytes) };
    }
}

/// The shards of `files` that hold the first `tokens` tokens, in order, the last one cut to fit. Fails if the files
/// hold fewer.
pub(crate) fn shards(files: &[String], tokens: usize) -> PyResult<Vec<Shard>> {
    use std::io::Read;
    use std::os::fd::AsRawFd;
    let mut out = Vec::new();
    let mut left = tokens;
    for path in files {
        if left == 0 {
            break;
        }
        let mut file = File::open(path)?;
        let mut header = [0u8; HEADER_BYTES];
        file.read_exact(&mut header)?;
        let word = |i: usize| i32::from_le_bytes(header[i..i + 4].try_into().unwrap());
        let n = word(8);
        let bytes = HEADER_BYTES + 2 * n.max(0) as usize;
        if word(0) != 20240520 || word(4) != 1 || n < 0 || file.metadata()?.len() != bytes as u64 {
            return Err(PyValueError::new_err(format!("invalid shard: {path}")));
        }
        let ptr = unsafe { libc::mmap(std::ptr::null_mut(), bytes, libc::PROT_READ, libc::MAP_SHARED, file.as_raw_fd(), 0) };
        if ptr == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error().into());
        }
        let len = (n as usize).min(left);
        left -= len;
        out.push(Shard { path: path.clone(), file, len, ptr: ptr as *const u8, bytes });
    }
    if left > 0 {
        return Err(PyValueError::new_err(format!("the files hold {} tokens, fewer than {tokens}", tokens - left)));
    }
    Ok(out)
}

/// A rayon pool of `threads` named "{name}-{i}", thread i pinned to cpus[i % len] if `cpus` is non-empty.
pub(crate) fn pinned_pool(threads: usize, cpus: &[usize], name: &'static str) -> PyResult<rayon::ThreadPool> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(move |i| format!("{name}-{i}"))
        .build()
        .map_err(|e| PyValueError::new_err(format!("{name} pool: {e}")))?;
    let pinned = pool.broadcast(|ctx| -> std::io::Result<()> {
        if cpus.is_empty() {
            return Ok(());
        }
        let cpu = cpus[ctx.index() % cpus.len()];
        let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
        unsafe { libc::CPU_SET(cpu, &mut set) };
        if unsafe { libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) } != 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(())
    });
    pinned.into_iter().collect::<std::io::Result<()>>()?;
    Ok(pool)
}

#[pymodule]
fn cache_ladder(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<hashmap::HashMapIndex>()?;
    m.add_class::<per_length::PerLengthTables>()?;
    m.add_class::<ckfast::FastChecksumIndex>()?;
    m.add_class::<slot_v1::SlotIndexV1>()?;
    m.add_class::<slot_branch_free::SlotIndexBranchFree>()?;
    m.add_class::<slot_ifma::SlotIndexIfma>()?;
    m.add_class::<slot_one_pass::SlotIndexOnePass>()
}
