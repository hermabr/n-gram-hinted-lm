//! NUMA helpers: the nodes' CPUs, memory bound to a node or interleaved over them (huge pages, faulted before the
//! clock; memory with a policy of its own is also left alone by the kernel's automatic NUMA balancing, whose page
//! migrations would otherwise land in the lookup), and per-node work queues run on every thread of the current rayon
//! pool (each thread takes its own node's items first).
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Per CPU its node (from /sys; one node if unknown).
pub(crate) fn cpu_nodes() -> Vec<usize> {
    let mut map = Vec::new();
    for node in 0..64 {
        let Ok(s) = std::fs::read_to_string(format!("/sys/devices/system/node/node{node}/cpulist")) else { continue };
        for part in s.trim().split(',').filter(|p| !p.is_empty()) {
            let (a, b) = part.split_once('-').unwrap_or((part, part));
            let (a, b): (usize, usize) = (a.parse().unwrap_or(0), b.parse().unwrap_or(0));
            if map.len() <= b {
                map.resize(b + 1, 0);
            }
            for c in a..=b {
                map[c] = node;
            }
        }
    }
    map
}

#[inline]
pub(crate) fn this_node(map: &[usize]) -> usize {
    let cpu = unsafe { libc::sched_getcpu() };
    if cpu < 0 {
        0
    } else {
        map.get(cpu as usize).copied().unwrap_or(0)
    }
}

/// The pages of [p, p + len) (page-aligned) bound to `node`, or (None) interleaved over the nodes with CPUs.
pub(crate) fn place(p: *mut u8, len: usize, node: Option<usize>) {
    const MPOL_BIND: i64 = 2;
    const MPOL_INTERLEAVE: i64 = 3;
    let mut mask = [0u64; 16];
    let mode = match node {
        Some(n) => {
            mask[n / 64] |= 1 << (n % 64);
            MPOL_BIND
        }
        None => {
            for n in cpu_nodes() {
                mask[n / 64] |= 1u64 << (n % 64);
            }
            MPOL_INTERLEAVE
        }
    };
    unsafe { libc::syscall(libc::SYS_mbind, p, len, mode, mask.as_ptr(), 1024u64, 0u32) };
}

/// `bytes` of anonymous memory as transparent huge pages, bound to `node` (None: interleaved); not faulted.
pub(crate) fn map_on(bytes: usize, node: Option<usize>) -> *mut u8 {
    let len = bytes.max(1).next_multiple_of(1 << 21);
    let flags = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE;
    let p = unsafe { libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ | libc::PROT_WRITE, flags, -1, 0) };
    assert!(p != libc::MAP_FAILED, "mmap of {bytes} bytes failed");
    unsafe { libc::madvise(p, len, libc::MADV_HUGEPAGE) };
    place(p.cast(), len, node);
    p.cast()
}

/// Memory of `n` T (zeroed, faulted in parallel now), bound to a node (None: interleaved); freed on drop.
pub(crate) struct NodeVec<T> {
    p: *mut T,
    n: usize,
}
unsafe impl<T> Send for NodeVec<T> {}
unsafe impl<T> Sync for NodeVec<T> {}

impl<T: Copy> NodeVec<T> {
    pub(crate) fn new(n: usize, node: Option<usize>) -> Self {
        let bytes = n * std::mem::size_of::<T>() + 64;
        let p = map_on(bytes, node);
        // fault every page now (zero fill), in parallel
        let pages = bytes.div_ceil(4096);
        let base = p as usize;
        (0..pages.div_ceil(512)).into_par_iter().for_each(|c| {
            for pg in c * 512..pages.min(c * 512 + 512) {
                unsafe { std::ptr::write_volatile((base + pg * 4096) as *mut u8, 0u8) };
            }
        });
        Self { p: p.cast(), n }
    }
    #[inline]
    pub(crate) fn ptr(&self) -> *mut T {
        self.p
    }
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.n
    }
    #[inline]
    pub(crate) fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.p, self.n) }
    }
    #[inline]
    #[allow(clippy::mut_from_ref)]
    pub(crate) unsafe fn slice_mut(&self, a: usize, b: usize) -> &mut [T] {
        std::slice::from_raw_parts_mut(self.p.add(a), b - a)
    }
}

impl<T> Drop for NodeVec<T> {
    fn drop(&mut self) {
        let len = (self.n * std::mem::size_of::<T>() + 64).max(1).next_multiple_of(1 << 21);
        unsafe { libc::munmap(self.p.cast(), len) };
    }
}

/// Runs `f(item, thread index)` for every item of every node's queue on all threads of the current pool: each thread
/// takes its node's items first (queue `q` serves the threads of node `nodes[q]`), then helps the other queues.
pub(crate) fn run_queues<F: Fn(usize, usize, usize) + Sync>(nodes: &[usize], lens: &[usize], f: F) {
    let map = cpu_nodes();
    let next: Vec<AtomicUsize> = lens.iter().map(|_| AtomicUsize::new(0)).collect();
    rayon::broadcast(|ctx| {
        let me = this_node(&map);
        let own = nodes.iter().position(|&n| n == me).unwrap_or(ctx.index() % lens.len().max(1));
        for k in 0..lens.len() {
            let q = (own + k) % lens.len();
            loop {
                let i = next[q].fetch_add(1, Ordering::Relaxed);
                if i >= lens[q] {
                    break;
                }
                f(q, i, ctx.index());
            }
        }
    });
}
