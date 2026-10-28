//! The validation-independent anchor index of the shards, built beside the training, and the after-step candidates
//! from it.
//!
//! Anchors: in each shard the w-grams starting at offsets a = j * step (step = 8 - w + 1), so every 8-context of the
//! shard contains exactly one (at offset o = a - its start, 0 <= o < step). The shards are split into one group per
//! NUMA node (consecutive shards, ~equal tokens); each group is an index of its own in its node's memory, built and
//! probed by that node's threads. Entry = the anchor w-gram's filter key (fkey) bits after its top CB bits (the
//! coarse bucket, implicit) | (w < 7) per offset o the top 2-3 fkey bits of the 8-context o tokens before the anchor
//! | local shard << low_bits | j. Per coarse bucket by fine buckets (the entry's top FB bits), within one by
//! sub-buckets (the next SB bits; a directory of u32 offsets per sub-bucket); the remaining hash bits stay in the entry
//! as a check. Build (training shards only; the anchors' keys 8 at a time with AVX-512 where the CPU has it): pass 1
//! counts per (piece, coarse bucket), pass 2 scatters the entries through write-combining lines (streaming stores),
//! pass 3 per coarse bucket by fine bucket into fixed-room regions of the thread's buffer (a spill for the rest), then
//! per fine bucket (in cache) by sub-bucket back. Every buffer is allocated and faulted before the clock.
//!
//! After the step: the distinct w-grams of the validation 8-contexts with, per offset, the set of the fingerprints of
//! the validation 8-contexts that have them there; per group and w-gram its sub-bucket's entries with its check bits;
//! per entry and offset whose fingerprint is in the set the corpus 8-context at that offset, tested against the
//! 8-contexts' filter and probed level by level as val_build's scan does.
use super::long::{Ext, VLevel, Work};
use super::numa::{cpu_nodes, run_queues, NodeVec};
use super::util::{fkey, hash, huge_zeroed, Filter};
use crate::rows::HEADER_BYTES;
use rayon::prelude::*;
use std::arch::x86_64::{_mm_loadu_si128, _mm_prefetch, _mm_sfence, _mm_stream_si128, __m128i, _MM_HINT_T0};
use std::fs::File;
use std::os::fd::AsRawFd;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

const CB: u32 = 13; // coarse bucket bits
const FB: u32 = 8; // fine bucket bits
const SB: u32 = 10; // sub-bucket bits
const PIECE_ANCHORS: usize = 1 << 23;
const K0: usize = 8;
const FBUF: usize = 1 << 16; // a fine bucket sorted in the thread's cache-sized buffer if at most this many entries
pub(crate) const EXT_LIM: usize = 32; // the direct path's longest chain (else table probes)

/// The shards mapped (MAP_POPULATE, before the clock): per shard its tokens.
pub(crate) struct Corpus {
    toks: Vec<(usize, usize)>, // (pointer, tokens)
    maps: Vec<(usize, usize)>,
}

impl Corpus {
    fn map(files: &[File], sizes: &[usize]) -> Self {
        let maps: Vec<(usize, usize)> = files
            .par_iter()
            .zip(sizes)
            .map(|(f, &n)| {
                let len = HEADER_BYTES + 2 * n;
                let p = unsafe { libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ, libc::MAP_SHARED | libc::MAP_POPULATE, f.as_raw_fd(), 0) };
                assert!(p != libc::MAP_FAILED, "shard mmap failed");
                (p as usize, len)
            })
            .collect();
        let toks = maps.iter().zip(sizes).map(|(&(p, _), &n)| (p + HEADER_BYTES, n)).collect();
        Self { toks, maps }
    }

    #[inline]
    pub(crate) fn shard(&self, s: usize) -> &[u16] {
        let (p, n) = self.toks[s];
        unsafe { std::slice::from_raw_parts(p as *const u16, n) }
    }
}

impl Drop for Corpus {
    fn drop(&mut self) {
        for &(p, len) in &self.maps {
            unsafe { libc::munmap(p as *mut libc::c_void, len) };
        }
    }
}

unsafe impl Send for Corpus {}
unsafe impl Sync for Corpus {}

/// Scatter through write-combining lines: per destination bucket its current line (8 entries) in `buf`, its next
/// position `cur` and its segment's start `seg` (lines wholly inside the segment are written with streaming stores,
/// the segment's partial first and last lines with plain stores).
struct Wc {
    buf: *mut u64,
    cur: *mut u64,
    seg: *mut u64,
}

impl Wc {
    #[inline(always)]
    unsafe fn push(&self, b: usize, e: u64, dst: *mut u64) {
        let pos = *self.cur.add(b);
        let line = self.buf.add(b * 8);
        *line.add(pos as usize & 7) = e;
        *self.cur.add(b) = pos + 1;
        if pos & 7 == 7 {
            let l0 = pos - 7;
            let s = *self.seg.add(b);
            if l0 >= s {
                let d = dst.add(l0 as usize) as *mut __m128i;
                let src = line as *const __m128i;
                _mm_stream_si128(d, *src);
                _mm_stream_si128(d.add(1), *src.add(1));
                _mm_stream_si128(d.add(2), *src.add(2));
                _mm_stream_si128(d.add(3), *src.add(3));
            } else {
                for k in s.saturating_sub(l0) as usize..8 {
                    *dst.add(l0 as usize + k) = *line.add(k);
                }
            }
        }
    }

    /// The buckets' last partial lines written.
    unsafe fn finish(&self, nb: usize, dst: *mut u64) {
        for b in 0..nb {
            let pos = *self.cur.add(b);
            if pos & 7 != 0 {
                let l0 = pos & !7;
                let from = l0.max(*self.seg.add(b));
                for q in from..pos {
                    *dst.add(q as usize) = *self.buf.add(b * 8 + (q & 7) as usize);
                }
            }
        }
        _mm_sfence();
    }
}

/// `n` u64 copied from `src` to `dst`, the whole lines of dst with streaming stores.
unsafe fn nt_copy(dst: *mut u64, src: *const u64, n: usize) {
    let mut i = 0;
    while i < n && (dst.add(i) as usize) & 63 != 0 {
        *dst.add(i) = *src.add(i);
        i += 1;
    }
    while i + 8 <= n {
        let d = dst.add(i) as *mut __m128i;
        let s = src.add(i) as *const __m128i;
        _mm_stream_si128(d, _mm_loadu_si128(s));
        _mm_stream_si128(d.add(1), _mm_loadu_si128(s.add(1)));
        _mm_stream_si128(d.add(2), _mm_loadu_si128(s.add(2)));
        _mm_stream_si128(d.add(3), _mm_loadu_si128(s.add(3)));
        i += 8;
    }
    while i < n {
        *dst.add(i) = *src.add(i);
        i += 1;
    }
}

/// The filter keys (fkey) of 8 anchors' w-grams at once (AVX-512): anchor k at tokens p + k * step. Reads 64 tokens
/// from p.
#[target_feature(enable = "avx512f,avx512bw,avx512dq")]
unsafe fn fkeys8(p: *const u16, ia: std::arch::x86_64::__m512i, ib: std::arch::x86_64::__m512i, n: u64, out: &mut [u64; 8]) {
    use std::arch::x86_64::*;
    let lo = _mm512_loadu_si512(p.cast());
    let hi = _mm512_loadu_si512(p.add(32).cast());
    let a = _mm512_permutex2var_epi16(lo, ia, hi);
    let b = _mm512_permutex2var_epi16(lo, ib, hi);
    let h = _mm512_mullo_epi64(a, _mm512_set1_epi64(0x9e3779b97f4a7c15u64 as i64));
    let h = _mm512_xor_si512(_mm512_xor_si512(h, b), _mm512_set1_epi64(n as i64));
    let h = _mm512_mullo_epi64(h, _mm512_set1_epi64(0xd6e8feb86659fd93u64 as i64));
    _mm512_storeu_si512(out.as_mut_ptr().cast(), h);
}

/// The permutation indices of fkeys8 for width w and step: per anchor k its first and last 4 tokens.
#[target_feature(enable = "avx512f,avx512bw,avx512dq")]
unsafe fn fkey_idx(w: usize, step: usize) -> (std::arch::x86_64::__m512i, std::arch::x86_64::__m512i) {
    let mut ia = [0u16; 32];
    let mut ib = [0u16; 32];
    for k in 0..8 {
        for i in 0..4 {
            ia[4 * k + i] = (k * step + i) as u16;
            ib[4 * k + i] = (k * step + w - 4 + i) as u16;
        }
    }
    (std::arch::x86_64::_mm512_loadu_si512(ia.as_ptr().cast()), std::arch::x86_64::_mm512_loadu_si512(ib.as_ptr().cast()))
}

fn avx512() -> bool {
    std::arch::is_x86_feature_detected!("avx512f") && std::arch::is_x86_feature_detected!("avx512bw") && std::arch::is_x86_feature_detected!("avx512dq")
}

/// Pass 1's count of a piece (anchors j0..j1 of tokens t) into c (2^cb counters).
#[target_feature(enable = "avx512f,avx512bw,avx512dq")]
unsafe fn count_avx(t: &[u16], j0: usize, j1: usize, w: usize, step: usize, cb: u32, c: &mut [u32]) {
    let (ia, ib) = fkey_idx(w, step);
    let mut hs = [0u64; 8];
    let mut j = j0;
    while j + 8 <= j1 && j * step + 64 <= t.len() {
        fkeys8(t.as_ptr().add(j * step), ia, ib, w as u64, &mut hs);
        for &h in &hs {
            *c.get_unchecked_mut((h >> (64 - cb)) as usize) += 1;
        }
        j += 8;
    }
    for j in j..j1 {
        let a = j * step;
        c[(fkey(&t[a..a + w]) >> (64 - cb)) as usize] += 1;
    }
}

/// Pass 2's scatter of a piece's entries (anchors j0..j1 of tokens tk; low = its local shard's bits).
#[target_feature(enable = "avx512f,avx512bw,avx512dq")]
#[allow(clippy::too_many_arguments)]
unsafe fn scatter_avx(tk: &[u16], j0: usize, j1: usize, w: usize, step: usize, cb: u32, pb: u32, kb: u32, fpb: u32, low: u64, wc: &Wc, dst: *mut u64) {
    let (ia, ib) = fkey_idx(w, step);
    // the 8-contexts at offsets o (from 4 tokens before the first anchor)
    let mut fidx = [(ia, ib); 8];
    for o in 0..step.min(8) {
        let mut a = [0u16; 32];
        let mut b = [0u16; 32];
        for k in 0..8 {
            for i in 0..4 {
                a[4 * k + i] = (k * step + 4 - o + i) as u16;
                b[4 * k + i] = (k * step + 8 - o + i) as u16;
            }
        }
        fidx[o] = (std::arch::x86_64::_mm512_loadu_si512(a.as_ptr().cast()), std::arch::x86_64::_mm512_loadu_si512(b.as_ptr().cast()));
    }
    let mut hs = [0u64; 8];
    let mut f8 = [0u64; 8];
    let mut fps = [0u64; 8];
    let mut j = j0.max(4usize.div_ceil(step));
    for j in j0..j.min(j1) {
        let a = j * step;
        let h = fkey(&tk[a..a + w]);
        wc.push((h >> (64 - cb)) as usize, ((h << cb) >> kb) << kb | fps_scalar(tk, a, step, fpb) << pb | low | j as u64, dst);
    }
    while j + 8 <= j1 && j * step + 64 <= tk.len() {
        let p = tk.as_ptr().add(j * step);
        fkeys8(p, ia, ib, w as u64, &mut hs);
        if fpb > 0 {
            fps = [0; 8];
            for (o, &(fa, fb)) in fidx.iter().enumerate().take(step) {
                fkeys8(p.sub(4), fa, fb, 8, &mut f8);
                for k in 0..8 {
                    fps[k] |= (f8[k] >> (64 - fpb)) << (o as u32 * fpb);
                }
            }
        }
        for (k, &h) in hs.iter().enumerate() {
            wc.push((h >> (64 - cb)) as usize, ((h << cb) >> kb) << kb | fps[k] << pb | low | (j + k) as u64, dst);
        }
        j += 8;
    }
    for j in j..j1 {
        let a = j * step;
        let h = fkey(&tk[a..a + w]);
        wc.push((h >> (64 - cb)) as usize, ((h << cb) >> kb) << kb | fps_scalar(tk, a, step, fpb) << pb | low | j as u64, dst);
    }
}

/// An anchor's fingerprints: per offset o (its 8-context starting o tokens before it) fpb top bits of its fkey.
#[inline]
fn fps_scalar(t: &[u16], a: usize, step: usize, fpb: u32) -> u64 {
    if fpb == 0 {
        return 0;
    }
    let mut f = 0u64;
    for o in 0..step {
        if a >= o && a - o + K0 <= t.len() {
            f |= (fkey(&t[a - o..a - o + K0]) >> (64 - fpb)) << (o as u32 * fpb);
        }
    }
    f
}

/// One NUMA node's index: shards shard0.. (consecutive) in the node's memory.
struct Group {
    shard0: usize,
    node: Option<usize>,
    pieces: Vec<(u32, usize, usize)>, // (local shard, j0, j1)
    entries: NodeVec<u64>,
    offs: NodeVec<u32>, // per sub-bucket its first entry, relative to its coarse bucket's
    cstart: Vec<u64>,   // per coarse bucket its first entry; + the end
    counts: NodeVec<u32>, // the build's per (piece, coarse bucket) counts, then cursors
}

/// A build thread's buffers (allocated and faulted before the clock on its node).
struct Scratch {
    wc_buf: NodeVec<u64>,
    wc_cur: NodeVec<u64>,
    wc_seg: NodeVec<u64>,
    tmp: NodeVec<u64>,
    fbuf: NodeVec<u64>,
    spill: NodeVec<u64>,
    spill2: NodeVec<u64>,
    dir: NodeVec<u32>,
}

pub struct AIndex {
    pub w: usize,
    step: usize,
    cb: u32,
    fb: u32,
    sb: u32,
    low_bits: u32,
    pos_bits: u32,
    /// per anchor offset o its 8-context's fingerprint (fpb bits of its fkey) in the entry, above the position
    fpb: u32,
    /// where the entry's hash bits start: pos_bits + step * fpb
    kb: u32,
    groups: Vec<Group>,
    nodes: Vec<Option<usize>>,
    pub corpus: Corpus,
    scratch: Vec<Mutex<Option<Scratch>>>, // per build thread
    gtable: Vec<u64>, // the validation w-grams (after the step; allocated before the clock)
    gmask: Vec<u32>, // per w-gram: bit (o << fpb | fp) for each offset o and fingerprint fp of a validation 8-context
}

#[derive(Clone, Copy)]
struct Ptr<T>(*mut T);
unsafe impl<T> Send for Ptr<T> {}
unsafe impl<T> Sync for Ptr<T> {}

/// The nodes with CPUs (the pools' threads are pinned to theirs; the process's own affinity may be one core).
fn node_list() -> Vec<usize> {
    let mut nodes = cpu_nodes();
    nodes.sort_unstable();
    nodes.dedup();
    nodes
}

impl AIndex {
    /// Before the clock: the shards mapped, the index's memory allocated and faulted (per NUMA node), the validation
    /// w-grams' table for up to `n_max` positions.
    pub fn alloc(files: &[File], sizes: &[usize], w: usize, n_max: usize) -> Self {
        assert!((2..=K0).contains(&w));
        let step = K0 - w + 1;
        let corpus = Corpus::map(files, sizes);
        let anchors = |n: usize| if n >= w { (n - w) / step + 1 } else { 0 };
        let max_a = sizes.iter().map(|&n| anchors(n)).max().unwrap_or(1).max(2);
        let low_bits = (max_a - 1).ilog2() + 1;
        // groups: one per node
        let nodes: Vec<Option<usize>> = {
            let n = node_list();
            if n.len() <= 1 { vec![None] } else { n.into_iter().map(Some).collect() }
        };
        let ng = nodes.len().min(sizes.len().max(1));
        let nodes = nodes[..ng].to_vec();
        let total_tokens: usize = sizes.iter().sum();
        let mut bounds = vec![0usize];
        let mut acc = 0usize;
        for (s, &n) in sizes.iter().enumerate() {
            acc += n;
            if bounds.len() < ng && acc * ng >= total_tokens * bounds.len() && s + 1 < sizes.len() {
                bounds.push(s + 1);
            }
        }
        while bounds.len() < ng {
            bounds.push(sizes.len());
        }
        bounds.push(sizes.len());
        let max_shards = bounds.windows(2).map(|b| b[1] - b[0]).max().unwrap().max(2);
        let shard_bits = (max_shards - 1).ilog2() + 1;
        let pos_bits = low_bits + shard_bits;
        let (cb, fb, sb) = (CB, FB, SB);
        // fingerprints at w < 7 (an anchor every 3 or 4 positions): up to 3 bits per offset, at least 2 check bits left
        let fpb = if w < 7 { ((64 - 2 - fb - sb).saturating_sub(pos_bits) / step as u32).min(3) } else { 0 };
        let kb = pos_bits + step as u32 * fpb;
        assert!(kb + fb + sb + 2 <= 64 && step << fpb <= 32, "too few hash bits left");
        let nc = 1usize << cb;
        let groups: Vec<Group> = (0..ng)
            .map(|g| {
                let (s0, s1) = (bounds[g], bounds[g + 1]);
                let mut pieces = Vec::new();
                for s in s0..s1 {
                    let na = anchors(sizes[s]);
                    let mut j = 0;
                    while j < na {
                        pieces.push(((s - s0) as u32, j, na.min(j + PIECE_ANCHORS)));
                        j += PIECE_ANCHORS;
                    }
                }
                let total: usize = (s0..s1).map(|s| anchors(sizes[s])).sum();
                let node = nodes[g];
                Group {
                    shard0: s0,
                    node,
                    entries: NodeVec::new(total, node),
                    offs: NodeVec::new(1 << (cb + fb + sb), node),
                    cstart: vec![0; nc + 1],
                    counts: NodeVec::new(pieces.len() * nc, node),
                    pieces,
                }
            })
            .collect();
        let gbits = ((n_max as f64 / 0.5).ceil() as usize).next_power_of_two().max(1024).trailing_zeros();
        Self { w, step, cb, fb, sb, low_bits, pos_bits, fpb, kb, groups, nodes, corpus, scratch: Vec::new(), gtable: huge_zeroed(1 << gbits), gmask: huge_zeroed(1 << gbits) }
    }

    /// Before the clock, in the build pool: each thread's buffers on its node.
    pub fn prepare(&mut self) {
        let map = cpu_nodes();
        let (cb, fb) = (self.cb, self.fb);
        let nc = 1usize << cb;
        // a coarse bucket's room in the thread's buffer: twice the average (bigger ones get their own, rarely)
        let biggest = self.groups.iter().map(|g| g.entries.len()).max().unwrap_or(0);
        let tmp_cap = (2 * biggest / nc).max(1 << 16);
        let grouped = self.nodes.iter().any(|n| n.is_some());
        let scratch: Vec<Mutex<Option<Scratch>>> = rayon::broadcast(|_| {
            let node = grouped.then(|| super::numa::this_node(&map));
            Mutex::new(Some(Scratch {
                wc_buf: NodeVec::new(nc * 8, node),
                wc_cur: NodeVec::new(nc.max(1 << fb), node),
                wc_seg: NodeVec::new(nc.max(1 << fb), node),
                tmp: NodeVec::new(tmp_cap, node),
                fbuf: NodeVec::new(FBUF, node),
                spill: NodeVec::new(tmp_cap / 8, node),
                spill2: NodeVec::new(tmp_cap / 8, node),
                dir: NodeVec::new(1 << SB, node),
            }))
        });
        self.scratch = scratch;
    }

    /// Beside the training, in the build pool: the index from the shards only. Returns the timings.
    pub fn build(&mut self) -> Vec<(String, f64)> {
        if self.scratch.len() != rayon::current_num_threads() {
            self.prepare();
        }
        let mut times = Vec::new();
        let t0 = Instant::now();
        let (cb, fb, sb) = (self.cb, self.fb, self.sb);
        let nc = 1usize << cb;
        let (w, step, pb, lb, fpb, kb) = (self.w, self.step, self.pos_bits, self.low_bits, self.fpb, self.kb);
        let avx = avx512();
        let gnodes: Vec<usize> = self.nodes.iter().enumerate().map(|(g, n)| n.unwrap_or(g)).collect();
        let lens: Vec<usize> = self.groups.iter().map(|g| g.pieces.len()).collect();
        let me = &*self;
        // pass 1: counts per piece and coarse bucket
        run_queues(&gnodes, &lens, |g, p, _| {
            let gr = &me.groups[g];
            let (s, j0, j1) = gr.pieces[p];
            let t = me.corpus.shard(gr.shard0 + s as usize);
            let c = unsafe { gr.counts.slice_mut(p * nc, (p + 1) * nc) };
            c.fill(0);
            if avx {
                unsafe { count_avx(t, j0, j1, w, step, cb, c) };
                return;
            }
            for j in j0..j1 {
                let a = j * step;
                c[(fkey(&t[a..a + w]) >> (64 - cb)) as usize] += 1;
            }
        });
        times.push(("build_count_s".into(), t0.elapsed().as_secs_f64()));
        // cursors: per coarse bucket its start, per piece its place in it (relative to the bucket's start)
        let t = Instant::now();
        for gr in self.groups.iter_mut() {
            let np = gr.pieces.len();
            let counts = Ptr(gr.counts.ptr());
            let tot: Vec<u64> = (0..nc)
                .into_par_iter()
                .map(|c| {
                    let mut at = 0u64;
                    for p in 0..np {
                        let q = unsafe { { counts }.0.add(p * nc + c) };
                        let n = unsafe { *q } as u64;
                        unsafe { *q = at as u32 };
                        at += n;
                    }
                    at
                })
                .collect();
            for c in 0..nc {
                gr.cstart[c + 1] = gr.cstart[c] + tot[c];
            }
            assert_eq!(gr.cstart[nc] as usize, gr.entries.len());
            assert!(tot.iter().all(|&n| n < 1 << 32), "a coarse bucket of 2^32 entries");
        }
        times.push(("build_cursors_s".into(), t.elapsed().as_secs_f64()));
        // pass 2: scatter
        let t = Instant::now();
        let me = &*self;
        run_queues(&gnodes, &lens, |g, p, th| {
            let gr = &me.groups[g];
            let (s, j0, j1) = gr.pieces[p];
            let tk = me.corpus.shard(gr.shard0 + s as usize);
            let mut sg = me.scratch[th].lock().unwrap();
            let sc = sg.as_mut().unwrap();
            let wc = Wc { buf: sc.wc_buf.ptr(), cur: sc.wc_cur.ptr(), seg: sc.wc_seg.ptr() };
            let row = &gr.counts.as_slice()[p * nc..(p + 1) * nc];
            for c in 0..nc {
                let at = gr.cstart[c] + row[c] as u64;
                unsafe {
                    *wc.cur.add(c) = at;
                    *wc.seg.add(c) = at;
                }
            }
            let dst = gr.entries.ptr();
            let low = (s as u64) << lb;
            if avx {
                unsafe { scatter_avx(tk, j0, j1, w, step, cb, pb, kb, fpb, low, &wc, dst) };
            } else {
                for j in j0..j1 {
                    let a = j * step;
                    let h = fkey(&tk[a..a + w]);
                    let c = (h >> (64 - cb)) as usize;
                    let e = ((h << cb) >> kb) << kb | fps_scalar(tk, a, step, fpb) << pb | low | j as u64;
                    unsafe { wc.push(c, e, dst) };
                }
            }
            unsafe { wc.finish(nc, dst) };
        });
        times.push(("build_scatter_s".into(), t.elapsed().as_secs_f64()));
        // pass 3: per coarse bucket, by fine bucket into the thread's buffer, then per fine bucket by sub-bucket back
        let t = Instant::now();
        let big = AtomicU64::new(0);
        let lens3: Vec<usize> = vec![nc; self.groups.len()];
        let me = &*self;
        let (nf, ns) = (1usize << fb, 1usize << sb);
        run_queues(&gnodes, &lens3, |g, c, th| {
            let gr = &me.groups[g];
            let (a, b) = (gr.cstart[c] as usize, gr.cstart[c + 1] as usize);
            let m = b - a;
            if m == 0 {
                let base = (c << fb) << sb;
                let offs = unsafe { gr.offs.slice_mut(base, base + (nf << sb)) };
                offs.fill(0);
                return;
            }
            let mut sg = me.scratch[th].lock().unwrap();
            let sc = sg.as_mut().unwrap();
            let v = unsafe { gr.entries.slice_mut(a, b) };
            let mut own: Option<NodeVec<u64>> = None;
            let own_used = m > sc.tmp.len();
            let tmp: *mut u64 = if !own_used {
                sc.tmp.ptr()
            } else {
                big.fetch_add(1, Ordering::Relaxed);
                own.insert(NodeVec::new(m + 8, gr.node)).ptr()
            };
            let wc = Wc { buf: sc.wc_buf.ptr(), cur: sc.wc_cur.ptr(), seg: sc.wc_seg.ptr() };
            // by fine bucket into regions of the thread's buffer of a fixed room each (line-aligned, a quarter over the
            // average); what does not fit goes to the spill buffer (a frequent w-gram's), counting-sorted after. If
            // the spill fills up too, the exact way: a histogram first, then the scatter.
            let room = ((m + m / 4) / nf + 64).next_multiple_of(8);
            let spill = sc.spill.ptr();
            let mut ns_spill = 0usize;
            let mut fits = !own_used && room * nf <= sc.tmp.len();
            let mut fill = [0usize; (1 << 10) + 1];
            if fits {
                for f in 0..nf {
                    unsafe {
                        *wc.cur.add(f) = (f * room) as u64;
                        *wc.seg.add(f) = (f * room) as u64;
                    }
                }
                let cap_spill = sc.spill.len();
                for &e in v.iter() {
                    let f = (e >> (64 - fb)) as usize;
                    if unsafe { *wc.cur.add(f) } < ((f + 1) * room) as u64 {
                        unsafe { wc.push(f, e, tmp) };
                    } else if ns_spill < cap_spill {
                        unsafe { *spill.add(ns_spill) = e };
                        ns_spill += 1;
                    } else {
                        fits = false;
                        break;
                    }
                }
                unsafe { wc.finish(nf, tmp) };
            }
            // per fine bucket: its entries in the region (from f * room) and its spill (sorted into spill2)
            let mut hist = [0usize; (1 << 10) + 1];
            let mut sp = [0usize; (1 << 10) + 1];
            let spill2 = sc.spill2.ptr();
            if fits {
                for f in 0..nf {
                    fill[f] = unsafe { *wc.cur.add(f) } as usize - f * room;
                }
                for q in 0..ns_spill {
                    sp[(unsafe { *spill.add(q) } >> (64 - fb)) as usize + 1] += 1;
                }
                for f in 0..nf {
                    sp[f + 1] += sp[f];
                }
                let mut at = sp;
                for q in 0..ns_spill {
                    let e = unsafe { *spill.add(q) };
                    let f = (e >> (64 - fb)) as usize;
                    unsafe { *spill2.add(at[f]) = e };
                    at[f] += 1;
                }
                for f in 0..nf {
                    hist[f + 1] = hist[f] + fill[f] + sp[f + 1] - sp[f];
                }
            } else {
                for &e in v.iter() {
                    hist[(e >> (64 - fb)) as usize + 1] += 1;
                }
                for f in 0..nf {
                    hist[f + 1] += hist[f];
                }
                for f in 0..nf {
                    unsafe {
                        *wc.cur.add(f) = hist[f] as u64;
                        *wc.seg.add(f) = hist[f] as u64;
                    }
                }
                for &e in v.iter() {
                    unsafe { wc.push((e >> (64 - fb)) as usize, e, tmp) };
                }
                unsafe { wc.finish(nf, tmp) };
            }
            let mut sub = [0usize; (1 << 12) + 1];
            let fbuf = sc.fbuf.ptr();
            let dirb = sc.dir.ptr();
            for f in 0..nf {
                let (h0, h1) = (hist[f], hist[f + 1]);
                // the fine bucket's entries: one slice, or its region's and its spill's
                let (s1, s2): (&[u64], &[u64]) = if fits {
                    unsafe { (std::slice::from_raw_parts(tmp.add(f * room), fill[f]), std::slice::from_raw_parts(spill2.add(sp[f]), sp[f + 1] - sp[f])) }
                } else {
                    (unsafe { std::slice::from_raw_parts(tmp.add(h0), h1 - h0) }, &[])
                };
                sub.fill(0);
                for &e in s1.iter().chain(s2) {
                    sub[((e >> (64 - fb - sb)) as usize & (ns - 1)) + 1] += 1;
                }
                for q in 0..ns {
                    sub[q + 1] += sub[q];
                }
                // the directory row, written with streaming stores
                let base = ((c << fb) | f) << sb;
                for q in 0..ns {
                    unsafe { *dirb.add(q) = (h0 + sub[q]) as u32 };
                }
                unsafe { nt_copy(gr.offs.ptr().add(base).cast(), dirb.cast(), ns / 2) };
                if h1 - h0 <= FBUF {
                    for &e in s1.iter().chain(s2) {
                        let q = (e >> (64 - fb - sb)) as usize & (ns - 1);
                        unsafe { *fbuf.add(sub[q]) = e };
                        sub[q] += 1;
                    }
                    unsafe { nt_copy(v.as_mut_ptr().add(h0), fbuf, h1 - h0) };
                } else {
                    let dst = &mut v[h0..h1];
                    for &e in s1.iter().chain(s2) {
                        let q = (e >> (64 - fb - sb)) as usize & (ns - 1);
                        dst[sub[q]] = e;
                        sub[q] += 1;
                    }
                }
            }
            unsafe { _mm_sfence() };
            drop(own);
        });
        times.push(("build_sort_s".into(), t.elapsed().as_secs_f64()));
        times.push(("build_big_buckets".into(), big.load(Ordering::Relaxed) as f64));
        times.push(("build_s".into(), t0.elapsed().as_secs_f64()));
        times.push(("build_entries".into(), self.groups.iter().map(|g| g.entries.len()).sum::<usize>() as f64));
        times.push(("build_groups".into(), self.groups.len() as f64));
        times
    }

    /// After the step: the occurrences of the validation contexts (positions i with avail[i] >= 8) pushed into the
    /// threads' lists (`work`, by thread index), as val_build's scan would.
    pub(crate) fn candidates(&self, x: &[u16], avail: &[u32], lv: &[VLevel], f2: &Filter, work: &[Mutex<Work>], ext: &Ext) -> Result<Vec<(String, f64)>, String> {
        let mut times = Vec::new();
        let t0 = Instant::now();
        let (w, step) = (self.w, self.step);
        let (cb, fb, sb) = (self.cb, self.fb, self.sb);
        // the distinct w-grams with their offsets' mask
        let (table, masks) = (&self.gtable, &self.gmask);
        let (kb, fpb) = (self.kb, self.fpb);
        let bits = table.len().trailing_zeros();
        if x.len() * 2 > table.len() {
            return Err("more validation positions than allocated for".into());
        }
        {
            let ta: &[AtomicU64] = unsafe { &*(table.as_slice() as *const [u64] as *const [AtomicU64]) };
            let ma: &[AtomicU32] = unsafe { &*(masks.as_slice() as *const [u32] as *const [AtomicU32]) };
            let mask = (1usize << bits) - 1;
            (0..x.len().div_ceil(64)).into_par_iter().for_each(|ch| {
                let mut hs = [(0u64, 0u32); 64 * K0];
                let mut m = 0;
                for i in ch * 64..x.len().min(ch * 64 + 64) {
                    if (avail[i] as usize) < K0 {
                        continue;
                    }
                    let ctx = &x[i + 1 - K0..=i];
                    let fp = if fpb > 0 { (fkey(ctx) >> (64 - fpb)) as u32 } else { 0 };
                    for o in 0..step {
                        // the index's hash bits only (w-grams agreeing in them share their entries)
                        let h = fkey(&ctx[o..o + w]) >> (kb - cb) << (kb - cb) | 1;
                        unsafe { _mm_prefetch(table.as_ptr().wrapping_add((h >> (64 - bits)) as usize).cast(), _MM_HINT_T0) };
                        hs[m] = (h, 1 << ((o as u32) << fpb | fp));
                        m += 1;
                    }
                }
                for &(h, b) in &hs[..m] {
                    let mut s = (h >> (64 - bits)) as usize;
                    loop {
                        match ta[s].compare_exchange(0, h, Ordering::Relaxed, Ordering::Relaxed) {
                            Ok(_) => break,
                            Err(v) if v == h => break,
                            Err(_) => s = (s + 1) & mask,
                        }
                    }
                    ma[s].fetch_or(b, Ordering::Relaxed);
                }
            });
        }
        times.push(("cand_grams_s".into(), t0.elapsed().as_secs_f64()));
        // per group (on its node's threads) and batch of w-grams: their directory words prefetched, then their
        // sub-buckets' first lines, then the entries of their keys collected (their tokens prefetched) and processed;
        // sub-buckets of more than CHUNK entries are left for a second pass, in CHUNK pieces
        const CHUNK: usize = 4096;
        const GCHUNK: usize = 1 << 12;
        let idx = |h: u64| (h >> (64 - cb - fb - sb)) as usize;
        let sub_mask = (1usize << (fb + sb)) - 1;
        let t1 = Instant::now();
        let gnodes: Vec<usize> = self.nodes.iter().enumerate().map(|(g, n)| n.unwrap_or(g)).collect();
        let nchunks = table.len().div_ceil(GCHUNK);
        let bigs: Vec<Mutex<Vec<(usize, usize, u64, u32)>>> = self.groups.iter().map(|_| Mutex::new(Vec::new())).collect();
        let probes = AtomicU64::new(0);
        run_queues(&gnodes, &vec![nchunks; self.groups.len()], |g, ci, th| {
            let gr = &self.groups[g];
            let range = |i: usize| {
                let c = i >> (fb + sb);
                let a = gr.cstart[c] as usize + unsafe { *gr.offs.ptr().add(i) } as usize;
                let b = if (i + 1) & sub_mask == 0 { gr.cstart[c + 1] as usize } else { gr.cstart[c] as usize + unsafe { *gr.offs.ptr().add(i + 1) } as usize };
                (a, b)
            };
            let mut wk = work[th].lock().unwrap();
            let mut p = Proc { ai: self, gr, lv, f2, ext, wk: &mut wk, probes: 0 };
            let mut big = Vec::new();
            let (s0, s1) = (ci * GCHUNK, table.len().min((ci + 1) * GCHUNK));
            let mut grams: [(u64, u32); 32] = [(0, 0); 32];
            let mut k = s0;
            while k < s1 {
                let mut m = 0;
                while m < 32 && k < s1 {
                    if table[k] != 0 {
                        grams[m] = (table[k], masks[k]);
                        m += 1;
                    }
                    k += 1;
                }
                let ch = &grams[..m];
                for &(h, _) in ch {
                    unsafe { _mm_prefetch(gr.offs.ptr().wrapping_add(idx(h)).cast(), _MM_HINT_T0) };
                }
                let mut se = [(0usize, 0usize); 32];
                for (q, &(h, _)) in ch.iter().enumerate() {
                    let (a, b) = range(idx(h));
                    unsafe { _mm_prefetch(gr.entries.ptr().wrapping_add(a).cast(), _MM_HINT_T0) };
                    se[q] = (a, b);
                }
                for (q, &(h, msk)) in ch.iter().enumerate() {
                    let (a, b) = se[q];
                    let key = (h << cb) >> kb;
                    if b - a > CHUNK {
                        let mut s = a;
                        while s < b {
                            big.push((s, b.min(s + CHUNK), key, msk));
                            s += CHUNK;
                        }
                    } else {
                        p.collect(a, b, key, msk);
                    }
                }
                p.run();
            }
            p.finish();
            probes.fetch_add(p.probes, Ordering::Relaxed);
            if !big.is_empty() {
                bigs[g].lock().unwrap().append(&mut big);
            }
        });
        times.push(("cand_small_s".into(), t1.elapsed().as_secs_f64()));
        let bigs: Vec<Vec<(usize, usize, u64, u32)>> = bigs.into_iter().map(|b| b.into_inner().unwrap()).collect();
        times.push(("cand_big_items".into(), bigs.iter().map(|b| b.len()).sum::<usize>() as f64));
        let lens: Vec<usize> = bigs.iter().map(|b| b.len()).collect();
        run_queues(&gnodes, &lens, |g, i, th| {
            let (a, b, key, m) = bigs[g][i];
            let mut wk = work[th].lock().unwrap();
            let mut p = Proc { ai: self, gr: &self.groups[g], lv, f2, ext, wk: &mut wk, probes: 0 };
            let mut s = a;
            while s < b {
                p.collect(s, b.min(s + 64), key, m);
                p.run();
                s += 64;
            }
            p.finish();
            probes.fetch_add(p.probes, Ordering::Relaxed);
        });
        times.push(("cand_s".into(), t1.elapsed().as_secs_f64()));
        times.push(("cand_passed".into(), probes.load(Ordering::Relaxed) as f64));
        Ok(times)
    }

    /// The probes of the candidates in `wk.cand` (shard, position of the next token), with the direct path: after the
    /// level-8 hit, the context's validation positions (if at most EXT_LIM; longer chains: those with the same two
    /// tokens before the 8-context, from the xmap) compared token by token with the corpus context's earlier tokens;
    /// the deeper levels' slots from the position with the longest match. Otherwise the table probes.
    fn probe(&self, wk: &mut Work, lv: &[VLevel], ext: &Ext) {
        let l0 = &lv[0];
        let k0 = l0.k;
        let Work { cand, hs, hits, fallback, deep, lists: acc, .. } = wk;
        hs.clear();
        for &(s, pp) in cand.iter() {
            let tk = self.corpus.shard(s as usize);
            let h = hash(&tk[pp as usize - k0..pp as usize]);
            l0.prefetch(h);
            hs.push(h);
        }
        hits.clear();
        for (&(s, pp), &h) in cand.iter().zip(hs.iter()) {
            if let Some(slot) = l0.find(h) {
                let tk = self.corpus.shard(s as usize);
                acc.push(0, slot >> l0.shift, (slot as u64) << 16 | tk[pp as usize] as u64);
                if pp as usize >= lv[1].k {
                    let head = (l0.slots[slot] & 0xFF_FFFF) as u32;
                    unsafe {
                        _mm_prefetch(ext.xbit.as_ptr().wrapping_add(slot / 64).cast(), _MM_HINT_T0);
                        _mm_prefetch(ext.next.as_ptr().wrapping_add(head as usize).cast(), _MM_HINT_T0);
                        _mm_prefetch(ext.x.as_ptr().wrapping_add((head as usize).saturating_sub(19)).cast(), _MM_HINT_T0);
                    }
                    hits.push((s, pp, head, slot as u32));
                }
            }
        }
        fallback.clear();
        deep.clear();
        for &(s, pp, head, slot) in hits.iter() {
            let tk = self.corpus.shard(s as usize);
            let pp = pp as usize;
            let most = pp.min(lv[lv.len() - 1].k);
            if ext.xbit[slot as usize / 64] >> (slot % 64) & 1 == 1 {
                let xv = ext.xo[ext.xo.partition_point(|e| e.0 < slot)].1;
                // a long chain: only its positions with the same two tokens before the 8-context can match deeper
                let arr = &ext.xarr[(xv >> 32) as usize..((xv >> 32) + (xv & 0xFFFF_FFFF)) as usize];
                let key = (tk[pp - 10] as u32) << 16 | tk[pp - 9] as u32;
                let lo = arr.partition_point(|e| e.0 < key);
                let hi = lo + arr[lo..].partition_point(|e| e.0 == key);
                if hi == lo {
                    continue;
                }
                if hi - lo > 4 * EXT_LIM {
                    fallback.push((s, pp as u32));
                    continue;
                }
                let (mut best, mut bpos) = (k0 + 2, arr[lo].1 as usize);
                for &(_, q) in &arr[lo..hi] {
                    let qi = q as usize;
                    let cap = (ext.cap[qi] as usize).min(pp);
                    let mut len = k0 + 2;
                    while len < cap && tk[pp - 1 - len] == ext.x[qi - len] {
                        len += 1;
                    }
                    if len > best {
                        best = len;
                        bpos = qi;
                        if best >= most {
                            break;
                        }
                    }
                }
                if best >= lv[1].k {
                    unsafe { _mm_prefetch(ext.rec.as_ptr().wrapping_add(bpos).cast(), _MM_HINT_T0) };
                    deep.push((s, pp as u32, best as u32, bpos as u32));
                }
                continue;
            }
            // the longest match of the corpus context's tokens before its 8-context with a chain position's, the
            // chain walked until a match reaches as far as any can (or past EXT_LIM positions: the table probes)
            let (mut best, mut bpos) = (k0, 0usize);
            let mut m = 0;
            let mut q = head;
            while q != u32::MAX && best < most {
                if m == EXT_LIM {
                    break;
                }
                let qi = q as usize;
                let cap = (ext.cap[qi] as usize).min(pp);
                let mut len = k0;
                while len < cap && tk[pp - 1 - len] == ext.x[qi - len] {
                    len += 1;
                }
                if len > best {
                    best = len;
                    bpos = qi;
                }
                m += 1;
                q = ext.next[qi];
            }
            if q != u32::MAX && best < most {
                fallback.push((s, pp as u32));
                continue;
            }
            if best >= lv[1].k {
                unsafe { _mm_prefetch(ext.rec.as_ptr().wrapping_add(bpos).cast(), _MM_HINT_T0) };
                deep.push((s, pp as u32, best as u32, bpos as u32));
            }
        }
        // the deeper levels' occurrences, from the matched validation position's slots
        for &(s, pp, best, bpos) in deep.iter() {
            let next_tok = self.corpus.shard(s as usize)[pp as usize] as u64;
            let r = ext.rec[bpos as usize];
            for (li, level) in lv.iter().enumerate().skip(1) {
                if (best as usize) < level.k {
                    break;
                }
                let sl = r[li - 1] as usize;
                acc.push(li, sl >> level.shift, (sl as u64) << 16 | next_tok);
            }
        }
        cand.clear();
        if !fallback.is_empty() {
            self.probe_tables(wk, lv);
        }
    }

    /// val_build's probes of the fallback candidates (level-8 hits), level by level from level 10 on.
    fn probe_tables(&self, wk: &mut Work, lv: &[VLevel]) {
        let Work { fallback: cand, hs, next, lists: acc, .. } = wk;
        for (i, level) in lv.iter().enumerate().skip(1) {
            if cand.is_empty() {
                break;
            }
            let k = level.k;
            hs.clear();
            for &(s, pp) in cand.iter() {
                let tk = self.corpus.shard(s as usize);
                let h = hash(&tk[pp as usize - k..pp as usize]);
                level.prefetch(h);
                hs.push(h);
            }
            next.clear();
            let kk = lv.get(i + 1).map_or(0, |l| l.k);
            for (&(s, pp), &h) in cand.iter().zip(hs.iter()) {
                if let Some(slot) = level.find(h) {
                    let tk = self.corpus.shard(s as usize);
                    acc.push(i, slot >> level.shift, (slot as u64) << 16 | tk[pp as usize] as u64);
                    if pp as usize >= kk {
                        next.push((s, pp));
                    }
                }
            }
            std::mem::swap(cand, next);
        }
        cand.clear();
    }

    /// Ready for another lookup: the validation w-grams' table zeroed.
    pub(crate) fn reset(&mut self) {
        self.gtable.par_chunks_mut(1 << 16).for_each(|c| c.fill(0));
        self.gmask.par_chunks_mut(1 << 16).for_each(|c| c.fill(0));
    }

    /// The index's memory in bytes: entries, directories (and the build's counts), the build threads' scratch.
    pub(crate) fn bytes(&self) -> (usize, usize, usize) {
        let e = self.groups.iter().map(|g| g.entries.len() * 8).sum();
        let d = self.groups.iter().map(|g| g.offs.len() * 4 + g.counts.len() * 4).sum();
        let s = self.scratch.iter().map(|m| m.lock().unwrap().as_ref().map_or(0, |s| (s.wc_buf.len() + s.wc_cur.len() + s.wc_seg.len() + s.tmp.len() + s.fbuf.len() + s.spill.len() + s.spill2.len()) * 8)).sum();
        (e, d, s)
    }
}

/// One thread's candidate processing in one group, a pipeline of batches: entries collected (their tokens
/// prefetched); a batch later their 8-contexts' filter keys computed (prefetched); a batch later the filter tested,
/// the passing candidates probed in batches.
struct Proc<'a, 'b> {
    ai: &'a AIndex,
    gr: &'a Group,
    lv: &'a [VLevel],
    f2: &'a Filter,
    ext: &'a Ext<'a>,
    wk: &'b mut Work,
    probes: u64,
}

impl Proc<'_, '_> {
    /// The entries of [a, b) with `key`: their tokens prefetched.
    #[inline]
    fn collect(&mut self, a: usize, b: usize, key: u64, gm: u32) {
        let ai = self.ai;
        let (pb, lb, kb, fpb) = (ai.pos_bits, ai.low_bits, ai.kb, ai.fpb);
        let low_mask = (1u64 << lb) - 1;
        let s0 = self.gr.shard0;
        for &en in unsafe { std::slice::from_raw_parts(self.gr.entries.ptr().add(a), b - a) } {
            if en >> kb == key {
                // the offsets whose 8-context's fingerprint some validation 8-context with this w-gram there has
                let m = if fpb == 0 {
                    gm as u8
                } else {
                    let fps = (en >> pb) as u32;
                    let mut m = 0u8;
                    for o in 0..ai.step as u32 {
                        let fp = fps >> (o * fpb) & ((1 << fpb) - 1);
                        m |= (((gm >> (o << fpb | fp)) & 1) as u8) << o;
                    }
                    if m == 0 {
                        continue;
                    }
                    m
                };
                let (s, a0) = (s0 + ((en & ((1u64 << pb) - 1)) >> lb) as usize, (en & low_mask) as usize * ai.step);
                let at = (ai.corpus.toks[s].0 as *const u16).wrapping_add(a0.saturating_sub(16));
                unsafe {
                    _mm_prefetch(at.cast(), _MM_HINT_T0);
                    _mm_prefetch(at.wrapping_add(32).cast(), _MM_HINT_T0);
                }
                self.wk.pend.push((s as u32, a0, m));
            }
        }
    }

    /// One pipeline step: the previous step's filter keys tested, the previous step's entries' filter keys computed,
    /// the entries collected since then queued.
    fn run(&mut self) {
        let f2 = self.f2;
        let wk = &mut *self.wk;
        wk.cand.extend(wk.c2.iter().filter(|c| f2.test(c.2)).map(|c| (c.0, c.1)));
        wk.c2.clear();
        let ai = self.ai;
        let step = ai.step;
        for &(s, a0, m) in &wk.got {
            let tk = ai.corpus.shard(s as usize);
            for o in 0..step {
                if m >> o & 1 == 1 && a0 >= o && a0 - o + K0 < tk.len() {
                    let pp = a0 - o + K0;
                    let h2 = fkey(&tk[pp - K0..pp]);
                    f2.prefetch(h2);
                    wk.c2.push((s, pp as u32, h2));
                }
            }
        }
        wk.got.clear();
        std::mem::swap(&mut wk.got, &mut wk.pend);
        if wk.cand.len() >= 32 {
            self.probe();
        }
    }

    fn probe(&mut self) {
        self.probes += self.wk.cand.len() as u64;
        self.ai.probe(self.wk, self.lv, self.ext);
    }

    fn finish(&mut self) {
        self.run();
        self.run();
        self.run();
        if !self.wk.cand.is_empty() {
            self.probe();
        }
    }
}
