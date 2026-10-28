"""The validation-side hint statistics on GPUs, timed from process start to the final rows: each rank holds its share
of the training shards in GPU memory, counts the (context, next token) occurrences of the validation contexts at
every level of ALL_LEVELS, and the ranks merge the counts exactly. The outputs are the count rows of
old_records.ValCountRows (tot, dist, cnt per position and level) and the hint rows of old_records.ValHintRows (cell,
top-8 tokens and weights) of each of val_hints.VARIANTS, with the same definitions (exact_match/src/kgram.rs).

Per level k (ascending), on every rank:
  1. the validation index (built once for all levels, before the scans): the distinct valid validation k-contexts,
     their ids (by key order) and an open-addressing table from key to (id, a representative position);
  2. scan: every training position whose (k')-context matched at the previous level k' (a bitmap; all positions at
     the first level) looks its k-context up; a hit is verified token by token (hashed levels) and its (context id,
     next token) counted in a pair hash table; the hits are the next level's bitmap. Big tables are split into
     L2-sized sub-tables: the hits' pairs are buffered, grouped by sub-table and counted sub-table by sub-table. The
     first level's scan runs shard by shard as the shards land in GPU memory (and the index builds meanwhile);
  3. local aggregation: the pair table's entries partitioned by owner rank (owner r holds a contiguous range of
     context ids);
  4. merge: all-to-all of the (pair, count) entries; the owner sorts them (CUB radix sort) and sums duplicates;
  5. per owned context: tot, dist, the next tokens ranked by count descending then token ascending (a stable CUB
     sort), its top-8, and per validation position whose context it owns: tot, dist, cnt (the target's count if among
     the TOP = 1024 most frequent);
then the hint rows of each variant from the per-level tops (the owner of the deepest matched level's context writes
the row), and everything reduced to rank 0.

Run with torchrun (one rank per GPU); see the arguments. Timings go to <out>/timings.json, the outputs to <out>/."""

import time

T_PROC = time.time()

import os  # noqa: E402
import sys  # noqa: E402

# --own-device: this rank sees only its own GPU (set before CUDA initializes: the driver then initializes one device
# per process instead of all of them in every process)
if "--own-device" in sys.argv and "LOCAL_RANK" in os.environ:
    _visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    _ids = _visible.split(",") if _visible else [str(i) for i in range(64)]
    os.environ["CUDA_VISIBLE_DEVICES"] = _ids[int(os.environ["LOCAL_RANK"])]

# --early-ctx: the CUDA driver initializes and creates this rank's primary context (which torch then uses) on a thread
# while the main thread imports torch
EARLY: dict = {}
if "--early-ctx" in sys.argv:
    import ctypes as _ct  # noqa: E402
    import threading as _th  # noqa: E402

    def _early_ctx() -> None:
        try:
            cu = _ct.CDLL("libcuda.so.1")
            dev, ctx = _ct.c_int(), _ct.c_void_p()
            index = 0 if "--own-device" in sys.argv else int(os.environ.get("LOCAL_RANK", 0))
            for rc in (cu.cuInit(0), cu.cuDeviceGet(_ct.byref(dev), index),
                       cu.cuDevicePrimaryCtxRetain(_ct.byref(ctx), dev)):
                if rc != 0:
                    raise RuntimeError(f"CUDA driver error {rc}")
            EARLY["ready"] = time.time() - T_PROC
        except Exception as e:  # noqa: BLE001
            EARLY["error"] = repr(e)

    EARLY["thread"] = _th.Thread(target=_early_ctx, daemon=True)
    EARLY["thread"].start()

import argparse
import ctypes
import hashlib
import json
import mmap
import shutil
import subprocess
import threading
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

T_IMPORT = time.time()

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "gpu_val_stats.cu"
ALL_LEVELS = (2, 3, 4, 5, 6, 8, 10, 12, 16, 20)
LEVELS = (2, 3, 4, 5, 6, 10, 20)  # the hint rows' levels and length buckets
VARIANTS = ((6, 10, 20), (2, 3, 6, 10, 20), LEVELS)
VAL_TOKENS = 10_485_760
HEADER_BYTES = 1024
BOS = 50256
GAP = 32  # SENT tokens before each shard (>= the longest context)
TOP = 1024  # val_hints.TOP: the cnt rule's most frequent next tokens
ENTRIES = 8
ARCHS = ("80", "86", "89", "90")
# The distinct (context, next token) pairs of n training tokens per level, to size the pair tables: U(n) = u *
# (n / 4e8) ** beta, fitted to 4 shards (4e8 tokens) and all 1028 (val_hints.ValContextCounts' distinct). A table
# that overflows doubles and its level reruns.
PAIRS = {2: (82.17e6, 0.620), 3: (62.30e6, 0.667), 4: (26.39e6, 0.694), 5: (8.99e6, 0.706), 6: (3.17e6, 0.705),
         8: (0.550e6, 0.671), 10: (0.223e6, 0.604), 12: (0.162e6, 0.577), 16: (0.121e6, 0.572), 20: (0.102e6, 0.581)}
SAFETY = 1.15
LOAD = 0.7  # the pair tables' load factor at the estimate


def lib_path() -> Path:
    """The kernels' library for this source: in $GVS_BUILD_DIR if set, else in .build/ next to this file."""
    digest = hashlib.sha256(SOURCE.read_bytes() + ",".join(ARCHS).encode()).hexdigest()[:12]
    return Path(os.environ.get("GVS_BUILD_DIR") or HERE / ".build") / f"gpu_val_stats_{digest}.so"


def find_nvcc() -> str:
    for c in (os.environ.get("CUDA_HOME", ""), "/usr/local/cuda-12.8", "/usr/local/cuda-12.6", "/usr/local/cuda"):
        if c and Path(c, "bin/nvcc").exists():
            return str(Path(c, "bin/nvcc"))
    raise FileNotFoundError("no nvcc: set CUDA_HOME")


def build() -> Path:
    """The kernels' shared library (nvcc, every arch of ARCHS), built once per source version."""
    out = lib_path()
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    prebuilt = HERE / ".build" / out.name
    if prebuilt.exists():  # built before, next to the source (e.g. in a code snapshot)
        shutil.copyfile(prebuilt, out)
        return out
    tmp = out.with_suffix(f".{os.getpid()}.tmp")
    gencode = [f"-gencode=arch=compute_{a},code=sm_{a}" for a in ARCHS]
    cmd = [find_nvcc(), "-O3", "-std=c++17", "-shared", "-Xcompiler", "-fPIC", *gencode, "-o", str(tmp), str(SOURCE)]
    subprocess.run(cmd, check=True)
    tmp.rename(out)
    return out


class Lib:
    """The kernels through ctypes: tensors as their data pointers, on the current stream; a nonzero return raises."""

    def __init__(self, path: Path):
        self.so = ctypes.CDLL(str(path))

    def __call__(self, name: str, *args) -> int:
        conv = []
        for a in args:
            if isinstance(a, torch.Tensor):
                conv.append(ctypes.c_void_p(a.data_ptr()))
            elif a is None:
                conv.append(ctypes.c_void_p(0))
            elif isinstance(a, ctypes._SimpleCData) or isinstance(a, ctypes.Array):
                conv.append(a)
            elif isinstance(a, int):
                conv.append(ctypes.c_int64(a))
            else:
                raise TypeError(f"{name}: argument {a!r}")
        conv.append(ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
        fn = getattr(self.so, name)
        fn.restype = ctypes.c_int
        rc = fn(*conv)
        if rc != 0:
            raise RuntimeError(f"{name} returned {rc}")
        return rc

    def cub(self, name: str, *args) -> None:
        """A CUB call: its temp storage queried, allocated (a torch tensor) and the call run."""
        size = ctypes.c_size_t(0)
        ptr = ctypes.c_void_p(ctypes.addressof(size))
        self(name, None, ptr, *args)
        temp = torch.empty(max(size.value, 1), dtype=torch.uint8, device="cuda")
        self(name, temp, ptr, *args)
        del temp


def i32(x: int) -> ctypes.c_int:
    return ctypes.c_int(x)


class Clock:
    """Phase timings: each phase ends with a device synchronize (its own time) and a barrier (the ranks' wait)."""

    def __init__(self, world: int):
        self.world = world
        self.t = time.time()
        self.phases: list[tuple[str, float, float]] = []  # name, own seconds, wait seconds

    def lap(self, name: str, barrier: bool = True) -> float:
        torch.cuda.synchronize()
        own = time.time()
        if barrier and self.world > 1:
            dist.barrier()
            torch.cuda.synchronize()
        end = time.time()
        self.phases.append((name, own - self.t, end - own))
        self.t = end
        return end - own


def shard_tokens(path: str) -> int:
    header = np.fromfile(path, dtype="<i4", count=256)
    if header[0] != 20240520 or header[1] != 1:
        raise ValueError(f"invalid shard {path}")
    return int(header[2])


def numa_cpus(device: int) -> set[int]:
    """The CPUs of the GPU's NUMA node that this process may use (empty if unknown)."""
    try:
        p = torch.cuda.get_device_properties(device)
        bus = f"{p.pci_domain_id:04x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}.0"
        node = int(Path(f"/sys/bus/pci/devices/{bus}/numa_node").read_text())
        if node < 0:
            return set()
        cpus: set[int] = set()
        for part in Path(f"/sys/devices/system/node/node{node}/cpulist").read_text().strip().split(","):
            a, _, b = part.partition("-")
            cpus.update(range(int(a), int(b or a) + 1))
        return cpus & os.sched_getaffinity(0)
    except Exception:  # noqa: BLE001
        return set()


def _cudart() -> ctypes.CDLL:
    """The CUDA runtime torch uses, through ctypes (which releases the GIL during the calls)."""
    rt = ctypes.CDLL(str(Path(torch.__file__).parent.parent / "nvidia/cuda_runtime/lib/libcudart.so.12"))
    rt.cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
    rt.cudaHostUnregister.argtypes = [ctypes.c_void_p]
    return rt


class Loader:
    """This rank's shards into GPU memory, in background threads (started at construction): concatenated with at least
    GAP SENT tokens before each (and after the last), each shard 32-aligned, the length a multiple of 32.

    mode "register": each shard file (in /dev/shm: tmpfs pages can be pinned) is mapped shared, pinned in place with
    cudaHostRegister and copied by DMA, no host copy; "pread": read in chunks into pinned buffers (two per thread) and
    copied. Each shard's copy is recorded as an event, and `ready[i]` set once it is enqueued."""

    def __init__(self, paths: list[str], tok_len: int, starts: list[int], sizes: list[int], threads: int, mode: str,
                 chunk_tokens: int, numa: bool = False, start: bool = True):
        self.paths, self.starts, self.sizes, self.mode = paths, starts, sizes, mode
        self.tok = torch.full((tok_len,), -1, dtype=torch.int16, device="cuda")
        self.ready = [threading.Event() for _ in paths]
        self.events: list = [None] * len(paths)
        self.stats = {"mode": mode, "threads": threads, "pin_s": [0.0] * threads, "read_s": [0.0] * threads,
                      "copy_enqueue_s": [0.0] * threads, "pin_alloc_s": 0.0}
        self.maps: list = []
        self.pinned: list[torch.Tensor] = []  # pread mode's buffers, reused for the outputs' D2H
        self.buffers: list = [None] * threads  # pread mode's per-thread pairs (preallocate() fills them up front)
        self.errors: list[BaseException] = []
        self.chunk_tokens = chunk_tokens
        self.lock = threading.Lock()
        self.device = torch.cuda.current_device()
        self.cpus = numa_cpus(self.device) if numa else set()
        self.t0 = time.time()
        self.t_done = 0.0
        self.threads = [threading.Thread(target=self._work, args=(t, threads), daemon=True) for t in range(threads)]
        self.started = False
        if start:
            self.start()

    def preallocate(self) -> None:
        """pread mode: every thread's two pinned buffers, now (e.g. before a clock) instead of in the threads."""
        if self.mode == "pread":
            for t in range(len(self.buffers)):
                if self.buffers[t] is None:
                    self.buffers[t] = [torch.empty(self.chunk_tokens, dtype=torch.int16, pin_memory=True) for _ in range(2)]
                    self.pinned += self.buffers[t]

    def start(self, stream_priority: int = 0) -> None:
        """Start the load threads (their copies on streams of `stream_priority`; lower is higher priority)."""
        self.stream_priority = stream_priority
        self.started = True
        self.t0 = time.time()
        for t in self.threads:
            t.start()

    def _work(self, t: int, threads: int) -> None:
        try:
            if self.cpus:
                os.sched_setaffinity(0, self.cpus)  # this thread only (Linux)
            torch.cuda.set_device(self.device)
            stream = torch.cuda.Stream(priority=getattr(self, "stream_priority", 0))
            cudart = _cudart()
            buffers, used = self.buffers[t], [None, None]
            for it, i in enumerate(range(t, len(self.paths), threads)):
                path, n, dst = self.paths[i], self.sizes[i], self.starts[i]
                if self.mode == "register":
                    r0 = time.time()
                    fd = os.open(path, os.O_RDWR)
                    try:
                        m = mmap.mmap(fd, 0, mmap.MAP_SHARED | mmap.MAP_POPULATE, mmap.PROT_READ | mmap.PROT_WRITE)
                    finally:
                        os.close(fd)
                    ptr = ctypes.addressof(ctypes.c_char.from_buffer(m))
                    err = cudart.cudaHostRegister(ptr, len(m), 0)
                    if err != 0:
                        raise RuntimeError(f"cudaHostRegister of {path}: error {err}")
                    with self.lock:
                        self.maps.append((m, ptr))
                    self.stats["pin_s"][t] += time.time() - r0
                    r0 = time.time()
                    src = torch.frombuffer(m, dtype=torch.int16, count=n, offset=HEADER_BYTES)
                    with torch.cuda.stream(stream):
                        self.tok[dst : dst + n].copy_(src, non_blocking=True)
                        ev = torch.cuda.Event()
                        ev.record(stream)
                    self.stats["copy_enqueue_s"][t] += time.time() - r0
                else:
                    if buffers is None:
                        r0 = time.time()
                        buffers = [torch.empty(self.chunk_tokens, dtype=torch.int16, pin_memory=True) for _ in range(2)]
                        with self.lock:
                            self.pinned += buffers
                        self.stats["pin_alloc_s"] = max(self.stats["pin_alloc_s"], time.time() - r0)
                    fd = os.open(path, os.O_RDONLY)
                    try:
                        for j, a in enumerate(range(0, n, self.chunk_tokens)):
                            c = min(n, a + self.chunk_tokens) - a
                            b = j % 2
                            if used[b] is not None:
                                used[b].synchronize()
                            r0 = time.time()
                            view = memoryview(buffers[b].numpy()).cast("B")[: 2 * c]
                            if os.preadv(fd, [view], HEADER_BYTES + 2 * a) != 2 * c:
                                raise OSError(f"short read of {path}")
                            self.stats["read_s"][t] += time.time() - r0
                            with torch.cuda.stream(stream):
                                self.tok[dst + a : dst + a + c].copy_(buffers[b][:c], non_blocking=True)
                                ev = torch.cuda.Event()
                                ev.record(stream)
                            used[b] = ev
                    finally:
                        os.close(fd)
                self.events[i] = ev
                self.ready[i].set()
            stream.synchronize()
            with self.lock:
                self.t_done = max(self.t_done, time.time())
        except BaseException as e:  # noqa: BLE001
            self.errors.append(e)
            for r in self.ready:
                r.set()

    def wait(self, i: int):
        """Shard i's copy event, once enqueued."""
        self.ready[i].wait()
        if self.errors:
            raise self.errors[0]
        return self.events[i]

    def join(self) -> None:
        for t in self.threads:
            t.join()
        if self.errors:
            raise self.errors[0]
        torch.cuda.synchronize()

    def release(self) -> None:
        """Unpin and unmap (after the clock)."""
        cudart = _cudart()
        for m, ptr in self.maps:
            cudart.cudaHostUnregister(ptr)
        self.maps.clear()


def d2h(tensors: dict[str, torch.Tensor], outs: dict[str, np.ndarray], bufs: list[torch.Tensor]) -> None:
    """The GPU tensors into the (pre-faulted) host arrays of the same bytes, in pieces through pinned buffers: the
    DMA of later pieces overlaps the host copy of earlier ones."""
    pieces = []
    for name, t in tensors.items():
        src = t.reshape(-1).view(torch.uint8)
        dst = outs[name].reshape(-1).view(np.uint8)
        assert src.numel() == dst.size, name
        size = bufs[0].numel() * bufs[0].element_size()
        for a in range(0, src.numel(), size):
            pieces.append((src, dst, a, min(src.numel(), a + size)))
    raw = [b.view(torch.uint8) for b in bufs]
    events: list = [None] * len(pieces)

    def issue(j: int) -> None:
        src, _, a, b = pieces[j]
        raw[j % len(raw)][: b - a].copy_(src[a:b], non_blocking=True)
        events[j] = torch.cuda.Event()
        events[j].record()

    for j in range(min(len(raw), len(pieces))):
        issue(j)
    for j, (_, dst, a, b) in enumerate(pieces):
        events[j].synchronize()
        np.copyto(dst[a:b], raw[j % len(raw)][: b - a].numpy())
        if j + len(raw) < len(pieces):
            issue(j + len(raw))


def layout(paths: list[str]) -> tuple[list[int], list[int], int]:
    """Each shard's token count and first position (32-aligned, >= GAP after the previous shard), and the length."""
    sizes = [shard_tokens(p) for p in paths]
    starts, at = [], GAP
    for n in sizes:
        starts.append(at)
        at = (at + n + GAP + 31) // 32 * 32
    return sizes, starts, at


def bench_io(paths: list[str], threads: int, chunk_tokens: int, tok: torch.Tensor, mode: str) -> dict:
    """Untimed micro-benchmarks after the run, each part alone: the read (page cache -> pinned host buffers, `threads`
    threads) and the H2D copy from pinned buffers; with mode "register", also pinning the shard files in place and the
    DMA copy from them."""
    sizes = [shard_tokens(p) for p in paths]
    tasks = [(p, a, min(n, a + chunk_tokens) - a) for p, n in zip(paths, sizes) for a in range(0, n, chunk_tokens)]
    buffers = [torch.empty(chunk_tokens, dtype=torch.int16, pin_memory=True) for _ in range(threads)]

    def work(t: int) -> None:
        for path, a, n in tasks[t::threads]:
            fd = os.open(path, os.O_RDONLY)
            os.preadv(fd, [memoryview(buffers[t].numpy()).cast("B")[: 2 * n]], HEADER_BYTES + 2 * a)
            os.close(fd)

    def run_threads(fn) -> float:
        t0 = time.time()
        ts = [threading.Thread(target=fn, args=(t,)) for t in range(threads)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        return time.time() - t0

    total = 2 * sum(sizes)
    res = {"bytes": total, "threads": threads}
    read = run_threads(work)
    res["pread_GBps"] = total / read / 1e9
    big = torch.empty(chunk_tokens * 4, dtype=torch.int16, pin_memory=True)
    reps = max(1, total // (2 * big.numel()))
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(reps):
        j = i * big.numel() % (tok.numel() - big.numel())
        tok[j : j + big.numel()].copy_(big, non_blocking=True)
    torch.cuda.synchronize()
    res["h2d_pinned_GBps"] = reps * 2 * big.numel() / (time.time() - t0) / 1e9
    if mode == "register":
        cudart = _cudart()
        maps: list = [None] * len(paths)

        def pin(t: int) -> None:
            for i in range(t, len(paths), threads):
                fd = os.open(paths[i], os.O_RDWR)
                m = mmap.mmap(fd, 0, mmap.MAP_SHARED | mmap.MAP_POPULATE, mmap.PROT_READ | mmap.PROT_WRITE)
                os.close(fd)
                ptr = ctypes.addressof(ctypes.c_char.from_buffer(m))
                cudart.cudaHostRegister(ptr, len(m), 0)
                maps[i] = (m, ptr)

        res["register_GBps"] = total / run_threads(pin) / 1e9
        torch.cuda.synchronize()
        t0 = time.time()
        at = 0
        for (m, _), n in zip(maps, sizes):
            n = min(n, tok.numel() - at)
            tok[at : at + n].copy_(torch.frombuffer(m, dtype=torch.int16, count=n, offset=HEADER_BYTES), non_blocking=True)
            at += n
        torch.cuda.synchronize()
        res["h2d_registered_GBps"] = 2 * at / (time.time() - t0) / 1e9
        t0 = time.time()
        for m, ptr in maps:
            cudart.cudaHostUnregister(ptr)
        res["unregister_s"] = time.time() - t0
    return res


# Slots per sub-table of a partitioned pair table, at least (512 KB; smaller ran faster on the L40S and the A5000, down
# to about 64K slots), and the most sub-table bits (main(): 13, or less where the device allows less).
SUB_SLOTS = 1 << 16
MAX_PBITS = 12


class Scanner:
    """One level's scan over position ranges (each a multiple of 32 positions from a 32-aligned start): counted
    directly in the pair table, or (partitioned) the hits' pair keys emitted into a buffer of up to `chunk` keys,
    then, per full buffer, grouped by sub-table and counted sub-table by sub-table (so the table's atomics stay in
    L2)."""

    def __init__(self, lib, k, vi, x, tok, bm_prev, bm_next, table, grid, partitioned, sparse, chunk, pbits):
        self.lib, self.k, self.vi, self.x, self.tok = lib, k, vi, x, tok
        self.bm_prev, self.bm_next, self.table, self.grid = bm_prev, bm_next, table, grid
        self.partitioned, self.sparse, self.chunk, self.pbits = partitioned, sparse, chunk, pbits
        self.words = False  # medium-density levels: a lane per bitmap word (set by the caller)
        self.times: dict[str, float] = {}
        self.events: list = []
        self.pending = 0
        if partitioned:
            self.emit = torch.empty(chunk, dtype=torch.int64, device="cuda")
            self.grouped = torch.empty(chunk, dtype=torch.int64, device="cuda")
            self.emitted = torch.zeros(1, dtype=torch.int64, device="cuda")
            self.counts = torch.zeros(1 << pbits, dtype=torch.int64, device="cuda")

    def scan(self, start: int, n: int) -> None:
        if self.partitioned and self.pending + n > self.chunk:
            self.flush()
        bm_in = None if self.bm_prev is None else self.bm_prev[start // 32 :]
        if self.partitioned:
            mode = 3 if bm_in is not None and self.words else 2
        else:
            mode = int(self.sparse and bm_in is not None)
        pk, pc, sub_cap, pbits, flags = self.table
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        self.lib("scan_level", self.tok[start:], n, i32(self.k), bm_in, self.bm_next[start // 32 :], self.vi.table,
                 ctypes.c_uint64(self.vi.cap), self.vi.rep, self.x, pk, pc, sub_cap, pbits, flags, i32(self.grid), i32(mode),
                 self.emit if self.partitioned else None, self.emitted if self.partitioned else None)
        e1.record()
        self.events.append(("scan", e0, e1))
        self.pending += n

    def flush(self) -> None:
        """Count the emitted keys: grouped by sub-table, then aggregated."""
        if not self.partitioned or self.pending == 0:
            return
        torch.cuda.current_stream().synchronize()
        t0 = time.time()
        n = int(self.emitted.item())
        pk, pc, sub_cap, pbits, flags = self.table
        e = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        e[0].record()
        self.counts.zero_()
        self.lib("subpart_count", self.emit, n, pbits, self.counts)
        cursor = torch.cumsum(self.counts, 0) - self.counts
        self.lib("subpart_scatter", self.emit, n, pbits, cursor, self.grouped)
        e[1].record()
        self.lib("aggregate", self.grouped, n, pk, pc, sub_cap, pbits, flags, i32(self.grid))
        e[2].record()
        torch.cuda.current_stream().synchronize()
        self.events += [("group", e[0], e[1]), ("aggregate", e[1], e[2])]
        self.times["flush_wall"] = self.times.get("flush_wall", 0.0) + time.time() - t0
        self.emitted.zero_()
        self.pending = 0


class IpcComm:
    """The ranks' GPU-to-GPU traffic over CUDA IPC (each rank copies what it needs out of its peers' buffers over
    NVLink/PCIe P2P), with a gloo process group for the small exchanges and the barriers: no NCCL communicator, so no
    NCCL init and no all-to-all connection setup (seconds each on the 8 H200s)."""

    def __init__(self, world: int, rank: int):
        self.world, self.rank = world, rank
        self.cap = 0
        self.arena_k = self.arena_c = None
        self.peer_k: list = []
        self.peer_c: list = []

    def share(self, tensors: list[torch.Tensor]) -> list[list[torch.Tensor]]:
        """Collective: per tensor, every rank's (this rank's own, the peers' opened from their IPC handles)."""
        from torch.multiprocessing.reductions import reduce_tensor

        mine = [reduce_tensor(t) for t in tensors]
        allh: list = [None] * self.world
        dist.all_gather_object(allh, mine)
        out = []
        for i, t in enumerate(tensors):
            out.append([t if r == self.rank else allh[r][i][0](*allh[r][i][1]) for r in range(self.world)])
        return out

    def arena(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Collective: this rank's send buffers (keys, counts) of at least n entries, shared with the peers."""
        need = torch.tensor([n, self.cap], dtype=torch.int64)
        alln = [torch.zeros(2, dtype=torch.int64) for _ in range(self.world)]
        dist.all_gather(alln, need)
        if any(int(a[0]) > int(a[1]) for a in alln):  # some rank grows: all re-share
            if n > self.cap:
                self.arena_k = self.arena_c = None
                self.cap = int(n * 1.1) + 1024
                self.arena_k = torch.empty(self.cap, dtype=torch.int64, device="cuda")
                self.arena_c = torch.empty(self.cap, dtype=torch.int32, device="cuda")
            self.peer_k, self.peer_c = self.share([self.arena_k, self.arena_c])
        return self.arena_k[:n], self.arena_c[:n]

    def exchange(self, send_counts: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Collective, after the arena holds this rank's entries grouped by owner: this rank's received entries."""
        mine = torch.tensor(send_counts, dtype=torch.int64)
        allc = [torch.zeros(self.world, dtype=torch.int64) for _ in range(self.world)]
        dist.all_gather(allc, mine)
        torch.cuda.synchronize()
        dist.barrier()  # every rank's arena is written
        n = sum(int(c[self.rank]) for c in allc)
        recv_k = torch.empty(n, dtype=torch.int64, device="cuda")
        recv_c = torch.empty(n, dtype=torch.int32, device="cuda")
        at = 0
        for r in range(self.world):
            m = int(allc[r][self.rank])
            off = int(allc[r][: self.rank].sum())
            if m:
                recv_k[at : at + m].copy_(self.peer_k[r][off : off + m], non_blocking=True)
                recv_c[at : at + m].copy_(self.peer_c[r][off : off + m], non_blocking=True)
            at += m
        torch.cuda.synchronize()
        dist.barrier()  # every rank has read: the arenas may be rewritten
        return recv_k, recv_c

    def sum_into(self, local: torch.Tensor, peers: list[torch.Tensor]) -> None:
        """local += every peer's (int32: the ranks' disjoint parts), copied over P2P."""
        tmp = torch.empty_like(local)
        for r, t in enumerate(peers):
            if r != self.rank:
                tmp.copy_(t)
                local += tmp


class ValIndex:
    """One level's validation contexts: per position its context id (-1 where invalid), the number of contexts, and
    the lookup table (8-byte slots, a context's id is its slot) with each slot's representative position."""

    def __init__(self, lib: Lib, x: torch.Tensor, avail: torch.Tensor, k: int):
        n = x.numel()
        keys = torch.empty(n, dtype=torch.int64, device="cuda")
        lib("val_keys", x, avail, n, i32(k), keys)
        pos = (keys != -1).nonzero().squeeze(1)
        # sorted as unsigned (the sign bit flipped): the order of the keys' home slots, ties by key
        sk, perm = torch.sort(keys[pos] ^ SIGN, stable=True)
        del keys
        spos = pos[perm]
        del pos, perm
        new = torch.ones_like(sk, dtype=torch.bool)
        new[1:] = sk[1:] != sk[:-1]
        gid = torch.cumsum(new, 0) - 1
        rep = spos[new].contiguous()
        ukeys = (sk[new] ^ SIGN).contiguous()
        del sk
        self.n = int(ukeys.numel())
        if bool((ukeys == -2).any()):
            raise RuntimeError(f"k={k}: a validation key collides with the empty marker")
        bad = torch.zeros(1, dtype=torch.int64, device="cuda")
        lib("val_verify", x, spos, gid, rep, int(spos.numel()), i32(k), bad)
        if int(bad.item()):
            raise RuntimeError(f"k={k}: {int(bad.item())} validation positions collide with another context's key")
        self.valid = int(spos.numel())
        # a context's id is its slot (< 2^24, the pair keys' bits), in a canonical layout (the same on every rank):
        # the keys in home order, each at max(its home, the previous key's slot + 1); no wrap-around
        self.cap = min(2 * self.n + 1024, (1 << 24) - 4096)
        home = torch.empty(self.n, dtype=torch.int64, device="cuda")
        lib("val_homes", ukeys, self.n, ctypes.c_uint64(self.cap), home)
        at = torch.arange(self.n, device="cuda", dtype=torch.int64)
        slot_of = torch.empty_like(home)
        lib.cub("inclusive_max", home - at, slot_of, self.n)
        slot_of += at
        size = max(self.cap, int(slot_of[-1].item()) + 1 if self.n else 0) + 1  # every home, then an empty slot
        if size >= 1 << 24:
            raise RuntimeError(f"k={k}: {self.n} contexts do not fit 24-bit ids")
        self.table = torch.full((size,), -1, dtype=torch.int64, device="cuda")
        self.table[slot_of] = ukeys
        self.rep = torch.zeros(size, dtype=torch.int32, device="cuda")
        self.rep[slot_of] = rep.to(torch.int32)
        self.ids = size  # the id space
        del home, at
        self.ctx_of = torch.full((n,), -1, dtype=torch.int32, device="cuda")
        self.ctx_of[spos] = slot_of[gid].to(torch.int32)


SIGN = -(1 << 63)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True, help="a file listing the training shard paths (all ranks)")
    ap.add_argument("--val", required=True, help="the validation shard")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8, help="loader threads per rank")
    ap.add_argument("--load-mode", choices=("register", "pread"), default="pread")
    ap.add_argument("--chunk-mb", type=int, default=16, help="pread mode's chunks")
    ap.add_argument("--sequential", action="store_true", help="no overlap: load, then the index, then the scans")
    ap.add_argument("--direct", action="store_true", help="count every level directly in its pair table")
    ap.add_argument("--pipeline", action="store_true", help="overlap each level's merge with the next level's scan")
    ap.add_argument("--async-init", action="store_true", help="initialize NCCL on a thread, overlapping the index")
    ap.add_argument("--nccl-warmup", action="store_true", help="a small all-to-all right after the NCCL init")
    ap.add_argument("--own-device", action="store_true", help="each rank sees only its GPU (CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--ipc", action="store_true", help="GPU traffic over CUDA IPC, gloo for the rest (no NCCL)")
    ap.add_argument("--early-ctx", action="store_true", help="create the CUDA context on a thread during the imports")
    ap.add_argument("--fused", action="store_true", help="scan each shard through every level as it lands (direct)")
    ap.add_argument("--numa", action="store_true", help="loader threads on the CPUs of the GPU's NUMA node")
    ap.add_argument("--words-density", type=float, default=0.0,
                    help="levels with fewer than 1/this active positions scan by bitmap words (0: never)")
    ap.add_argument("--chunk-positions", type=int, default=0, help="partitioned scans' buffer (0: by memory)")
    ap.add_argument("--bench-io", action="store_true")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--launch-time", type=float, default=float(os.environ.get("GVS_LAUNCH_TIME", "nan")))
    return ap.parse_args(argv)


def main() -> None:
    run(parse_args())


def run(args: argparse.Namespace, embed: "InProcess | None" = None) -> dict | None:
    """The whole build (main()'s), or with `embed` inside a process that has its process group and CUDA context
    already (InProcess.finish: no NCCL init, the loader and the kernels' library from `embed`, only `embed.variants`'
    hint rows, the outputs all-reduced and returned on every rank's GPU, nothing saved). Times are from process
    start (T_PROC), or from the call when embedded."""
    steps = _run(args, embed)
    try:
        while True:
            next(steps)
    except StopIteration as stop:
        return stop.value


def _run(args: argparse.Namespace, embed: "InProcess | None" = None):
    """run()'s body, as a generator: with `embed.split` it yields once, after the fused scans (before any collective:
    the part up to there may run on a background thread and stream, InProcess.start_background), and the caller
    resumes it for the merges and rows."""
    split = embed is not None and embed.split
    t_base = T_PROC if embed is None else time.time()
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    out = Path(args.out)
    timing: dict = {"rank": rank, "world": world, "proc_start": T_PROC, "import_s": T_IMPORT - T_PROC,
                    "sequential": args.sequential}
    marks: dict[str, float] = {}
    variants = VARIANTS if embed is None else embed.variants

    # ---- init; the shards start loading as soon as there is a CUDA context
    device = (0 if args.own_device else local) if embed is None else embed.device
    if "thread" in EARLY:
        EARLY.pop("thread").join()
        marks.update({f"early_ctx_{k}": v for k, v in EARLY.items()})
    torch.cuda.set_device(device)
    torch.empty(1, device="cuda")
    marks["cuda_ctx"] = time.time() - t_base
    if embed is None:
        paths = [p for p in Path(args.shards).read_text().split() if p]
        mine = paths[rank * len(paths) // world : (rank + 1) * len(paths) // world]
        sizes, starts, L = layout(mine)
        loader = Loader(mine, L, starts, sizes, args.threads, args.load_mode, args.chunk_mb << 19, args.numa)
    else:
        mine, sizes, starts, L, loader = embed.mine, embed.sizes, embed.starts, embed.L, embed.loader
        if not loader.started:
            loader.start()  # not started during the training (InProcess.start_load)
    n_tokens = sum(sizes)
    tok = loader.tok
    clock = Clock(world if not split else 1)  # split: no barriers before the yield
    clock.t = t_base

    def init_nccl() -> None:
        torch.cuda.set_device(device)
        t0 = time.time()
        dist.init_process_group("nccl", device_id=torch.device("cuda", device), timeout=timedelta(minutes=4))
        dist.barrier()
        marks["nccl_init_s"] = time.time() - t0
        if args.nccl_warmup:
            # the first all-to-all sets up every peer connection (seconds on 8 GPUs): done here, off the merge's path
            t0 = time.time()
            a = torch.zeros(world * 256, dtype=torch.int64, device="cuda")
            b = torch.empty_like(a)
            dist.all_to_all_single(b, a)
            dist.all_to_all_single(b, a, [256] * world, [256] * world)
            dist.all_reduce(a[:256])
            dist.reduce(a[:256], dst=0)
            torch.cuda.synchronize()
            marks["nccl_warmup_s"] = time.time() - t0
        marks["nccl_ready"] = time.time() - t_base

    nccl_thread = None
    comm = None
    if embed is not None:
        pass  # the process group exists (and its all-to-all connections are warm: InProcess.prepare)
    elif world > 1 and args.ipc:
        t0 = time.time()
        dist.init_process_group("gloo", timeout=timedelta(minutes=4))
        dist.barrier()
        marks["gloo_init_s"] = time.time() - t0
        comm = IpcComm(world, rank)
    elif world > 1 and args.async_init:
        # NCCL initializes on a thread while the index builds and the first level scans (no collective before the
        # first merge); phase barriers start once it is ready
        clock.world = 1
        nccl_thread = threading.Thread(target=init_nccl, daemon=True)
        nccl_thread.start()
    elif world > 1:
        init_nccl()
    lib = Lib(lib_path()) if embed is None else embed.lib
    clock.lap("init")
    props = torch.cuda.get_device_properties(device)
    timing["gpu"] = props.name
    sms = props.multi_processor_count
    if args.sequential:
        loader.join()
        clock.lap("load")

    # ---- the validation index, every level (overlapping the load)
    val = np.fromfile(args.val, dtype="<u2", count=VAL_TOKENS + 1 + HEADER_BYTES // 2)[HEADER_BYTES // 2 :]
    N = VAL_TOKENS
    x = torch.from_numpy(val[:N].view(np.int16).copy()).cuda()
    tgt = torch.from_numpy(val[1 : N + 1].view(np.int16).copy()).cuda()
    idx = torch.arange(N, device="cuda", dtype=torch.int64)
    xi = x.to(torch.int32) & 0xFFFF
    seg = torch.cummax(torch.where(xi == BOS, idx, torch.zeros_like(idx)), 0).values
    avail = (idx - seg + 1).to(torch.int32)
    del idx, xi, seg
    index = {}
    for k in ALL_LEVELS:
        t0 = time.time()
        index[k] = ValIndex(lib, x, avail, k)
        if os.environ.get("GVS_DEBUG_INDEX"):
            torch.cuda.synchronize()
            print(f"rank {rank} index k={k}: {time.time() - t0:.3f} s", flush=True)
    clock.lap("val_index", barrier=False)
    timing["val_index"] = {k: {"contexts": v.n, "valid": v.valid} for k, v in index.items()}

    # rank 0's host arrays for the outputs, allocated and faulted in by a thread while the GPUs scan
    host: dict[str, np.ndarray] = {}
    shapes = {"tot": ((N, len(ALL_LEVELS)), np.uint32), "dist": ((N, len(ALL_LEVELS)), np.uint16),
              "cnt": ((N, len(ALL_LEVELS)), np.uint32), "cell": ((len(VARIANTS), N), np.int16),
              "tokens": ((len(VARIANTS), N, ENTRIES), np.uint16), "weights": ((len(VARIANTS), N, ENTRIES), np.uint16)}

    def prefault() -> None:
        t0 = time.time()
        for name, (shape, dtype) in shapes.items():
            a = np.empty(shape, dtype)
            a.fill(0)
            host[name] = a
        marks["host_prefault_s"] = time.time() - t0

    host_thread = threading.Thread(target=prefault, daemon=True) if rank == 0 and embed is None else None
    if host_thread is not None:
        host_thread.start()

    # ---- per level: scan, local aggregation, merge, outputs
    out_cnts = torch.zeros((3, len(ALL_LEVELS), N), dtype=torch.int32, device="cuda")  # tot, dist, cnt
    mask = torch.zeros(N, dtype=torch.int32, device="cuda")
    rows_buf = torch.zeros((len(VARIANTS), N, 9), dtype=torch.int32, device="cuda") if args.ipc else None
    tops: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]] = {}
    nwords = L // 32
    bm_prev = None
    bm_next = torch.empty(nwords, dtype=torch.int32, device="cuda")
    popc = torch.zeros(1, dtype=torch.int64, device="cuda")
    levels_info = {}
    grid = sms * 32
    global SUB_SLOTS, MAX_PBITS
    SUB_SLOTS = int(os.environ.get("GVS_SUB_SLOTS", 0)) or SUB_SLOTS
    MAX_PBITS = min(int(os.environ.get("GVS_MAX_PBITS", 13)), lib.so.max_pbits())
    timing["sub_slots"], timing["max_pbits"] = SUB_SLOTS, MAX_PBITS
    # the partitioned scans' key buffers (two of 8 bytes per position): at most 512M positions, 15% of free memory
    free = torch.cuda.mem_get_info()[0] + torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    chunk_positions = args.chunk_positions or min(1 << 29, int(free * 0.15 / 16))
    chunk_positions = max(1 << 20, chunk_positions // 32 * 32)
    timing["chunk_positions"] = chunk_positions
    # --pipeline: each level's merge and outputs run on a thread and stream of their own, overlapping the next
    # level's scan (no per-phase barriers then: pipe_lap records each merge step's time on its stream)
    merge_stream = torch.cuda.Stream() if args.pipeline else None
    merger = ThreadPoolExecutor(1) if args.pipeline else None
    futures: list = []
    pipe_times: dict[str, float] = {}

    def pipe_lap(name: str) -> None:
        torch.cuda.current_stream().synchronize()
        now = time.time()
        last = pipe_times.get("_t", now)
        group = name.split(".")[-1]
        pipe_times[group] = pipe_times.get(group, 0.0) + now - last
        pipe_times["_t"] = now

    if args.pipeline:
        clock.lap = lambda name, barrier=True: None  # type: ignore[method-assign]
    # --fused (direct tables only): every level's pair table up front, and each shard scanned through all the levels
    # as soon as it lands (a level's bitmap only needs the same positions' previous level), so the scans overlap the
    # load; the merges follow, level by level
    fused: dict = {}
    if args.fused:
        for k in ALL_LEVELS:
            u, beta = PAIRS[k]
            est = min(SAFETY * u * (n_tokens / 4e8) ** beta, n_tokens)
            cap = max(1 << 20, int(est / LOAD))
            fused[k] = (torch.full((cap,), -1, dtype=torch.int64, device="cuda"),
                        torch.zeros((cap + 3) // 4, dtype=torch.int32, device="cuda"), cap,
                        torch.zeros(2, dtype=torch.int32, device="cuda"))
        bms = [bm_next, torch.zeros(nwords, dtype=torch.int32, device="cuda")]
        bms[0].zero_()
        if args.sequential:
            loader.join()
        for i, (st, n) in enumerate(zip(starts, sizes)):
            if not args.sequential:
                torch.cuda.current_stream().wait_event(loader.wait(i))
            n32 = (n + 31) // 32 * 32
            for li, k in enumerate(ALL_LEVELS):
                vi = index[k]
                pk, pc, cap, flags = fused[k]
                bm_in = None if li == 0 else bms[(li - 1) % 2][st // 32 :]
                mode = 0 if li == 0 or k <= 5 else 1
                lib("scan_level", tok[st:], n32, i32(k), bm_in, bms[li % 2][st // 32 :], vi.table,
                    ctypes.c_uint64(vi.cap), vi.rep, x, pk, pc, ctypes.c_uint64(cap), i32(0), flags, i32(grid),
                    i32(mode), None, None)
        loader.join()
        marks.setdefault("load_done", loader.t_done - t_base)
        torch.cuda.synchronize()
        bad = [k for k in ALL_LEVELS if int(fused[k][3][0].item())]
        if bad:
            raise RuntimeError(f"fused scan: the pair tables of levels {bad} overflowed")
        del bms
        marks["fused_scan_done"] = time.time() - t_base
        if nccl_thread is not None:
            nccl_thread.join()
            nccl_thread = None
            clock.world = world
        if split:  # the background part ends here; the caller resumes on its own thread and stream
            yield
            torch.cuda.current_stream().wait_stream(embed.stream)
            clock.world = world
            marks["resumed"] = time.time() - t_base
            clock.t = time.time()
        clock.lap("scan_all")
    for li, k in enumerate(ALL_LEVELS):
        info: dict = {}
        vi = index[k]

        bounds_l = [vi.ids * o // world for o in range(world + 1)]  # context ids are slots
        bounds = torch.tensor(bounds_l, dtype=torch.int64, device="cuda")
        lo, hi = bounds_l[rank], bounds_l[rank + 1]
        if fused:
            pk, pc, cap, flags = fused.pop(k)
            fl = flags.cpu().tolist()
            info.update(contexts=vi.n, cap=cap, false_hash_hits=fl[1], fused=True)
        else:
            if bm_prev is None:
                active = n_tokens
            else:
                popc.zero_()
                lib("popcount", bm_prev, nwords, popc)
                active = int(popc.item())
            free = torch.cuda.mem_get_info()[0] + torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
            u, beta = PAIRS[k]
            est = min(SAFETY * u * (n_tokens / 4e8) ** beta, active)
            cap = max(1 << 20, int(est / LOAD))
            cap = min(cap, int(free * 0.45 / 9))
            sparse = bm_prev is not None and active * 16 < L
            clock.lap(f"k{k}.prep", barrier=False)
            # partitioned aggregation for big tables (dense levels): sub-tables of about SUB_SLOTS slots
            partitioned = not sparse and cap > 4 * SUB_SLOTS and not args.direct
            reruns = 0
            while True:
                pbits = min(MAX_PBITS, max(1, (cap // SUB_SLOTS).bit_length())) if partitioned else 0
                sub_cap = -(-cap // (1 << pbits))
                cap = sub_cap << pbits
                pk = torch.full((cap,), -1, dtype=torch.int64, device="cuda")
                pc = torch.zeros((cap + 3) // 4, dtype=torch.int32, device="cuda")  # per slot a byte: its count's wraps
                flags = torch.zeros(2, dtype=torch.int32, device="cuda")
                table = (pk, pc, ctypes.c_uint64(sub_cap), i32(pbits), flags)
                first = bm_prev is None
                if first:
                    bm_next.zero_()
                    ranges = [(st, (n + 31) // 32 * 32, i) for i, (st, n) in enumerate(zip(starts, sizes))]
                else:
                    ranges = [(0, L, None)]
                overlap = first and not args.sequential and reruns == 0
                if first and not overlap:
                    loader.join()
                    marks.setdefault("load_done", loader.t_done - t_base)
                t_scan = time.time()
                scanner = Scanner(lib, k, vi, x, tok, bm_prev, bm_next, table, grid, partitioned, sparse, chunk_positions,
                                  pbits)
                scanner.words = active * args.words_density < L
                for st, n, i in ranges:
                    for a in range(0, n, chunk_positions):
                        if overlap and a == 0:
                            torch.cuda.current_stream().wait_event(loader.wait(i))
                        scanner.scan(st + a, min(n, a + chunk_positions) - a)
                scanner.flush()
                if overlap:
                    loader.join()
                    marks["load_done"] = loader.t_done - t_base
                torch.cuda.current_stream().synchronize()
                for name, e0, e1 in scanner.events:
                    scanner.times[name] = scanner.times.get(name, 0.0) + e0.elapsed_time(e1) / 1e3
                info["scan_parts_s"] = scanner.times
                fl = flags.cpu().tolist()
                info["scan_wall_s"] = time.time() - t_scan
                if fl[0] == 0:
                    break
                reruns += 1
                del pk, pc, scanner
                cap *= 2
            info.update(pbits=pbits, sub_cap=sub_cap)
            del scanner  # its key buffers
            popc.zero_()
            lib("popcount", bm_next, nwords, popc)
            hits = int(popc.item())
            if nccl_thread is not None:  # before the first collective
                nccl_thread.join()
                nccl_thread = None
                clock.world = world
            clock.lap(f"k{k}.scan")
            info.update(contexts=vi.n, active=active, hits=hits, cap=cap, reruns=reruns, sparse=sparse,
                        false_hash_hits=fl[1])

        def merge_level(k=k, li=li, vi=vi, pk=pk, pc=pc, cap=cap, bounds=bounds, lo=lo, hi=hi, info=info) -> None:
            """Local aggregation, merge and outputs of level k (on the merge thread's stream when pipelined)."""
            lap = clock.lap if not args.pipeline else pipe_lap
            # local aggregation: the table's entries by owner
            counts = torch.zeros(world, dtype=torch.int64, device="cuda")
            lib("part_count", pk, ctypes.c_uint64(cap), bounds, i32(world), counts)
            send_counts = counts.cpu().tolist()
            u_local = sum(send_counts)
            cursor = torch.zeros(world, dtype=torch.int64, device="cuda")
            cursor[1:] = torch.cumsum(counts, 0)[:-1]
            if comm is not None:
                send_k, send_c = comm.arena(u_local)
            else:
                send_k = torch.empty(u_local, dtype=torch.int64, device="cuda")
                send_c = torch.empty(u_local, dtype=torch.int32, device="cuda")
            lib("part_scatter", pk, pc, ctypes.c_uint64(cap), bounds, i32(world), cursor, send_k, send_c)
            del pk, pc
            lap(f"k{k}.local")
            # merge: all-to-all, then the owner's sort and sum
            if comm is not None:
                del send_k, send_c
                recv_k, recv_c = comm.exchange(send_counts)
            elif world > 1:
                recv_counts = torch.empty(world, dtype=torch.int64, device="cuda")
                dist.all_to_all_single(recv_counts, counts)
                rc = recv_counts.cpu().tolist()
                recv_k = torch.empty(sum(rc), dtype=torch.int64, device="cuda")
                recv_c = torch.empty(sum(rc), dtype=torch.int32, device="cuda")
                dist.all_to_all_single(recv_k, send_k, rc, send_counts)
                dist.all_to_all_single(recv_c, send_c, rc, send_counts)
                del send_k, send_c
            else:
                recv_k, recv_c = send_k, send_c
                del send_k, send_c
            lap(f"k{k}.comm")
            n_own = hi - lo
            ctx_bits = max(1, (max(n_own, 2) - 1).bit_length())
            recv_k -= lo << 16
            n_recv = recv_k.numel()
            sk = torch.empty_like(recv_k)
            sc = torch.empty_like(recv_c)
            lib.cub("sort_pairs", recv_k, sk, recv_c, sc, n_recv, i32(0), i32(16 + ctx_bits))
            del recv_k, recv_c
            if world > 1:
                uk = torch.empty_like(sk)
                agg = torch.empty_like(sc)
                nruns = torch.zeros(1, dtype=torch.int64, device="cuda")
                lib.cub("reduce_by_key", sk, uk, sc, agg, nruns, n_recv)
                U = int(nruns.item())
                del sk, sc
                uk, agg = uk[:U], agg[:U]
            else:
                uk, agg, U = sk, sc, n_recv
            lap(f"k{k}.merge")
            # per owned context: segments, totals, ranks, tops; per owned position: tot, dist, cnt
            start = torch.zeros(n_own, dtype=torch.int64, device="cuda")
            end = torch.zeros(n_own, dtype=torch.int64, device="cuda")
            lib("seg_bounds", uk, U, start, end)
            tot = torch.empty(n_own, dtype=torch.int64, device="cuda")
            lib("ctx_tot", start, end, agg, n_own, tot)
            key2 = torch.empty(U, dtype=torch.int64, device="cuda")
            val2 = torch.empty(U, dtype=torch.int32, device="cuda")
            lib("rank_keys", uk, agg, U, key2, val2)
            k2s = torch.empty_like(key2)
            sv = torch.empty_like(val2)
            lib.cub("sort_pairs", key2, k2s, val2, sv, U, i32(0), i32(32 + ctx_bits))
            del key2, val2, k2s
            rank_of = torch.empty(U, dtype=torch.int32, device="cuda")
            top_tok = top_cnt = None
            if k in LEVELS:
                top_tok = torch.zeros(n_own * ENTRIES, dtype=torch.int16, device="cuda")
                top_cnt = torch.zeros(n_own * ENTRIES, dtype=torch.int32, device="cuda")
            lib("rank_scatter", sv, U, uk, agg, start, rank_of, top_tok, top_cnt, i32(ENTRIES))
            del sv
            lib("pos_outputs", vi.ctx_of, N, lo, hi, tgt, uk, agg, rank_of, start, end, tot, i32(li), i32(TOP),
                out_cnts[0, li], out_cnts[1, li], out_cnts[2, li], mask)
            max_tot = int(tot.max().item()) if n_own else 0
            if max_tot >= 1 << 32:
                raise RuntimeError(f"k={k}: a total of {max_tot} does not fit uint32")
            if k in LEVELS:
                tops[k] = (top_tok, top_cnt, tot, lo, hi)
            del uk, agg, rank_of, start, end
            if k not in LEVELS:
                del tot
            lap(f"k{k}.rows")
            info.update(owned=n_own, pairs_local=u_local, pairs_recv=n_recv, pairs_owned=U,
                        mem_peak_gb=torch.cuda.max_memory_allocated() / 1e9)

        if args.pipeline:
            done = torch.cuda.Event()
            done.record()
            pk.record_stream(merge_stream)
            pc.record_stream(merge_stream)

            def job(fn=merge_level, done=done) -> None:
                with torch.cuda.stream(merge_stream):
                    merge_stream.wait_event(done)
                    merge_stream.synchronize()
                    pipe_times["_t"] = time.time()
                    fn()

            futures.append(merger.submit(job))
            del pk, pc
        else:
            merge_level()
            del pk, pc
        levels_info[k] = info
        bm_prev, bm_next = bm_next, (bm_prev if bm_prev is not None else torch.empty(nwords, dtype=torch.int32,
                                                                                         device="cuda"))
    del bm_prev, bm_next
    if args.pipeline:
        t0 = time.time()
        for f in futures:
            f.result()
        merger.shutdown()
        torch.cuda.synchronize()
        marks["pipeline_drain_s"] = time.time() - t0
        marks["scans_done"] = t0 - t_base
        pipe_times.pop("_t", None)
        timing["pipeline_merge_s"] = pipe_times
        clock.lap = Clock.lap.__get__(clock)  # type: ignore[method-assign]
        clock.lap("pipeline")
    timing["levels"] = levels_info

    # ---- hint rows of each variant, and the reduction to rank 0
    if comm is not None:
        # the outputs' buffers shared once; each rank sums the peers' masks, rank 0 their outputs
        peers_mask, peers_cnts, peers_rows = comm.share([mask, out_cnts, rows_buf])
        torch.cuda.synchronize()
        dist.barrier()
        mask_all = mask.clone()
        comm.sum_into(mask_all, peers_mask)
        torch.cuda.synchronize()
        dist.barrier()
        mask = mask_all
        rows = rows_buf
    elif world > 1:
        dist.all_reduce(mask)
    if comm is None:
        rows = torch.zeros((len(variants), N, 9), dtype=torch.int32, device="cuda")
    for v, levels in enumerate(variants):
        desc = []
        for k in levels:
            top_tok, top_cnt, tot, lo, hi = tops[k]
            desc += [ALL_LEVELS.index(k), LEVELS.index(k), index[k].ctx_of.data_ptr(), lo, hi, top_tok.data_ptr(),
                     top_cnt.data_ptr(), tot.data_ptr()]
        desc_t = torch.tensor(desc, dtype=torch.int64, device="cuda")
        lib("variant_rows", N, mask, i32(len(levels)), desc_t, rows[v])
    if comm is not None:
        torch.cuda.synchronize()
        dist.barrier()
        if rank == 0:
            comm.sum_into(out_cnts, peers_cnts)
            comm.sum_into(rows, peers_rows)
            torch.cuda.synchronize()
        dist.barrier()  # the peers' buffers stay until rank 0 has read them
        del peers_mask, peers_cnts, peers_rows
    elif world > 1 and embed is not None:  # every rank validates: the outputs on every GPU
        dist.all_reduce(out_cnts)
        dist.all_reduce(rows)
    elif world > 1:
        dist.reduce(out_cnts, dst=0)
        dist.reduce(rows, dst=0)
    clock.lap("final_reduce")
    result = None
    if embed is not None:
        torch.cuda.synchronize()
        marks["rows_on_gpu"] = time.time() - t_base
        timing["total_s"] = time.time() - t_base
        timing["phases"] = clock.phases
        timing["mem_peak_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9
        timing["marks"] = marks
        timing["levels"] = {k: {n: v for n, v in info.items() if not isinstance(v, dict)} for k, info in levels_info.items()}
        timing["load"] = {**{n: (max(v) if isinstance(v, list) else v) for n, v in loader.stats.items()},
                          "shards": len(mine), "bytes": 2 * n_tokens, "done_s": loader.t_done - t_base,
                          "start_s": loader.t0 - t_base}
        tops.clear()
        index.clear()
        loader.release()
        loader.tok = None  # the shards' 2 bytes per token: freed (with the pinned buffers)
        loader.pinned, loader.buffers = [], []
        del tok
        return {"counts": out_cnts, "rows": rows, "timing": timing}
    if rank == 0:
        gpu_out = {
            "tot": out_cnts[0].t().contiguous(),
            "dist": out_cnts[1].t().to(torch.int16).contiguous(),
            "cnt": out_cnts[2].t().contiguous(),
            "cell": rows[:, :, 0].to(torch.int16).contiguous(),
            "tokens": rows[:, :, 1:5].contiguous().view(torch.int16),
            "weights": rows[:, :, 5:9].contiguous().view(torch.int16),
        }
        torch.cuda.synchronize()
        marks["rows_on_gpu"] = time.time() - t_base
        host_thread.join()
        bufs = loader.pinned or [torch.empty(16 << 20, dtype=torch.int16, pin_memory=True) for _ in range(4)]
        d2h(gpu_out, host, bufs)
        result = host
        del gpu_out
    clock.lap("final")
    t_end = time.time()
    timing["total_s"] = t_end - t_base
    timing["launch_to_end_s"] = t_end - args.launch_time
    timing["phases"] = clock.phases
    timing["mem_peak_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9
    timing["mem_peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 1e9
    timing["mem_total_gb"] = torch.cuda.get_device_properties(device).total_memory / 1e9
    timing["tok_gb"] = tok.numel() * 2 / 1e9
    timing["marks"] = marks
    timing["load"] = {**loader.stats, "numa_cpus": len(loader.cpus), "shards": len(mine), "tokens": n_tokens, "bytes": 2 * n_tokens,
                      "done_s": loader.t_done - t_base, "start_s": loader.t0 - t_base}
    tops.clear()
    index.clear()
    del rows
    out_cnts = mask = None  # noqa: F841 (frees them)
    loader.release()

    if args.bench_io:
        if world > 1:
            dist.barrier()
        timing["bench_io"] = bench_io(mine, args.threads, args.chunk_mb << 19, tok, args.load_mode)
    del tok
    gathered = [None] * world
    if world > 1:
        dist.all_gather_object(gathered, timing)
    else:
        gathered = [timing]
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        (out / "timings.json").write_text(json.dumps(gathered, indent=1))
        if not args.no_save:
            assert result is not None
            for name in ("tot", "dist", "cnt"):
                np.save(out / f"{name}.npy", result[name])
            for v, levels in enumerate(VARIANTS):
                d = out / f"rows_{'_'.join(map(str, levels))}"
                d.mkdir(exist_ok=True)
                np.save(d / "cell.npy", result["cell"][v])
                np.save(d / "tokens.npy", result["tokens"][v])
                np.save(d / "weights.npy", result["weights"][v])
        print(json.dumps({"total_s": timing["total_s"], "phases": summarize(gathered)}, indent=1), flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


class InProcess:
    """The build inside a training process (a record port's HINT_GPU_VAL=1), on its process group (NCCL, every rank one
    GPU) and CUDA context, each rank loading its 1/world of `shard_paths`:
      - construct (every rank, before the clock): the shards' layout, the GPU token buffer and the pinned staging
        buffers allocated, the kernels' library loaded (built before: GVS_BUILD_DIR / .build, else nvcc now);
      - prepare() (collective, before the clock): the all-to-all's NCCL peer connections set up (seconds on 8 GPUs);
      - start_load() (optional, on the clock): the shard loads start in background threads, each on a CUDA stream of
        its own (a low-priority one), while the training runs; nothing of the validation is read yet;
      - finish() (collective, on the clock): waits for the loads (starting them if start_load was not called), then the
        validation index, the scans, merges and rows, as main()'s --fused --direct run; returns {"counts": int32
        [3, len(ALL_LEVELS), VAL_TOKENS] (tot, dist, cnt; tot and cnt as uint32 bits), "rows": int32 [len(variants),
        VAL_TOKENS, 9] (cell, then the 8 tokens and the 8 weights as int16 pairs), "timing": {...}}, on every rank's GPU.
    `variants`: the hint rows' level sets (each of LEVELS, whose index is a cell's length bucket)."""

    def __init__(self, shard_paths: list[str], val_path: str, variants=(LEVELS,), threads: int = 12, chunk_mb: int = 16,
                 flags: tuple[str, ...] = ("--fused", "--direct")):
        rank, world = dist.get_rank(), dist.get_world_size()
        self.device = torch.cuda.current_device()
        self.split, self.steps, self.thread, self.error = False, None, None, None
        self.variants = tuple(tuple(v) for v in variants)
        if not all(set(v) <= set(LEVELS) for v in self.variants):
            raise ValueError(f"variants {self.variants} must be of {LEVELS}")
        self.args = parse_args(["--shards", "-", "--val", val_path, "--out", "-", "--threads", str(threads),
                                "--chunk-mb", str(chunk_mb), *flags])
        self.mine = shard_paths[rank * len(shard_paths) // world : (rank + 1) * len(shard_paths) // world]
        self.sizes, self.starts, self.L = layout(self.mine)
        self.loader = Loader(self.mine, self.L, self.starts, self.sizes, threads, "pread", chunk_mb << 19, start=False)
        self.loader.preallocate()
        path = lib_path()
        self.lib = Lib(path if path.exists() else build())

    def prepare(self) -> None:
        """The NCCL all-to-all connections (main()'s --nccl-warmup), before the clock."""
        world = dist.get_world_size()
        a = torch.zeros(world * 256, dtype=torch.int64, device="cuda")
        b = torch.empty_like(a)
        dist.all_to_all_single(b, a)
        dist.all_to_all_single(b, a, [256] * world, [256] * world)
        dist.all_reduce(a[:256])
        dist.reduce(a[:256], dst=0)
        torch.cuda.synchronize()

    def start_load(self) -> None:
        self.loader.start(stream_priority=0)

    def start_background(self) -> None:
        """The loads, the validation index and the fused scans (all but the merges and rows) on a background thread
        and CUDA stream of their own (no collectives), from now; finish() then does the rest."""
        if not self.args.fused:
            raise ValueError("start_background needs --fused (its scans end before the first collective)")
        self.split = True
        self.stream = torch.cuda.Stream(device=self.device)
        self.loader.start()
        self.steps = _run(self.args, embed=self)

        def work() -> None:
            try:
                torch.cuda.set_device(self.device)
                with torch.cuda.stream(self.stream):
                    next(self.steps)
            except BaseException as e:  # noqa: BLE001
                self.error = e

        self.thread = threading.Thread(target=work, daemon=True)
        self.thread.start()

    def finish(self) -> dict:
        if self.steps is None:
            out = run(self.args, embed=self)
        else:
            t0 = time.time()
            self.thread.join()
            self.waited = time.time() - t0  # for the background part
            if self.error is not None:
                raise self.error
            try:
                while True:
                    next(self.steps)
            except StopIteration as stop:
                out = stop.value
        assert out is not None
        return out


def summarize(gathered: list[dict]) -> dict:
    """Per phase group: the max over ranks of own time, and of own + wait."""
    groups: dict[str, list[float]] = {}
    for t in gathered:
        per: dict[str, list[float]] = {}
        for name, own, wait in t["phases"]:
            g = name.split(".")[-1] if "." in name else name
            per.setdefault(g, [0.0, 0.0])
            per[g][0] += own
            per[g][1] += own + wait
        for g, (own, wall) in per.items():
            cur = groups.setdefault(g, [0.0, 0.0])
            cur[0] = max(cur[0], own)
            cur[1] = max(cur[1], wall)
    return {g: {"own_max": round(a, 4), "wall_max": round(b, 4)} for g, (a, b) in groups.items()}


if __name__ == "__main__":
    if os.environ.get("GVS_BUILD_ONLY"):
        print(build())
    else:
        main()
