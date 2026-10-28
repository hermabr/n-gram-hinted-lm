"""On-the-clock exact-match hints (HINT_ONLINE=1): the rows hint_runtime.Hints reads from precomputed files, computed
in the run instead, in parallel with training, as record #367 computes its own (track_1_short/retrieval.py).

All of the index work runs in a worker process that rank 0 starts before the clock (`python -m online_hints`; its
own process, so that the indexes' hundreds of GB of mappings never stall a training process: page-table walks, as
Slurm's smaps reads, and TLB shootdowns hit every thread of the process that maps them), on Rust thread pools
(exact_match). It talks to the ranks through one shared-memory file (rows and a header of flags). From the start of
the clock (`start`) it computes:
  1. training rows, exact_match.OnlineRows over the run's own training stream (the port's loader plan, every rank's
     batches): strictly causal in the steps, a step's rows count only the occurrences whose tokens were all trained
     (as inputs or targets) in earlier steps. Incremental (OnlineRows.advance: a state kept across calls), in phases of
     steps (PHASES), each published to the shared rows, which every rank reads its batches' rows from; a rank whose
     step's rows are not ready yet waits (on the clock).
  2. then (or alongside, HINT_ONLINE_SERIAL=0) the validation index over the first training shards, knowing nothing
     of the validation tokens:
     - SlotIndex (#367's table: 6-gram keys with 10- and 20-token fingerprints) over the first HINT_ONLINE_SLOT_SHARDS
       shards (HINT_ONLINE_SHARDS also sets it), the hint rows of the levels 6, 10 and 20;
     - CountTable (exact counts at its levels) over the first HINT_ONLINE_COUNT_SHARDS shards: the hint rows of the
       other levels of the variant (or all of them without a SlotIndex), and with HINT_NGRAM=1 the count rows
       (tot, dist, cnt at COUNT_LEVELS; at 6, 10 and 20 the SlotIndex's counts where it counts more occurrences,
       HINT_ONLINE_SLOT_COUNTS=1, the default).
The wait for the builds is on the clock (`finish`, at the last step). After the clock stops, the worker reads the
validation tokens, looks up every validation position and shares the rows (the retrieval part of the validation
forward pass, off the clock like the LM's own), then frees its memory and exits.

Env (all but HINT_ONLINE optional): HINT_LEVELS (the variant, e.g. 2,3,4,5,6,10,20; default 6,10,20), HINT_BUCKETS
(default LEVELS), HINT_ONLINE_INDEX (a text file of training shard paths, one per line, in order: the index shards),
HINT_ONLINE_SLOT_SHARDS / HINT_ONLINE_SHARDS, HINT_ONLINE_COUNT_SHARDS, HINT_ONLINE_COUNT_LEVELS (default
COUNT_LEVELS with HINT_NGRAM=1, else the variant's levels the SlotIndex lacks), HINT_ONLINE_TRAIN_THREADS (48),
HINT_ONLINE_THREADS (48, each validation build), HINT_ONLINE_PHASES (step bounds after step 1; default
2,3,4,6,8,11,16,22,32,45,64, then every 16 steps), HINT_ONLINE_RESERVE (0: unpinned build threads; N: pinned to the
allowed CPUs but the first N), HINT_ONLINE_SERIAL (1: the validation builds start once the training rows are done; 0:
alongside them from the start; by default 1 for streams of up to 150M tokens, whose rows take ~11 s, else 0).

HINT_GPU_VAL=1: the worker computes the training rows only, no validation index
and no validation rows (HINT_ONLINE_SLOT_SHARDS / HINT_ONLINE_COUNT_SHARDS must be 0): the port builds the validation
rows itself on the GPUs (old_records/gpu_val_stats.py's InProcess) and reads them instead of val_span's."""

import ctypes
import os
import pickle
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.distributed as dist

from hint_runtime import VAL_TOKENS, VOCAB, WEIGHT_ONE, HintRows, Hints, ngram_enabled

LEVELS = (2, 3, 4, 5, 6, 10, 20)  # val_hints.py's: the default buckets
COUNT_LEVELS = (2, 3, 4, 5, 6, 8, 10, 12, 16, 20)  # old_records.py's (ALL_LEVELS)
SLOT_LEVELS = (6, 10, 20)  # the SlotIndex's levels (its MIN and LEVELS)
ENTRIES = 8
BOS = 50256
COUNT_BINS, PURITY_BINS = 7, 6  # exact_match.COUNTS and PURITIES, + 1 each
PHASES = (2, 3, 4, 6, 8, 11, 16, 22, 32, 45, 64)  # then every 16 steps
# header words (int64)
TRAIN_READY, VAL_READY, ERROR, BUILT, WORKER_READY, START, FINISH = range(7)
HEADER = 8


def _ints(name: str, default: str) -> tuple[int, ...]:
    value = os.environ.get(name, default)
    return tuple(int(v) for v in value.split(",") if v.strip())


def _memfd(name: str) -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.memfd_create(name.encode(), 1)  # MFD_CLOEXEC
    if fd < 0:
        raise OSError(ctypes.get_errno(), "memfd_create")
    return fd


def _layout(n: int, ngram: bool) -> list[tuple[str, type, tuple[int, ...]]]:
    """The shared file's arrays: the header, the training rows of the n stream positions, the validation rows."""
    layout = [("header", np.int64, (HEADER,)),
              ("cell", np.int16, (n,)), ("next", np.uint16, (n, ENTRIES)), ("weights", np.uint16, (n, ENTRIES)),
              ("val_cell", np.int16, (VAL_TOKENS,)), ("val_next", np.uint16, (VAL_TOKENS, ENTRIES)),
              ("val_weights", np.uint16, (VAL_TOKENS, ENTRIES))]
    if ngram:
        L = len(COUNT_LEVELS)
        layout += [(f"{p}{name}", dtype, (size, L)) for p, size in (("", n), ("val_", VAL_TOKENS))
                   for name, dtype in (("tot", np.uint32), ("dist", np.uint16), ("cnt", np.uint32))]
    return layout


class _Shared:
    """Named arrays in one shared-memory file at `path` (a memfd, reached as /proc/<pid>/fd/<fd>)."""

    def __init__(self, path: str, layout: list[tuple[str, type, tuple[int, ...]]]):
        offsets, total = {}, 0
        for name, dtype, shape in layout:
            offsets[name] = total
            total += -(-int(np.prod(shape)) * np.dtype(dtype).itemsize // 64) * 64
        self.bytes = total
        if os.path.getsize(path) == 0:
            os.truncate(path, total)
        self.map = np.memmap(path, dtype=np.uint8, mode="r+", shape=(total,))
        self.arrays = {}
        for name, dtype, shape in layout:
            n = int(np.prod(shape)) * np.dtype(dtype).itemsize
            self.arrays[name] = self.map[offsets[name] : offsets[name] + n].view(dtype).reshape(shape)

    def populate(self):
        """Fault the whole file in (MADV_POPULATE_WRITE)."""
        libc = ctypes.CDLL(None, use_errno=True)
        libc.madvise.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
        if libc.madvise(ctypes.c_void_p(self.map.ctypes.data), ctypes.c_size_t(self.bytes), 23) != 0:
            self.map[:] = 0

    def __getitem__(self, name: str) -> np.ndarray:
        return self.arrays[name]


def _weights(counts: np.ndarray) -> np.ndarray:
    """old_records.py's _weights: counts [..., ENTRIES] as weights summing to about WEIGHT_ONE per row."""
    total = counts.sum(-1, keepdims=True, dtype=np.float64)
    return np.round(counts / np.maximum(total, 1) * WEIGHT_ONE).astype(np.uint16)


def slot_rows(compact: np.ndarray, n: int, buckets: tuple[int, ...], entries: int = ENTRIES):
    """A SlotIndex query's compact rows (lib.rs's compact: n cells, then [capacity, 2] entries (position, token |
    count << 16)) as precomputed rows: cell (int16 [n], the length bucket of the SlotIndex's depth d re-based on
    `buckets`' index of level SLOT_LEVELS[d]), the top `entries` candidates by (count descending, token ascending)
    (uint16 [n, entries]) and their weights (uint16 [n, entries])."""
    cells = compact[:n].astype(np.int64)
    pairs = compact[n:].reshape(-1, 2)
    count = (pairs[:, 1].astype(np.int64) >> 16) & 0x7FFF
    keep = count > 0
    at, token, count = pairs[keep, 0].astype(np.int64), pairs[keep, 1].astype(np.int64) & 0xFFFF, count[keep]
    key = np.sort(at << 32 | (0xFFFF - count) << 16 | token)
    at, count, token = key >> 32, 0xFFFF - ((key >> 16) & 0xFFFF), key & 0xFFFF
    rank = np.arange(len(at)) - np.searchsorted(at, at, side="left")
    sel = rank < entries
    nxt = np.zeros((n, entries), np.uint16)
    cnt = np.zeros((n, entries), np.uint64)
    nxt[at[sel], rank[sel]] = token[sel]
    cnt[at[sel], rank[sel]] = count[sel]
    per = COUNT_BINS * PURITY_BINS
    remap = np.array([0] + [1 + buckets.index(k) for k in SLOT_LEVELS], np.int64)  # by 1 + depth
    depth1 = np.minimum(cells // per, len(SLOT_LEVELS))
    cell = np.where(cells > 0, remap[depth1] * per + cells % per, 0).astype(np.int16)
    if not np.array_equal(cell > 0, cnt[:, 0] > 0):
        raise RuntimeError("SlotIndex rows: positions with a cell and positions with candidates differ")
    return cell, nxt, _weights(cnt)


class _Config:
    """The run's online hint settings (the env), checked; the worker gets a copy."""

    def __init__(self):
        self.ngram = ngram_enabled()
        self.levels = _ints("HINT_LEVELS", "6,10,20")
        self.buckets = _ints("HINT_BUCKETS", ",".join(map(str, LEVELS)))
        if not set(self.levels) <= set(self.buckets):
            raise ValueError(f"HINT_LEVELS {self.levels} must be of HINT_BUCKETS {self.buckets}")
        self.count_levels = COUNT_LEVELS if self.ngram else ()
        self.slot_shards = int(os.environ.get("HINT_ONLINE_SLOT_SHARDS", os.environ.get("HINT_ONLINE_SHARDS", "0")))
        self.count_shards = int(os.environ.get("HINT_ONLINE_COUNT_SHARDS", "0"))
        self.gpu_val = os.environ.get("HINT_GPU_VAL", "0") == "1"  # training rows only (the module docstring)
        if self.gpu_val and (self.slot_shards or self.count_shards):
            raise ValueError("HINT_GPU_VAL=1: no CPU validation index (HINT_ONLINE_SLOT_SHARDS/COUNT_SHARDS 0)")
        slot_levels = tuple(k for k in self.levels if k in SLOT_LEVELS) if self.slot_shards else ()
        self.row_levels = tuple(k for k in self.levels if k not in slot_levels)  # the CountTable's part of the rows
        default = COUNT_LEVELS if self.ngram else self.row_levels
        self.table_levels = _ints("HINT_ONLINE_COUNT_LEVELS", ",".join(map(str, default)))
        if self.row_levels and not self.count_shards and not self.gpu_val:
            raise ValueError(f"levels {self.row_levels} need a CountTable (HINT_ONLINE_COUNT_SHARDS)")
        if not set(self.row_levels) <= set(self.table_levels):
            raise ValueError(f"the CountTable's levels {self.table_levels} lack {self.row_levels}")
        if self.ngram and not self.gpu_val and (not self.count_shards or tuple(self.table_levels) != COUNT_LEVELS):
            raise ValueError(f"HINT_NGRAM=1 needs a CountTable at {COUNT_LEVELS} (HINT_ONLINE_COUNT_SHARDS)")
        if self.slot_shards and not set(self.levels) & set(SLOT_LEVELS):
            raise ValueError("a SlotIndex for a variant without its levels")
        self.train_threads = int(os.environ.get("HINT_ONLINE_TRAIN_THREADS", "48"))
        self.threads = int(os.environ.get("HINT_ONLINE_THREADS", "48"))
        self.phases = _ints("HINT_ONLINE_PHASES", ",".join(map(str, PHASES)))
        self.serial = os.environ.get("HINT_ONLINE_SERIAL")  # None: by the stream's size (OnlineHints)
        self.reserve = int(os.environ.get("HINT_ONLINE_RESERVE", "0"))
        self.slot_counts = os.environ.get("HINT_ONLINE_SLOT_COUNTS", "1") == "1"
        index = os.environ.get("HINT_ONLINE_INDEX")
        self.index_files = open(index).read().split() if index else None


class OnlineHints(Hints):
    """hint_runtime.Hints's interface (train_span, val_span, cells, entries, levels, slots, ngram, count_levels,
    unigram), with rows computed in the run (the module docstring). Collective: every rank constructs it, after
    init_process_group and before the clock.

    `plan`: every training batch of every rank, (step, shard, start, length): inputs [start, start + length) of
    training shard `shard` (0-based in `train_files`, the port's sorted files) trained in step `step` (0-based, the
    clock's steps), for the `steps` steps of the run and the loader's batches after them (step >= `steps`: loaded, never
    trained, so blank, as the compile warmup's). `val_file`: the validation shard."""

    def __init__(self, plan: list[tuple[int, int, int, int]], steps: int, train_files: list[str], val_file: str,
                 device: torch.device | str = "cuda"):
        self.device = torch.device(device)
        self.fake = False
        self.rank = dist.get_rank()
        config = _Config()
        self.config = config
        self.ngram, self.levels, self.buckets = config.ngram, config.levels, config.buckets
        self.cells = (len(self.buckets) + 1) * COUNT_BINS * PURITY_BINS
        self.entries, self.slots = ENTRIES, 1
        self.count_levels = config.count_levels
        self.train_files, self.val_file = list(train_files), val_file
        self.unigram = np.full(VOCAB, 1.0 / VOCAB, np.float32)  # replaced at start (on the clock) with shard 0's

        # the stream: each used shard's tokens up to its last trained target, in shard order
        self.steps = steps
        self.step_of = {(shard, start): step for step, shard, start, _ in plan}
        plan = [p for p in plan if p[0] < steps]
        ends: dict[int, int] = {}
        for _, shard, start, length in plan:
            ends[shard] = max(ends.get(shard, 0), start + length + 1)
        self.base, bounds = {}, [0]
        for shard in range(max(ends) + 1):
            size = ends.get(shard, 0)
            if size:
                size = min(size, int(np.fromfile(self.train_files[shard], dtype=np.int32, count=3)[2]))
            self.base[shard] = bounds[-1]
            bounds.append(bounds[-1] + size)
        self.bounds, self.n = bounds, bounds[-1]
        if config.serial is None:  # the builds after the training rows if those are short (~11 s for 128M tokens)
            config.serial = "1" if self.n <= 150_000_000 else "0"
        config.serial = config.serial == "1"
        spans = np.array([(self.base[sh] + st, self.base[sh] + st + ln, step) for step, sh, st, ln in plan], np.int64)
        self.started = False
        self.waited = {TRAIN_READY: 0.0, VAL_READY: 0.0, BUILT: 0.0}
        if self.ngram:  # shard 0's tokens staged on the GPU, for the unigram init's count at the start of the clock
            shard = np.memmap(self.train_files[0], dtype=np.int16, mode="c", offset=1024)  # writable for from_numpy, not copied
            self.shard0 = torch.from_numpy(shard).to(self.device)

        # rank 0: the shared file and the worker, which attaches and creates everything before the clock
        path, fd = [None], None
        if self.rank == 0:
            fd = _memfd("oldrec-online")
            path = [f"/proc/{os.getpid()}/fd/{fd}"]
            shared = _Shared(path[0], _layout(self.n, self.ngram))
            shared.populate()
            tmp = os.environ.get("TMPDIR", "/tmp")
            self.log_path = os.path.join(tmp, f"online-hints-{os.getpid()}.log")
            job = os.path.join(tmp, f"online-hints-{os.getpid()}.pkl")
            with open(job, "wb") as f:
                pickle.dump(dict(path=path[0], n=self.n, bounds=bounds, spans=spans, steps=steps, config=config,
                                 train_files=self.train_files, val_file=val_file, log=self.log_path), f)
            here = os.path.dirname(os.path.abspath(__file__))
            env = dict(os.environ, PYTHONPATH=os.pathsep.join([here, *sys.path[1:]]))
            self.worker = subprocess.Popen([sys.executable, "-m", "online_hints", job], stdin=subprocess.DEVNULL, env=env)
        dist.broadcast_object_list(path, src=0)
        self.shared = shared if self.rank == 0 else _Shared(path[0], _layout(self.n, self.ngram))
        self.header = self.shared["header"]
        dist.barrier()
        # the worker's indexes created (before the clock; minutes on a busy node): polled, not in a collective, which
        # could time out
        while self.header[WORKER_READY] == 0:
            if self.header[ERROR] or (self.rank == 0 and self.worker.poll() is not None):
                raise RuntimeError("the online hint worker failed before the clock" +
                                   (f": {self._worker_log()}" if self.rank == 0 else ""))
            time.sleep(0.01)
        if self.rank == 0:
            os.close(fd)  # every rank and the worker map it
        dist.barrier()

    def _worker_log(self) -> str:
        try:
            return open(self.log_path).read().strip().replace("\n", "; ")
        except OSError:
            return "(no log)"

    # ---- on the clock
    def start(self, ngram_mix=None):
        """At the start of the clock: the worker starts; with the n-gram, every rank sets the unigram base's init to
        shard 0's token frequencies (as the precomputed runs' unigram.npy), counted on its GPU."""
        self.started = True
        if self.rank == 0:
            self.header[START] = 1
        if ngram_mix is not None:
            frequency = torch.bincount(self.shard0.int() & 0xFFFF, minlength=VOCAB)[:VOCAB].double() + 0.5
            unigram = (frequency / frequency.sum()).float()
            with torch.no_grad():
                ngram_mix.base.copy_(torch.log(unigram.clamp_min(1e-12)))
            del self.shard0

    def _wait(self, word: int, value: int):
        t = time.perf_counter()
        while self.header[word] < value:
            if self.header[ERROR]:
                raise RuntimeError("the online hint worker failed" + (f": {self._worker_log()}" if self.rank == 0 else ""))
            time.sleep(0.0002)
        self.waited[word] += time.perf_counter() - t

    def train(self, shard: int, offsets) -> HintRows:
        raise NotImplementedError("online hints read contiguous batches (train_span)")

    def train_span(self, shard: int, start: int, length: int) -> HintRows:
        step = self.step_of[shard, start] if self.started else self.steps
        if step >= self.steps:  # the compile warmup's batches, and the loader's batches after the last step: never trained
            return self._blank(length)
        self._wait(TRAIN_READY, step + 1)
        a = self.base[shard] + start
        return self._rows("", slice(a, a + length))

    def finish(self):
        """At the last step, on the clock: wait for the index builds; then the worker may read the validation tokens."""
        self._wait(BUILT, 1)
        if self.rank == 0:
            self.header[FINISH] = 1

    # ---- after the clock
    def val(self, offsets) -> HintRows:
        raise NotImplementedError("online hints read contiguous batches (val_span)")

    def val_span(self, start: int, length: int) -> HintRows:
        self._wait(VAL_READY, 1)
        if start + length > VAL_TOKENS:
            raise IndexError(f"positions up to {start + length} are past the {VAL_TOKENS} validation positions")
        return self._rows("val_", slice(start, start + length))

    def _rows(self, prefix: str, index: slice) -> HintRows:
        s = self.shared
        rows = self._upload(s[f"{prefix}cell"][index], s[f"{prefix}next"][index], s[f"{prefix}weights"][index])
        if self.ngram:
            rows.counts = self._upload_counts(s[f"{prefix}tot"][index], s[f"{prefix}dist"][index], s[f"{prefix}cnt"][index])
        return rows

    def _blank(self, n: int) -> HintRows:
        rows = self._upload(np.zeros(n, np.int16), np.zeros((n, ENTRIES), np.uint16), np.zeros((n, ENTRIES), np.uint16))
        if self.ngram:
            L = len(COUNT_LEVELS)
            rows.counts = self._upload_counts(np.zeros((n, L), np.uint32), np.zeros((n, L), np.uint16), np.zeros((n, L), np.uint32))
        return rows

    def summary(self) -> str:
        """The worker's timings, and every rank's total wait for training rows and for the builds (collective: call it
        on every rank, after the validation)."""
        waits = [None] * dist.get_world_size()
        dist.all_gather_object(waits, (round(self.waited[TRAIN_READY], 3), round(self.waited[BUILT], 3)))
        log = self._worker_log() if self.rank == 0 else ""
        return (f"online hints: {log}; waits for training rows per rank {[w[0] for w in waits]} s, "
                f"for the builds {[w[1] for w in waits]} s")


# ---- the worker process (rank 0's child)

class _Worker:
    def __init__(self, job: dict):
        import exact_match

        self.job, c = job, job["config"]
        self.c = c
        self.log_file = open(job["log"], "a", buffering=1)
        self.shared = _Shared(job["path"], _layout(job["n"], c.ngram))
        self.header = self.shared["header"]
        self.parent = os.getppid()
        cpus = sorted(os.sched_getaffinity(0))[c.reserve:] if c.reserve else []
        t = time.perf_counter()
        files = c.index_files or job["train_files"]
        self.slot = self.table = None
        if c.slot_shards:
            fd = _memfd("oldrec-slot")
            self.slot = exact_match.SlotIndex.create(f"/proc/{os.getpid()}/fd/{fd}", self._files(files, c.slot_shards),
                                                     c.threads, cpus, sorted(os.sched_getaffinity(0)))
            os.close(fd)
        if c.count_shards:
            self.table = exact_match.CountTable.create(self._files(files, c.count_shards), list(c.table_levels), c.threads, cpus)
        t1 = time.perf_counter()
        bounds = job["bounds"]
        stream = np.concatenate([np.memmap(job["train_files"][i], dtype=np.uint16, mode="r", offset=1024)[: bounds[i + 1] - bounds[i]]
                                 for i in range(len(bounds) - 1)])
        self.rows = exact_match.OnlineRows(stream, bounds, job["spans"], c.train_threads, cpus)
        del stream
        self.rows.prepare(list(c.levels), list(c.buckets), ENTRIES, list(c.count_levels))
        self.log(f"before the clock: indexes created in {t1 - t:.1f} s, training rows prepared in {time.perf_counter() - t1:.1f} s")

    @staticmethod
    def _files(files: list[str], shards: int) -> list[str]:
        if len(files) < shards:
            raise ValueError(f"the index needs {shards} training shards, {len(files)} listed")
        return files[:shards]

    def log(self, line: str):
        self.log_file.write(line + "\n")

    def wait(self, word: int):
        while self.header[word] == 0:
            if os.getppid() != self.parent:
                raise SystemExit("rank 0 is gone")
            time.sleep(0.0002)

    def run(self):
        self.header[WORKER_READY] = 1
        self.wait(START)
        self.t0 = time.perf_counter()
        threads = [threading.Thread(target=self.guard, args=(self.train_rows,))]
        if not self.c.serial:
            threads.append(threading.Thread(target=self.guard, args=(self.builds,)))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if self.header[ERROR]:
            return
        self.header[BUILT] = 1
        self.log(f"builds ready {time.perf_counter() - self.t0:.2f} s after the clock started")
        self.wait(FINISH)
        self.guard(self.val_rows)

    def guard(self, work):
        try:
            work()
        except BaseException:
            self.log(traceback.format_exc())
            self.header[ERROR] = 1
            raise

    def train_rows(self):
        """The training rows, phase by phase (step 0's are blank: nothing was trained before it); then (serial) the
        builds."""
        c, s, steps = self.c, self.shared, self.job["steps"]
        self.header[TRAIN_READY] = 1
        phases = sorted({p for p in (*c.phases, *range(c.phases[-1] + 16, steps, 16)) if 1 < p < steps} | {steps})
        late = []
        for hi in phases:
            counts = (s["tot"], s["dist"], s["cnt"]) if c.ngram else (None, None, None)
            self.rows.advance(hi, list(c.levels), list(c.buckets), ENTRIES, list(c.count_levels),
                              s["cell"], s["next"], s["weights"], *counts)
            self.header[TRAIN_READY] = hi
            late.append(f"{hi}:{time.perf_counter() - self.t0:.2f}")
        self.log(f"training rows of steps < step:ready s {' '.join(late)}")
        # (self.rows is kept: freeing its hundreds of GB took ~15 s, which would delay the builds; the process exits)
        if c.serial:
            self.builds()

    def builds(self):
        for name, index in (("SlotIndex", self.slot), ("CountTable", self.table)):
            if index is not None:
                t = time.perf_counter()
                index.build()
                self.log(f"{name} built in {time.perf_counter() - t:.2f} s, ready {time.perf_counter() - self.t0:.2f} s "
                         f"after the clock started")

    def val_rows(self):
        """After the clock: the validation tokens read, every validation position's rows into the shared arrays."""
        t = time.perf_counter()
        c, s = self.c, self.shared
        if c.gpu_val:  # the port's GPU build has them
            return
        tokens = np.array(np.memmap(self.job["val_file"], dtype=np.uint16, mode="r", offset=1024)[: VAL_TOKENS + 1])
        if self.slot is not None:
            # in pieces that start at a BOS (a segment start either way, so the rows are the same), queried in parallel
            bos = np.flatnonzero(tokens[:VAL_TOKENS] == BOS)
            cuts = sorted({0, VAL_TOKENS, *(int(bos[min(np.searchsorted(bos, i * VAL_TOKENS // 32), len(bos) - 1)])
                                            for i in range(1, 32))})

            def piece(a: int, b: int):
                compact, lost = self.slot.query(tokens[a:b], 32 * (b - a))
                if lost:
                    raise RuntimeError(f"{lost} SlotIndex entries past the capacity")
                s["val_cell"][a:b], s["val_next"][a:b], s["val_weights"][a:b] = slot_rows(compact, b - a, c.buckets)

            with ThreadPoolExecutor(16) as pool:
                list(pool.map(piece, cuts[:-1], cuts[1:]))
            self.log(f"validation SlotIndex rows {time.perf_counter() - t:.2f} s, {(s['val_cell'] > 0).mean():.4f} hinted")
        if self.table is not None and c.row_levels:
            t1 = time.perf_counter()
            cell, nxt, weights, _ = self.table.query_rows(tokens, list(c.row_levels), list(c.buckets), ENTRIES)
            empty = s["val_cell"] == 0
            s["val_cell"][empty], s["val_next"][empty], s["val_weights"][empty] = cell[empty], nxt[empty], weights[empty]
            self.log(f"validation CountTable rows {time.perf_counter() - t1:.2f} s, {(s['val_cell'] > 0).mean():.4f} hinted in all")
        if c.ngram:
            t1 = time.perf_counter()
            tot, dist_, cnt = self.table.query_counts(tokens, 1024)
            if self.slot is not None and c.slot_counts:
                # the SlotIndex's counts at its levels (over far more shards; approximate as its rows are) where they
                # count more occurrences than the CountTable's (a choice by the context's counts only)
                st, sd, sc = (a.reshape(VAL_TOKENS, len(SLOT_LEVELS)) for a in self.slot.query_counts(tokens))
                for i, k in enumerate(SLOT_LEVELS):
                    j = COUNT_LEVELS.index(k)
                    use = st[:, i] > tot[:, j]
                    tot[use, j], dist_[use, j], cnt[use, j] = st[use, i], sd[use, i], sc[use, i]
            s["val_tot"][:], s["val_dist"][:], s["val_cnt"][:] = tot, dist_, cnt
            self.log(f"validation count rows {time.perf_counter() - t1:.2f} s" +
                     (" (levels 6/10/20 from the SlotIndex where it counts more)" if self.slot is not None and c.slot_counts else ""))
        self.log(f"validation rows {time.perf_counter() - t:.2f} s (off the clock)")
        self.header[VAL_READY] = 1


if __name__ == "__main__":
    with open(sys.argv[1], "rb") as f:
        job = pickle.load(f)
    worker = None
    try:
        worker = _Worker(job)
    except BaseException:
        with open(job["log"], "a") as f:
            f.write(traceback.format_exc())
        _Shared(job["path"], _layout(job["n"], job["config"].ngram))["header"][ERROR] = 1
        raise
    worker.run()
    os._exit(0)  # the indexes' memory goes with the process (no unmapping by hand)
