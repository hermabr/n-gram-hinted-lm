"""Exact-match retrieval: each position sees the next tokens that followed its context before.

Training rows come from a StreamIndex, strictly causal in the token stream: rank 0 resolves every step's rows from
earlier steps' documents, ahead of training, into shared memory, on a pinned pool of cores every rank sets aside.
Validation rows come from the exact counts over the count corpus (counts/build.py).

A position's match has a cell (length bucket, total count bin, top token share bin) and candidates: each next token
seen, with its count. A batch's rows are one int32 vector: the T cells, then [capacity, 2] entries (position,
token | count << 16), padded with (i % T, 0), which carry no weight.
"""
import glob
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import torch
import torch.distributed as dist
from exact_match import StreamIndex, memfd

BUILD_THREADS = 48  # rank 0's index build (the training rows), on cores every rank sets aside
# Candidate entries per position of a batch's rows: training steps hold 0.2 per position on average, at most 0.7;
# validation positions at most 8 (counts/build.py's ROW_K).
TRAIN_ENTRIES, VAL_ENTRIES = 2, 8


def blank(n: int, entries: int, device=None) -> torch.Tensor:
    """Rows of `n` positions without a match, with `entries` (padding) entries."""
    out = torch.zeros(n + 2 * entries, dtype=torch.int32, device=device)
    out[n::2] = torch.arange(entries, device=device) % n
    return out


@dataclass(frozen=True)
class CpuPlan:
    main: set[int]          # the main thread's core, its own from step 1
    rest: set[int]          # every other thread of this rank from step 1 (NCCL, gloo, loaders, the online index)
    build: list[int]        # this rank's share of the cores rank 0's index build runs on


def pin_cpus(local_rank: int, world_size: int) -> CpuPlan:
    """Each rank gets an equal share of the physical cores local to its GPU (within the allowed set: Slurm hands out
    arbitrary CPU ids), and is pinned to it before the process group starts, so every thread inherits it. Its last
    cores run rank 0's index build, one thread per core. From step 1 the main thread has the first core to itself."""
    cpulist = lambda path: {c for r in open(path).read().split(",") for a, _, b in [r.partition("-")] for c in range(int(a), int(b or a) + 1)}
    allowed = os.sched_getaffinity(0)
    cores = lambda cpus: sorted({frozenset(cpulist(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list")) & allowed
                                 for c in cpus}, key=min)
    local = [frozenset(cpulist("/sys/bus/pci/devices/{0.pci_domain_id:04x}:{0.pci_bus_id:02x}:{0.pci_device_id:02x}.0/local_cpulist"
                               .format(torch.cuda.get_device_properties(i)))) & allowed for i in range(torch.cuda.device_count())]
    if min(len(cores(cpus)) / local.count(cpus) for cpus in local) < 2:  # too few allowed cores near some GPU: ignore locality
        local = [frozenset(allowed)] * len(local)
    near = cores(local[local_rank])
    k, n = local[:local_rank].count(local[local_rank]), local.count(local[local_rank])
    mine = near[k * len(near) // n:(k + 1) * len(near) // n]
    spare = -(-BUILD_THREADS // world_size)
    build = [min(c) for c in mine[-spare:]] if len(mine) - spare >= 2 else []
    mine = mine[:len(mine) - len(build)]
    os.sched_setaffinity(0, set().union(*mine))
    return CpuPlan(main=set(mine[0]), rest=set().union(*mine[1:]) or set(mine[0]), build=build)


def pin_threads(plan: CpuPlan):
    """At step 1: the main thread to its own core, every other thread but the Rust pools (pinned already: the index's
    and the canonical mask's) to the rest."""
    for tid in map(int, os.listdir("/proc/self/task")):
        try:
            if not open(f"/proc/self/task/{tid}/comm").read().startswith(("streamindex-", "canonmask-")):
                os.sched_setaffinity(tid, plan.main if tid == threading.get_native_id() else plan.rest)
        except (ProcessLookupError, FileNotFoundError):
            pass


def build_cpus(plan: CpuPlan, group) -> list[int]:
    """The cores of rank 0's build pool: every rank's `plan.build`."""
    cpus = [None] * dist.get_world_size(group)
    dist.all_gather_object(cpus, plan.build, group=group)
    return sum(cpus, [])[:BUILD_THREADS]


def shared(group, create, attach):
    """An index in shared memory: rank 0 creates it in a new memfd, reached as /proc/<pid>/fd/<fd> (`create(path)`),
    then the other ranks open that path (`attach(path)`), and rank 0 closes the descriptor once every rank has it open.
    No mount limits its size, and the memory is freed once no rank maps it or holds it open, also after a crash."""
    rank = dist.get_rank(group)
    fd = memfd("exact-match") if rank == 0 else None
    path = [f"/proc/{os.getpid()}/fd/{fd}"]
    dist.broadcast_object_list(path, src=0, group=group)
    index = create(path[0]) if rank == 0 else None
    dist.barrier(group)
    if rank != 0:
        index = attach(path[0])
    dist.barrier(group)
    if rank == 0:
        os.close(fd)  # the path stops resolving; the mappings keep the memory
    return index


class OnlineCache:
    """Training rows, resolved on the clock ahead of training by rank 0 for every rank, and copied by each rank into a
    pinned ring allocated before the clock (a cudaHostAlloc on the clock holds the driver lock). A step's job is
    submitted when the loader fetches its batch, at most `lookahead` steps before the step trains, so a ring deeper
    than that never rewrites a slot whose step is still to train; a slot is rewritten only after its previous H2D
    retired (one event per slot)."""
    def __init__(self, files, schedule, rank, world, device, lookahead, group, build_cpus):
        files = sorted(glob.glob(files))
        entries = [TRAIN_ENTRIES * n for n, _ in schedule]
        self.index = shared(group, lambda path: StreamIndex.create(path, files, schedule, world, BUILD_THREADS, build_cpus, entries),
                            lambda path: StreamIndex.attach(path, files, schedule, rank, world, entries))
        self.rank = rank
        self.builder = ThreadPoolExecutor(max_workers=1)
        self.worker = ThreadPoolExecutor(max_workers=1, initializer=torch.cuda.set_device, initargs=(device,))
        self.ring = torch.zeros((lookahead + 2, max((1 + 2 * TRAIN_ENTRIES) * n for n, _ in schedule)), dtype=torch.int32, pin_memory=True)
        self.uploaded = [torch.cuda.Event() for _ in range(len(self.ring))]
        self.device = device
        self.jobs = {}  # step -> its rows' future, from fetch until the step trains
        self.step0 = None

    def start(self):
        """Rank 0 builds every rank's rows (a failure fails every rank's rows())."""
        if self.rank == 0:
            self.builder.submit(self.index.build)

    def submit(self, step, batch):
        """ScheduledBatches' fetch hook: `batch.docs` is the loader's (shard size, every rank's document starts, ends)."""
        self.jobs[step] = self.worker.submit(self._rows, step, *batch.docs)

    def _rows(self, step, size, starts, ends):
        rows = self.index.rows(step, size, starts, ends)
        i = step % len(self.ring)
        self.uploaded[i].synchronize()
        slot = self.ring[i, :len(rows)]
        slot.numpy()[:] = rows
        return slot

    def rows(self, step, n):
        """This step's rows on the device. Blank during warmup (no jobs) and at step 0: nothing was trained before it, so
        its rows are only checked to be blank, once step 1's are taken."""
        job = self.jobs.pop(step, None)
        if job is None or step == 0:
            self.step0 = (job, n)
            return blank(n, TRAIN_ENTRIES * n, self.device)
        if step == 1:
            job0, n0 = self.step0
            assert job0.result().equal(blank(n0, TRAIN_ENTRIES * n0)), "step 0 has retrieval matches"
        rows = job.result().to(self.device, non_blocking=True)  # the job also checks the loader's documents against the plan
        self.uploaded[step % len(self.ring)].record()
        return rows
