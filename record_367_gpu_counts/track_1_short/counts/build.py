"""Exact n-gram counts of the query contexts (the validation positions, and the fit positions of the count mixture) over
the count corpus, on the GPUs, on the clock: each rank holds 1/world of the corpus' shards, counts the (context, next
token) occurrences of every query context at each of LEVELS, and the ranks merge the counts exactly.

  - construct (before the clock): this rank's shards (round-robin over the ranks) copied from the /dev/shm staging into
    anonymous memory in transparent huge pages and pinned in place (cudaHostRegister); the kernels' library loaded;
  - prepare() (collective, before the clock): the all-to-all's NCCL buffers and the build's allocator blocks warmed;
  - start_load() at the clock's start, then pace() after every training step: the shards' host-to-device copies, an
    equal share per step on a low-priority stream, so the training's own copies never queue behind much of them;
  - counts(x, tgt, avail) (collective, after the last step): per level the query index (the distinct valid contexts in
    an open-addressing table, a representative position each), every shard scanned through its levels (a level looks
    up only the positions whose context matched at the previous one; hashed levels are verified token by token), then
    per level the local aggregation, the all-to-all, the owner's sort and sum and the per-position outputs, all-reduced
    to every rank. Also the record's validation rows (retrieval.py) from the levels ROW_LEVELS.

A query position's k-context is the k tokens ending at it, valid iff they lie in its segment (`avail`: tokens since
its segment's start, inclusive); an occurrence is any k tokens within one shard and the token after them. Level k is
counted over the first LEVEL_SHARDS[k] shards of the corpus (a shard skipped by the previous level is probed whole)."""
import ctypes
import fcntl
import hashlib
import mmap
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from track_1_short.data import HEADER_BYTES

HERE = Path(__file__).resolve().parent
BOS = 50256
GAP = 32  # 0xFFFF (not a GPT-2 token) before each shard and after the last: >= the longest context
LEVELS = (2, 3, 4, 5, 6, 10, 20)
# The short levels gain little from more tokens and cost the most to scan.
LEVEL_SHARDS = {2: 103, 3: 370, 4: 370, 5: 370}
# Distinct (context, next token) pairs of the validation contexts in n training tokens: u * (n / 4e8) ** beta (fit to
# 4 and 1028 shards); the fit positions add up to ~20% contexts, hence SAFETY.
PAIRS = {2: (82.17e6, 0.620), 3: (62.30e6, 0.667), 4: (26.39e6, 0.694), 5: (8.99e6, 0.706), 6: (3.17e6, 0.705),
         10: (0.223e6, 0.604), 20: (0.102e6, 0.581)}
SAFETY = 1.6
LOAD = 0.7
MAX_IDS = (1 << 25) - 4096  # context ids are slots of the query table, 25 bits of a pair key
SIGN = -(1 << 63)
CHUNK_BYTES = 8 << 20  # the paced copies' size
# The record's retrieval rows (exact_match/src/lib.rs's row): a match's length bucket is 1 + how many of ROW_LEVELS[1:]
# its deepest level reaches, its cell (bucket, total count bin, top token share bin); its candidates the ROW_K most
# frequent next tokens of its deepest level's context (count descending, then token).
ROW_LEVELS = (6, 10, 20)
ROW_K = 8
ROW_COUNTS = (2, 3, 5, 9, 17, 33)
ROW_PURITIES = (0.3, 0.5, 0.7, 0.85, 0.95)


def load_kernels() -> ctypes.CDLL:
    """kernels.cu built with nvcc for this GPU, once per source version (one rank builds, the others wait)."""
    arch = "".join(map(str, torch.cuda.get_device_capability()))
    digest = hashlib.sha256(HERE.joinpath("kernels.cu").read_bytes() + arch.encode()).hexdigest()[:12]
    out = HERE / ".build" / f"counts_{digest}.so"
    out.parent.mkdir(exist_ok=True)
    with open(out.parent / "lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not out.exists():
            nvcc = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"), "bin/nvcc")
            subprocess.run([nvcc, "-O3", "-std=c++17", "-shared", "-Xcompiler", "-fPIC", f"-gencode=arch=compute_{arch},code=sm_{arch}",
                            "-o", f"{out}.tmp", HERE / "kernels.cu"], check=True)
            Path(f"{out}.tmp").rename(out)
    return ctypes.CDLL(str(out))


class Lib:
    """The kernels through ctypes: tensors as their data pointers, on the current stream; a nonzero return raises."""

    def __init__(self):
        self.so = load_kernels()

    def __call__(self, name: str, *args) -> None:
        conv = [ctypes.c_void_p(a.data_ptr()) if isinstance(a, torch.Tensor) else ctypes.c_void_p(0) if a is None
                else ctypes.c_int64(a) if isinstance(a, int) else a for a in args]
        fn = getattr(self.so, name)
        fn.restype = ctypes.c_int
        if rc := fn(*conv, ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)):
            raise RuntimeError(f"{name} returned {rc}")

    def cub(self, name: str, *args) -> None:
        size = ctypes.c_size_t(0)
        ptr = ctypes.c_void_p(ctypes.addressof(size))
        self(name, None, ptr, *args)
        self(name, torch.empty(max(size.value, 1), dtype=torch.uint8, device="cuda"), ptr, *args)


def i32(x: int) -> ctypes.c_int:
    return ctypes.c_int(x)


class QueryIndex:
    """One level's query contexts: per position its context id (-1 where invalid), and the lookup table (8-byte slots,
    a context's id is its slot, canonical: the same on every rank) with each slot's representative position."""

    def __init__(self, lib: Lib, x: torch.Tensor, avail: torch.Tensor, k: int):
        n = x.numel()
        keys = torch.empty(n, dtype=torch.int64, device="cuda")
        lib("val_keys", x, avail, n, i32(k), keys)
        pos = (keys != -1).nonzero().squeeze(1)
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
        n_ctx = int(ukeys.numel())
        if bool((ukeys == -2).any()):
            raise RuntimeError(f"k={k}: a query key collides with the empty marker")
        bad = torch.zeros(1, dtype=torch.int64, device="cuda")
        lib("val_verify", x, spos, gid, rep, int(spos.numel()), i32(k), bad)
        if int(bad.item()):
            raise RuntimeError(f"k={k}: {int(bad.item())} query positions collide with another context's key")
        self.cap = min(2 * n_ctx + 1024, MAX_IDS)
        home = torch.empty(n_ctx, dtype=torch.int64, device="cuda")
        lib("val_homes", ukeys, n_ctx, ctypes.c_uint64(self.cap), home)
        at = torch.arange(n_ctx, device="cuda", dtype=torch.int64)
        slot_of = torch.empty_like(home)
        lib.cub("inclusive_max", home - at, slot_of, n_ctx)
        slot_of += at
        size = max(self.cap, int(slot_of[-1].item()) + 1 if n_ctx else 0) + 1
        if size >= MAX_IDS + 4096:
            raise RuntimeError(f"k={k}: {n_ctx} contexts do not fit 25-bit ids")
        self.table = torch.full((size,), -1, dtype=torch.int64, device="cuda")
        self.table[slot_of] = ukeys
        self.rep = torch.zeros(size, dtype=torch.int32, device="cuda")
        self.rep[slot_of] = rep.to(torch.int32)
        self.ids = size
        self.ctx_of = torch.full((n,), -1, dtype=torch.int32, device="cuda")
        self.ctx_of[spos] = slot_of[gid].to(torch.int32)


def queries(val: np.ndarray, blocks: list[np.ndarray], device, lib: Lib) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The query tokens, targets and `avail` of the validation stream `val` (its positions' inputs, then one more token:
    the last target) and then each fit block (likewise), each block a segment of its own; a BOS starts a segment."""
    parts = [val, *blocks]
    x = np.concatenate([p[:-1] for p in parts])
    tgt = np.concatenate([p[1:] for p in parts])
    first = np.zeros(len(x), dtype=bool)
    first[np.cumsum([0, *[len(p) - 1 for p in parts[:-1]]])] = True
    x_t = torch.from_numpy(x.view(np.int16)).to(device)
    idx = torch.arange(len(x), device=device)
    start = (x_t.to(torch.int32) & 0xFFFF) == BOS
    start |= torch.from_numpy(first).to(device)
    seg = torch.empty_like(idx)
    lib.cub("inclusive_max", torch.where(start, idx, torch.zeros_like(idx)), seg, len(x))  # torch.cummax: ~50x slower
    return x_t, torch.from_numpy(tgt.view(np.int16)).to(device), (idx - seg + 1).to(torch.int32)


class CountBuild:
    def __init__(self, shard_paths: list[str], rank: int, world: int):
        self.rank, self.world = rank, world
        self.level_shards = {k: min(len(shard_paths), LEVEL_SHARDS.get(k, len(shard_paths))) for k in LEVELS}
        self.mine = [(g, p) for g, p in enumerate(shard_paths) if g % world == rank]
        self.sizes = []
        for _, path in self.mine:
            header = np.fromfile(path, dtype="<i4", count=256)
            assert header[0] == 20240520 and header[1] == 1, f"invalid shard {path}"
            self.sizes.append(int(header[2]))
        self.starts, at = [], GAP
        for n in self.sizes:
            self.starts.append(at)
            at = (at + n + GAP + 31) // 32 * 32
        self.L = at
        self.tok = torch.full((self.L,), -1, dtype=torch.int16, device="cuda")
        self.rt = ctypes.CDLL(str(Path(torch.__file__).parent.parent / "nvidia/cuda_runtime/lib/libcudart.so.12"))
        self.rt.cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
        self.rt.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
        # One huge-page mapping per shard: few page-table entries, since Slurm's periodic smaps walk stalls a process
        # whose mappings have many (4K-page /dev/shm mappings stalled training for seconds).
        self.maps = []
        t0 = time.perf_counter()
        for (_, path), n in zip(self.mine, self.sizes):
            m = mmap.mmap(-1, -(-(HEADER_BYTES + 2 * n) // (2 << 20)) * (2 << 20), mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
            m.madvise(mmap.MADV_HUGEPAGE)
            m.madvise(mmap.MADV_DONTFORK)
            view, at = memoryview(m), 0
            with open(path, "rb", buffering=0) as f:
                while at < HEADER_BYTES + 2 * n:
                    got = f.readinto(view[at:HEADER_BYTES + 2 * n])
                    assert got, f"short read of {path}"
                    at += got
            del view
            ptr = ctypes.addressof(ctypes.c_char.from_buffer(m))
            if err := self.rt.cudaHostRegister(ptr, len(m), 0):
                raise RuntimeError(f"cudaHostRegister of {path}: error {err}")
            self.maps.append((m, ptr))
        self.pin_s = time.perf_counter() - t0
        self.stream = torch.cuda.Stream(priority=0)  # the lowest priority
        self.events: list[torch.cuda.Event] = []  # one per shard, once its copy is done
        self.lib = Lib()
        self.timing: dict[str, float] = {}

    def caps(self) -> dict[int, int]:
        """Each level's pair table slots: the estimated pairs of this rank's tokens of the level, at LOAD."""
        caps = {}
        for k in LEVELS:
            n_k = sum(n for (g, _), n in zip(self.mine, self.sizes) if g < self.level_shards[k])
            u, beta = PAIRS[k]
            caps[k] = max(1 << 20, int(min(SAFETY * u * (n_k / 4e8) ** beta, n_k) / LOAD))
        return caps

    def prepare(self) -> None:
        """Before the clock: the all-to-all's connections and NCCL buffers (small and 64 MB messages), and the caching
        allocator's blocks for the build (allocated once and freed, so the build's allocations make no cudaMalloc)."""
        w = self.world
        for size in (256, 1 << 23):
            a = torch.zeros(w * size, dtype=torch.int64, device="cuda")
            b = torch.empty_like(a)
            dist.all_to_all_single(b, a)
            dist.all_to_all_single(b, a, [size] * w, [size] * w)
            c = torch.zeros(1 << 22, dtype=torch.int32, device="cuda")
            dist.all_reduce(c)
            del a, b, c
        # the pair tables' bytes and a third more for the merges' buffers
        warm = torch.empty(int(1.33 * sum(9 * cap for cap in self.caps().values())), dtype=torch.uint8, device="cuda")
        del warm
        torch.cuda.synchronize()

    def start_load(self, steps: int) -> None:
        """At the clock's start: the copies split into CHUNK_BYTES pieces, the first of `steps` equal shares enqueued."""
        self.load_start = torch.cuda.Event(enable_timing=True)
        self.load_start.record(self.stream)
        self.pending = []  # (dst, src, bytes, whether it ends a shard), last first
        for (m, ptr), st, n in zip(self.maps, self.starts, self.sizes):
            dst, src, nbytes = self.tok.data_ptr() + 2 * st, ptr + HEADER_BYTES, 2 * n
            for a in range(0, nbytes, CHUNK_BYTES):
                c = min(nbytes, a + CHUNK_BYTES) - a
                self.pending.append((dst + a, src + a, c, a + c == nbytes))
        self.pending.reverse()
        self.per_step = -(-len(self.pending) // steps)
        self.pace()

    def pace(self) -> None:
        """After each training step: the next share of the copies."""
        s = ctypes.c_void_p(self.stream.cuda_stream)
        for _ in range(min(self.per_step, len(self.pending))):
            dst, src, c, last = self.pending.pop()
            if err := self.rt.cudaMemcpyAsync(ctypes.c_void_p(dst), ctypes.c_void_p(src), c, 1, s):
                raise RuntimeError(f"cudaMemcpyAsync: error {err}")
            if last:
                ev = torch.cuda.Event(enable_timing=True)
                ev.record(self.stream)
                self.events.append(ev)

    def load_ms(self) -> float:
        """The copies' time on the device from start_load (once they are done)."""
        return self.load_start.elapsed_time(self.events[-1])

    def counts(self, x: torch.Tensor, tgt: torch.Tensor, avail: torch.Tensor) -> torch.Tensor:
        """int32 [3, len(LEVELS), len(x)]: per query position and level its context's total occurrences (uint32 bits),
        distinct next tokens (at most 65535) and its target's count, on every rank; also the rows (batch_rows).
        Collective."""
        lib, world, rank, timing = self.lib, self.world, self.rank, self.timing

        def lap(name: str) -> None:
            torch.cuda.current_stream().synchronize()  # not the device: the copies' stream may still be loading
            now = time.perf_counter()
            timing[name] = timing.get(name, 0.0) + now - lap.t
            lap.t = now
        lap.t = time.perf_counter()

        N = x.numel()
        index = {k: QueryIndex(lib, x, avail, k) for k in LEVELS}
        lap("index")
        grid = torch.cuda.get_device_properties(x.device).multi_processor_count * 32
        tables = {k: (torch.full((cap,), -1, dtype=torch.int64, device="cuda"),
                      torch.zeros((cap + 3) // 4, dtype=torch.int32, device="cuda"), cap,
                      torch.zeros(1, dtype=torch.int32, device="cuda")) for k, cap in self.caps().items()}
        bms = [torch.zeros(self.L // 32, dtype=torch.int32, device="cuda") for _ in range(2)]
        for i, ((g, _), st, n) in enumerate(zip(self.mine, self.starts, self.sizes)):
            torch.cuda.current_stream().wait_event(self.events[i])
            scanned = [k for k in LEVELS if g < self.level_shards[k]]
            for li, k in enumerate(scanned):
                vi = index[k]
                pk, pc, cap, flags = tables[k]
                bm_in = None if li == 0 else bms[(li - 1) % 2][st // 32:]
                lib("scan_level", self.tok[st:], (n + 31) // 32 * 32, i32(k), bm_in, bms[li % 2][st // 32:], vi.table,
                    ctypes.c_uint64(vi.cap), vi.rep, x, pk, pc, ctypes.c_uint64(cap), flags, i32(grid), i32(int(li > 0 and k > 5)))
        lap("scan")
        if bad := [k for k in LEVELS if int(tables[k][3][0].item())]:
            raise RuntimeError(f"the pair tables of levels {bad} overflowed")
        del bms
        out = torch.zeros((3, len(LEVELS), N), dtype=torch.int32, device="cuda")
        for li, k in enumerate(LEVELS):
            vi = index.pop(k)
            pk, pc, cap, _ = tables.pop(k)
            bounds_l = [vi.ids * o // world for o in range(world + 1)]
            bounds = torch.tensor(bounds_l, dtype=torch.int64, device="cuda")
            lo, hi = bounds_l[rank], bounds_l[rank + 1]
            counts = torch.zeros(world, dtype=torch.int64, device="cuda")
            lib("part_count", pk, ctypes.c_uint64(cap), bounds, i32(world), counts)
            send_counts = counts.cpu().tolist()
            cursor = torch.zeros(world, dtype=torch.int64, device="cuda")
            cursor[1:] = torch.cumsum(counts, 0)[:-1]
            send_k = torch.empty(sum(send_counts), dtype=torch.int64, device="cuda")
            send_c = torch.empty(sum(send_counts), dtype=torch.int32, device="cuda")
            lib("part_scatter", pk, pc, ctypes.c_uint64(cap), bounds, i32(world), cursor, send_k, send_c)
            del pk, pc
            lap("local")
            recv_counts = torch.empty(world, dtype=torch.int64, device="cuda")
            dist.all_to_all_single(recv_counts, counts)
            rc = recv_counts.cpu().tolist()
            recv_k = torch.empty(sum(rc), dtype=torch.int64, device="cuda")
            recv_c = torch.empty(sum(rc), dtype=torch.int32, device="cuda")
            dist.all_to_all_single(recv_k, send_k, rc, send_counts)
            dist.all_to_all_single(recv_c, send_c, rc, send_counts)
            del send_k, send_c
            lap("comm")
            n_own = hi - lo
            ctx_bits = max(1, (max(n_own, 2) - 1).bit_length())
            recv_k -= lo << 16
            n_recv = recv_k.numel()
            sk, sc = torch.empty_like(recv_k), torch.empty_like(recv_c)
            lib.cub("sort_pairs", recv_k, sk, recv_c, sc, n_recv, i32(0), i32(16 + ctx_bits))
            del recv_k, recv_c
            uk, agg = torch.empty_like(sk), torch.empty_like(sc)
            nruns = torch.zeros(1, dtype=torch.int64, device="cuda")
            lib.cub("reduce_by_key", sk, uk, sc, agg, nruns, n_recv)
            U = int(nruns.item())
            del sk, sc
            uk, agg = uk[:U], agg[:U]
            lap("merge")
            start = torch.zeros(n_own, dtype=torch.int64, device="cuda")
            end = torch.zeros(n_own, dtype=torch.int64, device="cuda")
            lib("seg_bounds", uk, U, start, end)
            tot = torch.empty(n_own, dtype=torch.int64, device="cuda")
            lib("ctx_tot", start, end, agg, n_own, tot)
            lib("pos_outputs", vi.ctx_of, N, lo, hi, tgt, uk, agg, start, end, tot, out[0, li], out[1, li], out[2, li])
            if k in ROW_LEVELS:
                if k == ROW_LEVELS[0]:
                    row_entries = torch.zeros((len(ROW_LEVELS), N, ROW_K), dtype=torch.int32, device="cuda")
                    row_top = torch.zeros((len(ROW_LEVELS), N), dtype=torch.int32, device="cuda")
                # each context's ROW_K most frequent next tokens: its pairs sorted by count, descending
                key2 = torch.empty(U, dtype=torch.int64, device="cuda")
                val2 = torch.empty(U, dtype=torch.int32, device="cuda")
                lib("rank_keys", uk, agg, U, key2, val2)
                k2s, sv = torch.empty_like(key2), torch.empty_like(val2)
                lib.cub("sort_pairs", key2, k2s, val2, sv, U, i32(0), i32(32 + ctx_bits))
                del key2, val2, k2s
                top_tok = torch.zeros(n_own * ROW_K, dtype=torch.int16, device="cuda")
                top_cnt = torch.zeros(n_own * ROW_K, dtype=torch.int32, device="cuda")
                lib("rank_scatter", sv, U, uk, agg, start, top_tok, top_cnt, i32(ROW_K))
                del sv
                ri = ROW_LEVELS.index(k)
                ctx = vi.ctx_of.long()
                at = ((ctx >= lo) & (ctx < hi)).nonzero().squeeze(1)
                c = ctx[at] - lo
                tt = top_tok.view(-1, ROW_K)[c].to(torch.int32) & 0xFFFF
                tc = top_cnt.view(-1, ROW_K)[c].to(torch.int64)
                top = tc[:, :1].clamp_min(1)
                # the weights as the model reads them (15 bits): scaled down where the top count is larger
                w = torch.where(top > 32767, (tc * 32767 + top - 1) // top, tc)
                row_entries[ri, at] = torch.where(tc > 0, tt | (w.to(torch.int32) << 16), 0)
                row_top[ri, at] = tc[:, 0].clamp_max((1 << 31) - 1).to(torch.int32)
                del ctx, at, c, tt, tc, top, w, top_tok, top_cnt
            if n_own and int(tot.max().item()) >= 1 << 32:
                raise RuntimeError(f"k={k}: a total does not fit uint32")
            del uk, agg, start, end, tot, vi
            lap("rows")
        dist.all_reduce(out)
        dist.all_reduce(row_entries)
        dist.all_reduce(row_top)
        # per position its cell (0: no match at ROW_LEVELS[0]) and ROW_K candidates of its deepest matched level
        tot = out[0, [LEVELS.index(k) for k in ROW_LEVELS]].to(torch.int64) & 0xFFFFFFFF
        depth = ((tot > 0).to(torch.int64) * torch.arange(1, len(ROW_LEVELS) + 1, device=tot.device)[:, None]).amax(0) - 1
        sel = depth.clamp_min(0)
        pos = torch.arange(N, device=tot.device)
        total = tot[sel, pos].double()
        share = row_top[sel, pos].double() / total.clamp_min(1)
        count_bin = sum((total >= e).to(torch.int64) for e in ROW_COUNTS)
        purity = sum((share >= e).to(torch.int64) for e in ROW_PURITIES)
        cell = (((1 + depth) * (len(ROW_COUNTS) + 1) + count_bin) * (len(ROW_PURITIES) + 1) + purity) * (depth >= 0)
        self.row_cells = cell.to(torch.int32)
        self.row_entries = torch.where((depth >= 0)[:, None], row_entries[sel, pos], 0)
        lap("reduce")
        return out

    def batch_rows(self, at: int, n: int, capacity: int) -> torch.Tensor:
        """The record's rows (retrieval.py) of the n query positions from `at`: their cells, then `capacity` entries
        (position, token | count << 16), padded with (i % n, 0)."""
        cell, e = self.row_cells[at:at + n], self.row_entries[at:at + n]
        pos, k = (e != 0).nonzero(as_tuple=True)
        m = pos.numel()
        assert m <= capacity, f"{m} candidates overflow the rows' {capacity} entries"
        rows = torch.zeros(n + 2 * capacity, dtype=torch.int32, device=e.device)
        rows[:n] = cell
        ent = rows[n:].view(-1, 2)
        ent[:, 0] = torch.arange(capacity, device=e.device, dtype=torch.int32) % n
        ent[:m, 0] = pos.to(torch.int32)
        ent[:m, 1] = e[pos, k]
        return rows
