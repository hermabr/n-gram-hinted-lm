"""Record #1: llm.c's GPT-2 124M baseline (records/track_1_short/2024-10-13_llmc, https://github.com/karpathy/llm.c/discussions/481),
ported to PyTorch with record #367's exact-match hints (old_records/README.md, PORT.md).

The record ran llm.c's CUDA trainer, `train_gpt2cu -e d12 -b 64 -t 1024 -d 524288 -r 1 -z 1 -c 0.1 -l 0.0006 -q 0.0
-u 700`, for one epoch of FineWeb-10B: 19560 steps of 524,288 tokens on 8 GPUs. This is the same model and recipe in
PyTorch, after llm.c's own reference train_gpt2.py: GPT-2 124M (LayerNorm with bias, tanh-GELU MLP, learned position
embeddings, tied wte / lm_head, vocab padded to 50304 with the padding rows zero and outside the loss, as llm.c's CUDA
code), AdamW (0.9, 0.95), eps 1e-8, weight decay 0.1 on the 2D tensors, grad-norm clip 1.0, lr 6e-4 with a linear
warmup and a cosine decay to 0, bf16 autocast, torch.compile, flash SDPA, a sequential loader (rank r reads the r-th
B*T+1 chunk of each step's 8 chunks).

Run: python -m torch.distributed.run --standalone --nproc_per_node=8 train_gpt.py (old_records/ on PYTHONPATH,
DATA_PATH the directory with data/fineweb100B/). Env knobs: TRAIN_STEPS, LR_SCALE, SEED, HINTS / HINT_DIR, REC_* (PORT.md).

HINT_GPU_VAL=1 (with HINT_ONLINE=1): every hint row computed on the clock. The training rows come from the online
worker (old_records/online_hints.py, its training side only); the validation rows (the count rows at ALL_LEVELS and the
variant's hint rows, exactly ValCountRows' and ValHintRows') are built on these processes' GPUs from the training shards
listed in HINT_GPU_VAL_SHARDS (a file of paths, staged in /dev/shm before the clock) by old_records/gpu_val_stats.py's
InProcess, after the last step and before the clock stops (HINT_GPU_VAL_OVERLAP=1: the shards' host-to-GPU loads run
in background threads from the start of the clock, beside the training; the validation is not read before the last
step; HINT_GPU_VAL_OVERLAP=2: the loads, the validation index and the scans all run beside the training from the start
of the clock, on a thread and CUDA stream of their own, and only the merges and rows after the last step).
HINT_GPU_VAL_THREADS (12) / HINT_GPU_VAL_CHUNK_MB (16): the loads' threads per rank and chunk size.
HINT_GPU_VAL_CHECK=<dir> (smoke tests): after the run, the GPU-built rows compared with a reference's (gpu_val_stats_ref.py's
layout: tot/dist/cnt.npy, rows_<levels>/) as the model reads them, and the validation repeated with the reference rows.
"""

import contextlib
import glob
import math
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

import hint_runtime

# -----------------------------------------------------------------------------
# Configuration: the record's run, and the knobs

RECORD_STEPS = 19560  # one epoch of FineWeb-10B at 524,288 tokens per step (the record's log: s:0 .. s:19559, tel at s:19560)
RECORD_WARMUP = 700  # -u 700
DEFAULT_STEPS = 265  # ~38 s of timed training on 8 H200 (PORT.md); calibrate from the first real run


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


TRAIN_STEPS = env_int("TRAIN_STEPS", DEFAULT_STEPS)
STEP_SCALE = TRAIN_STEPS / RECORD_STEPS
LR_SCALE = env_float("LR_SCALE", 1.0)
SEED = env_int("SEED", 0)

LEARNING_RATE = 6e-4 * LR_SCALE  # -l 0.0006
WARMUP_STEPS = env_int("REC_WARMUP_STEPS", max(1, round(RECORD_WARMUP * STEP_SCALE)))
FINAL_LR_FRAC = env_float("REC_FINAL_LR_FRAC", 0.0)  # -q 0.0
WEIGHT_DECAY = env_float("REC_WEIGHT_DECAY", 0.1)  # -c 0.1
GRAD_CLIP = env_float("REC_GRAD_CLIP", 1.0)
BETAS = (0.9, env_float("REC_BETA2", 0.95))
LR_SCHEDULE = os.environ.get("REC_LR_SCHEDULE", "cosine")  # cosine (llm.c's), or wsd: warmup, constant, linear decay
assert LR_SCHEDULE in ("cosine", "wsd"), LR_SCHEDULE
COOLDOWN_FRAC = env_float("REC_COOLDOWN_FRAC", 0.5)  # wsd: the share of the steps in the final linear decay
HINT_LR = env_float("REC_HINT_LR", 0.008)  # the hint optimizer's base lr (hint_runtime.hint_optimizer), before LR_SCALE
EPS = 1e-8
B = env_int("REC_BATCH", 64)  # -b 64 sequences per GPU (keep 64 for real runs: 8 x 64 x 1024 = -d 524288)
T = 1024  # -t 1024
HINT_MID_LAYER = env_int("REC_HINT_MID_LAYER", round(7 * env_int("REC_N_LAYER", 12) / 12))  # site 1: after this many blocks (7 of 12 = 0.58 of the depth)
VAL_EVERY = env_int("REC_VAL_EVERY", 0)  # 0: only the final validation (all validations are off the clock)
LOG_EVERY = env_int("REC_LOG_EVERY", 25)  # steps between train-loss prints (each syncs the GPU)
COMPILE_WARMUP = env_int("REC_COMPILE_WARMUP", 3)  # off-the-clock steps that compile, then the state is restored
# REC_LM_AUX (found by the 1-GPU search): with an output mixture (HINT_NGRAM / HINT_MIX), add REC_LM_AUX x the LM's own
# cross-entropy to the training loss, so the LM learns at full gradient (in the mixture its gradient is scaled by its
# small posterior weight); validation reports the mixture's loss only
LM_AUX = float(os.environ.get("REC_LM_AUX", "0"))
VAL_TOKENS = 10_485_760  # the first 10 * 2**20 tokens of the validation shard, as llm.c's 20 val batches of 8 x 64 x 1024

VOCAB, PADDED_VOCAB = 50257, 50304
# model size knobs for the smaller-model interventions (default the record's 12 x 12 x 768)
N_LAYER, N_HEAD, N_EMBD = env_int("REC_N_LAYER", 12), env_int("REC_N_HEAD", 12), env_int("REC_N_EMBD", 768)

# -----------------------------------------------------------------------------
# The model: GPT-2 124M as llm.c's train_gpt2.py (with flash SDPA)


class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_attn = nn.Linear(N_EMBD, 3 * N_EMBD)
        self.c_proj = nn.Linear(N_EMBD, N_EMBD)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.size()
        q, k, v = self.c_attn(x).split(N_EMBD, dim=2)
        q, k, v = (z.view(b, t, N_HEAD, c // N_HEAD).transpose(1, 2) for z in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.c_proj(y.transpose(1, 2).contiguous().view(b, t, c))


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc = nn.Linear(N_EMBD, 4 * N_EMBD)
        self.c_proj = nn.Linear(4 * N_EMBD, N_EMBD)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # llm.c's NewGELU: 0.5 x (1 + tanh(sqrt(2 / pi) (x + 0.044715 x^3)))
        return self.c_proj(F.gelu(self.c_fc(x), approximate="tanh"))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1 = nn.LayerNorm(N_EMBD)
        self.attn = CausalSelfAttention()
        self.ln_2 = nn.LayerNorm(N_EMBD)
        self.mlp = MLP()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    def __init__(self, hint_cells: int | None, seed: int, ngram: dict | None = None):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(PADDED_VOCAB, N_EMBD),
            wpe=nn.Embedding(T, N_EMBD),
            h=nn.ModuleList([Block() for _ in range(N_LAYER)]),
            ln_f=nn.LayerNorm(N_EMBD),
        ))
        self.lm_head = nn.Linear(N_EMBD, PADDED_VOCAB, bias=False)
        self.lm_head.LLMC_SKIP_INIT = 1
        self.transformer.wte.weight = self.lm_head.weight  # tied
        self.init_rng = torch.Generator().manual_seed(seed)
        self.apply(self._init_weights)
        with torch.no_grad():  # llm.c's CUDA init: only the V real rows of the padded wte are drawn, the rest stay 0
            self.transformer.wte.weight[VOCAB:].zero_()
        # The exact-match hint (old_records/hint_runtime.py): its own optimizer, not the record's AdamW. `ngram`
        # (HINT_NGRAM=1): the count rows' ngram_levels and unigram, for the n-gram output mixture (NgramMix).
        self.hint = hint_runtime.HintEmbedding(N_EMBD, hint_cells, embed_std=0.02, **(ngram or {})) if hint_cells is not None else None
        self.per_position = False  # VAL_SUBSETS: forward returns the per-position losses (eager only, hint_runtime.report_val_subsets)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            # GPT-2's scaled init of the residual projections: 0.02 / sqrt(2 * n_layer)
            std = 0.02 / math.sqrt(2 * N_LAYER) if hasattr(module, "LLMC_RESIDUAL_SCALE_FLAG") else 0.02
            if not hasattr(module, "LLMC_SKIP_INIT"):
                nn.init.normal_(module.weight, mean=0.0, std=std, generator=self.init_rng)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02, generator=self.init_rng)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor, hint_cell=None, hint_tokens=None, hint_weights=None,
                hint_counts=None):
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        hint, rows = None, None
        if self.hint is not None:
            rows = hint_runtime.HintRows(cell=hint_cell, tokens=hint_tokens, weights=hint_weights, counts=hint_counts)
            hint = self.hint(self.transformer.wte.weight, rows).view(b, t, N_EMBD)
            x = x + (self.hint.site_scale(0) * hint).to(x.dtype)  # site 0: x = wte + wpe
        for i, block in enumerate(self.transformer.h):
            if hint is not None and i == HINT_MID_LAYER:  # site 1: after HINT_MID_LAYER blocks
                x = x + (self.hint.site_scale(1) * hint).to(x.dtype)
            x = block(x)
        if hint is not None:  # site 2: before ln_f
            x = x + (self.hint.site_scale(2) * hint).to(x.dtype)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)[..., :VOCAB]  # the padding columns are not in the softmax, as in llm.c
        if rows is None or not (self.hint.mix or self.hint.ngram):
            if self.per_position:
                return F.cross_entropy(logits.reshape(-1, VOCAB).float(), targets.reshape(-1), reduction="none")
            return F.cross_entropy(logits.reshape(-1, VOCAB).float(), targets.reshape(-1))
        # HINT_MIX=1: the loss of the mixture with the hints' candidate distributions (hint_runtime.mixed_nll);
        # HINT_NGRAM=1: with the full-count n-gram (hint_runtime.NgramMix)
        nll = F.cross_entropy(logits.reshape(-1, VOCAB).float(), targets.reshape(-1), reduction="none")
        if self.per_position:
            return self.hint.mixed_nll(nll, targets.reshape(-1), rows)
        mixed = self.hint.mixed_nll(nll, targets.reshape(-1), rows).mean()
        if LM_AUX > 0 and self.training:  # REC_LM_AUX
            return mixed + LM_AUX * nll.float().mean()
        return mixed


# -----------------------------------------------------------------------------
# Data: llm.c's sequential loader, also giving each chunk's (shard, offset)

HEADER_INTS = 256


def load_shard(path: str) -> np.ndarray:
    header = np.fromfile(path, dtype=np.int32, count=HEADER_INTS)
    assert header[0] == 20240520 and header[1] == 1, f"{path}: bad header"
    tokens = np.memmap(path, dtype=np.uint16, mode="r", offset=HEADER_INTS * 4)
    assert len(tokens) == header[2], f"{path}: {len(tokens)} tokens, header says {header[2]}"
    return tokens


class DataLoader:
    """Each step's rank r reads tokens[p + r * B * T : p + (r + 1) * B * T + 1] of the current shard; p advances by
    B * T * world and the next shard starts when the following step's chunks would not fit (llm.c's
    DistributedDataLoader, with the shard switch on the same step for every rank). next_batch gives x, y on the GPU and the chunk's (shard, offset): input position i of the
    chunk (flattened) is offset + i in the shard."""

    def __init__(self, files: list[str], rank: int, world: int, device: torch.device, accum: int = 1):
        assert files, "no data files"
        self.files, self.rank, self.world, self.device = files, rank, world, device
        # gradient accumulation (fewer than 8 GPUs): the rank reads `accum` chunks per step, those of the virtual
        # ranks rank * accum + m (m = 0 .. accum - 1) of a world * accum = 8 GPU run, so each step still covers the
        # same 8 chunks. accum = 1 is the plain loader (virtual rank = rank, stride B * T * world).
        self.accum, self.vworld = accum, world * accum
        self.reset()

    def reset(self) -> None:
        self.shard = 0
        self.tokens = load_shard(self.files[0])
        self.micro = 0
        self.position = self.rank * self.accum * B * T

    def next_batch(self):
        shard, start = self.shard, self.position
        buf = torch.from_numpy(self.tokens[start:start + B * T + 1].astype(np.int64)).pin_memory()
        buf = buf.to(self.device, non_blocking=True)
        self.micro += 1
        if self.micro < self.accum:  # the step's next chunk of this rank
            self.position += B * T
            return buf[:-1].view(B, T), buf[1:].view(B, T), shard, start
        self.micro = 0
        self.position += B * T * (self.vworld - self.accum + 1)  # = B * T * world without accumulation
        first = self.rank * self.accum * B * T  # the rank's first chunk's offset within a step
        # all ranks move on together: when the next step's 8 chunks (from rank 0's position) would not fit. (llm.c's
        # Python loader tests each rank's own position, so ranks 6 and 7 moved on a step early.)
        if self.position - first + (B * T * self.vworld + 1) > len(self.tokens):
            self.shard += 1
            if self.shard >= len(self.files):
                raise RuntimeError(f"ran out of training data after {len(self.files)} shards")
            self.tokens = load_shard(self.files[self.shard])
            self.position = first
        return buf[:-1].view(B, T), buf[1:].view(B, T), shard, start


def loader_plan(files: list[str], world: int, accum: int, steps: int) -> list[tuple[int, int, int, int]]:
    """HINT_ONLINE: every rank's DataLoader chunks from its reset, (step, shard, start, B * T), for the `steps` steps of
    `accum` chunks each, as next_batch moves (the loop fetches none after the last step)."""
    lens = [int(np.fromfile(f, dtype=np.int32, count=HEADER_INTS)[2]) for f in files]
    vworld, plan = world * accum, []
    for rank in range(world):
        first = rank * accum * B * T
        shard, position = 0, first
        for b in range(steps * accum):
            plan.append((b // accum, shard, position, B * T))
            if (b + 1) % accum:
                position += B * T
                continue
            position += B * T * (vworld - accum + 1)
            if position - first + (B * T * vworld + 1) > lens[shard]:
                shard, position = shard + 1, first
                if shard >= len(files):
                    raise RuntimeError(f"ran out of training data after {len(files)} shards")
    return plan


# -----------------------------------------------------------------------------
# HINT_GPU_VAL: the validation rows built on the GPUs (gpu_val_stats.InProcess), as hint_runtime.Hints's rows


class GpuValRows:
    """InProcess.finish()'s outputs (every rank's GPU) as the HintRows a validation batch reads: the variant's hint rows
    (cell, tokens, weights / WEIGHT_ONE) and, with `levels` > 0 (HINT_NGRAM), the count rows as float32 [n, 3, L], the
    conversions hint_runtime.Hints._upload / _upload_counts make of the precomputed files."""

    def __init__(self, built: dict, levels: int):
        rows = built["rows"][0]  # [N, 9] int32: cell, then 8 tokens and 8 weights as int16 pairs
        self.cell = rows[:, 0].long()
        self.tokens = rows[:, 1:5].contiguous().view(torch.int16).long() & 0xFFFF
        self.weights = (rows[:, 5:9].contiguous().view(torch.int16).int() & 0xFFFF).float() / hint_runtime.WEIGHT_ONE
        self.counts = None
        if levels:
            counts = built["counts"]  # [3, L, N] int32 (tot and cnt: uint32 bits)
            assert counts.shape[1] == levels, counts.shape
            self.counts = (counts.permute(2, 0, 1).long() & 0xFFFFFFFF).float().contiguous()

    def val_span(self, start: int, length: int) -> hint_runtime.HintRows:
        s = slice(start, start + length)
        return hint_runtime.HintRows(cell=self.cell[s], tokens=self.tokens[s], weights=self.weights[s],
                                     counts=None if self.counts is None else self.counts[s])

    def compare(self, other, chunk: int = 1 << 20) -> dict:
        """Positions whose rows differ from `other`'s (val_span), per part."""
        n, differ = len(self.cell), {"cell": 0, "tokens": 0, "weights": 0, "counts": 0}
        for a in range(0, n, chunk):
            mine, theirs = self.val_span(a, min(chunk, n - a)), other.val_span(a, min(chunk, n - a))
            differ["cell"] += int((mine.cell != theirs.cell).sum())
            differ["tokens"] += int((mine.tokens != theirs.tokens).any(-1).sum())
            differ["weights"] += int((mine.weights != theirs.weights).any(-1).sum())
            if self.counts is not None:
                differ["counts"] += int((mine.counts != theirs.counts).flatten(1).any(-1).sum())
        return differ


class RefValRows:
    """HINT_GPU_VAL_CHECK: a reference's validation rows (gpu_val_stats_ref.py's files) through the precomputed runs'
    upload path (hint_runtime.Hints._upload / _upload_counts)."""

    def __init__(self, directory: str, hints):
        d, self.hints = directory, hints
        rows = os.path.join(d, "rows_" + "_".join(map(str, hints.levels)))
        self.cell, self.tokens, self.weights = (np.load(os.path.join(rows, f"{n}.npy"), mmap_mode="r")
                                                for n in ("cell", "tokens", "weights"))
        self.counts = [np.load(os.path.join(d, f"{n}.npy"), mmap_mode="r") for n in ("tot", "dist", "cnt")] if hints.ngram else None

    def val_span(self, start: int, length: int) -> hint_runtime.HintRows:
        s = slice(start, start + length)
        rows = self.hints._upload(self.cell[s], self.tokens[s], self.weights[s])
        if self.counts is not None:
            rows.counts = self.hints._upload_counts(*(a[s] for a in self.counts))
        torch.cuda.synchronize()  # the ring's staging is reused
        return rows


def gval_summary(timing: dict, seconds: float, at_ms: float, overlap: int, waited: float | None) -> str:
    """The build's breakdown line (rank 0's times)."""
    marks = timing["marks"]
    phases = {name: own + wait for name, own, wait in timing["phases"]}
    peak = f"peak GPU memory {timing['mem_peak_allocated_gb']:.1f} GB; rank 0: {timing['load']['shards']} shards, {timing['load']['bytes'] / 1e9:.1f} GB"
    if overlap == 2:  # times from the clock's start (the background part's start)
        return (f"gpu val build: {seconds:.2f} s on the clock after the last step ({waited:.2f} waiting for the background "
                f"part, then merge {timing['total_s'] - marks['resumed']:.2f}); background from the clock start: loads done "
                f"{marks.get('load_done', 0.0):.2f} s, index {phases.get('val_index', 0.0):.2f}, scans done "
                f"{marks.get('fused_scan_done', 0.0):.2f} s; last step done {at_ms / 1000:.2f} s; {peak}")
    load = max(0.0, marks.get("load_done", 0.0))  # from the build's start until every shard was on the GPU
    scanned = marks.get("fused_scan_done", load)
    loads = ("loads from the clock start, done " + format(timing["load"]["done_s"] + at_ms / 1000, ".2f") + " s after it"
             if overlap else "loads after the last step")
    return (f"gpu val build: {seconds:.2f} s on the clock (load {load:.2f}, scan {scanned - load:.2f}, merge "
            f"{timing['total_s'] - scanned:.2f}; index {phases.get('val_index', 0.0):.2f} of it; {loads}), "
            f"started {at_ms / 1000:.2f} s after the clock started; {peak}")


# -----------------------------------------------------------------------------
# Main

def main() -> None:
    dist.init_process_group(backend="nccl")
    rank, local_rank, world = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    master = rank == 0
    print0 = lambda *a: print(*a, flush=True) if master else None

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # flash attention only (llm.c used cuDNN's flash attention)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(False)
    # compile flags as record #360's, the same for every port (no coordinate_descent_tuning: banned, 30 min compiles)

    use_hints = hint_runtime.hints_enabled()
    # HINT_ONLINE=1: the hint rows computed in the run, in parallel with training (old_records/online_hints.py); the
    # validation index is built while training runs, so only the final validation has rows
    online = use_hints and os.environ.get("HINT_ONLINE", "0") == "1"
    # HINT_GPU_VAL=1: the validation rows built on the GPUs in these processes, on the clock (the module docstring)
    gpu_val = online and os.environ.get("HINT_GPU_VAL", "0") == "1"

    data = os.path.join(os.environ.get("DATA_PATH", "."), "data/fineweb100B")
    train_files = sorted(glob.glob(os.path.join(data, "fineweb_train_*.bin")))
    val_file = os.path.join(data, "fineweb_val_000000.bin")
    # fewer than 8 GPUs: accumulate gradients over 8 // world micro-steps, so a step is the 8-GPU run's batch (its 8
    # chunks) and the math is the same; 8 GPUs (or more) do no accumulation
    accum = 8 // world if world < 8 else 1
    if online:
        from online_hints import OnlineHints
        assert VAL_EVERY == 0, "HINT_ONLINE: only the final validation has rows (REC_VAL_EVERY=0)"
        hints = OnlineHints(loader_plan(train_files, world, accum, TRAIN_STEPS), TRAIN_STEPS, train_files, val_file,
                            device=device)
    else:
        hints = hint_runtime.Hints(device=device) if use_hints else None
    assert world * accum == 8 or world >= 8, f"{world} GPUs: the batch of 8 GPUs needs a world size dividing 8"
    tokens_per_step = B * T * world * accum
    assert VAL_TOKENS % tokens_per_step == 0, f"{VAL_TOKENS} validation tokens are not whole steps of {tokens_per_step}"
    val_steps = VAL_TOKENS // tokens_per_step

    print0(f"torch {torch.__version__}, {world} GPUs ({torch.cuda.get_device_name(device)}), hints: "
           f"{os.environ.get('HINT_DIR') if use_hints else 'off'}")
    print0(f"TRAIN_STEPS {TRAIN_STEPS} (record {RECORD_STEPS}, STEP_SCALE {STEP_SCALE:.5f}), warmup {WARMUP_STEPS}, "
           f"lr {LEARNING_RATE:g} ({LR_SCHEDULE}{f' cooldown {COOLDOWN_FRAC}' if LR_SCHEDULE == 'wsd' else ''}), "
           f"final lr frac {FINAL_LR_FRAC}, wd {WEIGHT_DECAY}, betas {BETAS}, clip {GRAD_CLIP}, hint lr {HINT_LR}, "
           f"B {B} x T {T} x {world} = "
           f"{tokens_per_step} tokens/step{f' ({accum} micro-steps)' if accum > 1 else ''}, seed {SEED}, {len(train_files)} train shards")

    ngram = None
    if use_hints and hints.ngram:  # HINT_NGRAM=1: the n-gram output mixture over the count rows' levels
        ngram = {"ngram_levels": len(hints.count_levels), "unigram": hints.unigram}
        print0(f"n-gram mixing (HINT_NGRAM) over levels {hints.count_levels}")
    raw_model = GPT(hints.cells if use_hints else None, SEED, ngram).to(device)
    model = DDP(torch.compile(raw_model, dynamic=False, fullgraph=True), device_ids=[local_rank])

    record_params = [p for n, p in raw_model.named_parameters() if not n.startswith("hint.")]
    decay = [p for p in record_params if p.dim() >= 2]
    no_decay = [p for p in record_params if p.dim() < 2]
    print0(f"{sum(p.numel() for p in record_params):,} record parameters "
           f"({sum(p.numel() for p in decay):,} decayed, {sum(p.numel() for p in no_decay):,} not)")
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": WEIGHT_DECAY}, {"params": no_decay, "weight_decay": 0.0}],
        lr=LEARNING_RATE, betas=BETAS, eps=EPS, fused=True,
    )
    hint_opt = hint_runtime.hint_optimizer(raw_model.hint, lr=HINT_LR) if use_hints else None

    def lr_factor(step: int) -> float:
        """llm.c's cosine schedule (schedulers.h) as a fraction of the max lr."""
        if step < WARMUP_STEPS:
            return (step + 1) / WARMUP_STEPS
        if LR_SCHEDULE == "wsd":  # REC_LR_SCHEDULE=wsd: constant, then linear to FINAL_LR_FRAC over the last steps
            cooldown = max(1, round(COOLDOWN_FRAC * TRAIN_STEPS))
            frac = min(1.0, (TRAIN_STEPS - step) / cooldown)
            return FINAL_LR_FRAC + frac * (1.0 - FINAL_LR_FRAC)
        decay_ratio = (step - WARMUP_STEPS) / (TRAIN_STEPS - WARMUP_STEPS)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return FINAL_LR_FRAC + coeff * (1.0 - FINAL_LR_FRAC)

    train_loader = DataLoader(train_files, rank, world, device, accum)
    val_loader = DataLoader([val_file], rank, world, device, accum)
    val_rows = [hints]  # where the validation rows come from (HINT_GPU_VAL: GpuValRows once built)

    def batch(loader: DataLoader, train: bool):
        x, y, shard, start = loader.next_batch()
        if not use_hints:
            return x, y, ()
        rows = hints.train_span(shard, start, B * T) if train else val_rows[0].val_span(start, B * T)
        if rows.counts is not None:  # HINT_NGRAM=1
            return x, y, (rows.cell, rows.tokens, rows.weights, rows.counts)
        return x, y, (rows.cell, rows.tokens, rows.weights)

    def batches():
        """A step's micro-batches: one without accumulation, else the rank's accum chunks of the step."""
        return [batch(train_loader, True) for _ in range(accum)]

    def train_step(step: int, micro, fetch_next: bool):
        factor = lr_factor(step)
        for group in optimizer.param_groups:
            group["lr"] = LEARNING_RATE * factor
        if hint_opt is not None:
            hint_runtime.set_lr_factor(hint_opt, LR_SCALE * factor)
        if accum == 1:  # the 8-GPU path
            (x, y, hint_args), = micro
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y, *hint_args)
            nxt = batches() if fetch_next else None  # the host work overlaps the GPU's forward
            loss.backward()
        else:  # the mean of the micro-steps' losses (each chunk's mean, as each rank's of 8), DDP syncing at the last
            losses, nxt = [], None
            for m, (x, y, hint_args) in enumerate(micro):
                sync = m == accum - 1
                with model.no_sync() if not sync else contextlib.nullcontext():
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        loss = model(x, y, *hint_args)
                    if m == 0 and fetch_next:
                        nxt = batches()
                    (loss / accum).backward()
                losses.append(loss.detach())
            loss = torch.stack(losses).mean()
        torch.nn.utils.clip_grad_norm_(record_params, GRAD_CLIP)
        optimizer.step()
        if hint_opt is not None:
            hint_opt.step()
        model.zero_grad(set_to_none=True)
        return loss, nxt

    @torch.no_grad()
    def validate() -> float:
        model.eval()
        val_loader.reset()
        total = torch.zeros((), device=device)
        for _ in range(val_steps * accum):
            x, y, hint_args = batch(val_loader, False)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                total += model(x, y, *hint_args)
        dist.all_reduce(total, op=dist.ReduceOp.AVG)
        model.train()
        return total.item() / (val_steps * accum)

    # Compile off the clock, then restore the model, optimizers and loader.
    if COMPILE_WARMUP > 0:
        initial = {k: v.detach().clone() for k, v in raw_model.state_dict().items()}
        start = time.perf_counter()
        micro = batches()
        for step in range(COMPILE_WARMUP):
            _, micro = train_step(step, micro, fetch_next=True)
        torch.cuda.synchronize()
        print0(f"compile warmup: {COMPILE_WARMUP} steps in {time.perf_counter() - start:.1f} s")
        raw_model.load_state_dict(initial)
        del initial
        for opt in (optimizer, hint_opt):
            if opt is not None:
                opt.state.clear()
        model.zero_grad(set_to_none=True)
        train_loader.reset()
        dist.barrier()

    gval, gval_overlap, gval_line = None, env_int("HINT_GPU_VAL_OVERLAP", 0), ""
    if gpu_val:  # before the clock: the shards' GPU buffer and pinned staging allocated, NCCL's all-to-all warmed up
        import gpu_val_stats

        assert tuple(hints.buckets) == gpu_val_stats.LEVELS, f"HINT_GPU_VAL: buckets {hints.buckets}"
        start = time.perf_counter()
        gval_shards = open(os.environ["HINT_GPU_VAL_SHARDS"]).read().split()
        gval = gpu_val_stats.InProcess(gval_shards, val_file, variants=(hints.levels,),
                                       threads=env_int("HINT_GPU_VAL_THREADS", 12),
                                       chunk_mb=env_int("HINT_GPU_VAL_CHUNK_MB", 16))
        gval.prepare()
        dist.barrier()
        print0(f"gpu val build: {len(gval_shards)} shards ({len(gval.mine)} on rank 0, {2 * sum(gval.sizes) / 1e9:.1f} GB), "
               f"prepared in {time.perf_counter() - start:.1f} s before the clock"
               f"{['', ', loads overlapped with the training', ', loads, index and scans overlapped with the training'][gval_overlap]}")

    train_ms = 0.0
    val_loss = float("nan")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    if online:  # rank 0 starts the online hint work (and every rank the n-gram's unigram init)
        hints.start(raw_model.hint.ngram_mix if hints.ngram else None)
    if gval is not None and gval_overlap == 1:  # the shards' loads, in the background (nothing of the validation yet)
        gval.start_load()
    elif gval is not None and gval_overlap == 2:  # the loads, the validation index and the scans, in the background
        gval.start_background()
    micro = batches()
    for step in range(TRAIN_STEPS):
        last = step == TRAIN_STEPS - 1
        loss, nxt = train_step(step, micro, fetch_next=not last)
        if not last:
            micro = nxt
        done = step + 1
        if done % LOG_EVERY == 0 or last or (VAL_EVERY > 0 and done % VAL_EVERY == 0):
            loss_value = loss.item()  # syncs
            elapsed = train_ms + 1000 * (time.perf_counter() - t0)
            print0(f"step:{done}/{TRAIN_STEPS} train_loss:{loss_value:.4f} lr:{LEARNING_RATE * lr_factor(step):.3e} "
                   f"train_time:{elapsed:.0f}ms step_avg:{elapsed / done:.2f}ms")
        if (VAL_EVERY > 0 and done % VAL_EVERY == 0) or last:
            if online and last:  # the wait for the online index builds, on the clock
                hints.finish()
            if gval is not None and last:  # the GPU validation build, on the clock: its rows on every GPU
                torch.cuda.synchronize()
                start = time.perf_counter()
                at = 1000 * (start - t0)
                built = gval.finish()
                val_rows[0] = GpuValRows(built, len(hints.count_levels) if hints.ngram else 0)
                del built["counts"], built["rows"]
                torch.cuda.synchronize()
                gval_s = time.perf_counter() - start
                gval_line = gval_summary(built["timing"], gval_s, at, gval_overlap, getattr(gval, "waited", None))
            torch.cuda.synchronize()
            train_ms += 1000 * (time.perf_counter() - t0)
            val_loss = validate()  # off the clock
            print0(f"step:{done}/{TRAIN_STEPS} val_loss:{val_loss:.4f} train_time:{train_ms:.0f}ms "
                   f"step_avg:{train_ms / done:.2f}ms")
            torch.cuda.synchronize()
            t0 = time.perf_counter()

    print0(f"peak memory: {torch.cuda.max_memory_allocated(device) // 2**20} MiB")
    print0(f"FINAL val_loss:{val_loss:.4f} train_time_ms:{int(train_ms)} steps:{TRAIN_STEPS}")
    if online:
        print0(hints.summary())
    if gval_line:
        print0(gval_line)

    def subset_losses():  # VAL_SUBSETS (after the timer): validate()'s batches, per position, with the eager raw model
        val_loader.reset()
        for _ in range(val_steps * accum):
            start = val_loader.position  # the chunk's offset in the validation shard (next_batch's start)
            x, y, hint_args = batch(val_loader, False)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                losses = raw_model(x, y, *hint_args)
            yield start, losses

    hint_runtime.report_val_subsets(raw_model, subset_losses(), device, print0, val_loss)
    if gpu_val and os.environ.get("HINT_GPU_VAL_CHECK"):  # smoke tests: the rows against a reference's, as read
        reference = RefValRows(os.environ["HINT_GPU_VAL_CHECK"], hints)
        differ = val_rows[0].compare(reference)
        print0(f"HINT_GPU_VAL_CHECK rows vs {os.environ['HINT_GPU_VAL_CHECK']}: differing positions {differ}")
        val_rows[0] = reference
        print0(f"HINT_GPU_VAL_CHECK val_loss with the GPU rows {val_loss:.6f}, with the reference rows {validate():.6f}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
