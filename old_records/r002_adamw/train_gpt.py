"""Record #2: "Tuned learning rate & rotary embeddings" (06/06/24, 31.4 minutes; log in modded-nanogpt
records/track_1_short/2024-06-06_AdamW/), ported to torch 2.10 with record #367's exact-match hints
(old_records/README.md, PORT.md).

The record is modded-nanogpt-public's train_gpt2.py at 3e48a54 ("upgrade with rotary embeddings", unchanged through
7751345) run with 7751345's run.sh: GPT-2 124M with rotary embeddings, weightless fp32 rmsnorm, no biases, an attention
output scale 1/sqrt(24), tied wte / lm_head (nn.Linear's default init), AdamW lr 0.0018 betas (0.9, 0.95) wd 0.1 on
every parameter, each gradient normalized to unit norm, 256 warmup steps, constant lr, 2048 linear warmdown steps,
9536 steps of 8 x 64 x 1024 tokens. run.sh's arguments are the argparse defaults here. Everything the port changed is
marked `port:`.

Run: python -m torch.distributed.run --standalone --nproc_per_node=8 train_gpt.py (old_records/ on PYTHONPATH,
DATA_PATH the directory with data/fineweb100B/). Env knobs: TRAIN_STEPS, LR_SCALE, SEED, HINTS / HINT_DIR, REC_* (PORT.md).
"""

import os
import sys
import uuid
import math
import glob
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import hint_runtime

with open(sys.argv[0]) as f:
    code = f.read()

# -----------------------------------------------------------------------------
# port: the budget knobs (old_records/README.md item 4)

RECORD_STEPS = 9536  # run.sh --num_iterations (the log: s:0 .. s:9535, the last tel at s:9536)
RECORD_WARMUP = 256  # run.sh --warmup_iters
RECORD_WARMDOWN = 2048  # run.sh --warmdown_iters
DEFAULT_STEPS = 242  # ~38 s of timed training on 8 H200 at r001's 157 ms/step (PORT.md); calibrate from the first run


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


TRAIN_STEPS = env_int("TRAIN_STEPS", DEFAULT_STEPS)
STEP_SCALE = TRAIN_STEPS / RECORD_STEPS
LR_SCALE = env_float("LR_SCALE", 1.0)
SEED = env_int("SEED", 0)
HINT_MID_LAYER = env_int("REC_HINT_MID_LAYER", round(7 * env_int("REC_N_LAYER", 12) / 12))  # site 1: after this many blocks (7 of 12 = 0.58 of the depth)
LOG_EVERY = env_int("REC_LOG_EVERY", 25)  # steps between train-loss prints (each syncs the GPU, inside the timer)
COMPILE_WARMUP = env_int("REC_COMPILE_WARMUP", 3)  # off-the-clock steps that compile, then the state is restored
# REC_LM_AUX: with an output mixture, add REC_LM_AUX x the LM's own cross-entropy to the training loss (r001's)
LM_AUX = float(os.environ.get("REC_LM_AUX", "0"))
VAL_TOKENS = 10_485_760  # the record's 20 val batches of 8 x 64 x 1024
EMBED_STD = 1 / math.sqrt(3 * 768)  # the tied weight is lm_head's nn.Linear init U(+-1/sqrt(768)): std 0.02083

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model

class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos()
            self.sin_cached = freqs.sin()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4 # multihead attention
    d = x.shape[3]//2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

def rmsnorm(x0, eps=1e-6):
    x = x0.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x.type_as(x0)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim)
        q = q.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.attn_scale = (1 / math.sqrt(2 * config.n_layer))

    def forward(self, x):
        x = x + self.attn_scale * self.attn(rmsnorm(x))
        x = x + self.mlp(rmsnorm(x))
        return x

# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):

    def __init__(self, config, hint_cells=None, ngram=None):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying
        # port: record #367's hint (old_records/hint_runtime.py), a submodule (so DDP averages its gradients and
        # torch.compile compiles it with the model), outside the record's AdamW and gradient normalization. `ngram`
        # (HINT_NGRAM=1): the count rows' ngram_levels and unigram, for the n-gram output mixture (NgramMix)
        self.hint = None
        self.per_position = False  # port: VAL_SUBSETS, the loss per position (eager only, hint_runtime.report_val_subsets)
        if hint_cells is not None:
            assert 1 <= HINT_MID_LAYER < config.n_layer, f"REC_HINT_MID_LAYER={HINT_MID_LAYER} must be between blocks"
            self.hint = hint_runtime.HintEmbedding(config.n_embd, hint_cells, embed_std=EMBED_STD, **(ngram or {}))

    def forward(self, idx, targets=None, return_logits=True, hint_cell=None, hint_tokens=None, hint_weights=None,
                hint_counts=None):
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device) # shape (t)

        # forward the GPT model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        hint = None
        if self.hint is not None:
            # port: row i of the rank's B*T rows is for flattened input position i; cast to the stream's dtype (float32:
            # fp32 parameters, under bf16 autocast in training and without autocast in validation)
            rows = hint_runtime.HintRows(cell=hint_cell, tokens=hint_tokens, weights=hint_weights, counts=hint_counts)
            hint = self.hint(self.transformer.wte.weight, rows).view(b, t, -1)
            x = x + (self.hint.site_scale(0) * hint).to(x.dtype)  # port: site 0, the token embedding x0

        for i, block in enumerate(self.transformer.h):
            if hint is not None and i == HINT_MID_LAYER:  # port: site 1, after HINT_MID_LAYER blocks
                x = x + (self.hint.site_scale(1) * hint).to(x.dtype)
            x = block(x)
        if hint is not None:  # port: site 2, right before the final norm and head
            x = x + (self.hint.site_scale(2) * hint).to(x.dtype)
        x = rmsnorm(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            if hint is None or not (self.hint.mix or self.hint.ngram):
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1,
                                       **({"reduction": "none"} if self.per_position else {}))
            elif self.per_position:
                nll = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none")
                loss = self.hint.mixed_nll(nll, targets.view(-1), rows)
            else:
                # port: HINT_NGRAM=1 (or HINT_MIX=1): the loss of the mixture of the model's distribution with the
                # full-count n-gram (hint_runtime.NgramMix), per position from the record's own cross-entropy (the same
                # logits: bf16 under autocast in training, which autocast's cross_entropy computes in fp32; fp32 in
                # validation), then the mean; training and validation alike
                nll = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none")
                loss = self.hint.mixed_nll(nll, targets.view(-1), rows).mean()
                if LM_AUX > 0 and self.training:  # REC_LM_AUX
                    loss = loss + LM_AUX * nll.float().mean()
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss

    def record_parameters(self):
        # port: the record's parameters (its configure_optimizers and gradient normalization took self.parameters(),
        # which here would include the hint's)
        return [p for n, p in self.named_parameters() if not n.startswith("hint.")]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # port: fused=True (the same AdamW update in one kernel; the record used the default foreach implementation)
        optimizer = torch.optim.AdamW(self.record_parameters(), lr=learning_rate, weight_decay=weight_decay, betas=betas,
                                      fused=True)
        return optimizer

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
        print("---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try")
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2] # number of tokens (claimed)
    return ntok # for now just return the number of tokens

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2] # number of tokens (claimed)
    # port: memory-mapped instead of read whole (no stall on the clock at a shard switch; the same tokens)
    tokens = np.memmap(filename, dtype=np.uint16, mode="r", offset=256 * 4)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens

class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += shard_ntok
        self.ntok_total = ntok_total
        print0(f"DataLoader: total number of tokens: {ntok_total:,} across {len(self.files)} files")

        # kick things off
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self): # advance to next data shard
        # port: no wrap-around (the record wrapped with % len(files)): running out of shards is an error, since the
        # hint rows are per shard and OldRecordRun stages only port.json's train_shards
        self.current_shard = self.current_shard + 1
        if self.current_shard >= len(self.files):
            raise RuntimeError(f"ran out of training data after {len(self.files)} shards")
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        """port: also returns the batch's (shard, start): flattened input position i is offset start + i of shard
        `shard` (the index in the sorted file list)"""
        B = self.B
        T = self.T
        shard, start = self.current_shard, self.current_position
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance current position and load next shard if necessary
        self.current_position += B * T * self.num_processes
        # port: every rank moves on when the next step's chunks of all ranks would not fit (from rank 0's position), as
        # r001: the record tested each rank's own position, so ranks 6 and 7 moved on one step before ranks 0-5
        if self.current_position - self.process_rank * B * T + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x.cuda(), y.cuda(), shard, start

# -----------------------------------------------------------------------------
# int main

def print0(*args, **kwargs):
    # modified print that only prints from the master process
    # if this is not a distributed run, it's just a print
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs, flush=True)

if __name__ == "__main__":
    import argparse
    print0(f"Running pytorch {torch.version.__version__}")

    data_dir = os.path.join(os.environ.get("DATA_PATH", "."), "data/fineweb100B")
    parser = argparse.ArgumentParser()
    # port: the defaults are 7751345's run.sh (its arguments), on FineWeb-100B, with the step counts scaled by STEP_SCALE
    # file system input / output
    parser.add_argument("--input_bin", type=str, default=os.path.join(data_dir, "fineweb_train_*.bin"), help="input .bin to train on")
    parser.add_argument("--input_val_bin", type=str, default=os.path.join(data_dir, "fineweb_val_000000.bin"), help="input .bin to eval validation loss on")
    parser.add_argument("--output_dir", type=str, default="pylog124M", help="output directory to which to write logs and checkpoints")
    parser.add_argument("--model", type=str, default="d12", help="gpt2|gpt2-medium|gpt2-large|gpt2-xl|d12|d24|d36|d48")
    # token layout for each step of the optimization
    parser.add_argument("--batch_size", type=int, default=env_int("REC_BATCH", 64), help="batch size, in units of #batch dimensions")
    parser.add_argument("--sequence_length", type=int, default=1024, help="sequence length")
    parser.add_argument("--total_batch_size", type=int, default=None, help="total desired batch size, in units of #tokens (port: default batch_size * sequence_length * world, 524288 on 8 GPUs)")
    # workload (number of steps)
    parser.add_argument("--num_iterations", type=int, default=TRAIN_STEPS, help="number of iterations to run")
    # optimization
    parser.add_argument("--learning_rate", type=float, default=0.0018 * LR_SCALE, help="learning rate warmup iterations")
    parser.add_argument("--warmup_iters", type=int, default=env_int("REC_WARMUP_STEPS", max(1, round(RECORD_WARMUP * STEP_SCALE))), help="learning rate warmup iterations")
    parser.add_argument("--warmdown_iters", type=int, default=env_int("REC_WARMDOWN_STEPS", round(RECORD_WARMDOWN * STEP_SCALE)), help="learning rate warmdown iterations")
    parser.add_argument("--weight_decay", type=float, default=env_float("REC_WEIGHT_DECAY", 0.1), help="weight decay")
    # evaluation
    parser.add_argument("--val_loss_every", type=int, default=env_int("REC_VAL_EVERY", 0), help="every how mant steps to evaluate val loss? (port: 0 = only at the end; run.sh had 128)")
    parser.add_argument("--val_tokens", type=int, default=VAL_TOKENS, help="port: the record's 20 val batches of 8 x 64 x 1024, whatever the GPU count and batch")
    args = parser.parse_args()

    # args error checking and convenience variables
    B, T = args.batch_size, args.sequence_length
    assert 1 <= T <= 1024
    assert args.model in {"d12", "d24", "d36", "d48"}
    assert args.warmup_iters + args.warmdown_iters <= args.num_iterations, "warmup + warmdown longer than the run"

    # set up DDP (distributed data parallel). torchrun sets this env variable
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = 0 # each process gets the exact same seed
    print(f"using device: {device}")
    torch.manual_seed(SEED)  # port: SEED (the record had no seed); DDP broadcasts rank 0's init anyway
    np.random.seed(SEED)

    tokens_per_fwdbwd = B * T * ddp_world_size
    if args.total_batch_size is not None:
        assert args.total_batch_size == tokens_per_fwdbwd
    assert args.val_tokens % tokens_per_fwdbwd == 0, f"{args.val_tokens} val tokens are not whole steps of {tokens_per_fwdbwd}"
    val_steps = args.val_tokens // tokens_per_fwdbwd  # port: replaces --val_max_steps 20 (20 at 8 x 64 x 1024)

    # port: hint rows (old_records/README.md item 3); HINTS=0 is the plain record
    use_hints = hint_runtime.hints_enabled()
    hints = hint_runtime.Hints(device=device) if use_hints else None
    print0(f"{ddp_world_size} GPUs ({torch.cuda.get_device_name(device)}), hints: {os.environ.get('HINT_DIR') if use_hints else 'off'}")
    print0(f"TRAIN_STEPS {args.num_iterations} (record {RECORD_STEPS}, STEP_SCALE {STEP_SCALE:.5f}), warmup "
           f"{args.warmup_iters}, warmdown {args.warmdown_iters}, lr {args.learning_rate:g} (LR_SCALE {LR_SCALE}), wd "
           f"{args.weight_decay}, B {B} x T {T} x {ddp_world_size} = {tokens_per_fwdbwd} tokens/step, val_steps "
           f"{val_steps}, seed {SEED}, compile warmup {COMPILE_WARMUP}")

    # set up a context manager following the desired dtype and device
    ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

    # init the model from scratch
    model_config = {
        # model size knobs for the smaller-model interventions (default the record's 12 x 12 x 768)
        "d12": GPTConfig(block_size=1024, vocab_size=50257, n_layer=env_int("REC_N_LAYER", 12), n_head=env_int("REC_N_HEAD", 12), n_embd=env_int("REC_N_EMBD", 768)),
        "d24": GPTConfig(block_size=1024, vocab_size=50257, n_layer=24, n_head=16, n_embd=1024),
        "d36": GPTConfig(block_size=1024, vocab_size=50257, n_layer=36, n_head=20, n_embd=1280),
        "d48": GPTConfig(block_size=1024, vocab_size=50257, n_layer=48, n_head=25, n_embd=1600),
    }[args.model]
    ngram = None
    if use_hints and hints.ngram:  # port: HINT_NGRAM=1, the n-gram output mixture over the count rows' levels
        ngram = dict(ngram_levels=len(hints.count_levels), unigram=hints.unigram)
        print0(f"port: n-gram mixing (HINT_NGRAM) over levels {hints.count_levels}")
    model = GPT(model_config, hints.cells if use_hints else None, ngram)
    model = model.train().cuda()
    # port: fill each Rotary's cos/sin cache for T eagerly, with the record's own code, before compiling (as r003). In
    # the record the first compiled call set these plain attributes and the next call recompiled with the cache set;
    # under fullgraph=True the cache must exist before tracing. The values are the fp32 cache every forward reads.
    with torch.no_grad():
        for block in model.transformer.h:
            block.attn.rotary(torch.empty(1, T, 1, block.attn.head_dim, device=device))
    if use_hints:
        print0(f"port: hint embed_ratio {model.hint.embed_ratio:.4f}, site_scales {model.hint.site_scales.tolist()}, "
               f"site 1 after {HINT_MID_LAYER} blocks")
    # port: record #360's compile flags, as every port (the record set config.coordinate_descent_tuning = True; not here)
    print0("compiling the model...")
    model = torch.compile(model, dynamic=False, fullgraph=True)

    # load tokens
    train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
    val_loader = DistributedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size)
    # port: the record loaded its first batch here; the port loads it on the clock, after the compile warmup

    # here we wrap model into DDP container
    model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module._orig_mod # always contains the "raw" unwrapped model (port: below the compile wrapper)
    record_params = raw_model.record_parameters()
    print0(f"{sum(p.numel() for p in record_params):,} record parameters")

    # init the optimizer
    optimizer = raw_model.configure_optimizers(weight_decay=args.weight_decay,
                                               learning_rate=args.learning_rate, betas=(0.9, 0.95),
                                               device_type=device)
    # port: the hint's Adam (lr 0.008 x LR_SCALE, scales 10x, no wd), its lr following the record's schedule factor
    hint_opt = hint_runtime.hint_optimizer(raw_model.hint) if use_hints else None

    # learning rate decay scheduler (linear warmup and warmdown)
    def get_lr(it):
        assert it <= args.num_iterations
        # 1) linear warmup for warmup_iters steps
        if it < args.warmup_iters:
            return args.learning_rate * (it+1) / args.warmup_iters
        # 2) constant lr for a while
        elif it < args.num_iterations - args.warmdown_iters:
            return args.learning_rate
        # 3) linear warmdown
        else:
            decay_ratio = (args.num_iterations - it) / args.warmdown_iters
            return args.learning_rate * decay_ratio

    run_id = str(uuid.uuid4())

    # create the logging directory if it does not exist
    logfile = None
    if master_process and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        logfile = os.path.join(args.output_dir, "%s.log" % run_id)
        # create the log file "main.log" inside it, and wipe it clean
        with open(logfile, "w") as f:
            pass

    def log(line):
        if master_process and logfile is not None:
            with open(logfile, "a") as f:
                f.write(line + "\n")

    def batch(loader, train):
        """port: the next batch and its hint rows (train_span / val_span at the batch's (shard, start))"""
        x, y, shard, start = loader.next_batch()
        if not use_hints:
            return x, y, {}
        rows = hints.train_span(shard, start, B * T) if train else hints.val_span(start, B * T)
        hint = dict(hint_cell=rows.cell, hint_tokens=rows.tokens, hint_weights=rows.weights)
        if rows.counts is not None:  # port: HINT_NGRAM=1, the count rows for the n-gram output mixture
            hint["hint_counts"] = rows.counts
        return x, y, hint

    def train_step(step, x, y, hint_args, fetch_next):
        # --------------- TRAINING SECTION BEGIN -----------------
        model.train()
        # forward pass
        with ctx:
            _, loss = model(x, y, return_logits=False, **hint_args)
        # advance the dataset for the next batch
        nxt = batch(train_loader, True) if fetch_next else None
        # backward pass
        loss.backward()
        for p in record_params:  # port: the record's parameters only (the hint's Adam sees its raw averaged gradients)
            p.grad = p.grad / (p.grad.norm() + 1e-6)
        # determine and set the learning rate for this iteration
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if hint_opt is not None:
            # port: lr / args.learning_rate is the schedule factor; LR_SCALE scales the hint's lr too
            hint_runtime.set_lr_factor(hint_opt, LR_SCALE * get_lr(step) / args.learning_rate)
        # step the optimizer
        optimizer.step()
        if hint_opt is not None:
            hint_opt.step()
        optimizer.zero_grad(set_to_none=True)
        if hint_opt is not None:
            hint_opt.zero_grad(set_to_none=True)
        # --------------- TRAINING SECTION END -------------------
        return loss, lr, nxt

    def validate():
        # the record: no_grad, eval mode and no autocast (fp32); port: over val_tokens, averaged over the ranks (the
        # record printed rank 0's own 20 batches)
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss = torch.zeros((), device=device)
            for _ in range(val_steps):
                x_val, y_val, hint_val = batch(val_loader, False)
                _, loss = model(x_val, y_val, return_logits=False, **hint_val)
                val_loss += loss
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        model.train()
        return val_loss.item() / val_steps

    # port: compile off the clock, then restore the model, both optimizers and the loader (as r001)
    if COMPILE_WARMUP > 0:
        initial = {k: v.detach().clone() for k, v in raw_model.state_dict().items()}
        t_compile = time.perf_counter()
        x, y, hint_args = batch(train_loader, True)
        for step in range(COMPILE_WARMUP):
            _, _, (x, y, hint_args) = train_step(step, x, y, hint_args, fetch_next=True)
        torch.cuda.synchronize()
        print0(f"compile warmup: {COMPILE_WARMUP} steps in {time.perf_counter() - t_compile:.1f} s")
        raw_model.load_state_dict(initial)
        del initial
        for opt in (optimizer, hint_opt):
            if opt is not None:
                opt.state.clear()
        model.zero_grad(set_to_none=True)
        train_loader.reset()
        dist.barrier()

    # port: one clock over every training step, from the first batch's load and hint lookup to the last optimizer step;
    # stopped only for validation
    training_time_ms = 0.0
    val_loss = float("nan")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    x, y, hint_args = batch(train_loader, True)
    for step in range(args.num_iterations):
        last_step = step == args.num_iterations - 1
        loss, lr, nxt = train_step(step, x, y, hint_args, fetch_next=not last_step)
        if not last_step:
            x, y, hint_args = nxt
        done = step + 1
        validating = last_step or (args.val_loss_every > 0 and done % args.val_loss_every == 0)
        if done % LOG_EVERY == 0 or validating:
            # port: printed every LOG_EVERY steps (the record synced and printed every step)
            lossf = loss.item()  # syncs
            elapsed = training_time_ms + 1000 * (time.perf_counter() - t0)
            print0(f"step {done:4d}/{args.num_iterations} | train loss {lossf:.6f} | lr {lr:.2e} | "
                   f"train_time {elapsed:.0f}ms | step_avg {elapsed / done:.2f}ms")
            log("s:%d trl:%f" % (step, lossf))
        if validating:
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.perf_counter() - t0)
            val_loss = validate()  # off the clock
            print0(f"step {done:4d}/{args.num_iterations} | val loss {val_loss:.6f} | train_time {training_time_ms:.0f}ms "
                   f"| step_avg {training_time_ms / done:.2f}ms")
            log("s:%d tel:%f" % (done, val_loss))
            torch.cuda.synchronize()
            t0 = time.perf_counter()

    print0(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")
    # port: the record's checkpoint saves (every 5000 steps and final.pt) are dropped
    print0(f"FINAL val_loss:{val_loss:.4f} train_time_ms:{int(training_time_ms)} steps:{args.num_iterations}")

    def subset_losses():  # port: VAL_SUBSETS (after the timer), validate()'s batches per position with the eager model
        val_loader.reset()
        for _ in range(val_steps):
            start = val_loader.current_position  # the batch's offset in the validation shard
            x_val, y_val, hint_val = batch(val_loader, False)
            yield start, raw_model(x_val, y_val, return_logits=False, **hint_val)[1]

    hint_runtime.report_val_subsets(raw_model, subset_losses(), device, print0, val_loss)

    # -------------------------------------------------------------------------
    # clean up nice
    destroy_process_group()
