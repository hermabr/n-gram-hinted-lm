import os
import sys
import uuid
import glob
import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP
# port (old_records/README.md): record #367's exact-match hints, from old_records/ (on PYTHONPATH; added here as a fallback)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hint_runtime import Hints, HintRows, HintEmbedding, hint_optimizer, set_lr_factor, hints_enabled
import hint_runtime # port: VAL_SUBSETS (hint_runtime.report_val_subsets)

with open(sys.argv[0]) as f:
    code = f.read()

RECORD_STEPS = 7000 # the record's own --num_iterations (run.sh)
DEFAULT_STEPS = 176 # port: ~38 s of timed training on 8 H200 (PORT.md)
EMBED_STD = 1 / (3 * 768) ** 0.5 # port: the record's token embedding init: wte is tied to lm_head, whose nn.Linear init is U(+-1/sqrt(768)), std 0.0208

# -----------------------------------------------------------------------------
# OrthgonalNesterov optimizer

class OrthogonalNesterov(torch.optim.Optimizer):
    """
    Some warnings: This optimizer assumes that all parameters passed in are 2D.
    It shouldn't be used for the embedding layer, the final fully connected layer, or {0,1}-D
    parameters; those should be optimized by a standard method (e.g., AdamW).
    To use it with 4D convolutional filters, it works well to flatten their last 3 dimensions.
    """
    def __init__(self, params, lr=0.02, momentum=0.9, nesterov=True, zeropower_iters=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, zeropower_iters=zeropower_iters)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            for p in group['params']:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                state['steps'] = state.get('steps', 0) + 1
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group['nesterov'] else buf
                update = zeroth_power_via_newtonschulz5(g, steps=group['zeropower_iters'])
                p.data.add_(update, alpha=-lr)

@torch.compile
def zeroth_power_via_newtonschulz5(G, steps=5, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. It turns out
    to be empirically effective to keep increasing the slope of the quintic at zero even beyond the
    point where it no longer converges to one everywhere after repeated application (so long as it
    stays relatively close to 1 across the interval). Our usage of a Newton-Schulz iteration as the
    orthogonalization method traces to Bernstein & Newhouse (2024) https://arxiv.org/abs/2409.20325
    who suggested its use for computing the preconditioners of Shampoo.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16() / (G.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)

class CombinedOptimizer:

    def __init__(self, optimizers):
        assert all(len(opt.param_groups) == 1 for opt in optimizers)
        self.optimizers = optimizers
        self.param_groups = [pg for opt in self.optimizers for pg in opt.param_groups]
        self.base_lrs = [opt.param_groups[0]['lr'] for opt in self.optimizers]

    def step(self):
        for opt in self.optimizers:
            opt.step()

    def zero_grad(self, **kwargs):
        for opt in self.optimizers:
            opt.zero_grad(**kwargs)

    def scale_lrs(self, lr_scale):
        for base_lr, opt in zip(self.base_lrs, self.optimizers):
            opt.param_groups[0]['lr'] = base_lr * lr_scale

    def state_dict(self):
        return [opt.state_dict() for opt in self.optimizers]

    def load_state_dict(self, state_dicts): # port: for the restore after the compile warmup
        for opt, state in zip(self.optimizers, state_dicts):
            opt.load_state_dict(state)

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
        self.attn_scale = (1 / (2 * config.n_layer)**0.5)

    def forward(self, x):
        x = x + self.attn_scale * self.attn(rmsnorm(x))
        x = x + self.mlp(rmsnorm(x))
        return x

# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass
class GPTConfig:
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying
        # port: record #367's hint (old_records/hint_runtime.py), added at three sites in forward
        self.hint_embedding: HintEmbedding | None = None
        self.hint_mid = -1 # site 1 is after this many blocks (set with the hint)
        self.per_position = False # port: VAL_SUBSETS, the loss per position (eager only, hint_runtime.report_val_subsets)

    def forward(self, idx, targets=None, return_logits=True, hint_cell=None, hint_tokens=None, hint_weights=None,
                hint_counts=None):
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device) # shape (t)

        # forward the GPT model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        hint = None
        if self.hint_embedding is not None:
            # port: site 0, the hint added to the token embedding (the input of block 0); row i of the rank's B*T rows is
            # for flattened input position i, so it is viewed as (b, t, n_embd). Cast to the stream's dtype (float32:
            # the parameters are float32, under bf16 autocast in training and without autocast in validation)
            rows = HintRows(cell=hint_cell, tokens=hint_tokens, weights=hint_weights, counts=hint_counts)
            hint = self.hint_embedding(self.transformer.wte.weight, rows).view_as(x).type_as(x)
            x = x + self.hint_embedding.site_scale(0).type_as(x) * hint

        for i, block in enumerate(self.transformer.h):
            if hint is not None and i == self.hint_mid:
                # port: site 1, the residual stream after the first hint_mid blocks (7 of 12: ~0.6 of the depth)
                x = x + self.hint_embedding.site_scale(1).type_as(x) * hint
            x = block(x)
        if hint is not None:
            # port: site 2, right before the final norm and head
            x = x + self.hint_embedding.site_scale(2).type_as(x) * hint
        x = rmsnorm(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            logits = logits.float() # use tf32/fp32 for logits
            if hint is None or not (self.hint_embedding.mix or self.hint_embedding.ngram):
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1,
                                       **({"reduction": "none"} if self.per_position else {}))
            elif self.per_position:
                nll = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none")
                loss = self.hint_embedding.mixed_nll(nll, targets.view(-1), rows)
            else:
                # port: HINT_NGRAM=1 (or HINT_MIX=1): the loss of the mixture of the model's distribution (these fp32
                # logits, the record's own loss's) with the full-count n-gram (hint_runtime.NgramMix), per position
                # from the full-vocabulary cross-entropy, then the mean; training and validation alike
                nll = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none")
                loss = self.hint_embedding.mixed_nll(nll, targets.view(-1), rows).mean()
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            logits = logits.float() # use tf32/fp32 for logits
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        optimizer = CombinedOptimizer([
            torch.optim.AdamW(self.lm_head.parameters(), lr=learning_rate, betas=betas, weight_decay=0),
            OrthogonalNesterov(self.transformer.h.parameters(), lr=10 * learning_rate, momentum=0.95)
        ])
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
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens

class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes, hints=None, split="train"):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T
        # port: with hints, next_batch also looks up the rows of its input positions (a contiguous run of B*T tokens of
        # the current shard: hints.train_span for the training shards, hints.val_span for the validation shard)
        self.hints = hints
        self.split = split

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += int(shard_ntok)
        self.ntok_total = ntok_total

        # kick things off
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self): # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        # port: the inputs are tokens [start, start + B*T) of shard `shard` (0-based index in the sorted files); their
        # hint rows are looked up and queued for upload first, before the pageable .cuda() copies below, which can make
        # the host wait for the GPU, so the lookup overlaps the GPU's work on the current forward pass
        shard, start = self.current_shard, self.current_position
        hint = {}
        if self.hints is not None:
            rows = (self.hints.val_span(start, B * T) if self.split == "val"
                    else self.hints.train_span(shard, start, B * T))
            hint = dict(hint_cell=rows.cell, hint_tokens=rows.tokens, hint_weights=rows.weights)
            if rows.counts is not None: # port: HINT_NGRAM=1, the count rows for the n-gram output mixture
                hint["hint_counts"] = rows.counts
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance current position and load next shard if necessary
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x.cuda(), y.cuda(), hint

def loader_plan(files, B, T, world, accumulation, steps):
    """port: HINT_ONLINE, every rank's DistributedDataLoader batches from its reset, (step, shard, start, B * T), for the
    `steps` steps of `accumulation` batches each and the batches loaded after them (step `steps`), as next_batch moves:
    each rank advances to the next shard on its own position (so ranks may switch a step apart)."""
    lens = [int(_peek_data_shard(f)) for f in files]
    plan = []
    for rank in range(world):
        shard, position = 0, rank * B * T
        for b in range((steps + 1) * accumulation):
            plan.append((b // accumulation, shard, position, B * T))
            position += B * T * world
            if position + (B * T * world + 1) > lens[shard]:
                shard, position = (shard + 1) % len(files), rank * B * T
    return plan

# -----------------------------------------------------------------------------
# int main

def print0(*args, **kwargs):
    # modified print that only prints from the master process
    # if this is not a distributed run, it's just a print
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)

if __name__ == "__main__":
    import time
    import argparse
    print0(f"Running pytorch {torch.version.__version__}")

    # port: the defaults are the record's run.sh (upstream d4bc211), so the script runs without arguments; the data is
    # FineWeb-100B under $DATA_PATH, validated on the first val_tokens of its validation shard
    data_dir = os.path.join(os.environ.get("DATA_PATH", "."), "data/fineweb100B")
    parser = argparse.ArgumentParser()
    # file system input / output
    parser.add_argument("--input_bin", type=str, default=os.path.join(data_dir, "fineweb_train_*.bin"), help="input .bin to train on")
    parser.add_argument("--input_val_bin", type=str, default=os.path.join(data_dir, "fineweb_val_000000.bin"), help="input .bin to eval validation loss on")
    parser.add_argument("--model", type=str, default="d12", help="d12|d24|d36|d48")
    # token layout for each step of the optimization
    parser.add_argument("--batch_size", type=int, default=int(os.environ.get("REC_DEVICE_BATCH", 64)), help="batch size, in units of #batch dimensions")
    parser.add_argument("--accumulation", type=int, default=None, help="port: default 512 // (batch_size * world_size), the record's 8 x 64 sequences per step")
    parser.add_argument("--sequence_length", type=int, default=1024, help="sequence length")
    # workload (number of steps)
    parser.add_argument("--num_iterations", type=int, default=int(os.environ.get("TRAIN_STEPS", DEFAULT_STEPS)), help="number of iterations to run")
    # optimization
    parser.add_argument("--learning_rate", type=float, default=0.0018, help="learning rate warmup iterations")
    parser.add_argument("--warmup_iters", type=int, default=250, help="learning rate warmup iterations")
    parser.add_argument("--warmdown_iters", type=int, default=2000, help="learning rate warmdown iterations")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="weight decay") # (unused by configure_optimizers, as in the record)
    # evaluation
    parser.add_argument("--val_loss_every", type=int, default=int(os.environ.get("REC_VAL_EVERY", 128)), help="every how many steps to evaluate val loss?")
    parser.add_argument("--val_tokens", type=int, default=10485760, help="port: the record's 20 val batches of 8 x 64 x 1024")
    parser.add_argument("--save_every", type=int, default=0, help="every how many steps to save the checkpoint")
    args = parser.parse_args()

    # port: the budget knobs (old_records/README.md item 4). Every schedule length is the record's fraction of its own
    # 7000 steps: the lr warmup (250) and warmdown (2000). The record has no momentum warmup and no window or
    # batch-size schedule. LR_SCALE multiplies every lr: the record's two (AdamW lr, OrthogonalNesterov 10 x lr) and
    # the hint's.
    STEP_SCALE = args.num_iterations / RECORD_STEPS
    args.warmup_iters = round(float(os.environ.get("REC_WARMUP_FRAC", args.warmup_iters / RECORD_STEPS)) * args.num_iterations)
    args.warmdown_iters = max(1, round(float(os.environ.get("REC_WARMDOWN_FRAC", args.warmdown_iters / RECORD_STEPS)) * args.num_iterations))
    LR_SCALE = float(os.environ.get("LR_SCALE", 1.0))
    args.learning_rate *= LR_SCALE
    SEED = int(os.environ.get("SEED", 0))
    HINTS = hints_enabled()
    # port: HINT_ONLINE=1, the hint rows computed in the run, in parallel with training (old_records/online_hints.py); the
    # validation index is built while training runs, so only the final validation has rows
    HINT_ONLINE = HINTS and os.environ.get("HINT_ONLINE", "0") == "1"
    if HINT_ONLINE:
        args.val_loss_every = 0
    HINT_LR = float(os.environ.get("REC_HINT_LR", 0.008)) # the hint's Adam lr (#367's), scales at 10x; times LR_SCALE
    HINT_MID = int(os.environ.get("REC_HINT_MID", 7)) # site 1 after this many of the 12 blocks (7/12 ~ 0.6 of the depth)
    COMPILE_WARMUP = int(os.environ.get("REC_COMPILE_WARMUP", 10)) # compile steps before the clock, state restored after
    SAVE_CHECKPOINT = os.environ.get("REC_SAVE_CHECKPOINT", "0") == "1" # port: the record saved the final state; off by default

    # args error checking and convenience variables
    B, T = args.batch_size, args.sequence_length
    assert args.model in {"d12", "d24", "d36", "d48"}

    # set up DDP (distributed data parallel). torchrun sets this env variable
    assert torch.cuda.is_available()
    dist.init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    print(f"using device: {device}")
    master_process = (ddp_rank == 0) # this process will do logging, checkpointing etc.

    # port: the record's batch per step is 8 GPUs x 64 sequences x 1024, no accumulation; with fewer GPUs or a smaller
    # REC_DEVICE_BATCH the record's own --accumulation keeps 512 sequences per step
    if args.accumulation is None:
        assert 512 % (B * ddp_world_size) == 0, f"512 sequences per step do not split into {ddp_world_size} x {B}"
        args.accumulation = 512 // (B * ddp_world_size)
    # port: validation covers val_tokens whatever the GPU count and batch (the record: 20 steps of 8 x 64 x 1024)
    assert args.val_tokens % (B * T * ddp_world_size) == 0
    val_steps = args.val_tokens // (B * T * ddp_world_size)
    print0(f"port: TRAIN_STEPS={args.num_iterations} STEP_SCALE={STEP_SCALE:.4f} warmup={args.warmup_iters} "
           f"warmdown={args.warmdown_iters} LR_SCALE={LR_SCALE} lr={args.learning_rate:g} SEED={SEED} "
           f"HINTS={int(HINTS)} HINT_DIR={os.environ.get('HINT_DIR')} hint_lr={HINT_LR} hint_mid={HINT_MID} "
           f"compile_warmup={COMPILE_WARMUP} B={B} T={T} accumulation={args.accumulation} val_steps={val_steps} "
           f"world_size={ddp_world_size}")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # port: the hint rows (looked up on the host, uploaded from pinned memory by hint_runtime)
    if HINT_ONLINE:
        from online_hints import OnlineHints
        train_files = sorted(glob.glob(args.input_bin))
        hints = OnlineHints(loader_plan(train_files, B, T, ddp_world_size, args.accumulation, args.num_iterations),
                            args.num_iterations, train_files, args.input_val_bin, device=device)
    else:
        hints = Hints(device=device) if HINTS else None

    # load tokens
    train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size, hints=hints, split="train")
    print0(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
    val_loader = DistributedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size, hints=hints, split="val")
    print0(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")
    # port: the record loaded its first batch here; the port loads it after the compile warmup, on the clock

    # init the model from scratch
    num_vocab = 50257
    model_config = {
        "d12": GPTConfig(vocab_size=num_vocab, n_layer=12, n_head=12, n_embd=768),
        "d24": GPTConfig(vocab_size=num_vocab, n_layer=24, n_head=16, n_embd=1024),
        "d36": GPTConfig(vocab_size=num_vocab, n_layer=36, n_head=20, n_embd=1280),
        "d48": GPTConfig(vocab_size=num_vocab, n_layer=48, n_head=25, n_embd=1600),
    }[args.model]
    model = GPT(model_config)
    model = model.cuda()
    # port: fill each Rotary's cos/sin cache for T eagerly, with the record's own code, before compiling. In the record
    # the first compiled call set these plain attributes, and the next call recompiled once the cache was set (under
    # fullgraph=True and dynamic=False the cache must exist before tracing). The values are the fp32 cache that every
    # later forward of the record read
    with torch.no_grad():
        for block in model.transformer.h:
            block.attn.rotary(torch.empty(1, T, 1, block.attn.head_dim, device=device))
    if HINTS:
        # port: the hint is a submodule (float32 parameters), so torch.compile compiles it with the model and DDP averages
        # its gradients; its candidate mean is rescaled from this record's embedding (std 0.0208) to #367's (0.005)
        assert 1 <= HINT_MID < len(model.transformer.h), f"REC_HINT_MID={HINT_MID}: site 1 must be between blocks"
        # port: HINT_NGRAM=1 (the count rows staged): the n-gram output mixture (hint_runtime.NgramMix) over their levels
        ngram = dict(ngram_levels=len(hints.count_levels), unigram=hints.unigram) if hints.ngram else {}
        if hints.ngram:
            print0(f"port: n-gram mixing (HINT_NGRAM) over levels {hints.count_levels}")
        model.hint_embedding = HintEmbedding(model_config.n_embd, hints.cells, embed_std=EMBED_STD, **ngram).cuda()
        model.hint_mid = HINT_MID
        print0(f"port: hint embed_ratio={model.hint_embedding.embed_ratio:.4f} site_scales={model.hint_embedding.site_scales.tolist()}")
    # port: record #360's compile flags, as every port: torch.compile(model, dynamic=False, fullgraph=True) with the
    # default inductor config (the record set config.coordinate_descent_tuning = True; not here)
    print0("compiling the model...")
    model = torch.compile(model, dynamic=False, fullgraph=True)

    # here we wrap model into DDP container
    model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module # always contains the "raw" unwrapped model

    # set up a context manager following the desired dtype and device
    ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

    # init the optimizer (port: args.learning_rate is the record's 0.0018 times LR_SCALE, so both optimizers are scaled)
    optimizer = raw_model.configure_optimizers(weight_decay=args.weight_decay,
                                               learning_rate=args.learning_rate, betas=(0.9, 0.95))
    hint_opt = None
    if HINTS:
        # port: #367's Adam for the hint (scales at 10x, no wd), its lr following the record's schedule (set_lr_factor)
        hint_opt = hint_optimizer(raw_model.hint_embedding, lr=HINT_LR * LR_SCALE)

    # learning rate decay scheduler (linear warmup and warmdown)
    def get_lr(it):
        assert it <= args.num_iterations
        # 1) linear warmup for warmup_iters steps
        if it < args.warmup_iters:
            return (it+1) / args.warmup_iters
        # 2) constant lr for a while
        elif it < args.num_iterations - args.warmdown_iters:
            return 1.0
        # 3) linear warmdown
        else:
            decay_ratio = (args.num_iterations - it) / args.warmdown_iters
            return decay_ratio

    run_id = str(uuid.uuid4())
    logfile = None
    if master_process:
        os.makedirs('logs/%s' % run_id, exist_ok=True)
        logfile = 'logs/%s/log.txt' % run_id
        # create the empty log file
        with open(logfile, "w") as f:
            pass

    def train_step(step):
        """The record's training section: the accumulation steps (each loads the next batch after its forward pass,
        as the record's loop does), the lr schedule and the optimizer step. Returns the last micro-batch's loss."""
        global x, y, hint
        for _ in range(args.accumulation):
            # forward pass
            with ctx:
                _, loss = model(x, y, return_logits=False, **hint)
                train_loss = loss.detach()
            # advance the dataset for the next batch (port: with its hint rows)
            x, y, hint = train_loader.next_batch()
            # backward pass
            loss.backward()
        for p in model.parameters():
            p.grad /= args.accumulation
        # determine and set the learning rate for this iteration
        lr_scale = get_lr(step)
        optimizer.scale_lrs(lr_scale)
        if hint_opt is not None:
            set_lr_factor(hint_opt, lr_scale)
        # step the optimizer
        optimizer.step()
        if hint_opt is not None:
            hint_opt.step()
        optimizer.zero_grad(set_to_none=True)
        if hint_opt is not None:
            hint_opt.zero_grad(set_to_none=True)
        return train_loss, lr_scale

    # port: the record timed each step separately, its first one including the compile. Here COMPILE_WARMUP steps run
    # first, on the loader's first batches (with their hint rows); then the model, the optimizers (states and lrs) and
    # the loader are restored to their initial states, and the clock covers every one of the num_iterations steps
    initial_state = dict(model=copy.deepcopy(raw_model.state_dict()), optimizer=copy.deepcopy(optimizer.state_dict()),
                         hint_opt=copy.deepcopy(hint_opt.state_dict()) if hint_opt is not None else None)
    model.train()
    x, y, hint = train_loader.next_batch()
    t_warmup = time.time()
    for warmup_step in range(COMPILE_WARMUP):
        train_step(warmup_step)
    torch.cuda.synchronize()
    print0(f"port: compile warmup: {COMPILE_WARMUP} steps in {time.time() - t_warmup:.1f} s")
    raw_model.load_state_dict(initial_state["model"])
    optimizer.load_state_dict(initial_state["optimizer"])
    if hint_opt is not None:
        hint_opt.load_state_dict(initial_state["hint_opt"])
    del initial_state
    model.zero_grad(set_to_none=True)
    train_loader.reset()
    torch.cuda.synchronize()
    dist.barrier()

    training_time_ms = 0.0
    # start the clock (port: it runs through every training step, stopped only for validation and the checkpoint)
    torch.cuda.synchronize()
    t_clock = time.time()
    if HINT_ONLINE: # port: rank 0 starts the online hint work (and every rank the n-gram's unigram init)
        hints.start(raw_model.hint_embedding.ngram_mix if hints.ngram else None)
    x, y, hint = train_loader.next_batch()
    for step in range(args.num_iterations + 1):
        last_step = (step == args.num_iterations)

        # once in a while evaluate the validation dataset
        if (last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)):
            if HINT_ONLINE and last_step: # port: the wait for the online index builds, on the clock
                hints.finish()
            # port: stop the clock
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.time() - t_clock)
            model.eval()
            val_loader.reset()
            val_loss = 0.0
            for _ in range(val_steps):
                with torch.no_grad(): # I want to use ctx here but it causes a torch.compile error
                    x_val, y_val, hint_val = val_loader.next_batch()
                    _, loss = model(x_val, y_val, return_logits=False, **hint_val)
                    val_loss += loss
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
            val_loss /= val_steps
            # log val loss to console and to logfile
            print0(f"val loss {val_loss} (step {step}, train_time {training_time_ms:.0f} ms)")
            if master_process and logfile is not None:
                with open(logfile, "a") as f:
                    f.write("s:%d tel:%f\n" % (step, val_loss))
            # port: start the clock again
            torch.cuda.synchronize()
            t_clock = time.time()

        # save the state of the training process
        if master_process and SAVE_CHECKPOINT and (last_step or (args.save_every > 0 and step % args.save_every == 0)):
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.time() - t_clock)
            log = dict(step=step, args=args.__dict__, code=code, model=raw_model.state_dict(), optimizer=optimizer.state_dict())
            torch.save(log, 'logs/%s/state_step%06d.pt' % (run_id, step))
            torch.cuda.synchronize()
            t_clock = time.time()

        # bit confusing: we want to make sure to eval on 0th iteration
        # but also after the very last iteration. so we loop for step <= num_iterations
        # instead of just < num_iterations (one extra due to <=), only to do
        # the validation/sampling one last time, and then we break right here as we're done.
        if last_step:
            # port: the one result line (old_records/README.md item 6), after the final validation over all val_tokens
            print0(f"FINAL val_loss:{val_loss.item():.4f} train_time_ms:{int(training_time_ms)} steps:{args.num_iterations}")
            if HINT_ONLINE:
                print0(hints.summary())
            # port: VAL_SUBSETS (after the timer), the final validation's batches per position with the eager model
            def subset_losses():
                val_loader.reset()
                for _ in range(val_steps):
                    start = val_loader.current_position # the batch's offset in the validation shard
                    x_val, y_val, hint_val = val_loader.next_batch()
                    yield start, eager_model(x_val, y_val, return_logits=False, **hint_val)[1]
            eager_model = getattr(raw_model, "_orig_mod", raw_model)
            hint_runtime.report_val_subsets(eager_model, subset_losses(), torch.device("cuda", ddp_local_rank), print0,
                                            val_loss.item())
            break

        torch.cuda.synchronize()
        t0 = time.time()
        # --------------- TRAINING SECTION BEGIN -----------------
        model.train()
        train_loss, lr_scale = train_step(step)
        # --------------- TRAINING SECTION END -------------------
        # everything that follows now is just diagnostics, prints, logging, etc.
        # (port: on the clock, as all of the record's loop is)
        torch.cuda.synchronize()
        t1 = time.time()

        dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)
        tokens_per_second = args.accumulation * ddp_world_size * B * T / (t1 - t0)
        print0(f"step {step+1:4d}/{args.num_iterations} | train loss {train_loss.item():.4f} | lr_scale {lr_scale:.2e} | ({(t1-t0)*1000:.2f} ms | {tokens_per_second:.0f} tok/s)")
        # log training loss to logfile
        if master_process:
            with open(logfile, "a") as f:
                f.write("s:%d trl:%f\n" % (step, train_loss.item()))

    print0(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

    # -------------------------------------------------------------------------
    # clean up nice
    dist.destroy_process_group()
