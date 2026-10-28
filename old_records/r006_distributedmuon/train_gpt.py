import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import glob
import time
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
from hint_runtime import Hints, HintRows, HintEmbedding, hint_optimizer, hints_enabled
import hint_runtime # port: VAL_SUBSETS (hint_runtime.report_val_subsets)

RECORD_STEPS = 5100 # the record's own num_iterations
DEFAULT_STEPS = 245 # port: ~38 s of timed training on 8 H200 (PORT.md)
EMBED_STD = 1 / (3 * 768) ** 0.5 # port: the record's token embedding init: wte is tied to lm_head, whose nn.Linear init is U(+-1/sqrt(768)), std 0.0208

# -----------------------------------------------------------------------------
# Muon optimizer

def zeropower_via_svd(G, steps=None):
    U, S, V = G.svd()
    return U @ V.T

@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X

zeropower_backends = dict(svd=zeropower_via_svd, newtonschulz5=zeropower_via_newtonschulz5)

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer assumes that all parameters passed in are 2D.
    - It should not be used for the embedding layer, the final fully connected layer, or any {0,1}-D
    parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
    - We believe it is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.
    - We have not yet tried this optimizer for training scenarios larger than NanoGPT (124M).

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        backend: The chosen backend for the orthogonalization step. (recommended: 'newtonschulz5')
        backend_steps: The number of iteration steps to use in the backend, if it is iterative.
    """
    def __init__(self, params, lr=3e-4, momentum=0.95, nesterov=True,
                 backend='newtonschulz5', backend_steps=5,
                 rank=0, world_size=1):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, backend=backend, backend_steps=backend_steps)
        super().__init__(params, defaults)
        self.rank = rank
        self.world_size = world_size

    def step(self):

        for group in self.param_groups:

            lr = group['lr']
            momentum = group['momentum']
            zeropower_backend = zeropower_backends[group['backend']]

            # generate weight updates in distributed fashion
            total_params = sum(p.numel() for p in group['params'])
            updates_flat = torch.zeros(total_params, device='cuda', dtype=torch.bfloat16)
            curr_idx = 0
            for i, p in enumerate(group['params']):
                # luckily this will perfectly distribute a transformer with multiple of 4 layers to 8 GPUs
                if i % self.world_size == self.rank:
                    g = p.grad
                    if g is None:
                        continue
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    if group['nesterov']:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_backend(g, steps=group['backend_steps'])
                    g *= max(g.size(0), g.size(1))**0.5 # scale to have update.square().mean() == 1
                    updates_flat[curr_idx:curr_idx+p.numel()] = g.flatten()
                curr_idx += p.numel()

            # sync updates across devices. we are not memory-constrained so can do this simple deserialization
            dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            # deserialize and apply updates
            curr_idx = 0
            for p in group['params']:
                g = updates_flat[curr_idx:curr_idx+p.numel()].view_as(p.data).type_as(p.data)
                p.data.add_(g, alpha=-lr)
                curr_idx += p.numel()

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model

class Rotary(torch.nn.Module):

    def __init__(self, dim, base=10000):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos().bfloat16()
            self.sin_cached = freqs.sin().bfloat16()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4 # multihead attention
    d = x.shape[3]//2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),)) # QK norm suggested by @Grad62304977
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y = y.transpose(1, 2).contiguous().view_as(x) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(F.rms_norm(x, (x.size(-1),)))
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x

# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass
class GPTConfig:
    vocab_size : int = 50304
    n_layer : int = 12
    n_head : int = 6 # head dim 128 suggested by @Grad62304977
    n_embd : int = 768

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

    def forward(self, idx, targets=None, return_logits=True, hint_cell=None, hint_tokens=None, hint_weights=None):

        # forward the GPT model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        hint = None
        if self.hint_embedding is not None:
            # port: site 0, the hint added to the token embedding (the input of block 0); row i of the rank's B*T rows is
            # for flattened input position i, so it is viewed as (b, t, n_embd). Cast to the stream's dtype (float32:
            # the parameters are float32 under bf16 autocast, so the residual stream is float32)
            rows = HintRows(cell=hint_cell, tokens=hint_tokens, weights=hint_weights)
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
        x = F.rms_norm(x, (x.size(-1),))

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            logits = logits.float() # use tf32/fp32 for logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1,
                                   **({"reduction": "none"} if self.per_position else {}))
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            logits = logits.float() # use tf32/fp32 for logits
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss

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
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance current position and load next shard if necessary
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x.cuda(), y.cuda(), hint

# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    # data hyperparams (port: FineWeb-100B under $DATA_PATH; validation is the first val_tokens of its validation shard)
    input_bin : str = os.path.join(os.environ.get("DATA_PATH", "."), "data/fineweb100B/fineweb_train_*.bin") # input .bin to train on
    input_val_bin : str = os.path.join(os.environ.get("DATA_PATH", "."), "data/fineweb100B/fineweb_val_000000.bin") # input .bin to eval validation loss on
    # optimization hyperparams
    batch_size : int = 8*64 # batch size, in sequences, across all devices
    device_batch_size : int = int(os.environ.get("REC_DEVICE_BATCH", 64)) # batch size, in sequences, per device (port: REC_DEVICE_BATCH, for smoke tests on small GPUs; the global batch stays 8*64 by accumulation)
    sequence_length : int = 1024 # sequence length, in tokens
    num_iterations : int = int(os.environ.get("TRAIN_STEPS", DEFAULT_STEPS)) # number of iterations to run (port: TRAIN_STEPS; the record: 5100)
    learning_rate : float = 0.0036
    warmup_iters : int = 0
    warmdown_iters : int = 1450 # number of iterations of linear warmup/warmdown for triangular or trapezoidal schedule (port: scaled below)
    weight_decay : float = 0
    # evaluation and logging hyperparams
    val_loss_every : int = int(os.environ.get("REC_VAL_EVERY", 125)) # every how many steps to evaluate val loss? 0 for only at the end
    val_tokens : int = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    save_every : int = 0 # every how many steps to save the checkpoint? 0 for only at the end
args = Hyperparameters()

# port: the budget knobs (old_records/README.md item 4). Every schedule length is the record's fraction of its own 5100
# steps: the lr warmup (0 steps) and warmdown (1450). The record has no momentum warmup and no window or batch-size
# schedule. LR_SCALE multiplies every lr: the record's two optimizers' (AdamW 0.0036 on the tied head/embedding, Muon
# 0.1 * 0.0036 on the blocks) and the hint's.
STEP_SCALE = args.num_iterations / RECORD_STEPS
args.warmup_iters = round(args.warmup_iters * STEP_SCALE)
args.warmdown_iters = max(1, round(float(os.environ.get("REC_WARMDOWN_FRAC", args.warmdown_iters / RECORD_STEPS)) * args.num_iterations))
LR_SCALE = float(os.environ.get("LR_SCALE", 1.0))
args.learning_rate *= LR_SCALE
SEED = int(os.environ.get("SEED", 0))
HINTS = hints_enabled()
HINT_LR = float(os.environ.get("REC_HINT_LR", 0.008)) # the hint's Adam lr (#367's), scales at 10x; times LR_SCALE
HINT_MID = int(os.environ.get("REC_HINT_MID", 7)) # site 1 after this many of the 12 blocks (7/12 ~ 0.6 of the depth)
COMPILE_WARMUP = int(os.environ.get("REC_COMPILE_WARMUP", 10)) # compile steps before the clock, state restored after
SAVE_CHECKPOINT = os.environ.get("REC_SAVE_CHECKPOINT", "0") == "1" # port: the record saved the final state; off by default

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

# begin logging
logfile = None
if master_process:
    run_id = str(uuid.uuid4())
    logdir = 'logs/%s/' % run_id
    os.makedirs(logdir, exist_ok=True)
    logfile = 'logs/%s.txt' % run_id
    # create the log file
    with open(logfile, "w") as f:
        # begin the log by printing this file (the Python code)
        f.write('='*100 + '\n')
        f.write(code)
        f.write('='*100 + '\n')
def print0(s, logonly=False):
    if master_process:
        with open(logfile, "a") as f:
            if not logonly:
                print(s)
            f.write(s+'\n')
# log information about the hardware/software environment this is running on
# and print the full `nvidia-smi` to file
print0(f"Running pytorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}\nnvidia-smi:")
import subprocess
result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
print0(f'{result.stdout}', logonly=True)
print0('='*100, logonly=True)
print0(f"port: TRAIN_STEPS={args.num_iterations} STEP_SCALE={STEP_SCALE:.4f} warmup={args.warmup_iters} "
       f"warmdown={args.warmdown_iters} LR_SCALE={LR_SCALE} lr={args.learning_rate:g} SEED={SEED} "
       f"HINTS={int(HINTS)} HINT_DIR={os.environ.get('HINT_DIR')} hint_lr={HINT_LR} hint_mid={HINT_MID} "
       f"compile_warmup={COMPILE_WARMUP} device_batch={args.device_batch_size} world_size={ddp_world_size}")
torch.manual_seed(SEED)
np.random.seed(SEED)

# convenience variables
B, T = args.device_batch_size, args.sequence_length
# calculate the number of steps to take in the val loop.
assert args.val_tokens % (B * T * ddp_world_size) == 0
val_steps = args.val_tokens // (B * T * ddp_world_size)
# calculate the steps of gradient accumulation required to attain the desired global batch size.
assert args.batch_size % (B * ddp_world_size) == 0
train_accumulation_steps = args.batch_size // (B * ddp_world_size)
print0(f"port: B={B} T={T} train_accumulation_steps={train_accumulation_steps} val_steps={val_steps}")

# port: the hint rows (looked up on the host, uploaded from pinned memory by hint_runtime)
hints = Hints(device=device) if HINTS else None

# load tokens
train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size, hints=hints, split="train")
val_loader = DistributedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size, hints=hints, split="val")
print0(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
print0(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")
print0('='*100, logonly=True)

# there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency. suggested to me by @Grad62304977.
# this originates from Karpathy's experiments.
num_vocab = 50304
model = GPT(GPTConfig(vocab_size=num_vocab, n_layer=12, n_head=6, n_embd=768))
model = model.cuda()
# port: fill each Rotary's cos/sin cache for T eagerly, with the record's own code, before compiling. In the record the
# first compiled forward filled it, computing the frequencies on the CPU (inv_freq is a plain CPU attribute, and
# type_as moves the positions there), so the first graph would hold CPU ops and the next call would recompile once the
# cache is set. The values are the same cache that every later forward of the record read
with torch.no_grad():
    for block in model.transformer.h:
        block.attn.rotary(torch.empty(1, T, 1, block.attn.head_dim, device=device, dtype=torch.bfloat16))
if HINTS:
    # port: the hint is a submodule (float32 parameters), so torch.compile compiles it with the model and DDP averages
    # its gradients; its candidate mean is rescaled from this record's embedding (std 0.0208) to #367's (0.005)
    assert 1 <= HINT_MID < len(model.transformer.h), f"REC_HINT_MID={HINT_MID}: site 1 must be between blocks"
    model.hint_embedding = HintEmbedding(768, hints.cells, embed_std=EMBED_STD).cuda()
    model.hint_mid = HINT_MID
    print0(f"port: hint embed_ratio={model.hint_embedding.embed_ratio:.4f} site_scales={model.hint_embedding.site_scales.tolist()}")
# port: record #360's compile flags, as every port: torch.compile(model, dynamic=False, fullgraph=True) with the default
# inductor config (the record set coordinate_descent_tuning = True; not here)
model = torch.compile(model, dynamic=False, fullgraph=True)
# here we wrap model into DDP container
model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module # always contains the "raw" unwrapped model
ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

# init the optimizer(s) (port: args.learning_rate is the record's 0.0036 times LR_SCALE, so both are scaled)
optimizer1 = torch.optim.AdamW(raw_model.lm_head.parameters(), lr=args.learning_rate, betas=(0.9, 0.95),
                               weight_decay=args.weight_decay, fused=True)
optimizer2 = Muon(raw_model.transformer.h.parameters(), lr=0.1*args.learning_rate, momentum=0.95,
                  rank=ddp_rank, world_size=ddp_world_size)
optimizers = [optimizer1, optimizer2]
if HINTS:
    # port: #367's Adam for the hint (scales at 10x, no wd); a LambdaLR with the record's get_lr below, like the record's
    # optimizers, is hint_runtime.set_lr_factor(get_lr(step)) every step
    optimizers.append(hint_optimizer(raw_model.hint_embedding, lr=HINT_LR * LR_SCALE))
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
schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

def train_step():
    """The record's training section: the accumulation steps (each loads the next batch after its forward pass, as
    the record's loop does) and the optimizer steps. Returns the last micro-batch's loss."""
    global x, y, hint
    for i in range(1, train_accumulation_steps+1):
        # forward pass
        with ctx:
            _, loss = model(x, y, return_logits=False, **hint)
            train_loss = loss.detach()
        # advance the dataset for the next batch (port: with its hint rows)
        x, y, hint = train_loader.next_batch()
        # backward pass
        if i < train_accumulation_steps:
            with model.no_sync(): # there's no need to sync gradients every accumulation step
                loss.backward()
        else:
            loss.backward() # just sync on the last step
    for p in model.parameters():
        p.grad /= train_accumulation_steps
    # step the optimizers
    for opt in optimizers:
        opt.step()
    # null the gradients
    model.zero_grad(set_to_none=True)
    return train_loss

# port: the record timed from step 10 on ("the first 10 steps are slower"); here COMPILE_WARMUP steps run first, on the
# real first batches (with their hint rows), and then the model, the optimizers (states and lrs) and the loader are
# restored to their initial states, the alternative the record's own comment describes. The schedulers are not
# stepped in the warmup. The clock then covers every one of the num_iterations training steps.
initial_state = dict(model=copy.deepcopy(raw_model.state_dict()),
                     optimizers=[copy.deepcopy(opt.state_dict()) for opt in optimizers])
model.train()
x, y, hint = train_loader.next_batch()
t_warmup = time.time()
for warmup_step in range(COMPILE_WARMUP):
    train_step()
torch.cuda.synchronize()
print0(f"port: compile warmup: {COMPILE_WARMUP} steps in {time.time() - t_warmup:.1f} s")
raw_model.load_state_dict(initial_state["model"])
for opt, opt_state in zip(optimizers, initial_state["optimizers"]):
    opt.load_state_dict(opt_state)
del initial_state
model.zero_grad(set_to_none=True)
train_loader.reset()
torch.cuda.synchronize()
dist.barrier()

training_time_ms = 0
# start the clock
torch.cuda.synchronize()
t0 = time.time()
# port: the first batch's load (and hint lookup) on the clock. (The record loaded it before the clock and then reset
# the loader, so it trained on its first batch twice, at steps 0 and 1; here every step has its own batch.)
x, y, hint = train_loader.next_batch()
# begin training
for step in range(args.num_iterations + 1):
    last_step = (step == args.num_iterations)

    # once in a while evaluate the validation dataset
    if (last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        # run validation batches
        model.eval()
        val_loader.reset()
        val_loss = 0.0
        for _ in range(val_steps):
            x_val, y_val, hint_val = val_loader.next_batch()
            with ctx: # of course, we'd like to use no_grad() here too, but that creates a torch.compile error for some reason
                _, loss = model(x_val, y_val, return_logits=False, **hint_val)
                val_loss += loss.detach()
                del loss
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss /= val_steps
        # log val loss to console and to logfile
        print0(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step, 1):.2f}ms')
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.time()

    if master_process and SAVE_CHECKPOINT and (last_step or (args.save_every > 0 and step % args.save_every == 0)):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        # save the state of the training process
        log = dict(step=step, code=code, model=raw_model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
        torch.save(log, 'logs/%s/state_step%06d.pt' % (run_id, step))
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.time()

    # bit confusing: we want to make sure to eval on 0th iteration
    # but also after the very last iteration. so we loop for step <= num_iterations
    # instead of just < num_iterations (one extra due to <=), only to do
    # the validation/sampling one last time, and then we break right here as we're done.
    if last_step:
        # port: the one result line (old_records/README.md item 6), after the final validation over all val_tokens
        print0(f"FINAL val_loss:{val_loss.item():.4f} train_time_ms:{int(training_time_ms)} steps:{args.num_iterations}")
        # port: VAL_SUBSETS (after the timer), the final validation's batches per position with the eager model
        def subset_losses():
            val_loader.reset()
            for _ in range(val_steps):
                start = val_loader.current_position # the batch's offset in the validation shard
                x_val, y_val, hint_val = val_loader.next_batch()
                with ctx:
                    loss = eager_model(x_val, y_val, return_logits=False, **hint_val)[1]
                yield start, loss
        eager_model = getattr(raw_model, "_orig_mod", raw_model)
        hint_runtime.report_val_subsets(eager_model, subset_losses(), device, print0, val_loss.item())
        break

    # --------------- TRAINING SECTION BEGIN -----------------
    model.train()
    train_loss = train_step()
    # step the schedulers
    for sched in schedulers:
        sched.step()
    # --------------- TRAINING SECTION END -------------------
    # everything that follows now is just diagnostics, prints, logging, etc.

    #dist.all_reduce(train_loss, op=dist.ReduceOp.AVG) # all-reducing the training loss would be more correct in terms of logging, but slower
    if master_process: # port: as the record, only rank 0 syncs on train_loss.item()
        approx_time = training_time_ms + 1000 * (time.time() - t0)
        print0(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/(step+1):.2f}ms")

if master_process:
    print(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

# -------------------------------------------------------------------------
# clean up nice
dist.destroy_process_group()
