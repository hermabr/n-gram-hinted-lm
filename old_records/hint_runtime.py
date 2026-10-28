"""Exact-match hints for the old speedrun records (old_records/README.md): the rows a ported record reads for its
training and validation positions, and the record #367 hint mechanism (track_1_short/model/gpt.py) that turns them
into a vector added to the residual stream. Self-contained (numpy and torch), so a record's train_gpt.py imports it
with old_records/ on its PYTHONPATH.

HINT_DIR holds one hint variant's rows (src/retrieval_blog_post/old_records.py makes them):
    meta.json          {"levels", "cells", "entries", "train_shards", "val_tokens"}
    train/NNNN/        the rows of training shard NNNN (0-based, in the sorted order of the staged shard files)
    val/               the rows of the validation shard's first val_tokens positions
each with cell.npy (int16 [n], 0: no hint), tokens.npy (uint16 [n, entries]) and weights.npy (uint16 [n, entries],
the candidates' counts over their sum, in units of 1 / WEIGHT_ONE; 0 pads). Row i is the hint for position i's input
token, i.e. for predicting token i + 1. Blended variants (meta "slots" S > 1) hold S rows per position, its S deepest
matched levels, deepest first: cell [n, S], tokens and weights [n, S, entries]; HintRows then carry the slot axis
too, and HintEmbedding sums a hint per slot. HINT_DIR=fake gives random rows of the same shapes (HINT_FAKE_SLOTS
slots), for smoke tests.

N-gram count rows (HINT_NGRAM=1, NgramMix): meta.json then also has "count_levels" (ascending context lengths) and
"count_shards" (the training shards with count rows), unigram.npy (float32 [VOCAB], the unigram base's init, from
training tokens) is in HINT_DIR, and train/NNNN/ (the first count_shards) and val/ also hold tot.npy (uint32 [n, L]:
the position's k-context's count), dist.npy (uint16 [n, L]: its distinct next tokens) and cnt.npy (uint32 [n, L]:
how many of them were the position's own next token), per level k of count_levels; all 0 where the k-context did not
occur (or does not fit the position's document). HintRows then carry counts (float32 [T, 3, L]: tot, dist, cnt).
HINT_DIR=fake with HINT_NGRAM=1 gives random counts too."""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

WEIGHT_ONE = 65535
RECORD_EMBED_STD = 0.005  # record #367's token embedding init (tied to its head's), which its hint inits are tuned for
FAKE_CELLS, FAKE_ENTRIES = 336, 8
FAKE_COUNT_LEVELS = (2, 3, 4, 5, 6, 8, 10, 12, 16, 20)
VOCAB = 50257
VAL_TOKENS = 10_485_760  # the validation positions with rows (fake rows too, so smoke tests catch reads past them)


@dataclass
class HintRows:
    """The rows of T positions, on the GPU: cell (int64 [T], 0 where nothing matched), the candidates' tokens
    (int64 [T, entries]) and weights (float32 [T, entries], summing to 1 where cell > 0, all 0 elsewhere). With
    slots, cell is [T, S] and tokens and weights [T, S, entries]."""

    cell: torch.Tensor
    tokens: torch.Tensor
    weights: torch.Tensor
    counts: torch.Tensor | None = None  # HINT_NGRAM: float32 [T, 3, L], each level's tot, dist and cnt


class _Rows:
    """One shard's rows, mapped read-only."""

    def __init__(self, directory: Path):
        self.cell = np.load(directory / "cell.npy", mmap_mode="r")
        self.tokens = np.load(directory / "tokens.npy", mmap_mode="r")
        self.weights = np.load(directory / "weights.npy", mmap_mode="r")

    def __len__(self) -> int:
        return len(self.cell)


class _Counts:
    """One shard's count rows, mapped read-only."""

    def __init__(self, directory: Path):
        self.tot = np.load(directory / "tot.npy", mmap_mode="r")
        self.dist = np.load(directory / "dist.npy", mmap_mode="r")
        self.cnt = np.load(directory / "cnt.npy", mmap_mode="r")

    def __len__(self) -> int:
        return len(self.tot)


class Hints:
    """The rows of one hint variant (HINT_DIR, by default the environment's). Lookups take positions within one
    shard: a training shard's index and offsets into its tokens (after the header), or offsets into the validation
    shard."""

    def __init__(self, directory: str | os.PathLike | None = None, device: torch.device | str = "cuda"):
        directory = str(directory if directory is not None else os.environ["HINT_DIR"])
        self.device = torch.device(device)
        self.fake = directory == "fake"
        self.ngram = ngram_enabled()
        self.count_levels: tuple[int, ...] = ()
        self.unigram: np.ndarray | None = None
        if self.fake:
            self.cells, self.entries, self.levels = FAKE_CELLS, FAKE_ENTRIES, (6, 10, 20)
            self.slots = int(os.environ.get("HINT_FAKE_SLOTS", "1"))
            self._generator = np.random.default_rng(0)
            if self.ngram:
                self.count_levels = FAKE_COUNT_LEVELS
                self.unigram = self._generator.dirichlet(np.full(VOCAB, 0.3)).astype(np.float32)
            return
        root = Path(directory)
        meta = json.loads((root / "meta.json").read_text())
        self.cells, self.entries, self.levels = meta["cells"], meta["entries"], tuple(meta["levels"])
        self.slots = meta.get("slots", 1)
        self._train = [_Rows(root / "train" / f"{j:04d}") for j in range(meta["train_shards"])]
        self._val = _Rows(root / "val")
        if self.ngram:
            if "count_levels" not in meta:
                raise ValueError(f"HINT_NGRAM=1, but {root} has no count rows (meta.json lacks count_levels)")
            self.count_levels = tuple(meta["count_levels"])
            self.unigram = np.load(root / "unigram.npy").astype(np.float32)
            self._train_counts = [_Counts(root / "train" / f"{j:04d}") for j in range(meta["count_shards"])]
            self._val_counts = _Counts(root / "val")

    def train(self, shard: int, offsets) -> HintRows:
        """The rows of training shard `shard` at `offsets` (an int array, or a slice)."""
        if self.fake:
            return self._fake(offsets)
        if shard >= len(self._train):
            raise IndexError(f"training shard {shard} is past the {len(self._train)} shards with hint rows")
        counts = None
        if self.ngram:
            if shard >= len(self._train_counts):
                raise IndexError(f"training shard {shard} is past the {len(self._train_counts)} shards with count rows")
            counts = self._train_counts[shard]
        return self._gather(self._train[shard], offsets, counts)

    def train_span(self, shard: int, start: int, length: int) -> HintRows:
        """The rows of training shard `shard` at offsets [start, start + length)."""
        return self.train(shard, slice(start, start + length))

    def val(self, offsets) -> HintRows:
        """The rows of the validation shard at `offsets` (an int array, or a slice)."""
        if self.fake:
            stop = offsets.stop if isinstance(offsets, slice) else int(np.max(offsets, initial=-1)) + 1
            if stop > VAL_TOKENS:
                raise IndexError(f"positions up to {stop} are past the {VAL_TOKENS} validation positions with hint rows")
            return self._fake(offsets)
        return self._gather(self._val, offsets, self._val_counts if self.ngram else None)

    def val_span(self, start: int, length: int) -> HintRows:
        """The rows of the validation shard at offsets [start, start + length)."""
        return self.val(slice(start, start + length))

    def _gather(self, rows: _Rows, index, counts: _Counts | None = None) -> HintRows:
        if counts is not None:
            hint = self._gather(rows, index)
            stop = index.stop if isinstance(index, slice) else int(np.max(index, initial=-1)) + 1
            if stop > len(counts):
                raise IndexError(f"positions up to {stop} are past the {len(counts)} positions with count rows")
            hint.counts = self._upload_counts(*(_take(a, index) for a in (counts.tot, counts.dist, counts.cnt)))
            return hint
        if isinstance(index, slice):
            if index.stop > len(rows):
                raise IndexError(f"positions up to {index.stop} are past the {len(rows)} positions with hint rows")
        else:
            index = np.asarray(index)
            if len(index) and index.max() >= len(rows):
                raise IndexError(f"position {index.max()} is past the {len(rows)} positions with hint rows")
            if len(index) > 1:
                # a batch is a few contiguous runs of positions (documents, sequences): copying run slices is ~50x
                # faster than fancy-indexing the mapped arrays
                breaks = np.flatnonzero(np.diff(index) != 1) + 1
                if len(breaks) < len(index) // 16:
                    firsts, lasts = index[np.r_[0, breaks]], index[np.r_[breaks - 1, len(index) - 1]]
                    runs = [slice(int(a), int(b) + 1) for a, b in zip(firsts, lasts)]
                    cat = lambda a: np.concatenate([a[r] for r in runs])
                    return self._upload(cat(rows.cell), cat(rows.tokens), cat(rows.weights))
        cell, tokens, weights = rows.cell[index], rows.tokens[index], rows.weights[index]
        return self._upload(cell, tokens, weights)

    def _upload(self, cell: np.ndarray, tokens: np.ndarray, weights: np.ndarray) -> HintRows:
        # torch has no arithmetic on uint16: move the bits as int16 and widen on the GPU
        shapes = cell.shape, tokens.shape, weights.shape
        if self.device.type == "cpu":
            to = lambda a: torch.from_numpy(np.ascontiguousarray(a).view(np.int16))
            cell, tokens, weights = to(cell), to(tokens), to(weights)
        else:
            # one copy from a pinned staging buffer, so the host never waits for the GPU's queue (a copy from pageable
            # memory can); a ring of them, each reused once its copy is done
            a, b = cell.size, cell.size + tokens.size
            size = b + weights.size
            slot = self._next_slot = (getattr(self, "_next_slot", -1) + 1) % 4
            ring = self.__dict__.setdefault("_ring", [None] * 4)
            if ring[slot] is None or ring[slot][0].numel() < size:
                ring[slot] = (torch.empty(max(size, 1 << 20), dtype=torch.int16, pin_memory=True), torch.cuda.Event())
            buffer, done = ring[slot]
            done.synchronize()
            staged = buffer.numpy()
            staged[:a] = np.ascontiguousarray(cell).view(np.int16).reshape(-1)
            staged[a:b] = np.ascontiguousarray(tokens).view(np.int16).reshape(-1)
            staged[b:size] = np.ascontiguousarray(weights).view(np.int16).reshape(-1)
            flat = buffer[:size].to(self.device, non_blocking=True)
            done.record()
            cell, tokens, weights = flat[:a].view(shapes[0]), flat[a:b].view(shapes[1]), flat[b:].view(shapes[2])
        return HintRows(
            cell=cell.long(),
            tokens=tokens.long() & 0xFFFF,
            weights=(weights.int() & 0xFFFF).float() / WEIGHT_ONE,
        )

    def _upload_counts(self, tot: np.ndarray, dist: np.ndarray, cnt: np.ndarray) -> torch.Tensor:
        """Count rows [n, L] each as float32 [n, 3, L] on the device, through a pinned ring as _upload's."""
        n, levels = tot.shape
        if self.device.type == "cpu":
            return torch.from_numpy(np.stack([tot, dist, cnt], 1).astype(np.float32))
        size = 3 * n * levels
        slot = self._next_count_slot = (getattr(self, "_next_count_slot", -1) + 1) % 4
        ring = self.__dict__.setdefault("_count_ring", [None] * 4)
        if ring[slot] is None or ring[slot][0].numel() < size:
            ring[slot] = (torch.empty(max(size, 1 << 20), dtype=torch.float32, pin_memory=True), torch.cuda.Event())
        buffer, done = ring[slot]
        done.synchronize()
        staged = buffer.numpy()[:size].reshape(n, 3, levels)
        staged[:, 0], staged[:, 1], staged[:, 2] = tot, dist, cnt
        out = buffer[:size].to(self.device, non_blocking=True)
        done.record()
        return out.view(n, 3, levels)

    def _fake(self, index) -> HintRows:
        n = len(range(index.start, index.stop)) if isinstance(index, slice) else len(np.asarray(index))
        g, slots = self._generator, (self.slots,) if self.slots > 1 else ()
        cell = np.where(g.random((n, *slots)) < 0.5, g.integers(1, self.cells, (n, *slots)), 0).astype(np.int16)
        tokens = g.integers(0, 50257, (n, *slots, self.entries)).astype(np.uint16)
        raw = g.random((n, *slots, self.entries)) * (g.random((n, *slots, self.entries)) < 0.5)
        raw[..., 0] += 1e-3
        weights = np.round(raw / raw.sum(-1, keepdims=True) * WEIGHT_ONE) * (cell > 0)[..., None]
        hint = self._upload(cell, tokens, weights.astype(np.uint16))
        if self.ngram:  # counts that shrink with the level, matched up to a random depth, cnt and dist within tot
            levels = len(self.count_levels)
            depth = g.integers(0, levels + 1, (n, 1))
            tot = np.where(np.arange(levels) < depth, g.integers(1, 1 << 20, (n, levels)) >> np.arange(levels) * 2, 0)
            tot = np.maximum(tot, np.arange(levels) < depth).astype(np.uint32)
            dist = np.minimum(tot, g.integers(1, 30000, (n, levels))).astype(np.uint16)
            cnt = np.where(g.random((n, levels)) < 0.5, (g.random((n, levels)) * tot).astype(np.uint32), 0)
            hint.counts = self._upload_counts(tot, dist, cnt.astype(np.uint32))
        return hint


def _take(array: np.ndarray, index) -> np.ndarray:
    """array[index] for a slice or an int array; an int array of a few contiguous runs is copied run by run (as
    Hints._gather's rows)."""
    if isinstance(index, slice):
        return array[index]
    index = np.asarray(index)
    if len(index) > 1:
        breaks = np.flatnonzero(np.diff(index) != 1) + 1
        if len(breaks) < len(index) // 16:
            firsts, lasts = index[np.r_[0, breaks]], index[np.r_[breaks - 1, len(index) - 1]]
            return np.concatenate([array[int(a):int(b) + 1] for a, b in zip(firsts, lasts)])
    return array[index]


class NgramMix(nn.Module):
    """Infini-gram style output mixing (Liu et al., 2024) with full n-gram counts: the next-token distribution is
    lam * p_LM + (1 - lam) * q, where q is an interpolated Witten-Bell n-gram over the count levels (ascending):
    q_0 = softmax(base) (a unigram, its logits learned, initialized to log(unigram)), and at each level j whose context
    occurred (tot_j > 0)
        q_j(v) = (cnt_j(v) + g_j * dist_j * q_{j-1}(v)) / (tot_j + g_j * dist_j),
    else q_j = q_{j-1}; log g_j = gamma_level[j] + gamma_bin[bin(tot_j)] + gamma[j, bin(tot_j)] is learned per level
    and total-count bin (COUNT_EDGES; the per-level and per-bin terms let the cells rare in training, such as deep
    levels with large counts, which the causal training counts seldom have, borrow from the others). Each
    q_j sums to 1 over the vocabulary, as the counts of a context's next tokens sum to its total (the validation rows
    keep each context's 1024 most frequent next tokens: a target outside them counts 0, so q sums to at most 1 there).
    lam = sigmoid(lam_logit[d, b] + lam_level[d] + lam_sparse[d, s]), learned per gate (d, b) and per (d, s): d the
    deepest level with tot > 0 (+1; 0 where nothing matched, and then b = s = 0), b its total-count bin, s whether its
    context had a single distinct next token (infini-gram's "sparse" estimate, by which its lambda depends too). The
    per-level terms carry over to the gates the causal training counts seldom reach (deep levels with large counts,
    common in validation): no per-bin term shared over the levels, as a large count means a reliable estimate at a
    deep level but a generic context at a shallow one. gamma and lam see only the context's tot (and which levels matched), never cnt or the target: the
    mixture is a proper distribution evaluated at the target. Computed in float32."""

    COUNT_EDGES = (2, 3, 5, 9, 17, 33, 100, 1000, 10000)  # a total count's bin: how many edges are below it

    def __init__(self, levels: int, unigram: np.ndarray | torch.Tensor, lam_init: float = 0.0, gamma_init: float = 0.0):
        super().__init__()
        self.levels, self.bins = levels, len(self.COUNT_EDGES) + 1
        unigram = torch.as_tensor(np.asarray(unigram, dtype=np.float32))
        assert unigram.shape == (VOCAB,), unigram.shape
        self.base = nn.Parameter(torch.log(unigram.clamp_min(1e-12)))
        self.gamma = nn.Parameter(torch.full((levels * self.bins,), float(gamma_init)))
        self.gamma_level = nn.Parameter(torch.zeros(levels))
        self.gamma_bin = nn.Parameter(torch.zeros(self.bins))
        self.lam_logit = nn.Parameter(torch.full(((levels + 1) * self.bins,), float(lam_init)))
        self.lam_level = nn.Parameter(torch.zeros(levels + 1))
        self.lam_sparse = nn.Parameter(torch.zeros(2 * (levels + 1)))

    def _bins(self, tot: torch.Tensor) -> torch.Tensor:
        return sum((tot > e).long() for e in self.COUNT_EDGES)

    def gates(self, counts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Each position's gate (d * bins + b) and sparse index (d * 2 + s) (int64 [T] each, lam's docstring) from
        its counts [T, 3, L]: context-side only (tot and dist)."""
        tot, dist = counts[:, 0].float(), counts[:, 1].float()
        matched = tot > 0
        level = torch.arange(1, self.levels + 1, device=tot.device)
        deepest = (matched.long() * level).amax(1) - 1  # -1: nothing matched
        at = deepest.clamp_min(0)[:, None]
        dbin = torch.gather(self._bins(tot), 1, at)[:, 0]
        sparse = (torch.gather(dist, 1, at)[:, 0] == 1).long()
        d = deepest + 1
        hit = deepest >= 0
        return torch.where(hit, d * self.bins + dbin, 0), torch.where(hit, d * 2 + sparse, 0)

    def log_q(self, targets: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """log q(target) per position (float32 [T])."""
        tot, dist, cnt = counts.float().unbind(1)
        assert tot.shape[1] == self.levels, (tot.shape, self.levels)
        p = torch.log_softmax(self.base.float(), 0)[targets].exp()
        bins = self._bins(tot)
        idx = torch.arange(self.levels, device=tot.device) * self.bins + bins
        log_g = self.gamma.float()[idx] + self.gamma_level.float()[None, :] + self.gamma_bin.float()[bins]
        g = torch.exp(log_g.clamp(-20.0, 20.0))  # [T, L]
        for j in range(self.levels):
            matched = tot[:, j] > 0
            gd = g[:, j] * dist[:, j]
            den = torch.where(matched, tot[:, j] + gd, torch.ones_like(gd))
            p = torch.where(matched, (cnt[:, j] + gd * p) / den, p)
        return torch.log(p.clamp_min(1e-30))

    def forward(self, nll: torch.Tensor, targets: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """The mixture's loss per position, from the model's loss `nll` [T] (full-vocabulary cross-entropy)."""
        gate, sparse = self.gates(counts)
        a = self.lam_logit.float()[gate] + self.lam_level.float()[gate // self.bins] + self.lam_sparse.float()[sparse]
        lm = F.logsigmoid(a) - nll.float()
        ngram = F.logsigmoid(-a) + self.log_q(targets, counts)
        return -torch.logaddexp(lm, ngram)


class HintEmbedding(nn.Module):
    """Record #367's hint (track_1_short/model/gpt.py): the candidates' mean token embedding (weighted by their
    counts, read without a gradient), times a learned scale per cell, plus a learned embedding per cell; zero where
    nothing matched. The model adds site_scale(i) * hint at three sites: i = 0 after the token embedding, 1 in the
    residual stream at about 0.6 of the depth (the record's layer 7 of 11), 2 just before the final norm and head.
    The inits are the record's (tuned with the scales' 10x lr).

    `embed_std` is the port's token embedding init std. The candidates' mean embedding is divided by
    embed_std / RECORD_EMBED_STD (a constant), so the hint starts at the size it has in #367. Site 0's init is
    multiplied by the same ratio, because it is added to the raw embedding, which is that much larger. Sites 1 and 2
    add to the normed stream and keep #367's inits. `embed_ratio` overrides the ratio (REC_HINT_EMBED_RATIO).

    With slots (blended rows), the hint is the sum of each slot's hint, each with its own cell (which holds its level).

    `mix` (HINT_MIX=1 by default when unset) adds infini-gram style interpolation at the output (mixed_nll): the
    next-token distribution is a mixture of the model's and each slot's candidate distribution, with weights
    softmax(0, mix_logit[cell_1], ...) (the model's logit 0; empty slots out), as Liu et al. (2024) interpolate a
    neural LM with the infini-gram estimate, but with the weight learned per cell (length, count and top-share bins).
    `inject=False` keeps the residual-stream hint out (pure interpolation); the sites then read zeros.

    `ngram` (HINT_NGRAM=1; needs `ngram_levels` and `unigram`, Hints.count_levels and Hints.unigram) mixes the model's
    distribution with the full-count n-gram instead (NgramMix, self.ngram_mix; the rows then carry counts).
    HINT_NGRAM_LAM_INIT (0.0) is the initial logit of the model's weight, HINT_NGRAM_GAMMA_INIT (0.0) the initial
    log Witten-Bell gamma. Not combinable with `mix`."""

    def __init__(self, model_dim: int, cells: int, embed_std: float = RECORD_EMBED_STD,
                 embed_ratio: float | None = None, next_scale: float = 4.0,
                 site_scales: tuple[float, float, float] = (0.1, 8.0, 8.0), mix: bool | None = None,
                 inject: bool | None = None, mix_init: float = -2.0, ngram: bool | None = None,
                 ngram_levels: int = 0, unigram: np.ndarray | None = None):
        super().__init__()
        self.mix = mix if mix is not None else os.environ.get("HINT_MIX", "0") == "1"
        self.inject = inject if inject is not None else os.environ.get("HINT_INJECT", "1") == "1"
        self.ngram = ngram if ngram is not None else ngram_enabled()
        if self.ngram:
            if self.mix:
                raise ValueError("HINT_NGRAM and HINT_MIX are both output mixtures: use one")
            if ngram_levels <= 0 or unigram is None:
                raise ValueError("HINT_NGRAM needs the count rows' levels and unigram (Hints.count_levels, Hints.unigram)")
            self.ngram_mix = NgramMix(ngram_levels, unigram, lam_init=float(os.environ.get("HINT_NGRAM_LAM_INIT", "0")),
                                      gamma_init=float(os.environ.get("HINT_NGRAM_GAMMA_INIT", "0")))
        if self.mix:
            self.mix_logit = nn.Parameter(torch.full((cells,), float(os.environ.get("HINT_MIX_INIT", mix_init))))
        ratio = embed_ratio if embed_ratio is not None else float(os.environ.get("REC_HINT_EMBED_RATIO", embed_std / RECORD_EMBED_STD))
        self.embed_ratio = ratio
        self.next_scale = nn.Parameter(torch.full((cells,), float(next_scale)))
        self.cell_embed = nn.Parameter(torch.zeros(cells, model_dim))
        self.site_scales = nn.Parameter(torch.tensor((site_scales[0] * ratio, *site_scales[1:]), dtype=torch.float32))

    def forward(self, embed_weight: torch.Tensor, rows: HintRows) -> torch.Tensor:
        """The hint of each position, float32 [T, model_dim]; `embed_weight` is the input token embedding
        [vocab, model_dim]."""
        cell, tokens, weights = rows.cell, rows.tokens, rows.weights
        if not self.inject:  # zeros, still in the graph of every parameter (DDP wants a gradient for each)
            unused = (self.next_scale.sum() + self.cell_embed.sum()) * 0
            return embed_weight.new_zeros(cell.shape[0], embed_weight.shape[1], dtype=torch.float32) + unused
        if cell.dim() == 2:  # slots: a hint per slot, summed
            cell, tokens, weights = cell.flatten(), tokens.flatten(0, 1), weights.flatten(0, 1)
        weight = embed_weight.detach().float() * (1 / self.embed_ratio)
        mean = F.embedding_bag(tokens, weight, per_sample_weights=weights, mode="sum")
        hint = (mean * self.next_scale[cell, None] + self.cell_embed[cell]) * (cell > 0)[:, None]
        if rows.cell.dim() == 2:
            hint = hint.view(*rows.cell.shape, -1).sum(1)
        return hint

    def mixed_nll(self, nll: torch.Tensor, targets: torch.Tensor, rows: HintRows) -> torch.Tensor:
        """Per position, the loss of the mixture of the model's next-token distribution (its loss `nll` [T], from the
        full-vocabulary cross-entropy) and the slots' candidate distributions, at `targets` [T]; `nll` itself without
        `mix`. A proper distribution's loss: the mixture weights sum to 1 and every component is normalized (a
        candidate distribution's weights sum to 1 over its top entries). With `ngram`, NgramMix's loss (rows.counts)."""
        if self.ngram:
            assert rows.counts is not None, "HINT_NGRAM: the rows lack counts"
            return self.ngram_mix(nll, targets, rows.counts)
        if not self.mix:
            return nll
        cell, tokens, weights = rows.cell, rows.tokens, rows.weights
        if cell.dim() == 1:
            cell, tokens, weights = cell[:, None], tokens[:, None], weights[:, None]
        q = (weights * (tokens == targets[:, None, None])).sum(-1)  # [T, S]: each slot's probability of the target
        logits = torch.where(cell > 0, self.mix_logit[cell], float("-inf"))
        log_pi = torch.log_softmax(torch.cat([logits.new_zeros(len(cell), 1), logits], 1), 1)
        terms = torch.cat([(log_pi[:, :1] - nll.float()[:, None]), log_pi[:, 1:] + torch.log(q)], 1)
        return -torch.logsumexp(terms, 1)

    def site_scale(self, site: int) -> torch.Tensor:
        return self.site_scales[site]


def hint_optimizer(module: HintEmbedding, lr: float = 0.008, betas=(0.9, 0.99), eps: float = 1e-10) -> torch.optim.Adam:
    """Adam for a HintEmbedding as the record trains it: the scales at 10x `lr`, no weight decay. Each group keeps its
    base lr in "initial_lr", for the record's schedule (set_lr_factor)."""
    scales = [module.next_scale, module.site_scales, *([module.mix_logit] if module.mix else [])]
    ngram = getattr(module, "ngram", False)
    if ngram:  # the n-gram's gammas and gates at the scales' lr, its unigram logits at the base lr
        m = module.ngram_mix
        scales += [m.gamma, m.gamma_level, m.gamma_bin, m.lam_logit, m.lam_level, m.lam_sparse]
    groups = [
        {"params": scales, "lr": 10 * lr, "initial_lr": 10 * lr},
        {"params": [module.cell_embed, *([module.ngram_mix.base] if ngram else [])], "lr": lr, "initial_lr": lr},
    ]
    return torch.optim.Adam(groups, betas=betas, eps=eps, weight_decay=0.0)


def set_lr_factor(optimizer: torch.optim.Optimizer, factor: float) -> None:
    """Every group's lr to its initial_lr times `factor` (the record's lr schedule at this step)."""
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * factor


def allreduce_grads(module: nn.Module) -> None:
    """Average the module's gradients over the ranks (for records whose own gradient sync does not cover it)."""
    for p in module.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)


def ngram_enabled() -> bool:
    """Whether the run mixes the model's output with the full-count n-gram (HINT_NGRAM=1; NgramMix)."""
    return os.environ.get("HINT_NGRAM", "0") == "1"


def hints_enabled() -> bool:
    """Whether the run uses hints: HINTS=1 (the default when HINT_DIR is set)."""
    return os.environ.get("HINTS", "1" if "HINT_DIR" in os.environ else "0") == "1"


# -----------------------------------------------------------------------------
# VAL_SUBSETS: the final validation's loss on the deduplicated subsets of the validation targets, an extra per-position
# pass after the run's last validation (off the clock; extra output only)

SUBSET_NAMES = ("all", "noexact", "nonear")


def val_subsets_enabled() -> bool:
    """Whether to run the subsets pass: VAL_SUBSETS=<labels.npy> (src/retrieval_blog_post/old_records.py's
    ValSubsetLabels)."""
    return bool(os.environ.get("VAL_SUBSETS"))


class ValSubsets:
    """Sums of per-position validation losses over the subsets of the targets, by the labels in `path` (VAL_SUBSETS;
    uint8 [VAL_TOKENS], per validation position i, i.e. its target val token i + 1: 0 in a document with an exact copy
    in the training shards (or a target outside every document's body), 1 in one without, but with a near-copy
    (13-word n-gram Jaccard >= 0.8), 2 in one with neither). The subsets: "all" (every label), "noexact" (1 and 2),
    "nonear" (2)."""

    def __init__(self, path: str | os.PathLike | None = None, device: torch.device | str = "cuda"):
        self.labels = np.load(path if path is not None else os.environ["VAL_SUBSETS"], mmap_mode="r")
        self.device = torch.device(device)
        self.sums = torch.zeros(3, dtype=torch.float64, device=self.device)  # per label
        self.counts = torch.zeros(3, dtype=torch.float64, device=self.device)

    def add(self, start: int, losses: torch.Tensor) -> None:
        """The per-position losses (any shape, flattened in order) of validation positions start, start + 1, ..."""
        losses = losses.detach().reshape(-1).double()
        stop = start + losses.numel()
        if start < 0 or stop > len(self.labels):
            raise IndexError(f"validation positions {start}..{stop} are past the {len(self.labels)} with subset labels")
        labels = torch.from_numpy(np.array(self.labels[start:stop])).to(self.device).long()
        self.sums.index_add_(0, labels, losses.to(self.device))
        self.counts.index_add_(0, labels, torch.ones_like(losses, device=self.device))

    def means(self) -> dict[str, tuple[float, int]]:
        """Each subset's mean loss and positions over all ranks (all_reduce when distributed)."""
        totals = torch.stack([self.sums, self.counts])
        if dist.is_available() and dist.is_initialized():
            totals = totals.clone()
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        sums, counts = totals.tolist()
        subset = lambda v: (v[0] + v[1] + v[2], v[1] + v[2], v[2])
        return {name: (s / max(c, 1.0), int(c)) for name, s, c in zip(SUBSET_NAMES, subset(sums), subset(counts))}

    @staticmethod
    def line(means: dict[str, tuple[float, int]], seconds: float | None = None) -> str:
        """VAL_SUBSETS all:L noexact:L nonear:L tokens:N/N/N[ time:Ts], the subsets' mean losses and positions
        (`means`, from means())."""
        losses = " ".join(f"{name}:{loss:.4f}" for name, (loss, _) in means.items())
        tokens = "/".join(str(n) for _, n in means.values())
        return f"VAL_SUBSETS {losses} tokens:{tokens}" + (f" time:{seconds:.1f}s" if seconds is not None else "")


def report_val_subsets(model: nn.Module, losses, device: torch.device | str, print0=print,
                       val_loss: float | None = None) -> dict[str, tuple[float, int]] | None:
    """The subsets pass (VAL_SUBSETS; a no-op without it): with `model` (the uncompiled raw model, so nothing
    recompiles) in eval mode, under no_grad and with model.per_position = True (its forward then returns per-position
    losses), consume `losses`, an iterable of (start, per-position loss) over the run's validation batches (each rank
    its own; a generator calling the model, made by the port), then print one line on rank 0 (ValSubsets.line).
    `val_loss` (the run's) is checked against "all". Call it after the timer stops. Restores the model's mode and
    per_position. Returns the subsets' (mean, positions), or None without VAL_SUBSETS."""
    if not val_subsets_enabled():
        return None
    import time

    sync = (lambda: torch.cuda.synchronize(device)) if torch.device(device).type == "cuda" else (lambda: None)
    sync()
    t0 = time.perf_counter()
    subsets = ValSubsets(device=device)
    training = model.training
    model.eval()
    model.per_position = True
    try:
        with torch.no_grad():
            for start, loss in losses:
                subsets.add(int(start), loss)
    finally:
        model.per_position = False
        model.train(training)
    means = subsets.means()
    line = ValSubsets.line(means, time.perf_counter() - t0)
    if val_loss is not None and abs(means["all"][0] - val_loss) > 2e-3:
        line += f" (all differs from val_loss {val_loss:.4f})"
    print0(line)
    return means
