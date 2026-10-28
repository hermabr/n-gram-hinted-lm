"""Per-token validation losses (VAL_TOKEN_LOSSES, not part of a record run): off the clock, the final validation's
loss at every token and the hint the model read there, then the same model's loss with every row blank (no hints),
written by rank 0 to VAL_TOKEN_LOSSES/token_losses.npz (src/retrieval_blog_post/token_losses.py reads it). Every
array is in val-stream order: entry i is the target at position i + 1 of the validation shard, predicted from the
positions before it. record_360/track_1_short/token_losses.py is the same without the hints."""
import glob
import os

import numpy as np
import torch
import torch.distributed as dist

from track_1_short.data import HEADER_BYTES
from track_1_short.retrieval import VAL_ENTRIES, blank


class TokenLosses:
    """A validation pass's per-token columns, batch by batch, on this rank's device."""

    def __init__(self):
        self.columns = {}

    def add(self, losses, targets, rows):
        """A batch's per-token `losses` (the eval forward's) of its `targets`, and the retrieval `rows` it read: per
        position its cell (0: no match) and, over its candidates, their total count, the top one's count and the
        target's count (0: not among them)."""
        n = len(targets)
        cell, entries = rows[:n], rows[n:].view(-1, 2)
        at, token, count = entries[:, 0].long(), entries[:, 1] & 0xFFFF, entries[:, 1] >> 16
        zeros = torch.zeros(n, dtype=torch.int32, device=rows.device)
        batch = {
            "target": targets.int(),
            "loss": losses.float(),
            "hint_cell": cell,
            "hint_total": zeros.index_add(0, at, count),
            "hint_top_count": zeros.scatter_reduce(0, at, count, "amax"),
            "hint_target_count": zeros.index_add(0, at, count * (token == targets[at])),
        }
        for name, column in batch.items():
            self.columns.setdefault(name, []).append(column)

    def gather(self):
        """Every rank's batches in val-stream order, on rank 0 (None on the others). A collective."""
        out = {}
        for name, parts in self.columns.items():
            local = torch.stack(parts)  # [batches, n]
            every = torch.empty((dist.get_world_size(), *local.shape), dtype=local.dtype, device=local.device)
            dist.all_gather_into_tensor(every, local)
            # batch b of rank r holds the targets b * B + r * n + 1 ... b * B + (r + 1) * n
            out[name] = every.transpose(0, 1).reshape(-1).cpu().numpy()
        return out if dist.get_rank() == 0 else None


def write_token_losses(directory, val_files, hinted, model, forward_args, ngram_table, val_loader, val_steps, device):
    """Evaluate `model` again with blank rows, then write the final validation's columns (`hinted`) and the new pass's
    losses (loss_without_hints) to `directory`/token_losses.npz, after checking the targets against the val stream."""
    loader = val_loader()
    batches = [next(loader) for _ in range(val_steps)]
    del loader
    pulls = ngram_table.eval_pulls([batch.ngram_ids_cpu for batch in batches])
    blank_pass = TokenLosses()
    model.eval()
    with torch.no_grad():
        for batch, pull in zip(batches, pulls):
            n = len(batch.inputs_cpu)
            rows = blank(n, VAL_ENTRIES * n, device)
            ngram_table.land(pull)
            losses = model(batch.inputs, batch.targets, batch.cum_seqlens, ngram_table.slots(pull, batch.ngram_ids),
                           forward_args, ret=rows)
            blank_pass.add(losses, batch.targets, rows)
    columns, without = hinted.gather(), blank_pass.gather()
    if columns is None:
        return
    (val_file,) = sorted(glob.glob(val_files))
    stream = np.memmap(val_file, dtype="<u2", mode="r", offset=HEADER_BYTES)
    assert np.array_equal(columns["target"], stream[1:len(columns["target"]) + 1]), "targets out of val-stream order"
    assert np.array_equal(columns["target"], without["target"])
    os.makedirs(directory, exist_ok=True)
    np.savez(os.path.join(directory, "token_losses.npz"), **columns, loss_without_hints=without["loss"])
