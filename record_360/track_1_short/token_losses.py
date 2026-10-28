"""Per-token validation losses (VAL_TOKEN_LOSSES, not part of a record run): off the clock, the final validation's
loss at every token, written by rank 0 to VAL_TOKEN_LOSSES/token_losses.npz (src/retrieval_blog_post/token_losses.py
reads it). Every array is in val-stream order: entry i is the target at position i + 1 of the validation shard,
predicted from the positions before it. The retrieval record's track_1_short/token_losses.py adds its hints."""
import glob
import os

import numpy as np
import torch
import torch.distributed as dist

from track_1_short.data import HEADER_BYTES


class TokenLosses:
    """A validation pass's per-token columns, batch by batch, on this rank's device."""

    def __init__(self):
        self.columns = {}

    def add(self, losses, targets):
        """A batch's per-token `losses` (the eval forward's) of its `targets`."""
        for name, column in {"target": targets.int(), "loss": losses.float()}.items():
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


def write_token_losses(directory, val_files, token_losses):
    """Write the final validation's columns to `directory`/token_losses.npz, after checking the targets against the
    val stream."""
    columns = token_losses.gather()
    if columns is None:
        return
    (val_file,) = sorted(glob.glob(val_files))
    stream = np.memmap(val_file, dtype="<u2", mode="r", offset=HEADER_BYTES)
    assert np.array_equal(columns["target"], stream[1:len(columns["target"]) + 1]), "targets out of val-stream order"
    os.makedirs(directory, exist_ok=True)
    np.savez(os.path.join(directory, "token_losses.npz"), **columns)
