"""The validation-cache sweep (VAL_CACHE_SWEEP, not part of a record run): after the final validation, off the clock,
the trained model is evaluated again with validation indexes over the first N training tokens, one N at a time, and
with no hints for N = 0. Only the index changes: the model, the validation batches and their n-gram rows stay."""
import glob
import os
import tempfile
from concurrent.futures import Future

import numpy as np
import torch
import torch.distributed as dist

from track_1_short.data import HEADER_BYTES
from track_1_short.retrieval import VAL_ENTRIES, ValidationCache, blank


def first_tokens(train_files, tokens, scratch) -> list[str]:
    """The shard files of the first `tokens` training tokens: whole shards, then a copy of the next one cut to the
    rest, written to `scratch`."""
    files = []
    for path in sorted(glob.glob(train_files)):
        if tokens == 0:
            break
        n = (os.path.getsize(path) - HEADER_BYTES) // 2
        if n > tokens:
            header = np.fromfile(path, dtype="<i4", count=HEADER_BYTES // 4)
            header[2] = tokens
            cut = os.path.join(scratch, os.path.basename(path))
            with open(cut, "wb") as f:
                f.write(header.tobytes())
                f.write(np.fromfile(path, dtype="<u2", count=tokens, offset=HEADER_BYTES).tobytes())
            path, n = cut, tokens
        files.append(path)
        tokens -= n
    assert tokens == 0, f"the training shards hold {tokens} tokens fewer than the sweep asks for"
    return files


def sweep_val_caches(sizes, train_files, model, forward_args, ngram_table, val_loader, val_steps, group, build_cpus,
                     lookup_cpus, device, print0):
    """Evaluate `model` once per size of `sizes` (training tokens; 0: no hints) and log each as
    `val_cache_tokens:N val_loss:L`. Each index is built, looked up and dropped before the next is created."""
    loader = val_loader()
    batches = [next(loader) for _ in range(val_steps)]
    del loader
    tokens = [batch.inputs_cpu.astype(np.uint16) for batch in batches]
    pulls = ngram_table.eval_pulls([batch.ngram_ids_cpu for batch in batches])
    built = Future()
    built.set_result(None)
    model.eval()
    for size in sizes:
        with tempfile.TemporaryDirectory() as scratch:
            cache = None
            if size == 0:
                rows = (blank(len(t), VAL_ENTRIES * len(t), device) for t in tokens)
            else:
                # only rank 0 reads the shards (the other ranks attach to its table)
                files = first_tokens(train_files, size, scratch) if dist.get_rank(group) == 0 else []
                cache = ValidationCache(files, group, build_cpus, lookup_cpus)
                cache.start(built)
                cache.wait()
                rows = cache.rows(tokens, device)
            val_loss = 0
            with torch.no_grad():
                for batch, pull, r in zip(batches, pulls, rows):
                    ngram_table.land(pull)
                    val_loss += model(batch.inputs, batch.targets, batch.cum_seqlens, ngram_table.slots(pull, batch.ngram_ids),
                                      forward_args, ret=r).mean()
            val_loss /= val_steps
            dist.reduce(val_loss, 0, op=dist.ReduceOp.AVG)
            print0(f"val_cache_tokens:{size} val_loss:{val_loss:.4f}", console=True)
            if cache is not None:
                cache.worker.shutdown()  # rank 0's release
            del cache, rows
