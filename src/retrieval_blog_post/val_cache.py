"""How the #367 record's validation loss depends on the size of its validation cache: one model trained as the record
trains it (its training rows from the online cache, over the tokens trained on so far), then evaluated with validation
indexes over the first N training tokens (train_gpt.py's VAL_CACHE_SWEEP, off the clock after the record's own
validation), on the 8 GPUs of a miso node. The remake, at 37B tokens, of the online-trained line of the cache-size
figure in artifacts/ablation-findings.html (10.26B tokens of FineWeb10B)."""

import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import furu

from retrieval_blog_post.config import LOCAL_DISK, TORCH_VENV
from retrieval_blog_post.fineweb import FinewebShards
from retrieval_blog_post.kgram import SHARD_TOKENS

# The record's code in the worker's snapshot: train_gpt.py, track_1_short/, exact_match/.
REPO = Path(__file__).resolve().parents[2]
# The environment the record runs in on miso (config.TORCH_VENV; LOCAL_DISK, the node's NVMe, is where the shards are
# copied: NFS reads at 1.2 GB/s, the NVMe at 31): the record's stack (torch 2.10.0+cu128, triton 3.6.0, kernels 0.16.1,
# huggingface-hub 1.29.0; libcudart.so.13 is the node's), made with
#   uv venv --python 3.12 $TORCH_VENV && uv pip install --python $TORCH_VENV/bin/python \
#     --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple \
#     --index-strategy unsafe-best-match torch==2.10.0 kernels==0.16.1 huggingface-hub==1.29.0 \
#     typing-extensions==4.15.0 numpy tqdm tiktoken setuptools maturin
CARGO_BIN = Path.home() / ".cargo/bin"
INDEX_SHARDS = 370  # the record's validation index (track_1_short/config.py's val_index_shards): 37B tokens
# The cache sizes N: doublings from 75M, about what the record trains on (kgram.py's RECORD_367), to the record's own
# 37B.
SIZES = (
    *(75_000_000 << i for i in range(9)),  # 75M to 19.2B
    INDEX_SHARDS * SHARD_TOKENS,
)


@dataclass(frozen=True)
class ValCacheLosses:
    """The validation loss of one trained model per cache size, without hints, and with the record's own index."""

    tokens: tuple[int, ...]
    losses: tuple[float, ...]
    no_hints: float
    record: float
    train_time_ms: int


class ValCacheSweep(furu.Spec[ValCacheLosses]):
    """The record trained once with seed `seed`, its validation loss with indexes over the first `tokens` (each)
    training tokens."""

    tokens: tuple[int, ...] = SIZES
    seed: int = 0

    def metadata(self) -> furu.Metadata:
        return furu.Metadata(requires=furu.Requires(gpus=8))

    def create(self) -> ValCacheLosses:
        if max(self.tokens) > INDEX_SHARDS * SHARD_TOKENS:
            raise ValueError(
                f"the sweep goes past the {INDEX_SHARDS} shards the record stages"
            )
        text = train_record(
            self.directory.data,
            INDEX_SHARDS,
            {
                "TRAIN_SEED": str(self.seed),
                "VAL_CACHE_SWEEP": ",".join(map(str, (0, *self.tokens))),
            },
            self.logger,
        )
        record = re.search(r"val_loss:([\d.]+) train_time:(\d+)ms", text)
        losses = {
            int(n): float(loss)
            for n, loss in re.findall(r"val_cache_tokens:(\d+) val_loss:([\d.]+)", text)
        }
        if record is None or set(losses) != {0, *self.tokens}:
            raise RuntimeError("the run log lacks validation losses")
        return ValCacheLosses(
            tokens=self.tokens,
            losses=tuple(losses[n] for n in self.tokens),
            no_hints=losses[0],
            record=float(record[1]),
            train_time_ms=int(record[2]),
        )


def train_record(
    out: Path,
    train_shards: int,
    run_env: dict[str, str],
    logger,
    script: Path = REPO / "train_gpt.py",
) -> str:
    """Run `script` (by default the record's train_gpt.py in the worker's snapshot) on this node's 8 GPUs, with its first
    `train_shards` training shards and the validation shard copied to the NVMe and `run_env` added to the record's
    environment; the console goes to out/console.txt and the run log to out/logs/. Returns the run log."""
    shutil.rmtree(out / "logs", ignore_errors=True)  # an earlier attempt's
    LOCAL_DISK.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=LOCAL_DISK) as local:
        local = Path(local)
        data = local / "data/fineweb100B"
        data.mkdir(parents=True)
        shards = [
            *FinewebShards(shards=train_shards).create(),
            *FinewebShards(shards="val").create(),
        ]
        with ThreadPoolExecutor(32) as pool:
            list(
                pool.map(lambda s: shutil.copyfile(s.path, data / s.path.name), shards)
            )
        logger.info(f"copied {len(shards)} shards to {data}")
        env = {
            **os.environ,
            "VIRTUAL_ENV": str(TORCH_VENV),
            "PATH": f"{TORCH_VENV}/bin:{os.environ['PATH']}",
            "PYTHONPATH": str(build_extension(local)),
            # skips kernels' publisher check: the FA3 kernel is in the HF cache already
            "HF_HUB_OFFLINE": "1",
            "DATA_PATH": str(local),
            "TMPDIR": str(local),  # the sweep's cut shard
            "OMP_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
            "TORCHINDUCTOR_CACHE_DIR": str(local / "inductor"),
            "TRITON_CACHE_DIR": str(local / "triton"),
            **run_env,
        }
        with open(out / "console.txt", "w") as console:
            subprocess.run(
                [
                    TORCH_VENV / "bin/python",
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--nproc_per_node=8",
                    script,
                ],
                cwd=out,  # the run log goes to out/logs/
                env=env,
                stdout=console,
                stderr=subprocess.STDOUT,
                check=True,
            )
    (log,) = (out / "logs").glob("*.txt")
    return log.read_text()


def build_extension(local: Path) -> Path:
    """The exact_match extension of the snapshot, built for TORCH_VENV (as exp/run.sh builds it) in `local`, unpacked
    into a directory to put on PYTHONPATH."""
    subprocess.run(
        [
            TORCH_VENV / "bin/maturin",
            "build",
            "-q",
            "--release",
            "-m",
            REPO / "exact_match/Cargo.toml",
            "-i",
            TORCH_VENV / "bin/python",
            "-o",
            local / "wheel",
        ],
        env={
            **os.environ,
            "CARGO_NET_OFFLINE": "true",
            "CARGO_TARGET_DIR": str(local / "target"),
            "PATH": f"{CARGO_BIN}:{os.environ['PATH']}",
        },
        check=True,
    )
    (wheel,) = (local / "wheel").glob("*.whl")
    zipfile.ZipFile(wheel).extractall(local / "ext")
    return local / "ext"
