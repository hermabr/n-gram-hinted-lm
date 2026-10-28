"""Record #1's llm.c trainer (old_records/r001_llmc_c) on a miso node's 8 H200s, through its run_hints.py (4x512, 470
steps): with the hints, their validation rows built on CPUs, and without them. A run returns (val loss, seconds on the
clock).

    uv run src/retrieval_blog_post/llmc_cpu_val.py      # keep it running: it coordinates the worker
"""

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import furu

from retrieval_blog_post import llmc_onclock
from retrieval_blog_post.config import NVCC, TORCH_VENV
from retrieval_blog_post.fineweb import TRAIN_SHARDS, FinewebShards
from retrieval_blog_post.llmc_onclock import CODE, STEPS_PER_SHARD, clear_stale_shm, source_hash, tail
from retrieval_blog_post.old_records import ValSubsetLabels
from retrieval_blog_post.val_cache import CARGO_BIN


class LlmcBuild(furu.Spec[Path]):
    """build.sh's build/train_gpt2cu of old_records/r001_llmc_c with the sources hashed `source`."""

    source: str = source_hash()

    def metadata(self) -> furu.Metadata:  # 8 GPUs, so that the runs' worker builds it
        return furu.Metadata(requires=furu.Requires(gpus=8))

    def create(self) -> Path:
        if source_hash() != self.source:
            raise RuntimeError(f"the snapshot's sources hash to {source_hash()}, not {self.source}")
        build = self.directory.data / "build"
        env = {**os.environ, "BUILD": str(build), "PATH": f"{CARGO_BIN}:{os.environ['PATH']}", "NVCC": str(NVCC),
               "NVIDIA": str(TORCH_VENV / "lib/python3.12/site-packages/nvidia")}
        subprocess.run(["bash", CODE / "build.sh"], env=env, check=True)
        return build / "train_gpt2cu"


class LlmcCpuValRun(furu.Spec[tuple[float, float]]):
    """run_hints.py for `steps` steps with seed `seed`, with or without the hints. The clock is llm.c's per-step
    CUDA-event times, summed: every step, and with the hints the wait for the CPU build after the last step. Off it:
    the setup (a forward and backward of the first batch), staging the shards in /dev/shm, the validation lookups and
    the final validation."""

    hints: bool = True
    seed: int = 0
    steps: int = 470

    def metadata(self) -> furu.Metadata:
        return furu.Metadata(requires=furu.Requires(gpus=8))

    def _train(self) -> FinewebShards:
        return FinewebShards(shards=TRAIN_SHARDS if self.hints else math.ceil(self.steps / STEPS_PER_SHARD))

    @furu.dependency
    def deps(self) -> list[furu.Spec]:
        return [LlmcBuild(), self._train(), FinewebShards(shards="val"), *([ValSubsetLabels()] if self.hints else [])]

    def create(self) -> tuple[float, float]:
        binary = LlmcBuild().create()
        shards = [*self._train().create(), *FinewebShards(shards="val").create()]
        out = self.directory.data
        clear_stale_shm("llmc-cpu-val-")
        need, free = sum(s.path.stat().st_size for s in shards), shutil.disk_usage("/dev/shm").free
        if free < 1.2 * need:
            raise RuntimeError(f"/dev/shm has {free / 1e9:.0f} GB free; the shards need {need / 1e9:.0f} GB")
        with tempfile.TemporaryDirectory(dir="/dev/shm", prefix="llmc-cpu-val-") as stage:
            start = time.time()
            with ThreadPoolExecutor(32) as pool:
                copies = [["numactl", "--interleave=all", "cp", s.path, Path(stage) / s.path.name] for s in shards]
                list(pool.map(lambda argv: subprocess.run(argv, check=True), copies))
            self.logger.info(f"staged {len(shards)} shards in {stage} in {time.time() - start:.0f} s")
            argv = [sys.executable, CODE / "run_hints.py", "--bin", binary, "--shards", stage, "--out", out,
                    "--steps", str(self.steps), "--seed", str(self.seed)]
            if self.hints:
                argv += ["--subsets", ValSubsetLabels().create().directory / "labels.npy"]
            else:
                argv.append("--no-hints")
            self.logger.info(subprocess.list2cmdline(map(str, argv)))
            with open(out / "run.log", "w") as log:
                done = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT)
            shutil.rmtree(out / "data", ignore_errors=True)  # the links into the stage
        if done.returncode != 0:
            raise RuntimeError(f"run_hints.py failed ({done.returncode}):\n{tail(out / 'run.log')}")
        res = json.loads((out / "result.json").read_text())
        return res["val"], res["clock_ms"] / 1000


create = llmc_onclock.create  # one miso node for the runs (scancel it when done)

# with and without the hints, 3 seeds each
BATCH: list[furu.Spec] = [LlmcCpuValRun(hints=hints, seed=seed) for hints in (True, False) for seed in (0, 1, 2)]


if __name__ == "__main__":
    from retrieval_blog_post import llmc_cpu_val  # the specs' module by name, not __main__

    for spec, (val, seconds) in zip(llmc_cpu_val.BATCH, llmc_cpu_val.create(llmc_cpu_val.BATCH)):
        print(f"hints={spec.hints} seed={spec.seed} steps={spec.steps}: ({val:.4f}, {seconds:.2f})")
