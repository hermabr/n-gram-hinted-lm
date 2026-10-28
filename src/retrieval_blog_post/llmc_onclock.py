"""Record #1's own llm.c trainer (old_records/r001_llmc_c/train_gpt2.cu, built by llmc_cpu_val.LlmcBuild) on a node's
8 H200s with the exact-match hints, every statistic of which is computed on the clock (the validation rows on the GPUs).

    uv run src/retrieval_blog_post/llmc_onclock.py      # keep it running (e.g. in tmux): it coordinates the worker

A run stages its training shards and the validation shard on the local disk, and all TRAIN_SHARDS training shards in
/dev/shm, read once through before the clock (the validation rows count over them). The 210-step results were
measured on an earlier build of this trainer."""

import hashlib
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import furu
from furu.resources import ResourceFloor
from furu.worker.backends.slurm.backend import SlurmWorkerBackend

from retrieval_blog_post.config import LOCAL_DISK
from retrieval_blog_post.fineweb import TRAIN_SHARDS, FinewebShards
from retrieval_blog_post.old_records import ValSubsetLabels
from retrieval_blog_post.config import gpu_node
from retrieval_blog_post.val_cache import REPO

CODE = REPO / "old_records/r001_llmc_c"
BUILT = (".cu", ".cuh", ".h", ".cpp", ".rs", ".toml", ".lock", ".sh")
STEPS_PER_SHARD = 190  # 100M-token shards at 524,288 tokens per step (llm.c skips a shard's tail of under a step)
TIMEOUT = 30 * 60
# the PyTorch port's best on-clock recipe (r001_llmc: 4x512, aux 1, wsd cooldown 0.1, LR 3, warmup 50, the hint lr
# 0.002667 x LR 3); one validation, after the last step
HINTED = ("-e", "custom:4:512:8", "-b", "64", "-t", "1024", "-d", "524288", "-r", "1", "-z", "1", "-c", "0.1",
          "-l", "0.0018", "-q", "0.0", "-u", "50", "-k", "wsdlin", "-n", "0", "-v", "100000", "-m", "20", "-s", "0",
          "-h", "0")
HINT_ENV = (("HINTS", "1"), ("HINT_LR", "0.008"), ("HINT_AUX", "1"))


@dataclass(frozen=True)
class LlmcResult:
    """A run's final validation loss, its clock (ms) and steps; the losses on the validation subsets (hints only);
    the mean step time (ms, steps 1.. as llm.c's average); the on-clock row builds (ms)."""

    val_loss: float
    train_time_ms: int
    steps: int
    noexact: float | None = None
    nonear: float | None = None
    step_ms: float | None = None
    train_rows_ms: float | None = None
    val_rows_ms: float | None = None
    lines: tuple[str, ...] = field(default=())


def source_hash() -> str:
    """The hash of the sources build.sh compiles (LlmcBuild's `source`)."""
    digest = hashlib.sha256()
    for path in sorted(CODE.rglob("*")):
        rel = path.relative_to(CODE)
        if path.is_file() and path.suffix in BUILT and not {"build", "target"} & set(rel.parts):
            digest.update(f"{rel}\0".encode() + path.read_bytes() + b"\0")
    return digest.hexdigest()[:16]


class LlmcRun(furu.Spec[LlmcResult]):
    """LlmcBuild(source=source)'s binary for `steps` steps with llm.c's flags `args` (later flags override earlier
    ones) and `env` added to its environment, seed `seed`; with HINTS=1 in env the hinted run (its validation shards
    staged). `check` was the old builds' row check (HINT_DUMP), which the committed train_gpt2.cu does not have."""

    steps: int
    source: str = field(default_factory=source_hash)
    args: tuple[str, ...] = HINTED
    env: tuple[tuple[str, str], ...] = HINT_ENV
    seed: int = 0
    check: bool = False

    migrations = (furu.Renamed("label", to="source"),)

    def metadata(self) -> furu.Metadata:
        return furu.Metadata(requires=furu.Requires(gpus=8))

    def _hints(self) -> bool:
        return dict(self.env).get("HINTS") == "1"

    @furu.dependency
    def deps(self) -> list[furu.Spec]:
        deps = [FinewebShards(shards=3), FinewebShards(shards="val")]
        if self._hints():
            deps += [FinewebShards(shards=TRAIN_SHARDS), ValSubsetLabels()]
        return [self._build(), *deps]

    def _build(self) -> furu.Spec[Path]:
        from retrieval_blog_post.llmc_cpu_val import LlmcBuild

        return LlmcBuild(source=self.source)

    def create(self) -> LlmcResult:
        out = self.directory.data
        if self.check:
            raise ValueError("the committed train_gpt2.cu has no HINT_DUMP")
        binary = self._build().create()
        clear_stale_shm("llmc-onclock-")
        n = math.ceil(self.steps / STEPS_PER_SHARD)
        LOCAL_DISK.mkdir(parents=True, exist_ok=True)
        stage = None
        with tempfile.TemporaryDirectory(dir=LOCAL_DISK) as local:
            local = Path(local)
            data = local / "data"
            data.mkdir()
            assert n <= 3, f"{self.steps} steps need {n} shards, more than FinewebShards(shards=3)"
            train = FinewebShards(shards=3).create()[:n]
            copies = [(s.path, data / s.path.name) for s in [*train, *FinewebShards(shards="val").create()]]
            nccl = {"NCCL_CUMEM_ENABLE": "0", "NCCL_NVLS_ENABLE": "0"}
            env = {**os.environ, "SEED": str(self.seed), **nccl, **dict(self.env)}
            try:
                if self._hints():
                    every = FinewebShards(shards=TRAIN_SHARDS).create()
                    need = sum(s.path.stat().st_size for s in every)
                    root = Path("/dev/shm") if shutil.disk_usage("/dev/shm").free > 1.2 * need else local
                    stage = tempfile.TemporaryDirectory(dir=root, prefix="llmc-onclock-")
                    staged = [Path(stage.name) / s.path.name for s in every]
                    copies += list(zip((s.path for s in every), staged))
                    (local / "val_shards.txt").write_text("".join(f"{p}\n" for p in staged))
                    labels = ValSubsetLabels().create().directory / "labels.npy"
                    env |= {"HINT_VAL_SHARDS": str(local / "val_shards.txt"), "HINT_VAL_SUBSETS": str(labels)}
                start = time.time()
                with ThreadPoolExecutor(32) as pool:
                    list(pool.map(lambda c: shutil.copyfile(*c), copies))
                self.logger.info(f"copied {len(copies)} files to {local} in {time.time() - start:.0f} s")
                if stage is not None:  # read once through (a raw copy, before the clock)
                    def prefault(path: Path) -> int:
                        with open(path, "rb", buffering=0) as f:
                            got = 0
                            while chunk := f.read(64 << 20):
                                got += len(chunk)
                        return got

                    start = time.time()
                    with ThreadPoolExecutor(32) as pool:
                        read = sum(pool.map(prefault, staged))
                    self.logger.info(f"read {read / 1e9:.0f} GB of staged shards in {time.time() - start:.0f} s")
                argv = [str(binary), "-i", str(data / "fineweb_train_*.bin"),
                        "-j", str(data / "fineweb_val_000000.bin"), *self.args, "-x", str(self.steps)]
                command = subprocess.list2cmdline(argv)
                self.logger.info(f"llmc {self.source}: {command} env {dict(self.env)} seed {self.seed}")
                _launch(argv, local / "nccl", out, env, self.logger)
            finally:
                if stage is not None:
                    stage.cleanup()
        return _result(out / "console.txt", self.steps)


def _result(console: Path, steps: int) -> LlmcResult:
    text = console.read_text(errors="replace")
    step_ms = [float(m) for m in re.findall(r"^step +\d+/\d+ \|.*\| ([\d.]+) ms \|", text, re.MULTILINE)]
    mean = sum(step_ms[1:]) / max(1, len(step_ms) - 1) if step_ms else None
    keep = tuple(re.findall(r"^(?:FINAL|VAL_SUBSETS|hints:|val loss|total average).*$", text, re.MULTILINE))
    if final := re.findall(r"FINAL val_loss:([\d.]+) train_time_ms:(\d+) steps:(\d+)", text):
        loss, ms, n = final[-1]
        sub = re.findall(r"VAL_SUBSETS all:([\d.]+) noexact:([\d.]+) nonear:([\d.]+)", text)
        rows = re.findall(r"hints: training rows in (\d+) ms", text)
        val = re.findall(r"hints: validation rows in (\d+) ms", text)
        return LlmcResult(float(loss), int(ms), int(n), float(sub[-1][1]) if sub else None,
                          float(sub[-1][2]) if sub else None, mean, float(rows[-1]) if rows else None,
                          float(val[-1]) if val else None, keep)
    losses = re.findall(r"^val loss ([\d.]+)", text, re.MULTILINE)
    if not losses or mean is None:
        raise RuntimeError(f"the run's console lacks its results:\n{tail(console)}")
    # llm.c as is: its last val loss, and its own clock (the mean step, step 0 excluded) x the steps
    return LlmcResult(float(losses[-1]), int(mean * steps), steps, step_ms=mean, lines=keep)


def tail(path: Path, lines: int = 40) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError as error:
        return repr(error)


def clear_stale_shm(prefix: str) -> None:
    """Remove /dev/shm/<prefix>* left by killed runs (one worker per node, so none is in use)."""
    for path in Path("/dev/shm").glob(f"{prefix}*"):
        shutil.rmtree(path, ignore_errors=True)


def _launch(argv: list[str], rendezvous: Path, out: Path, env: dict[str, str], logger) -> None:
    """The 8 ranks of `argv` (llm.c's fs rendezvous in `rendezvous`: rank 0 first, the others once it has written
    NCCL's id), rank r's output in out/console.txt (r = 0) or console_rank{r}.txt; raises if any rank fails."""
    rendezvous.mkdir(parents=True, exist_ok=True)
    procs, files = [], []
    deadline = time.time() + TIMEOUT

    def start(rank: int) -> None:
        console = open(out / ("console.txt" if rank == 0 else f"console_rank{rank}.txt"), "w")
        files.append(console)
        flags = ["-pi", "fs", "-pf", str(rendezvous), "-pn", "8", "-pr", str(rank), "-pg", "8"]
        procs.append(subprocess.Popen([*argv, *flags], cwd=out, env=env, stdout=console, stderr=subprocess.STDOUT))

    def failed() -> int | None:
        return next((r for r, p in enumerate(procs) if p.poll() not in (None, 0)), None)

    try:
        start(0)
        nccl_id = rendezvous / "ncclUniqueId.sync"
        while not (nccl_id.exists() and nccl_id.stat().st_size >= 128):
            if failed() is not None or procs[0].poll() is not None:
                raise RuntimeError(f"rank 0 exited before the others started:\n{tail(out / 'console.txt')}")
            if time.time() > deadline:
                raise TimeoutError("rank 0 wrote no NCCL id")
            time.sleep(0.2)
        for rank in range(1, 8):
            start(rank)
        logger.info("started 8 ranks")
        while any(p.poll() is None for p in procs):
            if (rank := failed()) is not None:
                console = out / ("console.txt" if rank == 0 else f"console_rank{rank}.txt")
                raise RuntimeError(f"rank {rank} failed:\n{tail(console)}")
            if time.time() > deadline:
                raise TimeoutError(f"the run took over {TIMEOUT} s:\n{tail(out / 'console.txt')}")
            time.sleep(1)
        if (rank := failed()) is not None:
            raise RuntimeError(f"rank {rank} failed:\n{tail(out / 'console.txt')}")
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
                p.wait()
        for f in files:
            f.close()


def create(specs: list[furu.Spec]) -> list:
    """One GPU node (config.gpu_node) for the runs; it stays up for 10 idle minutes (scancel it when done)."""
    return furu.create(
        specs,
        on=(
            SlurmWorkerBackend(
                max_workers=1,
                resources=gpu_node(),
                job_name="llmc-onclock-8gpu",
                reserve_for=ResourceFloor(gpus=8),
                worker_idle_timeout=10 * 60,
                max_failures_per_worker=100,  # a failing run must not cost the node
            ),
        ),
    )


# 210 steps, 3 seeds, and 200 and 220
BATCH: list[furu.Spec] = [
    *(LlmcRun(steps=210, seed=seed) for seed in (0, 1, 2)),
    *(LlmcRun(steps=steps) for steps in (200, 220)),
]

if __name__ == "__main__":
    from retrieval_blog_post import llmc_onclock  # the specs' module by name, not __main__

    for spec, result in zip(llmc_onclock.BATCH, llmc_onclock.create(llmc_onclock.BATCH)):
        print(spec, result)
