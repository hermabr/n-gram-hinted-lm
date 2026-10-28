"""Record #1 run by llm.c's own trainer (train_gpt2.cu) with the first version of old_records/r001_llmc_c's hints,
whose rows are precomputed; llmc_onclock.py runs the later versions, which compute every row on the clock. The
first version's binaries are prebuilt in SCRATCH (BINARY, PRISTINE); its source is not published.

LlmcRun is OldRecordRun's non-online, non-late run with the binary in place of the PyTorch port
(old_records/r001_llmc/train_gpt.py): the same staged rows, the same knobs in the environment (the ones llm.c takes as
flags turned into its flags as the port derives them), and the same FINAL line. One deliberate difference from the
record: the data order is sequential (LLMC_SEQUENTIAL=1, STEPS_PER_SHARD steps per shard), not llm.c's shuffled shards
and documents, so that the training rows stay causal in the order the run reads them.

The runs are in old_records.py's BATCH (which imports this module after BEST_1 is defined)."""

import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import furu

from retrieval_blog_post import llmc_onclock  # by module: circular via old_records
from retrieval_blog_post.config import LOCAL_DISK, SCRATCH
from retrieval_blog_post.fineweb import FinewebShards
from retrieval_blog_post.kgram import VAL_TOKENS
from retrieval_blog_post.old_records import (
    BEST_1,
    COUNT_LEVELS,
    COUNT_NAMES,
    ENTRIES,
    TRAIN_HINT_SHARDS,
    CountRowFiles,
    HintRowFiles,
    RunResult,
    TrainCountRows,
    TrainHintRows,
    ValCountRows,
    ValHintRows,
)
from retrieval_blog_post.val_hints import LEVELS, cells

BINARY = str(SCRATCH / "llmc-hints/bin/train_gpt2cu_hints")
PRISTINE = str(SCRATCH / "llmc-hints/bin/train_gpt2cu_pristine")  # llm.c at 7ecd8906 as is (shuffled data, cosine)
DEFAULT_STEPS = 265  # the port's default (r001_llmc/train_gpt.py)
RECORD_STEPS = 19560  # llm.c's -x for one epoch of FineWeb-10B
RECORD_WARMUP = 700  # its -u
TIMEOUT = 40 * 60  # the whole run, staging excluded

# v2 is the port's best recipe (4x512, 650 steps), v1 its pass at 12x768 (with a 0.1 cooldown), and v1 cosine
# llm.c's own recipe at the steps that fit
LLMC_V2 = BEST_1
LLMC_V1 = (
    ("TRAIN_STEPS", "245"), ("REC_LM_AUX", "1"), ("REC_LR_SCHEDULE", "wsd"), ("REC_COOLDOWN_FRAC", "0.1"),
    ("REC_WARMUP_STEPS", "30"), ("LR_SCALE", "1.5"), ("REC_HINT_LR", "0.005333"),
)
LLMC_V1_COSINE = (("TRAIN_STEPS", "248"),)


class LlmcRun(furu.Spec[RunResult]):
    """llm.c's trainer (`binary`), seed `seed`, labelled `version`, with the hint rows of the variant `levels` (None:
    without hints) and `env` added to its environment (later pairs override earlier ones). `counts` C > 0 also stages
    the n-gram count rows of the first C training shards and sets HINT_NGRAM=1 unless `env` sets it, as OldRecordRun.
    The run stages the shards it reads and one more where it has hint rows (llm.c checks every shard its glob
    matches)."""

    version: str
    levels: tuple[int, ...] | None = None
    env: tuple[tuple[str, str], ...] = ()
    seed: int = 0
    counts: int = 0
    binary: str = BINARY

    def metadata(self) -> furu.Metadata:
        return furu.Metadata(requires=furu.Requires(gpus=8))

    def _steps(self) -> int:
        return int(dict(self.env).get("TRAIN_STEPS", DEFAULT_STEPS))

    def _read_shards(self) -> int:
        """The training shards the run reads."""
        return max(1, math.ceil(self._steps() / llmc_onclock.STEPS_PER_SHARD))

    def _train_shards(self) -> int:
        """The training shards staged: the read ones and one more, within the hint rows' TRAIN_HINT_SHARDS."""
        n = self._read_shards()
        return n + 1 if self.levels is None or n + 1 <= TRAIN_HINT_SHARDS else n

    def check(self) -> None:
        """Raises ValueError if the run would read past its rows."""
        n = self._read_shards()
        if self.counts > 0 and self.levels is None:
            raise ValueError("count rows need a hint variant's levels")
        if self.levels is not None and n > TRAIN_HINT_SHARDS:
            raise ValueError(f"the run reads {n} shards, past the {TRAIN_HINT_SHARDS} with hint rows")
        if self.counts > 0 and n > self.counts:
            raise ValueError(f"the run reads {n} shards, past the {self.counts} with count rows")
        if dict(self.env).get("REC_LR_SCHEDULE", "cosine") not in ("cosine", "wsd"):
            raise ValueError(f"REC_LR_SCHEDULE {dict(self.env)['REC_LR_SCHEDULE']!r}: cosine or wsd")

    @furu.dependency
    def shards(self) -> list[furu.Spec]:
        return [FinewebShards(shards=self._train_shards()), FinewebShards(shards="val")]

    def _row_specs(self) -> tuple[ValHintRows, TrainHintRows]:
        assert self.levels is not None
        return ValHintRows(levels=self.levels, slots=1, buckets=LEVELS), TrainHintRows(
            levels=self.levels, slots=1, buckets=LEVELS, shards=TRAIN_HINT_SHARDS
        )

    def _count_specs(self) -> tuple[ValCountRows, TrainCountRows]:
        return ValCountRows(), TrainCountRows(shards=self.counts)

    @furu.dependency
    def rows(self) -> list[furu.Spec[HintRowFiles]]:
        return [] if self.levels is None else list(self._row_specs())

    @furu.dependency
    def count_rows(self) -> list[furu.Spec[CountRowFiles]]:
        return list(self._count_specs()) if self.levels is not None and self.counts > 0 else []

    def create(self) -> RunResult:
        self.check()
        out = self.directory.data
        n = self._train_shards()
        LOCAL_DISK.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=LOCAL_DISK) as local:
            local = Path(local)
            data = local / "data/fineweb100B"
            data.mkdir(parents=True)
            (local / "nccl").mkdir()  # the fs rendezvous of NCCL's id (-pi fs -pf)
            copies = [
                (s.path, data / s.path.name)
                for s in [*FinewebShards(shards=n).create(), *FinewebShards(shards="val").create()]
            ]
            if self.levels is not None:
                hints = local / "hints"
                val, train = (spec.create() for spec in self._row_specs())
                for name in ("cell", "tokens", "weights"):
                    copies.append((val.directory / f"{name}.npy", hints / "val" / f"{name}.npy"))
                    for j in range(n):
                        copies.append((train.directory / f"{j:04d}/{name}.npy", hints / f"train/{j:04d}/{name}.npy"))
                meta = {
                    "levels": list(self.levels),
                    "cells": cells(LEVELS),
                    "entries": ENTRIES,
                    "train_shards": n,
                    "val_tokens": VAL_TOKENS,
                    "slots": 1,
                    "buckets": list(LEVELS),
                }
                if self.counts > 0:
                    val_counts, train_counts = (spec.create() for spec in self._count_specs())
                    for name in COUNT_NAMES:
                        copies.append((val_counts.directory / f"{name}.npy", hints / "val" / f"{name}.npy"))
                        for j in range(self.counts):
                            copies.append(
                                (train_counts.directory / f"{j:04d}/{name}.npy", hints / f"train/{j:04d}/{name}.npy")
                            )
                    copies.append((train_counts.directory / "unigram.npy", hints / "unigram.npy"))
                    meta |= {"count_levels": list(COUNT_LEVELS), "count_shards": self.counts}
                hints.mkdir()
                (hints / "meta.json").write_text(json.dumps(meta))
            for _, target in copies:
                target.parent.mkdir(parents=True, exist_ok=True)
            start = time.time()
            with ThreadPoolExecutor(32) as pool:
                list(pool.map(lambda c: shutil.copyfile(*c), copies))
            self.logger.info(f"copied {len(copies)} files to {local} in {time.time() - start:.0f} s")
            commands = [llmc_command(self, local, rank) for rank in range(8)]
            argv, env = commands[0]
            self.logger.info(f"llmc {self.version}: {_added(env)}")
            self.logger.info(f"llmc rank 0: {subprocess.list2cmdline(argv)}")
            _launch(commands, out, self.logger)
        text = (out / "console.txt").read_text(errors="replace")
        final = re.findall(r"FINAL val_loss:([\d.]+) train_time_ms:(\d+) steps:(\d+)", text)
        if not final and (avg := re.findall(r"total average iteration time: ([\d.]+) ms", text)):
            # the pristine llm.c binary (PRISTINE): its last "val loss" and its mean step time (step 0 excluded) x steps
            losses = re.findall(r"^val loss ([\d.]+)", text, re.MULTILINE)
            final = [(losses[-1], str(int(float(avg[-1]) * self._steps())), str(self._steps()))]
        if not final:
            raise RuntimeError(f"the run's console lacks its final line:\n{llmc_onclock.tail(out / 'console.txt')}")
        loss, ms, steps = final[-1]
        result = RunResult(val_loss=float(loss), train_time_ms=int(ms), steps=int(steps))
        self.logger.info(f"llmc {self.version} {self.levels} {dict(self.env)}: {result}")
        for line in re.findall(r"^step_avg_ms:.*$", text, re.MULTILINE)[-1:]:
            self.logger.info(f"llmc {self.version}: {line}")
        return result


def llmc_command(spec: LlmcRun, local: Path, rank: int) -> tuple[list[str], dict[str, str]]:
    """Rank `rank`'s argv and environment for `spec`'s run staged in `local`, the flags derived from its env as
    r001_llmc/train_gpt.py derives its settings."""
    spec.check()
    knobs = dict(spec.env)
    steps = spec._steps()
    lr = 6e-4 * float(knobs.get("LR_SCALE", "1.0"))
    warmup = int(knobs["REC_WARMUP_STEPS"]) if "REC_WARMUP_STEPS" in knobs else max(
        1, round(RECORD_WARMUP * steps / RECORD_STEPS)
    )
    shape = tuple(int(knobs.get(k, d)) for k, d in (("REC_N_LAYER", 12), ("REC_N_EMBD", 768), ("REC_N_HEAD", 12)))
    model = "d12" if shape == (12, 768, 12) else "custom:{}:{}:{}".format(*shape)
    if "LLMC_INIT" in knobs:  # an llm.c checkpoint to start from (e.g. the PyTorch port's initial weights, same seed)
        model = knobs["LLMC_INIT"]
    schedule = "wsdlin" if knobs.get("REC_LR_SCHEDULE") == "wsd" else "cosine"
    data = local / "data/fineweb100B"
    argv = [
        spec.binary,
        "-i", str(data / "fineweb_train_*.bin"),  # llm.c globs it itself
        "-j", str(data / "fineweb_val_000000.bin"),
        "-e", model,
        "-b", "64", "-t", "1024", "-d", "524288",
        "-r", "1", "-z", "1", "-c", "0.1",
        "-l", "%.8g" % lr, "-q", "0.0", "-u", str(warmup), "-k", schedule,
        "-x", str(steps), "-v", "100000", "-m", "20", "-s", "0", "-h", "0", "-n", "0",
        "-pi", "fs", "-pf", str(local / "nccl"), "-pn", "8", "-pr", str(rank), "-pg", "8",
    ]
    env = {**os.environ, "SEED": str(spec.seed), "LLMC_SEQUENTIAL": "1", **knobs}
    if spec.levels is None:
        env["HINTS"] = "0"
    else:
        env["HINTS"] = knobs.get("HINTS", "1")
        env["HINT_DIR"] = str(local / "hints")
        if spec.counts > 0:
            env.setdefault("HINT_NGRAM", "1")
    return argv, env


def _added(env: dict[str, str]) -> dict[str, str]:
    """`env`'s entries not in (or different from) os.environ."""
    return {k: v for k, v in env.items() if os.environ.get(k) != v}


def _console(out: Path, rank: int) -> Path:
    return out / ("console.txt" if rank == 0 else f"console_rank{rank}.txt")


def _launch(commands: list[tuple[list[str], dict[str, str]]], out: Path, logger) -> None:
    """Runs the ranks' commands in `out`, rank r's output in _console(out, r). Rank 0 first, the others once it has
    written NCCL's id (llm.c's fs rendezvous could read it half-written); raises if any rank fails or after TIMEOUT."""
    procs: list[subprocess.Popen] = []
    files = []
    launched = time.time()
    deadline = launched + TIMEOUT

    def start(rank: int) -> None:
        argv, env = commands[rank]
        console = open(_console(out, rank), "w")
        files.append(console)
        procs.append(subprocess.Popen(argv, cwd=out, env=env, stdout=console, stderr=subprocess.STDOUT))

    def failed() -> int | None:
        return next((r for r, p in enumerate(procs) if p.poll() not in (None, 0)), None)

    def fail(rank: int) -> RuntimeError:
        return RuntimeError(
            f"rank {rank} exited with {procs[rank].returncode}:\n{llmc_onclock.tail(_console(out, rank))}"
            + (f"\nrank 0:\n{llmc_onclock.tail(_console(out, 0))}" if rank != 0 else "")
        )

    try:
        start(0)
        nccl_id = Path(commands[0][0][commands[0][0].index("-pf") + 1]) / "ncclUniqueId.sync"
        while not (nccl_id.exists() and nccl_id.stat().st_size >= 128):  # sizeof(ncclUniqueId)
            if (rank := failed()) is not None:
                raise fail(rank)
            if procs[0].poll() == 0:
                raise RuntimeError(f"rank 0 exited before the others started:\n{llmc_onclock.tail(_console(out, 0))}")
            if time.time() > launched + 300:
                raise TimeoutError(f"rank 0 wrote no NCCL id in 300 s:\n{llmc_onclock.tail(_console(out, 0))}")
            time.sleep(0.2)
        for rank in range(1, len(commands)):
            start(rank)
        logger.info(f"started {len(procs)} ranks")
        while any(p.poll() is None for p in procs):
            if (rank := failed()) is not None:
                raise fail(rank)
            if time.time() > deadline:
                raise TimeoutError(f"the run took over {TIMEOUT} s:\n{llmc_onclock.tail(_console(out, 0))}")
            time.sleep(2)
        if (rank := failed()) is not None:
            raise fail(rank)
    finally:
        live = [p for p in procs if p.poll() is None]
        for p in live:
            p.terminate()
        stop = time.time() + 30
        for p in live:
            try:
                p.wait(timeout=max(0.0, stop - time.time()))
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
        for f in files:
            f.close()
