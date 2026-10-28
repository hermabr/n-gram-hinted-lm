"""How fast the validation hint statistics build on a node's 8 GPUs: old_records/gpu_val_stats.py loads the training
shards, 1/8 per GPU, counts the validation contexts' next tokens at every level of ALL_LEVELS, merges the GPUs' counts,
and makes the count rows (ValCountRows') and each variant's hint rows (ValHintRows'), timed from process start to the
rows in rank 0's host memory. Staging the shards in /dev/shm and building the kernels are before the clock.

Each of `runs` is one torchrun with gpu_val_stats.py's flags (see its --help), NAME=value environment variables, and
"+hold": a process holding a CUDA context on every GPU during the run, as the training would. The first run's outputs
are checked against ValCountRows and ValHintRows (exactly). It runs in old_records.py's BATCH."""

import json
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import furu

from retrieval_blog_post.fineweb import TRAIN_SHARDS, FinewebShards
from retrieval_blog_post.val_cache import LOCAL_DISK, REPO, TORCH_VENV
from retrieval_blog_post.val_hints import VARIANTS

# old_records.py's specs are imported where used: old_records.py imports this module (its BATCH)
OLD = REPO / "old_records"
SCRIPT = OLD / "gpu_val_stats.py"
BEST = "NCCL_NVLS_ENABLE=0 NCCL_CUMEM_ENABLE=0"
FAST = "--load-mode pread --async-init --nccl-warmup --direct"


@dataclass(frozen=True)
class ValStatsTimes:
    """Per run: the end-to-end seconds (process start to the rows on rank 0), and whether the first run's outputs
    equal the reference rows; timings.json, compare.json and the outputs are in `directory` (run0/, run1/, ...)."""

    directory: Path
    totals: tuple[float, ...]
    exact: bool


class ValStatsTiming(furu.Spec[ValStatsTimes]):
    """gpu_val_stats.py on `gpus` GPUs over the first `shards` training shards, once per entry of `runs`. `version`
    only distinguishes reruns of changed code."""

    shards: int = TRAIN_SHARDS
    gpus: int = 8
    # NCCL_CUMEM_ENABLE=0: NCCL's legacy cudaIpc peer mappings set up in about 1 s on the H200s, against about 6 s with
    # cuMem (NCCL 2.27's default) and 8.7 s with torch's CUDA IPC
    runs: tuple[str, ...] = (
        f"{BEST} +hold {FAST} --early-ctx --fused",
        f"{BEST} NCCL_IB_DISABLE=1 +hold {FAST} --early-ctx --fused",
        f"{BEST} NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo +hold {FAST} --early-ctx --fused",
        f"{BEST} NCCL_IB_DISABLE=1 +hold {FAST} --early-ctx --fused --chunk-mb 64",
        f"{BEST} NCCL_IB_DISABLE=1 {FAST} --early-ctx --fused",
        f"{BEST} NCCL_IB_DISABLE=1 +hold {FAST} --early-ctx --fused --bench-io",
    )
    threads: int = 12
    version: int = 12

    def metadata(self) -> furu.Metadata:
        return furu.Metadata(requires=furu.Requires(gpus=self.gpus))

    @furu.dependency
    def data(self) -> list[furu.Spec]:
        return [FinewebShards(shards=self.shards), FinewebShards(shards="val")]

    @furu.dependency
    def reference(self) -> list[furu.Spec]:
        from retrieval_blog_post.old_records import ValCountRows, ValHintRows

        if self.shards != TRAIN_SHARDS:
            return []
        return [ValCountRows(), *(ValHintRows(levels=v) for v in VARIANTS)]

    def _hold(self, env: dict) -> subprocess.Popen:
        """A process with a CUDA context on every GPU, once it has them (until killed)."""
        code = (
            "import sys, time, torch\n"
            "[torch.zeros(1, device=f'cuda:{i}') for i in range(torch.cuda.device_count())]\n"
            "print('ready', flush=True)\n"
            "time.sleep(3600)\n"
        )
        holder = subprocess.Popen([TORCH_VENV / "bin/python", "-c", code], env=env, stdout=subprocess.PIPE, text=True)
        assert holder.stdout is not None
        line = holder.stdout.readline()
        if line.strip() != "ready":
            holder.kill()
            raise RuntimeError(f"the GPU holder failed: {line!r}")
        return holder

    def create(self) -> ValStatsTimes:
        out = self.directory.data
        train = FinewebShards(shards=self.shards).create()
        (val,) = FinewebShards(shards="val").create()
        need = sum(s.path.stat().st_size for s in [*train, val])
        # /dev/shm (the worker's memory holds it; "register" pins tmpfs pages), else the node's NVMe
        stage = Path("/dev/shm")
        if shutil.disk_usage(stage).free < 1.2 * need:
            LOCAL_DISK.mkdir(parents=True, exist_ok=True)
            stage = LOCAL_DISK
        notes: dict = {"stage_dir": str(stage), "stage_bytes": need}
        with tempfile.TemporaryDirectory(dir=stage, prefix="val-stats-") as local:
            local = Path(local)
            data = local / "data"
            data.mkdir()
            copies = [(s.path, data / s.path.name) for s in [*train, val]]
            start = time.time()
            with ThreadPoolExecutor(32) as pool:
                list(pool.map(lambda c: shutil.copyfile(*c), copies))
            notes["stage_copy_s"] = time.time() - start

            def prefault(path: Path) -> int:
                with open(path, "rb", buffering=0) as f:
                    n = 0
                    while chunk := f.read(64 << 20):
                        n += len(chunk)
                return n

            start = time.time()
            with ThreadPoolExecutor(32) as pool:
                read = sum(pool.map(prefault, [c[1] for c in copies]))
            notes["prefault_s"] = time.time() - start
            notes["prefault_GBps"] = read / notes["prefault_s"] / 1e9
            self.logger.info(f"staged {len(copies)} files in {local}: {notes}")
            (local / "shards.txt").write_text("".join(f"{data / s.path.name}\n" for s in train))
            env = {
                **os.environ,
                "VIRTUAL_ENV": str(TORCH_VENV),
                "PATH": f"{TORCH_VENV}/bin:{os.environ['PATH']}",
                "PYTHONUNBUFFERED": "1",
                "OMP_NUM_THREADS": "1",
                "TMPDIR": str(local),
                "GVS_BUILD_DIR": str(local / "build"),
            }
            start = time.time()
            subprocess.run([TORCH_VENV / "bin/python", SCRIPT], env={**env, "GVS_BUILD_ONLY": "1"}, check=True)
            notes["build_s"] = time.time() - start
            totals = []
            for i, mode in enumerate(self.runs):
                run = out / f"run{i}"
                tokens = mode.split()
                hold = "+hold" in tokens
                tokens = [t for t in tokens if not t.startswith("+")]
                run_env = dict(t.split("=", 1) for t in tokens if "=" in t and not t.startswith("-"))
                args = [
                    "--shards", str(local / "shards.txt"),
                    "--val", str(data / val.path.name),
                    "--out", str(run),
                    "--threads", str(self.threads),
                    *(t for t in tokens if t.startswith("-") or "=" not in t),
                ]
                if i > 0:
                    args.append("--no-save")
                holder = self._hold(env) if hold else None
                launch = time.time()
                with open(out / f"console{i}.txt", "w") as console:
                    done = subprocess.run(
                        [
                            TORCH_VENV / "bin/python", "-m", "torch.distributed.run", "--standalone",
                            f"--nproc_per_node={self.gpus}", SCRIPT, *args,
                        ],
                        env={**env, **run_env, "GVS_LAUNCH_TIME": repr(launch)},
                        cwd=out,
                        stdout=console,
                        stderr=subprocess.STDOUT,
                    )
                if holder is not None:
                    holder.kill()
                    holder.wait()
                notes[f"run{i}"] = {"mode": mode, "subprocess_s": time.time() - launch, "returncode": done.returncode}
                if done.returncode != 0:
                    if i == 0:
                        raise RuntimeError(f"run 0 ({mode}) failed: see {out / 'console0.txt'}")
                    self.logger.warning(f"run {i} ({mode}) failed: see console{i}.txt")
                    totals.append(float("nan"))
                    continue
                timings = json.loads((run / "timings.json").read_text())
                totals.append(max(t["total_s"] for t in timings))
                notes[f"run{i}"]["total_s"] = totals[-1]
                self.logger.info(f"run {i} ({mode}): {totals[-1]:.3f} s end to end, {notes[f'run{i}']}")
                (out / "notes.json").write_text(json.dumps(notes, indent=1))
        exact = False
        if self.shards == TRAIN_SHARDS:
            from retrieval_blog_post.old_records import ValCountRows, ValHintRows

            counts = ValCountRows().create()
            refs = [f"{'_'.join(map(str, v))}={ValHintRows(levels=v).create().directory}" for v in VARIANTS]
            result = subprocess.run(
                [TORCH_VENV / "bin/python", OLD / "gpu_val_stats_compare.py", str(out / "run0"), str(counts.directory),
                 *refs],
                check=True, capture_output=True, text=True,
            )
            report = json.loads(result.stdout)
            cells = [v for per in report["counts"].values() for v in (per.values() if isinstance(per, dict) else [per])]
            cells += [v for r in report["rows"].values() for v in r.values()]
            exact = all(isinstance(v, dict) and v.get("differ", v.get("rows_differ")) == 0 for v in cells)
            self.logger.info(f"exact: {exact}; {json.dumps(report)}")
        (out / "notes.json").write_text(json.dumps({**notes, "totals": totals, "exact": exact}, indent=1))
        return ValStatsTimes(out, tuple(totals), exact)

