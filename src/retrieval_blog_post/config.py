"""Everything specific to the cluster the results were computed on: the paths outside the repo, and the Slurm workers
that compute the specs. To run elsewhere, edit this file (or set RETRIEVAL_BLOG_POST_<NAME> for a path)."""

import os
from collections.abc import Sequence
from pathlib import Path

import furu
from furu.resources import ResourceFloor
from furu.worker.backends.slurm.backend import SlurmWorkerBackend
from furu.worker.backends.slurm.resources import MemoryPerNode, SlurmResources

# Set on the first import in the process that submits specs, from this file's location: the checkout (or worktree) it
# runs from. Furu's workers inherit that environment, so a Slurm worker, which runs the code from a snapshot, keeps
# writing to the submitter's checkout rather than into its snapshot.
os.environ.setdefault(
    "RETRIEVAL_BLOG_POST_FINAL_RESULTS",
    str(Path(__file__).resolve().parents[2] / "final-results"),
)


def final_results_root() -> Path:
    """Storage for specs whose results belong with the code (pass it as Metadata.storage)."""
    return Path(os.environ["RETRIEVAL_BLOG_POST_FINAL_RESULTS"])


def _path(name: str, default: str) -> Path:
    return Path(os.environ.get(f"RETRIEVAL_BLOG_POST_{name}", default))


# Paths outside the repo, as on the cluster the results were computed on: set RETRIEVAL_BLOG_POST_<NAME> to run
# elsewhere. Some end up in specs' fields (a binary's path), so their defaults keep those specs' identities.
DATA = _path("DATA", "/sphinx/u/brunborg/data/fineweb100B")  # a local copy of FineWeb100B's shards (fineweb.py)
SCRATCH = _path("SCRATCH", "/sphinx/u/brunborg/.tmp")  # builds and checkpoints the specs read
TORCH_VENV = _path("TORCH_VENV", "/sphinx/u/brunborg/venvs/record-torch210")  # the records' stack (val_cache.py)
NVCC = _path("NVCC", "/usr/local/cuda-12.6/bin/nvcc")  # CUDA 12.6's, for llm.c's build.sh
LOCAL_DISK = _path("LOCAL_DISK", "/scr/brunborg")  # a node's local disk, which runs stage their shards on
MODDED_NANOGPT = _path("MODDED_NANOGPT", "/sphinx/u/brunborg/modded-nanogpt")  # a modded-nanogpt checkout
# brunborg.com's checkout: blog.py's publish() replaces POST/furu with the figures (copied to STAGING first, on the
# same file system); twitter.py's publish() writes the thread's page into SITE, whose fonts twitter_media.py uses if
# present
SITE = _path("SITE", "/sphinx/u/brunborg/brunborg-com")
POST = _path("POST", str(SITE / "src/lib/blog/nanogpt-speedrun-retrieval-hints"))
STAGING = _path("STAGING", str(SCRATCH / "brunborg-com/src/lib/blog/nanogpt-speedrun-retrieval-hints/furu"))

# Slurm: the account, the partitions for whole GPU nodes, for large CPU workers, and for any CPU worker, and the nodes
# to avoid on the last.
ACCOUNT = "miso"
GPU_PARTITION = "miso"
CPU_PARTITION = "miso-lo"
SHARED_CPU_PARTITIONS = "sc-loprio,sphinx-lo,sphinx,miso-lo"
SHARED_CPU_EXCLUDE = (
    "tiger[6-8],jagupard[19-20,26-30,32-36],john[1-15,17],cocoflops[1-2],iliad6,pasteur[5-7],viscam[5,9]"
)


def _resources(
    partition: str, cpus: int, mem_gib: int, time_limit: str, gpus: int = 0, extra: tuple[str, ...] = ()
) -> SlurmResources:
    return SlurmResources(
        account=ACCOUNT,
        partition=partition,
        cpus_per_worker=cpus,
        gpus=gpus,
        memory=MemoryPerNode(mem_gib),
        time_limit=time_limit,
        extra_sbatch_args=extra,
    )


def cpu(*, n_cpus: int = 16, mem_gib: int = 64, time_limit: str = "2-00:00:00") -> SlurmResources:
    """A CPU worker on CPU_PARTITION, which has the large-memory nodes."""
    return _resources(CPU_PARTITION, n_cpus, mem_gib, time_limit)


def gpu_node(*, n_gpus: int = 8, time_limit: str = "6:00:00") -> SlurmResources:
    """A whole node of 8 H200s, as the record's runs hold one (160 CPUs: 192 never schedules)."""
    return _resources(GPU_PARTITION, 20 * n_gpus, 300 * n_gpus, time_limit, gpus=n_gpus)


def shared_cpu(*, n_cpus: int = 16, mem_gib: int = 64, time_limit: str = "2-00:00:00") -> SlurmResources:
    """A CPU worker on any of SHARED_CPU_PARTITIONS."""
    return _resources(SHARED_CPU_PARTITIONS, n_cpus, mem_gib, time_limit, extra=(f"--exclude={SHARED_CPU_EXCLUDE}",))


def create[T](specs: Sequence[furu.Spec[T]], job_name: str) -> list[T]:
    """Create `specs` and their missing dependencies on Slurm workers; a spec runs on any pool whose workers have what
    its metadata `requires`. This process coordinates the workers until everything is done, so keep it running (e.g.
    in tmux)."""
    from retrieval_blog_post.ladder import WORKER_GIB

    return furu.create(
        specs,
        on=(
            # for the build timings (ladder.BuildTime), one at a time per worker, nothing else on them
            SlurmWorkerBackend(
                max_workers=2,
                # a short limit lets the backfill scheduler start one between larger pending jobs
                resources=cpu(n_cpus=96, mem_gib=WORKER_GIB, time_limit="3:00:00"),
                job_name=f"{job_name}-{WORKER_GIB}g",
                reserve_for=ResourceFloor(memory_gib=WORKER_GIB),
            ),
            # for the record runs (ValCacheSweep, DataScaling and its NoHintsTraining runs) and the llm.c runs
            SlurmWorkerBackend(
                max_workers=5,
                resources=gpu_node(),
                job_name=f"{job_name}-8gpu",
                reserve_for=ResourceFloor(gpus=8),
            ),
            # for the jobs that ask for over 100 GB (KGramMatchStats)
            SlurmWorkerBackend(
                max_workers=8,
                resources=cpu(n_cpus=32, mem_gib=256),
                job_name=f"{job_name}-256g",
                reserve_for=ResourceFloor(memory_gib=101),
            ),
            SlurmWorkerBackend(max_workers=10, resources=cpu(), job_name=f"{job_name}-cpu"),
            SlurmWorkerBackend(max_workers=128, resources=shared_cpu(), job_name=f"{job_name}-shared"),
        ),
    )
