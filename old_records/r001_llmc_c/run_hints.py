"""One run of build/train_gpt2cu on this node's 8 GPUs: with the hints, their validation rows built on CPUs
(HINT_CPU_VAL=1, llmc/cpu_val.h), or with --no-hints the same recipe and pinning on the same clock (HINTS=0,
CLOCK_EVERY_STEP=1). Standard library only.

    python3 run_hints.py --shards DIR --out OUT [--subsets labels.npy] [--steps 470] [--seed 0] [--no-hints]

DIR holds the training shards and fineweb_val_000000.bin, best staged in /dev/shm interleaved over the sockets
(numactl --interleave=all cp). The training reads the first ceil(steps / 190) shards, the hints' corpus all of them.
Each rank runs on one physical core of its GPU's NUMA node, the CPU build beside the training on the other CPUs, the
lookup after the clock on all of them. --subsets (old_records.ValSubsetLabels' labels.npy) adds the losses on the
validation subsets. The ranks' logs go to OUT/rank{r}.log, the result to OUT/result.json and stdout.
"""

import argparse
import glob
import json
import math
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
# the PyTorch port's best on-clock recipe (README): 4x512, aux 1, wsdlin, lr 3 x 6e-4, the hint lr 3 x 0.002667
HINTED = ["-e", "custom:4:512:8", "-b", "64", "-t", "1024", "-d", "524288", "-r", "1", "-z", "1", "-c", "0.1",
          "-l", "0.0018", "-q", "0.0", "-u", "50", "-k", "wsdlin", "-n", "0", "-v", "100000", "-m", "20", "-s", "0",
          "-h", "0"]
HINT_ENV = {"HINTS": "1", "HINT_LR": "0.008", "HINT_AUX": "1", "HINT_CPU_VAL": "1", "NCCL_CUMEM_ENABLE": "0",
            "NCCL_NVLS_ENABLE": "0"}
NO_HINT_ENV = {"HINTS": "0", "CLOCK_EVERY_STEP": "1", "NCCL_CUMEM_ENABLE": "0", "NCCL_NVLS_ENABLE": "0"}
STEPS_PER_SHARD = 190
TIMEOUT = 900


def cpu_list(text: str) -> list[int]:
    out = []
    for item in text.strip().split(","):
        a, _, b = item.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return out


def cpu_ranges(cpus: list[int]) -> str:
    out, i = [], 0
    while i < len(cpus):
        j = i
        while j + 1 < len(cpus) and cpus[j + 1] == cpus[j] + 1:
            j += 1
        out.append(f"{cpus[i]}-{cpus[j]}" if j > i else str(cpus[i]))
        i = j + 1
    return ",".join(out)


def gpu_nodes(n: int) -> list[int]:
    """Each GPU's NUMA node (nvidia-smi's PCI bus ids and sysfs), -1 if unknown."""
    try:
        ids = subprocess.run(["nvidia-smi", "--query-gpu=pci.bus_id", "--format=csv,noheader"], capture_output=True,
                             text=True, check=True).stdout.split()
        nodes = []
        for bus in ids[:n]:
            with open(f"/sys/bus/pci/devices/{bus.lower()[-12:]}/numa_node") as f:
                nodes.append(int(f.read()))
        return nodes
    except Exception:
        return [-1] * n


def pinning(nproc: int) -> tuple[list[list[int]], list[int]]:
    """Per rank one whole physical core of its GPU's NUMA node (the first free one of the job's CPUs), and the rest."""
    allowed = sorted(os.sched_getaffinity(0))
    nodes = {int(re.search(r"node(\d+)", p).group(1)): cpu_list(open(p).read())
             for p in glob.glob("/sys/devices/system/node/node*/cpulist")}
    free = set(allowed)
    ranks = []
    for node in gpu_nodes(nproc):
        cpu = next(c for c in [*nodes.get(node, []), *allowed] if c in free)
        try:
            core = cpu_list(open(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list").read())
        except OSError:
            core = [cpu]
        mine = sorted(c for c in core if c in free)
        free -= set(mine)
        ranks.append(mine)
    return ranks, sorted(free)


def parse(log: str, steps: int) -> dict:
    text = open(log, errors="replace").read()
    res: dict = {"steps": steps}
    if m := re.findall(r"FINAL val_loss:([\d.]+) train_time_ms:(\d+) steps:(\d+)", text):
        res["val"], res["clock_ms"] = float(m[-1][0]), int(m[-1][1])
    if m := re.findall(r"VAL_SUBSETS all:([\d.]+) noexact:([\d.]+) nonear:([\d.]+)", text):
        res["all"], res["noexact"], res["nonear"] = map(float, m[-1])
    step_ms = [float(x) for x in re.findall(r"^step +\d+/\d+ \|.*\| ([\d.]+) ms \|", text, re.M)]
    if step_ms:
        res["step0_ms"] = step_ms[0]
        res["step_ms_mean"] = sum(step_ms[1:]) / max(1, len(step_ms) - 1)
    if m := re.findall(r"hints: training rows in (\d+) ms", text):
        res["train_rows_ms"] = int(m[-1])
    # on the clock the wait for the build (build_wait_ms); after the clock the rows (val_rows_ms: rank 0 reading the
    # validation tokens and looking them up, the copies)
    if m := re.findall(r"validation rows \(CPU\) built (\d+) ms \(wall\) after the build's start \(waited for (\d+) ms "
                       r"after the last step\); after the clock, on the GPUs in (\d+) ms \(waiting (\d+): reading the "
                       r"validation tokens (\d+), the lookup (\d+); copies (\d+)\)", text):
        (res["build_end_ms"], res["build_wait_ms"], res["val_rows_ms"], res["val_wait_ms"], res["val_read_ms"],
         res["val_lookup_ms"], res["val_copy_ms"]) = map(int, m[-1])
    if m := re.findall(r"hints: cpu val build: (.*)$", text, re.M):
        res["cpu_build"] = {k: float(v) for k, v in (kv.split("=", 1) for kv in m[-1].split())}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=os.path.join(HERE, "build", "train_gpt2cu"))
    ap.add_argument("--shards", required=True)
    ap.add_argument("--steps", type=int, default=470)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--subsets", help="old_records.ValSubsetLabels' labels.npy")
    ap.add_argument("--env", action="append", default=[], help="K=V added to the ranks' environment")
    ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--no-hints", action="store_true", help="the recipe without the hints, same clock")
    a = ap.parse_args()
    data = os.path.join(a.out, "data")
    os.makedirs(data, exist_ok=True)
    every = sorted(glob.glob(os.path.join(a.shards, "fineweb_train_*.bin")))
    for p in every[:math.ceil(a.steps / STEPS_PER_SHARD)] + [os.path.join(a.shards, "fineweb_val_000000.bin")]:
        dst = os.path.join(data, os.path.basename(p))
        if not os.path.exists(dst):
            os.symlink(p, dst)
    shard_list = os.path.join(a.out, "val_shards.txt")
    with open(shard_list, "w") as f:
        f.write("".join(p + "\n" for p in every))
    ranks, rest = pinning(a.nproc)
    env = {**os.environ, **HINT_ENV, "SEED": str(a.seed), "HINT_VAL_SHARDS": shard_list,
           **({"HINT_VAL_SUBSETS": a.subsets} if a.subsets else {}), "HINT_CPU_VAL_CPUS": cpu_ranges(rest),
           "HINT_CPU_VAL_LOOKUP_CPUS": cpu_ranges(sorted(os.sched_getaffinity(0))),
           "HINT_CPU_VAL_SHM": f"/dev/shm/llmc_cpuval_{os.getpid()}"}
    if a.no_hints:
        env = {**os.environ, **NO_HINT_ENV, "SEED": str(a.seed)}
    env |= dict(kv.split("=", 1) for kv in a.env)
    argv = [a.bin, "-i", os.path.join(data, "fineweb_train_*.bin"), "-j", os.path.join(data, "fineweb_val_000000.bin"),
            *HINTED, "-x", str(a.steps)]
    rdv = os.path.join(a.out, "nccl")
    os.makedirs(rdv, exist_ok=True)
    for f in glob.glob(os.path.join(rdv, "*")):
        os.unlink(f)
    info = {"argv": argv, "env": {k: v for k, v in env.items() if k.startswith(("HINT", "SEED", "NCCL", "CLOCK"))},
            "rank_cpus": [cpu_ranges(c) for c in ranks]}
    print(json.dumps(info), flush=True)
    procs = []
    start = time.time()
    try:
        for r in range(a.nproc):
            cmd = ["taskset", "-c", cpu_ranges(ranks[r]), *argv, "-pi", "fs", "-pf", rdv, "-pn", str(a.nproc),
                   "-pr", str(r), "-pg", str(a.nproc)]
            log = open(os.path.join(a.out, f"rank{r}.log"), "w")
            procs.append(subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=a.out))
            if r == 0:  # the others once rank 0 has written NCCL's id
                while not (os.path.exists(f"{rdv}/ncclUniqueId.sync") and os.path.getsize(f"{rdv}/ncclUniqueId.sync") >= 128):
                    if procs[0].poll() is not None:
                        raise RuntimeError("rank 0 exited before the others started")
                    if time.time() - start > TIMEOUT:
                        raise TimeoutError(f"rank 0 wrote no NCCL id in {TIMEOUT} s")
                    time.sleep(0.1)
        while any(p.poll() is None for p in procs):
            if any(p.poll() not in (None, 0) for p in procs) or time.time() - start > TIMEOUT:
                break
            time.sleep(0.5)
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
                p.wait()
        if os.path.exists(env.get("HINT_CPU_VAL_SHM", "")):  # a run that failed before mapping it everywhere
            os.unlink(env["HINT_CPU_VAL_SHM"])
    rcs = [p.returncode for p in procs]
    res = parse(os.path.join(a.out, "rank0.log"), a.steps) | {"seed": a.seed, "rcs": rcs, **info}
    with open(os.path.join(a.out, "result.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res), flush=True)
    if any(rcs) or "val" not in res:
        os.system(f"tail -20 {a.out}/rank0.log")
        sys.exit(1)


if __name__ == "__main__":
    main()
