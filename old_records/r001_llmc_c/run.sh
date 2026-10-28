#!/bin/bash
# run.sh DATA [STEPS]: the record's command (https://github.com/karpathy/llm.c/discussions/481) on this node's 8 GPUs,
# DATA holding fineweb_train_*.bin and fineweb_val_000000.bin; STEPS (-x) defaults to one epoch of the shards.
# The ranks meet through llm.c's file-system rendezvous (-pi fs), rank 0 first.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
DATA=$1; STEPS=${2:--1}; NP=${NP:-8}
BIN=${BIN:-$HERE/build/train_gpt2cu}
RDV=$(mktemp -d)
ARGS=(-i "$DATA/fineweb_train_*.bin" -j "$DATA/fineweb_val_000000.bin"
      -e d12 -b 64 -t 1024 -d 524288 -r 1 -z 1 -c 0.1 -l 0.0006 -q 0.0 -u 700 -n 5000 -v 250 -s 20000 -h 1 -x "$STEPS")
pids=()
for r in $(seq 0 $((NP - 1))); do
  "$BIN" "${ARGS[@]}" -pi fs -pf "$RDV" -pn $NP -pr $r -pg $NP > "rank$r.log" 2>&1 &
  pids+=($!)
  if [ $r = 0 ]; then until [ -s "$RDV/ncclUniqueId.sync" ]; do sleep 0.2; done; fi
done
for p in "${pids[@]}"; do wait $p; done
tail -3 rank0.log
