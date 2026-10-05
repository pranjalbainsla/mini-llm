#!/usr/bin/env bash
# Run one config over several seeds. init_seed and train_seed move together;
# eval_seed stays fixed so every run is scored on the same eval batches.
# Usage: ./run_seeds.sh <tag> [extra toy_train.py args...]
#   ./run_seeds.sh baseline
#   SEEDS=1 ./run_seeds.sh gqa --attention=gqa      # single-seed screen (seed 1)
#   SEEDS="2 3 4 5" ./run_seeds.sh gqa --attention=gqa  # extra seeds, only if borderline
set -e
mkdir -p results
tag=$1; shift
for seed in ${SEEDS:-1 2 3 4 5}; do
  python toy_train.py configs/base.py --init_seed=$seed --train_seed=$seed \
    --run_name=${tag}_s${seed} "$@" \
    > results/${tag}_s${seed}.log 2>&1
  echo "done $tag seed $seed"
done
