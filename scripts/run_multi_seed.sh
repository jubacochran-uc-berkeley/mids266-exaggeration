#!/bin/bash
# scripts/run_multi_seed.sh
# Usage: bash scripts/run_multi_seed.sh configs/roberta_full.yaml
#        bash scripts/run_multi_seed.sh configs/roberta_full.yaml 30

CONFIG=$1
N_RUNS=${2:-100}

export PYTHONPATH=$(pwd):$PYTHONPATH
python src/train.py --config "$CONFIG" --multi-seed --n-runs "$N_RUNS"