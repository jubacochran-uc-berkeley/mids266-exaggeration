#!/bin/bash
# Run a single experiment
# Usage: bash scripts/run_experiment.sh configs/roberta_full.yaml
set -e

CONFIG=${1:?Usage: bash scripts/run_experiment.sh <config_path>}

echo "Running experiment: $CONFIG"
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):$PYTHONPATH"
python src/train.py --config "$CONFIG"