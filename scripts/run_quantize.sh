#!/bin/bash
# scripts/run_quantize.sh
export PYTHONPATH=$(pwd):$PYTHONPATH
python src/quantize.py --config "$1"