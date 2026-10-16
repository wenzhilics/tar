#!/usr/bin/env bash
set -euo pipefail

MODEL=$1
TAG=$2
OUT=eval/results/${TAG}
export TOKENIZERS_PARALLELISM=false

python3 eval/math_eval.py --model "${MODEL}" --output_dir "${OUT}/pass1" --n_sampling 1
python3 eval/math_eval.py --model "${MODEL}" --output_dir "${OUT}/pass8" --n_sampling 8
python3 eval/diversity.py --result_dir "${OUT}/pass8"
