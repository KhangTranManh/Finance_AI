#!/usr/bin/env bash
set -euo pipefail

JOB_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$JOB_DIR/venv"
OUTPUT_DIR="$JOB_DIR/output"
PAIR_FILE="$JOB_DIR/data/phase2c_teacher_v1_dpo_candidates_provisional.jsonl"

mkdir -p "$OUTPUT_DIR" "$JOB_DIR/logs"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  python3 -m venv --system-site-packages "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install --upgrade --no-cache-dir \
  "trl[peft]==0.29.0" \
  "transformers>=5.0.0,<6.0.0" \
  "datasets>=4.4.0,<6.0.0" \
  "peft>=0.18.0,<0.21.0" \
  accelerate bitsandbytes huggingface_hub qwen-vl-utils

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1

"$VENV_DIR/bin/python" "$JOB_DIR/scripts/train_phase2c_dpo_remote_4090.py" \
  --pairs "$PAIR_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --hub-model-id "Kxck/Finance_500_v1_DPO_386_provisional" \
  --max-visual-tokens 1024
