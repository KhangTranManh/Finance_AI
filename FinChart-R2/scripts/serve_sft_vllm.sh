#!/usr/bin/env bash
set -euo pipefail

ROOT="${FINCHART_SFT_VLLM_ROOT:-/root/finchart_sft_vllm}"
VENV="${ROOT}/venv"
MODEL_DIR="${ROOT}/run/merged_finchartsft_bf16"
API_KEY_FILE="${ROOT}/runtime/api_key"

if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
  echo "Missing merged model: ${MODEL_DIR}" >&2
  exit 1
fi
if [[ ! -s "${API_KEY_FILE}" ]]; then
  echo "Missing API key file: ${API_KEY_FILE}" >&2
  exit 1
fi

export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${HF_HOME:-${ROOT}/cache/huggingface}"

exec "${VENV}/bin/vllm" serve "${MODEL_DIR}" \
  --host 127.0.0.1 \
  --port 8999 \
  --api-key "$(<"${API_KEY_FILE}")" \
  --served-model-name FinChart-SFT-408 \
  --dtype bfloat16 \
  --max-model-len 2048 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.80 \
  --limit-mm-per-prompt '{"image": 1}' \
  --generation-config vllm \
  --trust-remote-code
