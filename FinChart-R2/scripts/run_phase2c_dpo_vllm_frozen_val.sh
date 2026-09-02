#!/usr/bin/env bash
set -euo pipefail

# Portable Linux-GPU launcher for the frozen ChartQA val[0:500] evaluation.
# It deliberately does not source .env. Public Hub artifacts need no token;
# otherwise export HF_TOKEN in the shell before running this script.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_ROOT="${FINCHART_VLLM_RUN_ROOT:-${PROJECT_DIR}/.artifacts/phase2c_dpo_vllm}"
VENV_DIR="${RUN_ROOT}/venv"
MERGED_DIR="${RUN_ROOT}/merged_sft_dpo_386_bf16"
OUTPUT_DIR="${RUN_ROOT}/val_0_500"
ADAPTER_ID="${FINCHART_DPO_ADAPTER:-Kxck/Finance_500_v1_DPO_386_provisional}"
ADAPTER_REVISION="${FINCHART_DPO_REVISION:-2a6df004d4522cb5b7a072fd6b8dea4d54d7b64d}"

mkdir -p "${RUN_ROOT}" "${OUTPUT_DIR}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  python3 -m venv "${VENV_DIR}"
fi
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install -r "${SCRIPT_DIR}/requirements-phase2c-vllm.txt"

if [[ ! -f "${MERGED_DIR}/config.json" ]]; then
  "${VENV_DIR}/bin/python" "${SCRIPT_DIR}/merge_phase2c_dpo_for_vllm.py" \
    --adapter "${ADAPTER_ID}" \
    --adapter-revision "${ADAPTER_REVISION}" \
    --output "${MERGED_DIR}"
fi

# vLLM 0.28 otherwise selects a FlashInfer JIT sampler that requires a system
# CUDA toolkit. Greedy decoding is unchanged when the PyTorch sampler is used.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

"${VENV_DIR}/bin/python" "${SCRIPT_DIR}/evaluate_phase2c_dpo_vllm_frozen_val.py" \
  --model "${MERGED_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size "${FINCHART_VLLM_BATCH_SIZE:-8}" \
  --gpu-memory-utilization "${FINCHART_VLLM_GPU_MEMORY:-0.82}" \
  --resume

echo "Evaluation artifacts: ${OUTPUT_DIR}"
