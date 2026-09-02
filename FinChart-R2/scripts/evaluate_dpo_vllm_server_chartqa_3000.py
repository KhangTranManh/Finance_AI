#!/usr/bin/env python3
"""Evaluate the cumulative SFT+DPO adapter through the vLLM HTTP server.

This deliberately reuses the SFT evaluator implementation so both adapters use
the same ChartQA slices, prompt, decoding, matcher, retry, and resume logic.
"""

from __future__ import annotations

import asyncio

import evaluate_sft_vllm_server_chartqa_3000 as evaluator


evaluator.MODEL_NAME = "FinChart-SFT-DPO-386"
evaluator.SOURCE_ADAPTER = "Kxck/Finance_500_v1_DPO_386_provisional"
evaluator.OUTPUT_STEM = "dpo_vllm_chartqa_val500_test2500"


if __name__ == "__main__":
    asyncio.run(evaluator.main_async())
