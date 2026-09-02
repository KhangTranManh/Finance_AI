#!/usr/bin/env python3
"""Merge the FinChart SFT adapter into Qwen3-VL for complete vLLM inference."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from huggingface_hub import HfApi
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor


BASE_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
SFT_ADAPTER = "Kxck/Finance_500_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", default=SFT_ADAPTER)
    parser.add_argument("--adapter-revision", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter_path = Path(args.adapter).expanduser()
    adapter_source = (
        str(adapter_path.resolve()) if adapter_path.exists() else str(args.adapter)
    )
    adapter_revision = args.adapter_revision
    if not adapter_path.exists() and not adapter_revision:
        adapter_revision = HfApi().model_info(adapter_source).sha

    print(
        json.dumps(
            {
                "base_model": BASE_MODEL,
                "adapter": adapter_source,
                "adapter_revision": adapter_revision,
                "output": str(output_dir),
                "dtype": "bfloat16",
            },
            indent=2,
        ),
        flush=True,
    )

    model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
    )
    adapter_kwargs = {"is_trainable": False}
    if adapter_revision:
        adapter_kwargs["revision"] = adapter_revision
    model = PeftModel.from_pretrained(model, adapter_source, **adapter_kwargs)
    model = model.merge_and_unload(safe_merge=True)
    model.config.use_cache = True
    model.save_pretrained(output_dir, safe_serialization=True, max_shard_size="4GB")
    AutoProcessor.from_pretrained(BASE_MODEL).save_pretrained(output_dir)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": BASE_MODEL,
        "source_adapter": adapter_source,
        "source_adapter_revision": adapter_revision,
        "merge_dtype": "bfloat16",
        "safe_merge": True,
        "includes_complete_multimodal_lora": True,
        "purpose": "FinChart SFT-only vLLM serving and evaluation",
    }
    (output_dir / "merge_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Merged SFT model saved to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
