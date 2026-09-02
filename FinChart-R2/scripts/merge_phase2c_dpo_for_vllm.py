#!/usr/bin/env python3
"""Merge the Phase 2C multimodal DPO LoRA into Qwen3-VL for vLLM.

The DPO adapter contains LoRA tensors for both the language backbone and the
vision tower.  Merging before vLLM inference guarantees that both groups are
applied; it avoids relying on experimental multimodal-tower LoRA support.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor


BASE_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
DEFAULT_ADAPTER = "Kxck/Finance_500_v1_DPO_386_provisional"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter",
        default=DEFAULT_ADAPTER,
        help="Local adapter directory or Hugging Face repository ID.",
    )
    parser.add_argument(
        "--adapter-revision",
        default=None,
        help="Optional immutable Hugging Face revision for a remote adapter.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter_path = Path(args.adapter).expanduser()
    if adapter_path.exists():
        adapter_source = str(adapter_path.resolve())
        if not (adapter_path / "adapter_config.json").is_file():
            raise FileNotFoundError(f"Missing adapter_config.json in {adapter_path}")
    else:
        adapter_source = str(args.adapter)
    output_dir = args.output.resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "base_model": BASE_MODEL,
                "adapter": adapter_source,
                "adapter_revision": args.adapter_revision,
                "output": str(output_dir),
                "merge_dtype": "bfloat16",
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
    if args.adapter_revision:
        adapter_kwargs["revision"] = args.adapter_revision
    model = PeftModel.from_pretrained(model, adapter_source, **adapter_kwargs)
    model = model.merge_and_unload(safe_merge=True)
    model.config.use_cache = True
    model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size="4GB",
    )

    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    processor.save_pretrained(output_dir)

    manifest = {
        "base_model": BASE_MODEL,
        "source_adapter": adapter_source,
        "source_adapter_revision": args.adapter_revision,
        "merge_dtype": "bfloat16",
        "safe_merge": True,
        "purpose": "vLLM frozen ChartQA val[0:500] evaluation",
    }
    (output_dir / "merge_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Merged model saved to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
