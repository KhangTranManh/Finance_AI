"""Run the FinChart-R2 Phase 2C provisional multimodal DPO pilot.

Designed for an isolated remote environment with one 24 GB NVIDIA GPU.
The script never reads ChartQA validation data.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib
import json
import os
import random
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from transformers.trainer_utils import get_last_checkpoint
import trl
from trl import DPOConfig
import trl.trainer.dpo_trainer as dpo_module


BASE_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
SFT_ADAPTER_ID = "Kxck/Finance_500_v1"
CHARTQA_DATASET = "HuggingFaceM4/ChartQA"
DEFAULT_HUB_MODEL_ID = "Kxck/Finance_500_v1_DPO_386_provisional"
EXPECTED_PAIRS = 386
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hub-model-id", default=DEFAULT_HUB_MODEL_ID)
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument("--max-visual-tokens", type=int, default=1024)
    return parser.parse_args()


def patch_trl_029_for_qwen3vl():
    """Repair the two TRL 0.29 model-forward allowlists for Qwen3-VL."""
    global dpo_module
    source_path = Path(dpo_module.__file__).resolve()
    source = source_path.read_text(encoding="utf-8")
    old = (
        'for key in ("pixel_values", "pixel_attention_mask", "image_grid_thw", '
        '"image_sizes", "token_type_ids"):'
    )
    new = (
        'for key in ("pixel_values", "pixel_attention_mask", "image_grid_thw", '
        '"image_sizes", "token_type_ids", "mm_token_type_ids"):'
    )
    if new not in source:
        count = source.count(old)
        if count != 2:
            raise RuntimeError(
                f"Unexpected TRL source at {source_path}: found {count} forward allowlists"
            )
        source_path.write_text(source.replace(old, new), encoding="utf-8")
        importlib.invalidate_caches()
        dpo_module = importlib.reload(dpo_module)
    elif old in source:
        raise RuntimeError("TRL is only partially patched; recreate the isolated environment")
    patched = source_path.read_text(encoding="utf-8")
    if patched.count(new) != 2:
        raise RuntimeError("Both TRL DPO forward paths were not patched")
    return dpo_module.DPOTrainer, dpo_module.DataCollatorForVisionPreference


def read_and_validate_pairs(path: Path) -> tuple[list[dict], str, Counter]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    required = {
        "dataset_index",
        "image_split",
        "image_index",
        "prompt",
        "chosen",
        "rejected",
        "ground_truth",
        "manual_audit_status",
    }
    if len(rows) != EXPECTED_PAIRS:
        raise ValueError(f"Expected {EXPECTED_PAIRS} pairs, found {len(rows)}")
    seen = set()
    for line_number, row in enumerate(rows, 1):
        missing = required - set(row)
        if missing:
            raise ValueError(f"Line {line_number} is missing {sorted(missing)}")
        if row["image_split"] != "train":
            raise RuntimeError(f"Validation leakage at line {line_number}")
        reference = (row["image_split"], int(row["image_index"]))
        if reference in seen:
            raise ValueError(f"Duplicate image reference at line {line_number}: {reference}")
        seen.add(reference)
        if str(row["chosen"]).strip() == str(row["rejected"]).strip():
            raise ValueError(f"Identical chosen/rejected at line {line_number}")
    statuses = Counter(str(row["manual_audit_status"]) for row in rows)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return rows, digest, statuses


class Qwen3VLVisionPreferenceCollator:
    """Align Qwen3-VL multimodal token types after chosen/rejected concatenation."""

    def __init__(self, processor, base_collator_cls):
        self.base_collator = base_collator_cls(processor=processor)

    def __call__(self, examples):
        if len(examples) != 1:
            raise RuntimeError("The guarded Qwen3-VL collator requires device batch size 1")
        batch = self.base_collator(examples)
        if "mm_token_type_ids" not in batch:
            raise RuntimeError("The Qwen3-VL processor did not return mm_token_type_ids")
        mm_ids = batch["mm_token_type_ids"]
        input_ids = batch["input_ids"]
        missing_tokens = input_ids.shape[1] - mm_ids.shape[1]
        if mm_ids.shape[0] != input_ids.shape[0] or missing_tokens < 0:
            raise RuntimeError("Invalid mm_token_type_ids dimensions")
        if missing_tokens:
            completion_types = torch.zeros(
                (mm_ids.shape[0], missing_tokens), dtype=mm_ids.dtype, device=mm_ids.device
            )
            batch["mm_token_type_ids"] = torch.cat((mm_ids, completion_types), dim=1)
        if batch["mm_token_type_ids"].shape != input_ids.shape:
            raise RuntimeError("Failed to align complete Qwen3-VL token types")
        return batch


def main() -> None:
    args = parse_args()
    args.pairs = args.pairs.resolve()
    args.output_dir = args.output_dir.resolve()
    checkpoints = args.output_dir / "checkpoints"
    adapter_dir = args.output_dir / "adapter_sft_dpo_386"
    cache_dir = args.output_dir / "cache"
    for directory in (args.output_dir, checkpoints, adapter_dir, cache_dir):
        directory.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible CUDA GPU is required")
    gpu = torch.cuda.get_device_properties(0)
    if gpu.total_memory < 20 * 2**30:
        raise RuntimeError(f"At least 20 GiB VRAM is required; found {gpu.total_memory / 2**30:.1f}")
    if trl.__version__ != "0.29.0":
        raise RuntimeError(f"Expected TRL 0.29.0, found {trl.__version__}")

    DPOTrainer, BaseVisionCollator = patch_trl_029_for_qwen3vl()
    records, pair_sha256, audit_statuses = read_and_validate_pairs(args.pairs)
    warnings.warn(
        f"PROVISIONAL PILOT: audit statuses are {dict(audit_statuses)}; teacher labels are not ground truth."
    )

    random.seed(SEED)
    torch.manual_seed(SEED)
    chartqa_train = load_dataset(
        CHARTQA_DATASET, split="train", cache_dir=str(cache_dir / "datasets")
    )
    examples = []
    for row in records:
        image_index = int(row["image_index"])
        source = chartqa_train[image_index]
        question = str(source.get("query", source.get("question", ""))).strip()
        if question != str(row["prompt"]).strip():
            raise ValueError(f"Question mismatch at image_index={image_index}")
        examples.append(
            {
                "image": source["image"].convert("RGB"),
                "prompt": [{"role": "user", "content": str(row["prompt"])}],
                "chosen": [{"role": "assistant", "content": str(row["chosen"])}],
                "rejected": [{"role": "assistant", "content": str(row["rejected"])}],
            }
        )
    preference_dataset = Dataset.from_list(examples)
    splits = preference_dataset.train_test_split(test_size=0.10, seed=SEED, shuffle=True)

    bf16 = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if bf16 else torch.float16
    max_pixels = args.max_visual_tokens * 28 * 28
    processor = AutoProcessor.from_pretrained(
        BASE_MODEL,
        min_pixels=256 * 28 * 28,
        max_pixels=max_pixels,
        cache_dir=str(cache_dir / "models"),
    )
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    base_model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL,
        device_map={"": 0},
        dtype=compute_dtype,
        attn_implementation="sdpa",
        quantization_config=quantization,
        cache_dir=str(cache_dir / "models"),
    )
    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)
    model = PeftModel.from_pretrained(
        base_model,
        SFT_ADAPTER_ID,
        is_trainable=True,
        cache_dir=str(cache_dir / "models"),
    )
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    collator = Qwen3VLVisionPreferenceCollator(processor, BaseVisionCollator)
    training_args = DPOConfig(
        output_dir=str(checkpoints),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        learning_rate=2e-6,
        warmup_steps=3,
        beta=0.1,
        loss_type=["sigmoid"],
        max_length=None,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        max_grad_norm=1.0,
        bf16=bf16,
        fp16=not bf16,
        tf32=True,
        logging_steps=1,
        logging_first_step=True,
        eval_strategy="epoch",
        save_strategy="steps",
        save_steps=10,
        save_total_limit=3,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        report_to="none",
        seed=SEED,
        precompute_ref_log_probs=True,
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=splits["train"],
        eval_dataset=splits["test"],
        processing_class=processor,
        data_collator=collator,
    )

    batch = next(iter(trainer.get_train_dataloader()))
    required = {"input_ids", "attention_mask", "completion_mask", "pixel_values", "mm_token_type_ids"}
    missing = required - set(batch)
    if missing:
        raise RuntimeError(f"Multimodal preflight is missing {sorted(missing)}")
    if batch["mm_token_type_ids"].shape != batch["input_ids"].shape:
        raise RuntimeError("Multimodal token types are not aligned")
    prepared_batch = trainer._prepare_inputs(batch)
    metrics_snapshot = copy.deepcopy(trainer._metrics)
    model.eval()
    with torch.no_grad():
        smoke_loss = trainer.compute_loss(model, prepared_batch)
    if not torch.isfinite(smoke_loss).item():
        raise RuntimeError(f"Non-finite DPO preflight loss: {smoke_loss.item()}")
    trainer._metrics = metrics_snapshot
    model.train()
    print("DPO preflight passed", float(smoke_loss))
    del batch, prepared_batch, smoke_loss, metrics_snapshot
    gc.collect()
    torch.cuda.empty_cache()

    last_checkpoint = get_last_checkpoint(str(checkpoints))
    print("Resume checkpoint:", last_checkpoint)
    result = trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))
    trainer.save_state()

    manifest = {
        "experiment": "FinChart Phase 2C multimodal DPO-386 provisional remote pilot",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": BASE_MODEL,
        "source_adapter": SFT_ADAPTER_ID,
        "hub_model_id": args.hub_model_id,
        "pairs": len(records),
        "pair_sha256": pair_sha256,
        "audit_statuses": dict(audit_statuses),
        "dataset_split": "ChartQA train only",
        "train_pairs": len(splits["train"]),
        "eval_pairs": len(splits["test"]),
        "max_visual_tokens": args.max_visual_tokens,
        "gpu": gpu.name,
        "gpu_memory_gib": round(gpu.total_memory / 2**30, 2),
        "precompute_ref_log_probs": True,
        "training_metrics": result.metrics,
        "teacher_is_ground_truth": False,
        "manual_audit_complete": False,
        "required_next_step": "Frozen ChartQA val[0:500] evaluation against SFT-408.",
    }
    (args.output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    if not args.no_push:
        api = HfApi()
        identity = api.whoami().get("name")
        target_owner = args.hub_model_id.split("/", 1)[0]
        if identity != target_owner:
            raise RuntimeError(f"Authenticated as {identity}, cannot push to {target_owner}")
        api.create_repo(args.hub_model_id, repo_type="model", exist_ok=True)
        api.upload_folder(
            repo_id=args.hub_model_id,
            repo_type="model",
            folder_path=str(adapter_dir),
            commit_message="FinChart Phase 2C DPO-386 provisional remote pilot",
        )
        api.upload_file(
            repo_id=args.hub_model_id,
            repo_type="model",
            path_or_fileobj=str(args.output_dir / "training_manifest.json"),
            path_in_repo="training_manifest.json",
            commit_message="Add provisional DPO training manifest",
        )
        print("Uploaded model:", args.hub_model_id)
    print("Saved adapter:", adapter_dir)


if __name__ == "__main__":
    main()
