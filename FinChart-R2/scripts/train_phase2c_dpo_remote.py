#!/usr/bin/env python3
"""Run the FinChart-R2 Phase 2C multimodal DPO pilot on an isolated GPU host."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib
import json
import os
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi, get_token
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from transformers.trainer_utils import get_last_checkpoint


BASE_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
SFT_ADAPTER_ID = "Kxck/Finance_500_v1"
DATASET_NAME = "HuggingFaceM4/ChartQA"
HUB_MODEL_ID = "Kxck/Finance_500_v1_DPO_386_provisional"
EXPECTED_PAIRS = 386
EVAL_FRACTION = 0.10
SEED = 42
MAX_IMAGE_PIXELS = 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--push-to-hub", action="store_true")
    return parser.parse_args()


def patch_trl_qwen3vl_forwarding():
    """Patch the two TRL 0.29 model-input allowlists omitted for Qwen3-VL."""
    import trl
    import trl.trainer.dpo_trainer as dpo_module

    if trl.__version__ != "0.29.0":
        raise RuntimeError(f"Expected TRL 0.29.0, found {trl.__version__}")
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
                f"Refusing unexpected TRL source patch: found {count}/2 allowlists in {source_path}"
            )
        source_path.write_text(source.replace(old, new), encoding="utf-8")
        importlib.invalidate_caches()
        dpo_module = importlib.reload(dpo_module)
    elif old in source:
        raise RuntimeError("TRL source is partially patched")
    if source_path.read_text(encoding="utf-8").count(new) != 2:
        raise RuntimeError("TRL policy/reference forwarding patch failed")
    return trl, dpo_module


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    pairs_path = args.pairs.resolve()
    checkpoint_dir = run_root / "checkpoints"
    adapter_dir = run_root / "outputs" / "adapter_sft_dpo_386"
    for directory in (checkpoint_dir, adapter_dir):
        directory.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected exactly one CUDA GPU, found {torch.cuda.device_count()}")
    gpu = torch.cuda.get_device_properties(0)
    bf16 = torch.cuda.is_bf16_supported()
    print(
        json.dumps(
            {
                "gpu": gpu.name,
                "gpu_memory_gib": round(gpu.total_memory / 2**30, 2),
                "bf16": bf16,
                "run_root": str(run_root),
            },
            indent=2,
        ),
        flush=True,
    )

    token = get_token()
    if not token:
        raise RuntimeError("No cached Hugging Face token is available")
    identity = HfApi(token=token).whoami()
    hf_user = identity.get("name") or identity.get("fullname")
    if args.push_to_hub and hf_user != HUB_MODEL_ID.split("/", 1)[0]:
        raise RuntimeError(f"Authenticated as {hf_user}; refusing to push to {HUB_MODEL_ID}")
    print(f"Authenticated Hugging Face account: {hf_user}", flush=True)

    records = [
        json.loads(line)
        for line in pairs_path.read_text(encoding="utf-8").splitlines()
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
    if len(records) != EXPECTED_PAIRS:
        raise ValueError(f"Expected {EXPECTED_PAIRS} pairs, found {len(records)}")
    for line_number, row in enumerate(records, 1):
        missing = required - set(row)
        if missing:
            raise ValueError(f"Line {line_number} missing fields: {sorted(missing)}")
        if row["image_split"] != "train":
            raise RuntimeError(f"Validation leakage at line {line_number}")
        if str(row["chosen"]).strip() == str(row["rejected"]).strip():
            raise ValueError(f"Identical chosen/rejected at line {line_number}")
    pair_hash = hashlib.sha256(pairs_path.read_bytes()).hexdigest()
    print(
        {
            "pairs": len(records),
            "audit_status": dict(Counter(str(row["manual_audit_status"]) for row in records)),
            "sha256": pair_hash,
        },
        flush=True,
    )

    chartqa_train = load_dataset(DATASET_NAME, split="train", token=token)
    examples = []
    for row in records:
        image_index = int(row["image_index"])
        source = chartqa_train[image_index]
        source_question = str(source.get("query", source.get("question", ""))).strip()
        if source_question != str(row["prompt"]).strip():
            raise ValueError(f"Question mismatch at dataset_index={row['dataset_index']}")
        examples.append(
            {
                "image": source["image"].convert("RGB"),
                "prompt": [{"role": "user", "content": str(row["prompt"])}],
                "chosen": [{"role": "assistant", "content": str(row["chosen"])}],
                "rejected": [{"role": "assistant", "content": str(row["rejected"])}],
            }
        )
    preference_dataset = Dataset.from_list(examples)
    split = preference_dataset.train_test_split(test_size=EVAL_FRACTION, seed=SEED, shuffle=True)
    train_dataset, eval_dataset = split["train"], split["test"]
    print({"train_pairs": len(train_dataset), "eval_pairs": len(eval_dataset)}, flush=True)

    compute_dtype = torch.bfloat16 if bf16 else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    processor = AutoProcessor.from_pretrained(BASE_MODEL, token=token)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    image_size = getattr(processor.image_processor, "size", None)
    try:
        original_max_pixels = int(image_size["longest_edge"])
    except (KeyError, TypeError, AttributeError):
        try:
            original_max_pixels = int(image_size.longest_edge)
        except (TypeError, AttributeError) as error:
            raise RuntimeError(
                f"Cannot safely cap Qwen3-VL image pixels; size={image_size!r}"
            ) from error
    capped_max_pixels = min(original_max_pixels, MAX_IMAGE_PIXELS)
    try:
        image_size["longest_edge"] = capped_max_pixels
    except (KeyError, TypeError, AttributeError):
        image_size.longest_edge = capped_max_pixels
    print(
        {
            "original_max_image_pixels": original_max_pixels,
            "training_max_image_pixels": capped_max_pixels,
        },
        flush=True,
    )
    base_model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL,
        device_map={"": 0},
        dtype=compute_dtype,
        attn_implementation="sdpa",
        quantization_config=quantization,
        token=token,
    )
    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)
    model = PeftModel.from_pretrained(
        base_model, SFT_ADAPTER_ID, is_trainable=True, token=token
    )
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    trl, dpo_module = patch_trl_qwen3vl_forwarding()
    DataCollatorForVisionPreference = dpo_module.DataCollatorForVisionPreference
    DPOTrainer = dpo_module.DPOTrainer
    DPOConfig = trl.DPOConfig

    class Qwen3VLVisionPreferenceCollator:
        def __init__(self, vision_processor):
            self.base_collator = DataCollatorForVisionPreference(processor=vision_processor)

        def __call__(self, batch_examples):
            if len(batch_examples) != 1:
                raise RuntimeError("Qwen3-VL repair requires per-device batch size 1")
            batch = self.base_collator(batch_examples)
            # TRL 0.29's vision collator drops cached reference scores. Preserve
            # them so precompute_ref_log_probs=True works for multimodal DPO.
            if "ref_chosen_logps" in batch_examples[0]:
                batch["ref_chosen_logps"] = torch.tensor(
                    [float(example["ref_chosen_logps"]) for example in batch_examples],
                    dtype=torch.float32,
                )
            if "ref_rejected_logps" in batch_examples[0]:
                batch["ref_rejected_logps"] = torch.tensor(
                    [float(example["ref_rejected_logps"]) for example in batch_examples],
                    dtype=torch.float32,
                )
            if "mm_token_type_ids" not in batch:
                raise RuntimeError("Processor omitted mm_token_type_ids")
            mm_ids, input_ids = batch["mm_token_type_ids"], batch["input_ids"]
            missing_tokens = input_ids.shape[1] - mm_ids.shape[1]
            if missing_tokens < 0:
                raise RuntimeError("mm_token_type_ids is longer than input_ids")
            if missing_tokens:
                completion_types = torch.zeros(
                    (mm_ids.shape[0], missing_tokens), dtype=mm_ids.dtype, device=mm_ids.device
                )
                batch["mm_token_type_ids"] = torch.cat((mm_ids, completion_types), dim=1)
            if batch["mm_token_type_ids"].shape != input_ids.shape:
                raise RuntimeError("Failed to align mm_token_type_ids")
            return batch

    dpo_args = DPOConfig(
        output_dir=str(checkpoint_dir),
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
        save_steps=5,
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
        args=dpo_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        data_collator=Qwen3VLVisionPreferenceCollator(processor),
    )

    batch = next(iter(trainer.get_train_dataloader()))
    required_batch = {
        "input_ids",
        "attention_mask",
        "completion_mask",
        "pixel_values",
        "mm_token_type_ids",
    }
    missing_batch = required_batch - set(batch)
    if missing_batch:
        raise RuntimeError(f"Multimodal preflight missing: {sorted(missing_batch)}")
    missing_reference = {"ref_chosen_logps", "ref_rejected_logps"} - set(batch)
    if missing_reference:
        raise RuntimeError(
            f"Precomputed reference scores were dropped: {sorted(missing_reference)}"
        )
    if batch["mm_token_type_ids"].shape != batch["input_ids"].shape:
        raise RuntimeError("Preflight mm_token_type_ids shape mismatch")
    prepared_batch = trainer._prepare_inputs(batch)
    metrics_snapshot = copy.deepcopy(trainer._metrics)
    model.eval()
    with torch.no_grad():
        smoke_loss = trainer.compute_loss(model, prepared_batch)
    if not torch.isfinite(smoke_loss).item():
        raise RuntimeError(f"Non-finite DPO smoke loss: {smoke_loss.item()}")
    trainer._metrics = metrics_snapshot
    model.train()
    print(f"Multimodal DPO preflight passed: loss={float(smoke_loss):.6f}", flush=True)
    del batch, prepared_batch, smoke_loss, metrics_snapshot
    gc.collect()
    torch.cuda.empty_cache()

    random.seed(SEED)
    torch.manual_seed(SEED)
    last_checkpoint = get_last_checkpoint(str(checkpoint_dir))
    print(f"Resume checkpoint: {last_checkpoint}", flush=True)
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))
    trainer.save_state()

    manifest = {
        "experiment": "FinChart Phase 2C multimodal DPO-386 remote pilot",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": BASE_MODEL,
        "source_sft_adapter": SFT_ADAPTER_ID,
        "output_adapter": HUB_MODEL_ID,
        "pairs": len(records),
        "train_pairs": len(train_dataset),
        "eval_pairs": len(eval_dataset),
        "pair_sha256": pair_hash,
        "teacher_is_ground_truth": False,
        "manual_audit_status": dict(
            Counter(str(row["manual_audit_status"]) for row in records)
        ),
        "training_metrics": train_result.metrics,
        "gpu": gpu.name,
        "trl_version": trl.__version__,
        "precomputed_reference_logps": True,
        "max_image_pixels": capped_max_pixels,
        "frozen_chartqa_val_evaluation_complete": False,
    }
    manifest_path = adapter_dir / "training_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)

    if args.push_to_hub:
        api = HfApi(token=token)
        api.create_repo(HUB_MODEL_ID, repo_type="model", exist_ok=True, private=False)
        api.upload_folder(
            repo_id=HUB_MODEL_ID,
            repo_type="model",
            folder_path=str(adapter_dir),
            commit_message="Add FinChart Phase 2C DPO-386 provisional adapter",
        )
        print(f"Uploaded adapter to https://huggingface.co/{HUB_MODEL_ID}", flush=True)


if __name__ == "__main__":
    main()
