"""Build the Colab-only Phase 2C multimodal DPO diagnostic notebook."""

from __future__ import annotations

import json
from pathlib import Path


OUTPUT = Path("FinChart-R2/notebooks/05_FinChart_R2_Phase2C_DPO_Colab_Leakage_Gated.ipynb")


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


cells = [
    markdown(
        """# FinChart-R2 - Phase 2C: Multimodal DPO on SFT-408

**Colab-only diagnostic notebook.** It continues the public SFT adapter Kxck/Finance_500_v1 with 51 structured visual preference pairs using TRL's VLM-supported DPO path.

Research boundary: all 51 pairs were derived from frozen ChartQA val[0:500]. This run may validate the DPO path and inspect training metrics, but it must not be evaluated on or compared with the frozen 69.0% SFT-408 benchmark. A reportable experiment requires equivalent pairs derived solely from ChartQA train.
"""
    ),
    markdown(
        """## 1. Colab runtime

In Colab select **Runtime -> Change runtime type -> T4 GPU** (or stronger). This notebook does not use the local machine GPU.
"""
    ),
    code("!pip install -q -U unsloth unsloth_zoo trl transformers accelerate bitsandbytes peft datasets pillow\n"),
    markdown(
        """## 2. Mount Google Drive

Copy phase2c_orpo_51_pairs.jsonl into MyDrive/FinChart-R2/data/. Despite the historical filename, its schema is generic preference data (prompt, chosen, rejected) and is used directly by DPO.
"""
    ),
    code(
        """from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path

ROOT = Path('/content/drive/MyDrive')
PROJECT_DIR = ROOT / 'FinChart-R2'
DATA_DIR = PROJECT_DIR / 'data'
RUN_DIR = PROJECT_DIR / 'phase2c' / 'dpo_51_leaky_diagnostic'
ADAPTER_DIR = RUN_DIR / 'adapter_sft_dpo_51'
CHECKPOINT_DIR = RUN_DIR / 'checkpoints'
for path in [RUN_DIR, ADAPTER_DIR, CHECKPOINT_DIR]:
    path.mkdir(parents=True, exist_ok=True)

PAIR_SOURCE = DATA_DIR / 'phase2c_orpo_51_pairs.jsonl'
SFT_ADAPTER_ID = 'Kxck/Finance_500_v1'
BASE_MODEL = 'unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit'
print('Pair source:', PAIR_SOURCE)
print('SFT adapter:', SFT_ADAPTER_ID)
"""
    ),
    markdown("## 3. Load pairs and enforce the research boundary\n"),
    code(
        """import json

def read_jsonl(path):
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]

if not PAIR_SOURCE.exists():
    raise FileNotFoundError(f'Upload the pair artifact first: {PAIR_SOURCE}')

records = read_jsonl(PAIR_SOURCE)
assert len(records) == 51, f'Expected 51 pairs, got {len(records)}'
required = {'prompt', 'chosen', 'rejected', 'image_split', 'image_index', 'dataset_index'}
for row in records:
    assert required <= row.keys(), f'Missing required fields: {required - row.keys()}'
    assert row['chosen'].strip() != row['rejected'].strip(), 'Degenerate preference pair.'

split_counts = {}
for row in records:
    split_counts[row['image_split']] = split_counts.get(row['image_split'], 0) + 1
uses_frozen_validation = any(
    row['image_split'] == 'val' and int(row['image_index']) < 500 for row in records
)
assert uses_frozen_validation, 'This notebook is intentionally scoped to the current validation-derived diagnostic.'
print('Pair count:', len(records))
print('Pair splits:', split_counts)
print('LEAKY DIAGNOSTIC: frozen evaluation is disabled for this adapter.')
"""
    ),
    markdown(
        """## 4. Build a VLM preference dataset

TRL DPO receives the chart as a top-level image column plus conversational prompt, chosen, and rejected columns. Do not use UnslothVisionDataCollator here: DPO selects its own vision-preference collator.
"""
    ),
    code(
        """from datasets import Dataset, load_dataset

DATASET_NAME = 'HuggingFaceM4/ChartQA'
datasets_by_split = {
    split: load_dataset(DATASET_NAME, split=split)
    for split in sorted({row['image_split'] for row in records})
}

examples = []
for row in records:
    image = datasets_by_split[row['image_split']][int(row['image_index'])]['image']
    examples.append({
        'prompt': [{
            'role': 'user',
            'content': [
                {'type': 'image', 'image': image},
                {'type': 'text', 'text': row['prompt']},
            ],
        }],
        'chosen': [{'role': 'assistant', 'content': row['chosen']}],
        'rejected': [{'role': 'assistant', 'content': row['rejected']}],
        'image': image,
        'dataset_index': int(row['dataset_index']),
    })

dpo_dataset = Dataset.from_list(examples)
assert len(dpo_dataset) == 51
assert 'image' in dpo_dataset.column_names
print(dpo_dataset)
print('Chosen example:\\n', records[0]['chosen'])
print('Rejected example:\\n', records[0]['rejected'])
"""
    ),
    markdown("## 5. Load the SFT-408 policy adapter\n"),
    code(
        """import torch
from peft import PeftModel
from unsloth import FastVisionModel, is_bfloat16_supported

if not torch.cuda.is_available():
    raise RuntimeError('Start a Colab GPU runtime before continuing.')

model, processor = FastVisionModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=2048,
    load_in_4bit=True,
    use_gradient_checkpointing='unsloth',
)
model = PeftModel.from_pretrained(model, SFT_ADAPTER_ID, is_trainable=True)
processor.tokenizer.padding_side = 'left'
FastVisionModel.for_training(model)
print('Loaded trainable SFT-408 policy adapter.')
"""
    ),
    markdown(
        """## 6. DPO VLM preflight

TRL DPO supports a top-level image column for Vision-Language Models. max_length=None is required here so image tokens are not truncated. The batch must contain pixel_values before training.
"""
    ),
    code(
        """from trl import DPOConfig, DPOTrainer

DPO_BETA = 0.05
dpo_args = DPOConfig(
    output_dir=str(CHECKPOINT_DIR),
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=5e-6,
    warmup_steps=5,
    beta=DPO_BETA,
    max_length=None,
    remove_unused_columns=False,
    logging_steps=1,
    save_strategy='epoch',
    optim='adamw_8bit',
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    report_to='none',
)

# ref_model=None makes DPO retain the initial policy state as its reference.
trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=dpo_args,
    train_dataset=dpo_dataset,
    processing_class=processor,
)

batch = next(iter(trainer.get_train_dataloader()))
assert 'pixel_values' in batch, 'DPO VLM preflight failed: image tensors are absent.'
print({key: tuple(value.shape) for key, value in batch.items() if hasattr(value, 'shape')})
print('DPO VLM preflight passed.')
"""
    ),
    markdown(
        """## 7. One-epoch DPO diagnostic

This trains and saves an SFT + DPO diagnostic adapter. Track rewards/accuracies and rewards/margins; positive and improving values show the preference objective is separating chosen from rejected completions. They do not replace task accuracy.
"""
    ),
    code(
        """trainer_stats = trainer.train()
model.save_pretrained(str(ADAPTER_DIR))
processor.save_pretrained(str(ADAPTER_DIR))

metadata = {
    'source_adapter': SFT_ADAPTER_ID,
    'pair_source': str(PAIR_SOURCE),
    'pair_count': len(records),
    'method': 'DPO',
    'beta': DPO_BETA,
    'epochs': 1,
    'training_data_boundary': 'VALIDATION_DERIVED_LEAKY_DIAGNOSTIC_ONLY',
    'frozen_evaluation_permitted': False,
}
(RUN_DIR / 'dpo_run_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
print('Saved SFT + DPO adapter:', ADAPTER_DIR)
"""
    ),
    markdown(
        """## 8. Frozen evaluation is intentionally blocked

Do not use ChartQA val[0:500] after this run. Rebuild preference pairs from ChartQA train, rerun the same notebook with uses_frozen_validation == False, then reuse the unchanged Phase 1 evaluator from notebook 03.
"""
    ),
    code(
        """raise RuntimeError(
    'Frozen evaluation blocked: this DPO adapter was trained on preference pairs derived from val[0:500]. '
    'Its score cannot be compared to the reported 69.0% SFT-408 frozen benchmark.'
)
"""
    ),
]

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "colab": {"name": OUTPUT.name, "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "cells": cells,
}

OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"Created {OUTPUT}")
