"""Build the Colab notebook for Phase 2C train-only preference mining."""

from __future__ import annotations

import json
from pathlib import Path


OUTPUT = Path(
    "FinChart-R2/notebooks/04_FinChart_R2_Phase2C_DPO_Train_Preference_Mining_Colab.ipynb"
)


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
        """# FinChart-R2 - Phase 2C: Train Preference Mining for DPO

This Colab-only notebook runs the existing SFT-408 adapter on an unseen ChartQA train slice, saves every prediction in JSONL, and exports only incorrect predictions as candidates for teacher annotation and DPO-pair construction.

Research boundary: ChartQA train only. The frozen ChartQA val[0:500] set is never read by this notebook. The default source slice begins at index 500 to avoid the train[0:500] pool used by the original Phase 2A pilot.
"""
    ),
    markdown(
        """## 1. Runtime

Select Runtime -> Change runtime type -> T4 GPU (or stronger), Runtime Version 2026.07. This notebook uses the Colab GPU, not the local machine.
"""
    ),
    code(
        """import sys

assert sys.version_info[:2] == (3, 12), (
    f'Expected Colab Runtime Version 2026.07 (Python 3.12), found Python {sys.version.split()[0]}. '
    'Change the runtime version, then reconnect.'
)
print('Python runtime:', sys.version)
"""
    ),
    code(
        """!pip install -q -U --no-cache-dir unsloth unsloth_zoo transformers accelerate bitsandbytes peft datasets tqdm
!pip install -q --no-cache-dir --upgrade "torchvision>=0.28.0" "pillow==11.3.0"
"""
    ),
    markdown(
        """### Restart required

After installation finishes, select Runtime -> Restart session. Then continue from Mount Google Drive. This avoids mixed compiled Python packages.
"""
    ),
    markdown("## 2. Mount Drive and configure a resumable run\n"),
    code(
        """from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path

ROOT = Path('/content/drive/MyDrive')
PROJECT_DIR = ROOT / 'FinChart-R2'
RUN_DIR = PROJECT_DIR / 'phase2c' / 'train_preference_mining'
RUN_DIR.mkdir(parents=True, exist_ok=True)

SFT_ADAPTER_ID = 'Kxck/Finance_500_v1'
BASE_MODEL = 'unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit'
DATASET_NAME = 'HuggingFaceM4/ChartQA'

# The initial SFT pilot used the first train pool. Mine an unseen train-only slice.
TRAIN_START = 500
MINE_N = 2000
MAX_NEW_TOKENS = 64
SAVE_EVERY = 25

RUN_TAG = f'train_{TRAIN_START}_{TRAIN_START + MINE_N}'
ALL_PATH = RUN_DIR / f'phase2c_{RUN_TAG}_sft_predictions.jsonl'
ERROR_PATH = RUN_DIR / f'phase2c_{RUN_TAG}_sft_errors.jsonl'
CORRECT_PATH = RUN_DIR / f'phase2c_{RUN_TAG}_sft_correct.jsonl'
MANIFEST_PATH = RUN_DIR / f'phase2c_{RUN_TAG}_manifest.json'

print('All predictions:', ALL_PATH)
print('DPO/teacher candidates:', ERROR_PATH)
"""
    ),
    markdown("## 3. Load ChartQA train only and resume safely\n"),
    code(
        """import json
from datasets import load_dataset

chartqa_train = load_dataset(DATASET_NAME, split='train')
assert TRAIN_START >= 500, 'Keep the mining slice outside the original Phase 2A pilot pool.'
assert TRAIN_START + MINE_N <= len(chartqa_train), 'Requested train range is out of bounds.'
mine_ds = chartqa_train.select(range(TRAIN_START, TRAIN_START + MINE_N))

def read_jsonl(path):
    if not path.exists():
        return []
    with path.open(encoding='utf-8') as handle:
        rows = []
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
        return rows

existing_rows = read_jsonl(ALL_PATH)
completed_indices = {int(row['dataset_index']) for row in existing_rows}
expected_indices = set(range(TRAIN_START, TRAIN_START + MINE_N))
assert completed_indices <= expected_indices, 'Existing output belongs to a different run range.'

print('ChartQA train size:', len(chartqa_train))
print('Mining range:', TRAIN_START, 'to', TRAIN_START + MINE_N - 1)
print('Already complete:', len(completed_indices))
print('Remaining:', MINE_N - len(completed_indices))
"""
    ),
    markdown("## 4. Load the SFT-408 adapter for deterministic inference\n"),
    code(
        """import unsloth
import torch
from peft import PeftModel
from unsloth import FastVisionModel

if not torch.cuda.is_available():
    raise RuntimeError('Start a Colab GPU runtime before continuing.')

model, processor = FastVisionModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=2048,
    load_in_4bit=True,
)
model = PeftModel.from_pretrained(model, SFT_ADAPTER_ID)
FastVisionModel.for_inference(model)
print('Loaded SFT-408 adapter for inference.')
"""
    ),
    markdown("## 5. Phase 1-compatible answer matcher and inference prompt\n"),
    code(
        """import re
import string

def unwrap_answer(value):
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value

def normalize_answer(text):
    text = '' if text is None else str(text)
    text = text.lower().strip()
    text = re.sub(r'^\\s*final\\s+answer\\s*:\\s*', '', text)
    text = re.sub(r'^\\s*answer\\s*:\\s*', '', text)
    text = re.sub(r'\\s+', ' ', text)
    text = re.sub(r'\\s*/\\s*', '/', text)
    return text.strip(string.whitespace + '.,;:!?')

def try_parse_number(text):
    try:
        return float(normalize_answer(text).replace(',', '').replace('%', '').strip())
    except (ValueError, TypeError):
        return None

def deterministic_match(prediction, ground_truth, tolerance=1e-6):
    pred_norm = normalize_answer(prediction)
    gt_norm = normalize_answer(ground_truth)
    if pred_norm == gt_norm:
        return True
    pred_num = try_parse_number(pred_norm)
    gt_num = try_parse_number(gt_norm)
    return (
        pred_num is not None
        and gt_num is not None
        and abs(pred_num - gt_num) <= tolerance
    )

def build_messages(image, question):
    return [{
        'role': 'user',
        'content': [
            {'type': 'image', 'image': image},
            {
                'type': 'text',
                'text': (
                    'Look carefully at the chart and answer the question.\\n\\n'
                    f'Question: {question}\\n\\n'
                    'Return only the final answer.'
                ),
            },
        ],
    }]
"""
    ),
    markdown("## 6. Mine SFT errors and checkpoint JSONL\n"),
    code(
        """from tqdm.auto import tqdm

def append_jsonl(path, row):
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + '\\n')

pending = [
    (dataset_index, chartqa_train[dataset_index])
    for dataset_index in range(TRAIN_START, TRAIN_START + MINE_N)
    if dataset_index not in completed_indices
]

for position, (dataset_index, example) in enumerate(
    tqdm(pending, desc='Mining ChartQA train predictions'),
    start=1,
):
    question = str(example.get('query', example.get('question', ''))).strip()
    ground_truth = str(unwrap_answer(example.get('label', example.get('answer', '')))).strip()
    messages = build_messages(example['image'], question)
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=prompt,
        images=example['image'],
        return_tensors='pt',
    ).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
        )
    prediction = processor.batch_decode(
        output_ids[:, inputs['input_ids'].shape[1]:],
        skip_special_tokens=True,
    )[0].strip()
    is_correct = deterministic_match(prediction, ground_truth)
    row = {
        'dataset_index': dataset_index,
        'image_dataset': DATASET_NAME,
        'image_split': 'train',
        'image_index': dataset_index,
        'question': question,
        'ground_truth': ground_truth,
        'sft_prediction': prediction,
        'ground_truth_normalized': normalize_answer(ground_truth),
        'prediction_normalized': normalize_answer(prediction),
        'deterministic_correct': is_correct,
        'candidate_status': 'MODEL_CORRECT' if is_correct else 'CANDIDATE_INCORRECT',
        'source_adapter': SFT_ADAPTER_ID,
        'prompt_protocol': 'phase1_final_answer_v1',
    }
    append_jsonl(ALL_PATH, row)

    if position % SAVE_EVERY == 0:
        print(f'Checkpoint: {position}/{len(pending)} new samples')

print('Mining pass complete:', ALL_PATH)
"""
    ),
    markdown("## 7. Export teacher/DPO candidate queues as JSONL\n"),
    code(
        """all_rows = read_jsonl(ALL_PATH)
assert len({row['dataset_index'] for row in all_rows}) == MINE_N, 'Run is incomplete; resume Cell 6.'

errors = [row for row in all_rows if not row['deterministic_correct']]
correct = [row for row in all_rows if row['deterministic_correct']]

def write_jsonl(path, rows):
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\\n')

write_jsonl(ERROR_PATH, errors)
write_jsonl(CORRECT_PATH, correct)
manifest = {
    'dataset': DATASET_NAME,
    'source_split': 'train',
    'train_start': TRAIN_START,
    'mine_n': MINE_N,
    'source_adapter': SFT_ADAPTER_ID,
    'prompt_protocol': 'phase1_final_answer_v1',
    'total_predictions': len(all_rows),
    'deterministic_correct': len(correct),
    'candidate_incorrect': len(errors),
    'outputs': {
        'all': str(ALL_PATH),
        'errors': str(ERROR_PATH),
        'correct': str(CORRECT_PATH),
    },
    'next_step': 'Teacher annotation and audit of candidate_incorrect only; create schema-matched DPO pairs.',
}
MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

print(json.dumps(manifest, ensure_ascii=False, indent=2))
"""
    ),
    markdown(
        """## Next step

Use the exported errors JSONL only as a teacher-review queue. After teacher validation and a manual audit, build DPO pairs with the same response schema in chosen and rejected. Do not use this notebook to touch or evaluate frozen ChartQA val[0:500].
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
