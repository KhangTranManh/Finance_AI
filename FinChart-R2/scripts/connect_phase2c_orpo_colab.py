"""Connect the Phase 2C Colab notebook to the exported ORPO preference pairs.

Run from the repository root.  This keeps notebook JSON edits deterministic.
"""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = Path("FinChart-R2/notebooks/04_FinChart_R2_Phase2C_ORPO_Colab_Leakage_Gated.ipynb")


def lines(source: str) -> list[str]:
    return source.splitlines(keepends=True)


def replace_cell(cells: list[dict], marker: str | tuple[str, ...], source: str) -> None:
    markers = (marker,) if isinstance(marker, str) else marker
    for cell in cells:
        if any(item in "".join(cell.get("source", [])) for item in markers):
            cell["source"] = lines(source)
            return
    raise RuntimeError(f"Notebook cell marker not found: {markers}")


notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
cells = notebook["cells"]

replace_cell(
    cells,
    ("Copy `phase2c_best_clean_grounding_v1.jsonl`", "Copy `phase2c_orpo_51_pairs.jsonl`"),
    """## 2. Mount Google Drive and paths\n\nCopy `phase2c_orpo_51_pairs.jsonl` from the local project into `MyDrive/FinChart-R2/data/` before running. This is the exported `prompt` / `chosen` / `rejected` artifact.\n""",
)

replace_cell(
    cells,
    "PAIR_SOURCE =",
    """from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path
ROOT = Path('/content/drive/MyDrive')
PROJECT_DIR = ROOT / 'FinChart-R2'
DATA_DIR = PROJECT_DIR / 'data'
RUN_DIR = PROJECT_DIR / 'phase2c' / 'orpo_51'
ADAPTER_DIR = RUN_DIR / 'adapter_sft_orpo_51'
CHECKPOINT_DIR = RUN_DIR / 'checkpoints'
for path in [RUN_DIR, ADAPTER_DIR, CHECKPOINT_DIR]:
    path.mkdir(parents=True, exist_ok=True)

PAIR_SOURCE = DATA_DIR / 'phase2c_orpo_51_pairs.jsonl'
SFT_ADAPTER_ID = 'Kxck/Finance_500_v1'
BASE_MODEL = 'unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit'
print('Pair source:', PAIR_SOURCE)
print('SFT adapter:', SFT_ADAPTER_ID)
""",
)

replace_cell(
    cells,
    ("ALLOW_LEAKY_DIAGNOSTIC = False", "ALLOW_LEAKY_DIAGNOSTIC = EXPERIMENT_MODE"),
    """import json

def read_jsonl(path):
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]

if not PAIR_SOURCE.exists():
    raise FileNotFoundError(f'Upload the ORPO pair artifact to Drive first: {PAIR_SOURCE}')
records = read_jsonl(PAIR_SOURCE)
assert len(records) == 51, f'Expected 51 ORPO pairs, got {len(records)}'
required_fields = {'prompt', 'chosen', 'rejected', 'image_split', 'image_index'}
for row in records:
    missing = required_fields - set(row)
    assert not missing, f'Missing ORPO fields: {missing}'
    assert row['chosen'].strip() != row['rejected'].strip(), 'Degenerate preference pair.'

split_counts = {}
for row in records:
    split_counts[row['image_split']] = split_counts.get(row['image_split'], 0) + 1
print('Pair splits:', split_counts)

# The requested 51 pairs are validation-derived.  This enables only a diagnostic
# ORPO run; frozen val[0:500] evaluation remains hard-blocked below.
EXPERIMENT_MODE = 'LEAKY_DIAGNOSTIC'
ALLOW_LEAKY_DIAGNOSTIC = EXPERIMENT_MODE == 'LEAKY_DIAGNOSTIC'
uses_frozen_validation = any(
    row['image_split'] == 'val' and int(row['image_index']) < 500 for row in records
)
if uses_frozen_validation and not ALLOW_LEAKY_DIAGNOSTIC:
    raise RuntimeError(
        'BLOCKED: current 51 pairs are derived from frozen ChartQA val[0:500]. '
        'Rebuild equivalent preference pairs from ChartQA train for a reported experiment.'
    )
if uses_frozen_validation:
    print('LEAKY_DIAGNOSTIC enabled: training may run, but frozen evaluation is prohibited.')
else:
    print('Train-only pairs detected: frozen evaluation is permitted after ORPO.')
""",
)

replace_cell(
    cells,
    ("def text_or_none(value):", "# The pair text is exported verbatim"),
    """from datasets import Dataset, load_dataset

DATASET_NAME = 'HuggingFaceM4/ChartQA'

# The pair text is exported verbatim from phase2c_orpo_51_pairs.jsonl.
# Images are loaded by their stable ChartQA split/index references.
datasets_by_split = {
    split: load_dataset(DATASET_NAME, split=split)
    for split in sorted({row['image_split'] for row in records})
}
pairs = []
for row in records:
    image = datasets_by_split[row['image_split']][int(row['image_index'])]['image']
    prompt = [{
        'role': 'user',
        'content': [
            {'type': 'image', 'image': image},
            {'type': 'text', 'text': row['prompt']},
        ],
    }]
    pairs.append({
        'prompt': prompt,
        'chosen': [{'role': 'assistant', 'content': row['chosen']}],
        'rejected': [{'role': 'assistant', 'content': row['rejected']}],
        'image': image,
        'dataset_index': int(row['dataset_index']),
    })

preference_dataset = Dataset.from_list(pairs)
print(preference_dataset)
print('PROMPT:\\n', records[0]['prompt'])
print('CHOSEN:\\n', records[0]['chosen'])
print('REJECTED:\\n', records[0]['rejected'])
""",
)

replace_cell(
    cells,
    ("Run only after the leakage gate", "This 51-pair run is explicitly"),
    """## 7. Light ORPO train\n\nThis 51-pair run is explicitly a **leaky diagnostic** requested for pair-format and training-path validation: 1 epoch, LR `5e-6`, beta `0.05`. It may save an adapter, but its score must never be compared with the 69.0% frozen SFT-408 result.\n""",
)

replace_cell(
    cells,
    ("Leaky pairs detected. This cell is intentionally blocked", "trainer_stats = trainer.train()"),
    """if uses_frozen_validation and not ALLOW_LEAKY_DIAGNOSTIC:
    raise RuntimeError('Leaky pairs detected. Enable diagnostic mode only for a non-reported smoke test.')

trainer_stats = trainer.train()
model.save_pretrained(str(ADAPTER_DIR))
processor.save_pretrained(str(ADAPTER_DIR))
metadata = {
    'source_adapter': SFT_ADAPTER_ID,
    'pair_source': str(PAIR_SOURCE),
    'pair_count': len(records),
    'experiment_mode': EXPERIMENT_MODE,
    'training_data_boundary': 'VALIDATION_DERIVED_LEAKY_DIAGNOSTIC_ONLY' if uses_frozen_validation else 'TRAIN_ONLY',
    'frozen_evaluation_permitted': not uses_frozen_validation,
}
(RUN_DIR / 'orpo_run_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
print('Saved SFT + ORPO adapter:', ADAPTER_DIR)
print('Saved run metadata:', RUN_DIR / 'orpo_run_metadata.json')
""",
)

NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"Updated {NOTEBOOK}")
