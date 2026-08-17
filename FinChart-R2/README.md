# FinChart-R2

Phase 2 of FinChart tests whether targeted, teacher-assisted structured supervision can improve chart reasoning in **Qwen3-VL-4B-Instruct**, while keeping the deployable model compact. The teacher is used offline only to create supervision; it is never required at inference time.

## Research scope

Phase 1 froze the evaluator and ChartQA `val[0:500]` subset, then identified four principal failure modes: numerical reasoning, visual grounding/value extraction, counting, and logical reasoning. Phase 2 uses **only ChartQA train** for SFT; the Phase 1 validation subset remains training-free.

Phase 2A runs a teacher-assisted data-engineering pipeline:

```text
ChartQA train -> normalization -> VLM teacher -> structured annotation
-> deterministic validation -> quality filtering/audit -> SFT-ready data
```

Each annotation has a closed task taxonomy and fields for target series/category, relevant values, operation, calculation, answer, and confidence. Only `VALIDATED` examples that pass strict filtering are eligible for the pilot SFT dataset. The current local pilot output contains 408 strict-clean examples from a 500-example annotation run; it is intentionally not committed to Git.

Phase 2B has trained a QLoRA pilot on the 408 approved examples. Its preliminary deterministic validation result is 69.0% (345/500), versus the 63.4% Phase 1 base score. The result is tracked as preliminary because the pilot generation prompt differs from the frozen Phase 1 prompt; a protocol-identical rerun and semantic error analysis remain required before scaling data.

## Repository layout

```text
FinChart-R2/
|- configs/       # non-secret reproducible settings
|- docs/          # research design and data contract
|- notebooks/     # Colab entry point (outputs stripped for Git)
|- scripts/       # reproducible Phase 2A pipeline stages
|- results/       # local generated data; ignored by Git
|- .env.template  # copy to .env and fill locally
`- requirements.txt
```

## Quick start: Phase 2A

1. Create a local environment file: `Copy-Item .env.template .env`, then set the teacher credentials locally.
2. Install dependencies: `pip install -r requirements.txt`.
3. In Colab, upload and run [the Phase 2A notebook](notebooks/02_FinChart_R2_Phase2A_Teacher_Assisted_Annotation.ipynb), or run the local stages in order:

   ```powershell
   python scripts/run_phase2a_pilot.py
   python scripts/rebuild_phase2a_v2.py
   python scripts/approve_and_build_train_v3.py
   ```

Generated annotations, logs, audits, and SFT data stay under `results/` and are excluded from version control.

## Phase 2B pilot SFT

Use [the Phase 2B pilot notebook](notebooks/03_FinChart_R2_Phase2B_Pilot_SFT_408.ipynb) after Phase 2A produces the approved `v3_train_clean` JSONL. It trains the QLoRA pilot and saves adapters, checkpoints, metrics, and predictions locally. The tracked [pilot report](../reports/phase2b_pilot_408_evaluation.md) contains the extracted result; raw artifacts are intentionally ignored.

## Guardrails

- Never train on the frozen Phase 1 validation subset.
- The teacher is a primary annotator, not ground truth.
- Validate schema, taxonomy, answer representation, arithmetic, confidence, and semantic conflicts before SFT.
- Do not use unrestricted chain-of-thought; targets are concise, inspectable structured supervision.
- Scale from 500 to 2k-3k+ examples only after frozen evaluation shows a pilot improvement.
