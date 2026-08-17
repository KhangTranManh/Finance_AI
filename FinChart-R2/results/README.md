# Local generated outputs

This directory is deliberately excluded from Git except for this note. It may contain teacher checkpoints, request logs, audits, candidate datasets, approved pilot SFT data, and evaluations.

Recreate the Phase 2A pilot locally with the scripts listed in the root README. Do not commit raw teacher responses, provider metadata, API logs, or generated training data.

At the current pilot stage, `phase2a_pilot_500_v3_train_clean.jsonl` is the only retained training artifact. It contains the strict-clean examples approved for a small SFT experiment; it is not a final full-train dataset.
