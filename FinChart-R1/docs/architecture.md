# Phase 1 architecture

Phase 1 separates model prediction from evaluation so later SFT and Vision GRPO experiments can use exactly the same evaluator.

```text
ChartQA validation subset
  -> Qwen3-VL-4B greedy baseline
  -> deterministic evaluator
     -> match: CORRECT
     -> mismatch: vision LLM examiner
          -> CORRECT | INCORRECT | AMBIGUOUS | POSSIBLE_LABEL_ERROR
          -> transport/API/JSON failure: retry + backoff -> JUDGE_ERROR
  -> coverage-aware summary and error analysis
```

The baseline CSV is an immutable prediction record. The judge checkpoint can be rerun after a runtime/API failure without re-running Qwen. Final analysis derives `final_verdict` from both files.

`AMBIGUOUS` is a successful semantic inspection with no confident resolution. `JUDGE_ERROR` is a technical failure and must remain distinct.
