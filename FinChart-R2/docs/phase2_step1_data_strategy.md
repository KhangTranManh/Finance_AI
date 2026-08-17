# Phase 2A, Step 1 — failure-driven data strategy

## Decision

Phase 2 will use the ChartQA **training** split only to improve the skills that the frozen Phase 1 evaluation identified as weak. It will not train directly on the raw question-answer pairs until label-quality rules and a structured-supervision schema have been approved.

The desired solving process is:

```text
interpret question → identify relevant chart elements → extract values → choose operation → calculate or compare → answer
```

## Evidence and starting curriculum

| Primary skill | Phase 1 confirmed errors | Error share | Initial target after filtering |
| --- | ---: | ---: | ---: |
| Numerical reasoning | 54 | 47.4% | 45% |
| Visual grounding | 34 | 29.8% | 30% |
| Counting | 21 | 18.4% | 15% |
| Logical reasoning | 5 | 4.4% | 10% |

These are soft sampling targets, not quotas. Logical reasoning is modestly oversampled because it composes grounding and arithmetic skills and matters for robust chart understanding.

## Taxonomy and sample requirements

| Skill | Failure addressed | Subtypes | Required quality evidence |
| --- | --- | --- | --- |
| Numerical reasoning | Wrong arithmetic, scale, ratio direction, or values from a wrong series. | Sum, difference, average, median, ratio, percentage, percentage change, min/max plus arithmetic, units, multi-step calculation. | Every value is visually or metadata-grounded; calculation recomputes; unit and rounding match the answer. |
| Visual grounding | Wrong legend, series, bar, point, row, or column selected in dense charts. | Legend mapping, color, series, bar/segment, point, row/column, value extraction, extrema. | Target element is unique, visible, and sufficiently described; reject subjective colors and unreadable labels. |
| Counting | Under/over-counting qualifying visual elements. | Bars, points, occurrences, values above/below threshold, categories, colors. | Eligible universe and threshold boundary are explicit; enumerate qualifying elements during audit. |
| Logical reasoning | Incorrect comparisons, ranking, intervals, or conditions. | Greater/lower, ranking, interval, `A > B + C`, trend/crossing. | All operands are grounded; ties, periods, and inclusive boundaries are unambiguous. |

Use concise structured reasoning when it improves supervision. For an arithmetic question, later examples may state `Values: 72, 71, 61, 47; Operation: average; Calculation: 251 / 4 = 62.75; Answer: 62.75.` For direct extraction, use only the visual reference and answer.

## Curriculum

1. Start with direct grounding and one-step arithmetic.
2. Add composed tasks: extrema plus arithmetic, threshold counting, ratios, and comparisons.
3. Add dense multi-series charts, stacked bars, unit conversion, global counting, and explicit multi-step questions.

Each future example will have one primary task for balancing and may receive secondary tags for its other required skills.

## Risks and constraints

- The Phase 1 review found 28 possible label errors in 500 validation cases; raw labels require quality gates.
- Percentage/proportion representation may be normalizable, but semantic corrections require stronger evidence and must never be silently invented.
- Overweighting arithmetic without grounding can teach fluent calculations over the wrong values.
- Long free-form reasoning is unnecessary, costly, and harder to validate; use short inspectable fields instead.
- Never include the fixed R1 validation subset in SFT data. R1’s evaluator and subset remain frozen for the base-versus-SFT comparison.

## Step 1 exit gate

Proceed to Step 2 only after approving the taxonomy, the 45/30/15/10 starting mix, concise structured supervision, label-quality gates, and strict train/evaluation separation.
