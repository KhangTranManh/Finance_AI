"""Run the 500-sample Phase 2A notebook pipeline outside Colab.

The source of truth remains the Colab notebook. This runner executes its code
cells in order with local project paths so artifacts are saved to results/.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "02_FinChart_R2_Phase2A_Teacher_Assisted_Annotation.ipynb"
PHASE1_ROOT = ROOT.parent / "FinChart-R1"

# The Drive-mount cell is deliberately omitted. All other code cells are run
# exactly in notebook order.
CELL_ORDER = [
    3, 5, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 32, 34, 36, 38,
    40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68,
]


def fallback_display(value: object) -> None:
    """Keep notebook display calls useful in a standard Python process."""
    try:
        try:
            rendered = value.to_string(max_rows=30, max_cols=20)
        except TypeError:
            rendered = value.to_string(max_rows=30)
    except AttributeError:
        rendered = str(value)
    # Windows PowerShell may use a legacy output code page. Audit content can
    # contain Unicode chart symbols, so make display-only output loss-tolerant.
    print(rendered.encode("ascii", "backslashreplace").decode("ascii"))


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    namespace: dict[str, object] = {
        "__name__": "__main__",
        "__file__": str(NOTEBOOK),
        "display": fallback_display,
    }

    for directory in (
        ROOT / "results",
        ROOT / "results" / "audit",
        ROOT / "results" / "logs",
        ROOT / "results" / "checkpoints",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    # Equivalent to the Colab workspace cell, without mounting Drive.
    namespace.update({
        "PHASE1_PROJECT_DIR": PHASE1_ROOT,
        "PROJECT_DIR": ROOT,
        "RESULTS_DIR": ROOT / "results",
        "DATA_DIR": ROOT / "results",
        "AUDIT_DIR": ROOT / "results" / "audit",
        "LOG_DIR": ROOT / "results" / "logs",
        "CHECKPOINT_DIR": ROOT / "results" / "checkpoints",
    })

    for index in CELL_ORDER:
        cell = notebook["cells"][index]
        if cell["cell_type"] != "code":
            raise TypeError(f"Notebook cell {index} is not code.")
        source = "".join(cell["source"])
        print(f"\n=== Running notebook cell {index} ===", flush=True)
        exec(compile(source, f"{NOTEBOOK.name}:cell_{index}", "exec"), namespace)


if __name__ == "__main__":
    main()
