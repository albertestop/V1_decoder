#!/usr/bin/env python3
"""Export neural-autoencoder run metrics as a table ready to paste into PowerPoint.

Run example
-------
python tests/export_neural_autoencoder_runs_table.py 203 204 205
-------
python tests/export_neural_autoencoder_runs_table.py 203 204 205 \\
    --labels "1024 -> 512, t_w = 10, id_w = 0.1" "256, t_w = 1" "64, t_w = 0" \\
    --output outputs/neural_autoencoder/comparison.tsv

Open the generated TSV file, select all, copy, then paste into a PowerPoint
table with eight columns.  Each tab becomes a new cell and each line a new row.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "neural_autoencoder" / "run_comparison.tsv"


def load_presentation_helper():
    """Load the collector used for the .pptx version, without making tests a package."""
    path = Path(__file__).with_name("generate_neural_autoencoder_runs_ppt.py")
    spec = importlib.util.spec_from_file_location("ppt_run_comparison", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load metric collector from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="Run numbers or names, e.g. 203 run_204.")
    parser.add_argument("--labels", nargs="+", help="Optional labels, one per run in the same order.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help=f"TSV output path (default: {DEFAULT_OUTPUT}).")
    arguments = parser.parse_args()
    if arguments.labels is not None and len(arguments.labels) != len(arguments.runs):
        parser.error("--labels must contain exactly one label for each run.")

    helper = load_presentation_helper()
    labels = arguments.labels or [None] * len(arguments.runs)
    rows = [
        helper.collect_row(run.removeprefix("run_"), label)
        for run, label in zip(arguments.runs, labels)
    ]
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    # Use single-line headers: quoted/multiline TSV headers paste unreliably in PowerPoint.
    headers = [header.replace("\n", " ") for header in helper.HEADERS]
    with arguments.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, dialect="excel-tab", lineterminator="\n")
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"Created {arguments.output.resolve()} with {len(rows)} run(s).")


if __name__ == "__main__":
    main()
