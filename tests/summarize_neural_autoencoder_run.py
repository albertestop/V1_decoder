#!/usr/bin/env python3
"""Print key validation metrics for a neural-autoencoder run.

Run without arguments to be prompted for a run number, or pass it directly:
    python tests/summarize_neural_autoencoder_run.py 203
"""

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIRECTORY = PROJECT_ROOT / "outputs" / "neural_autoencoder"
MISSING_VALUE = "non-existent"


def format_value(value: float | str) -> str:
    if isinstance(value, str):
        return value
    return f"{value:.6g}"


def load_json(path: Path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def read_outlier_counts(path: Path) -> dict[str, str]:
    counts = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        token, count = line.split(":", maxsplit=1)
        counts[token.strip()] = count.strip()
    return counts


def get_run_identifier(argument: str | None) -> str:
    if argument is not None:
        return argument.removeprefix("run_")

    run_identifier = input("Run number: ").strip().removeprefix("run_")
    if not run_identifier:
        raise ValueError("A run number is required.")
    return run_identifier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?", help="Run number, for example 203")
    arguments = parser.parse_args()

    run_identifier = get_run_identifier(arguments.run)
    run_directory = RUNS_DIRECTORY / f"run_{run_identifier}"
    validation_path = run_directory / "validation_error_stats.json"
    counts_path = run_directory / "high_mae_trial_plots" / "plot_counts.txt"
    history_path = run_directory / "history.json"

    if validation_path.is_file():
        validation_stats = load_json(validation_path)
        token_1 = validation_stats["token_1"]
        token_2 = validation_stats["token_2"]
        token_1_error = format_value(token_1["pred_global_distance_to_target"])
        token_1_std = format_value(token_1["global_distance_std"])
        token_1_misalignment = format_value(token_1["mean_align_dist"])
        token_2_error = format_value(token_2["pred_global_distance_to_target"])
        token_2_std = format_value(token_2["global_distance_std"])
    else:
        token_1_error = token_1_std = token_1_misalignment = MISSING_VALUE
        token_2_error = token_2_std = MISSING_VALUE

    if counts_path.is_file():
        outlier_counts = read_outlier_counts(counts_path)
        token_1_outliers = outlier_counts.get("token_1", "not found")
        token_2_outliers = outlier_counts.get("token_2", "not found")
    else:
        token_1_outliers = token_2_outliers = MISSING_VALUE

    if history_path.is_file():
        history = load_json(history_path)
        if not history:
            raise ValueError(f"No epoch history found in {history_path}")

        best_epoch = min(history, key=lambda epoch: epoch["val_loss"])
        best_validation_epoch = f"{best_epoch['epoch']:g}"
        overfitting = format_value(best_epoch["val_loss"] / best_epoch["train_loss"])
        time_loss_ratio = format_value(best_epoch["val_loss_time"] / best_epoch["train_loss_time"])
    else:
        best_validation_epoch = overfitting = time_loss_ratio = MISSING_VALUE

    print(f"Run {run_identifier} ({run_directory})")
    print(f"Token 1 error:         {token_1_error} ± {token_1_std}")
    print(f"Token 1 misalignment:  {token_1_misalignment}")
    print(f"Token 2 error:         {token_2_error} ± {token_2_std}")
    print(f"Token 1 outliers:      {token_1_outliers}")
    print(f"Token 2 outliers:      {token_2_outliers}")
    print(f"Best validation epoch:  {best_validation_epoch}")
    print(f"Overfitting (val/train):      {overfitting}")
    print(f"Time loss ratio (val/train):  {time_loss_ratio}")


if __name__ == "__main__":
    main()
