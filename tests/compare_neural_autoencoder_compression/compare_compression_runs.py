#!/usr/bin/env python3
"""Compare neural-autoencoder validation metrics across compression levels.

Runs are grouped when their ``config.toml`` files are identical after removing
``custom_model.latent_num_tokens``, ``train.epochs``, ``data.val_trial_selection``, ``data.root``, ``train.device``, and ``output.dir``.
The latter is a run-specific destination rather than a training parameter.
Metrics are read from the same files used by
``tests/summarize_neural_autoencoder_run.py``.

Example:
    python tests/compare_neural_autoencoder_compression/compare_compression_runs.py

Set ``INPUT_NEURONS`` and ``IGNORE_SINGLE_RUN_PARAMETER_SETS`` below to choose
the dataset size and whether parameter sets represented by just one run are
excluded. Only runs whose ``summary.json`` reports the configured number of
input neurons (the second dimension of ``dataset_shape``) are compared.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

try:
    import tomllib
except ModuleNotFoundError as error:  # pragma: no cover - only for Python < 3.11
    raise SystemExit("Python 3.11 or later is required (tomllib is unavailable).") from error


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_DIRECTORY = PROJECT_ROOT / "outputs" / "neural_autoencoder"
DEFAULT_OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "results"

# Comparison settings.
INPUT_NEURONS = 1024
IGNORE_SINGLE_RUN_PARAMETER_SETS = True


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        return tomllib.load(file)


def parameter_set(config: dict[str, Any]) -> dict[str, Any]:
    """Return configuration fields shared by a comparable compression set."""
    parameters = copy.deepcopy(config)
    parameters.get("custom_model", {}).pop("latent_num_tokens", None)
    parameters.get("train", {}).pop("epochs", None)
    parameters.get("train", {}).pop("device", None)
    parameters.get("data", {}).pop("val_trial_selection", None)
    parameters.get("data", {}).pop("data_root_path", None)
    parameters.get("output", {}).pop("dir", None)
    return parameters


def canonical_config(config: dict[str, Any]) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)


def read_outlier_counts(path: Path) -> dict[str, float]:
    counts: dict[str, float] = {}
    if not path.is_file():
        return counts
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        token, value = line.split(":", maxsplit=1)
        try:
            counts[token.strip()] = float(value.strip())
        except ValueError:
            continue
    return counts


def run_number(run_directory: Path) -> int | str:
    suffix = run_directory.name.removeprefix("run_")
    return int(suffix) if suffix.isdigit() else suffix


def run_sort_key(run_directory: Path) -> tuple[int, int | str]:
    identifier = run_number(run_directory)
    return (0, identifier) if isinstance(identifier, int) else (1, identifier)


def inspect_run(run_directory: Path, input_neurons: int) -> dict[str, Any] | None:
    config_path = run_directory / "config.toml"
    stats_path = run_directory / "validation_error_stats.json"
    summary_path = run_directory / "summary.json"
    if not config_path.is_file() or not stats_path.is_file() or not summary_path.is_file():
        return None

    config = load_config(config_path)
    latent_num_tokens = config.get("custom_model", {}).get("latent_num_tokens")
    if latent_num_tokens is None:
        return None

    dataset_shape = load_json(summary_path).get("dataset_shape")
    if not isinstance(dataset_shape, list) or len(dataset_shape) < 2:
        return None
    try:
        run_input_neurons = int(dataset_shape[1])
    except (TypeError, ValueError):
        return None
    if run_input_neurons != input_neurons:
        return None

    validation_stats = load_json(stats_path)
    try:
        token_1_error = float(validation_stats["token_1"]["pred_global_distance_to_target"])
        token_2_error = float(validation_stats["token_2"]["pred_global_distance_to_target"])
    except (KeyError, TypeError, ValueError):
        return None

    outliers = read_outlier_counts(run_directory / "high_mae_trial_plots" / "plot_counts.txt")
    return {
        "run": run_number(run_directory),
        "directory": str(run_directory),
        "input_neurons": run_input_neurons,
        "latent_num_tokens": int(latent_num_tokens),
        "token_1_error": token_1_error,
        "token_2_error": token_2_error,
        "token_1_outliers": outliers.get("token_1"),
        "token_2_outliers": outliers.get("token_2"),
        "parameters": parameter_set(config),
    }


def collect_groups(
    runs_directory: Path,
    input_neurons: int,
    ignore_single_run_parameter_sets: bool,
) -> list[list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run_directory in sorted(runs_directory.glob("run_*"), key=run_sort_key):
        if not run_directory.is_dir():
            continue
        record = inspect_run(run_directory, input_neurons)
        if record is not None:
            grouped[canonical_config(record["parameters"])].append(record)

    # Optionally retain individual runs, e.g. to include their metrics in the
    # report even though they do not provide a compression-level comparison.
    groups = [
        sorted(group, key=lambda record: (record["latent_num_tokens"], record["run"]))
        for group in grouped.values()
        if not ignore_single_run_parameter_sets or len(group) > 1
    ]
    return sorted(groups, key=lambda group: (min(record["run"] for record in group), len(group)))


def mean_points(group: list[dict[str, Any]], metric: str) -> list[dict[str, float | int]]:
    """Average repeated runs at each compression level for one plotted metric."""
    values_by_lnt: dict[int, list[float]] = defaultdict(list)
    for record in group:
        value = record[metric]
        if value is not None:
            values_by_lnt[record["latent_num_tokens"]].append(float(value))
    return [
        {"latent_num_tokens": latent_num_tokens, metric: sum(values) / len(values)}
        for latent_num_tokens, values in sorted(values_by_lnt.items())
    ]


def plot_metric(groups: list[list[dict[str, Any]]], metric: str, ylabel: str, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 6))
    plotted = 0
    for group_index, group in enumerate(groups, start=1):
        points = mean_points(group, metric)
        if not points:
            continue
        axis.plot(
            [record["latent_num_tokens"] for record in points],
            [record[metric] for record in points],
            marker="o",
            linewidth=1.8,
            label=f"Parameter set {group_index}",
        )
        plotted += 1

    axis.set_xlabel("latent_num_tokens (LNT)")
    axis.set_ylabel(ylabel)
    axis.set_title(f"{ylabel} by compression level")
    axis.grid(True, alpha=0.3)
    if plotted:
        axis.legend(title="Configuration", fontsize="small")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def write_parameter_details(groups: list[list[dict[str, Any]]], path: Path) -> None:
    lines = [
        "Parameter sets are identical except for custom_model.latent_num_tokens, train.epochs, data.val_trial_selection, data.data_root_path, train.device, and output.dir.",
        "Outliers are absent when high_mae_trial_plots/plot_counts.txt does not exist.",
        "",
    ]
    for group_index, group in enumerate(groups, start=1):
        lines.extend((f"Parameter set {group_index}", "Parameters:", json.dumps(group[0]["parameters"], indent=2, sort_keys=True), "Runs:"))
        for record in group:
            lines.append(
                "  run_{run}: LNT={latent_num_tokens}, token_1_error={token_1_error:.8g}, "
                "token_2_error={token_2_error:.8g}, token_1_outliers={token_1_outliers}, "
                "token_2_outliers={token_2_outliers}".format(**record)
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIRECTORY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    arguments = parser.parse_args()

    if INPUT_NEURONS <= 0:
        raise ValueError("INPUT_NEURONS must be positive")

    groups = collect_groups(
        arguments.runs_dir,
        INPUT_NEURONS,
        IGNORE_SINGLE_RUN_PARAMETER_SETS,
    )
    output_directory = arguments.output_dir / f"input_neurons_{INPUT_NEURONS}"
    output_directory.mkdir(parents=True, exist_ok=True)
    write_parameter_details(groups, output_directory / "parameter_sets.txt")
    plot_metric(groups, "token_1_error", "Token 1 error", output_directory / "token_1_error_vs_lnt.png")
    plot_metric(groups, "token_2_error", "Token 2 error", output_directory / "token_2_error_vs_lnt.png")
    plot_metric(groups, "token_1_outliers", "Token 1 outliers", output_directory / "token_1_outliers_vs_lnt.png")
    plot_metric(groups, "token_2_outliers", "Token 2 outliers", output_directory / "token_2_outliers_vs_lnt.png")

    print(f"Found {len(groups)} parameter sets.")
    print(f"Wrote plots and parameter details to: {output_directory}")


if __name__ == "__main__":
    main()
