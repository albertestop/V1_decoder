#!/usr/bin/env python3
"""Compare complete-trial neural-AE outliers in one run's train and validation splits.

Example:
    /home/albertestop/miniconda3/envs/sci_albert/bin/python \
        tests/check_train_val_outliers/analyze_run_outliers.py --run run_193 --device cuda:0

The default output directory is ``tests/<run name>/``.  Each row in
``outliers.csv`` is one outlying (split, trial, neuron, token) combination;
``row_start`` and ``row_end_exclusive`` identify every dataset row used to
compute that trial-level MAE.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from v1tovideo.neural_autoencoder.config_parser import parse_neural_ae_experiment_config
from v1tovideo.neural_autoencoder.models import build_model_from_target


# These are deliberately the same thresholds and strict comparison used by
# trainer_sc.save_high_mae_trial_plots.
OUTLIER_THRESHOLDS = {1: 0.3, 2: 0.485}
DEFAULT_DATA_ROOT = Path("/home/albertestop/data/processed_data/tba")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save train/validation trial outliers for a saved neural-AE run."
    )
    parser.add_argument(
        "--run",
        required=True,
        help="Run name (for example run_193) or path to a saved run directory.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "neural_autoencoder",
        help="Directory containing run_* directories when --run is a name.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Local processed-data root used when the run config references an unavailable path.",
    )
    parser.add_argument("--device", default=None, help="Model device; defaults to the run's configured device.")
    parser.add_argument("--batch-size", type=int, default=8, help="Rows evaluated per prediction batch.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to tests/<run name>.",
    )
    return parser.parse_args()


def resolve_run_dir(run: str, runs_root: Path) -> Path:
    candidate = Path(run).expanduser()
    run_dir = candidate if candidate.is_dir() else runs_root / run
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    for name in ("config.toml", "model.pt", "val_indices.json"):
        if not (run_dir / name).is_file():
            raise FileNotFoundError(f"{run_dir.name} is missing required artifact: {name}")
    return run_dir.resolve()


def load_model_config(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        model_config = summary.get("model_config")
        if isinstance(model_config, dict):
            return model_config
    return parse_neural_ae_experiment_config(run_dir / "config.toml").model


def resolve_dataset_dir(configured_path: Path, data_root: Path) -> Path:
    if configured_path.is_dir():
        return configured_path
    local_path = data_root.expanduser() / configured_path.name
    if local_path.is_dir():
        return local_path
    raise FileNotFoundError(
        f"Dataset from run config is unavailable ({configured_path}); local replacement also does not exist: {local_path}"
    )


def load_trials(dataset_dir: Path, val_rows: set[int]) -> list[dict[str, Any]]:
    inputs = np.load(dataset_dir / "data" / "responses.npy", allow_pickle=True).astype(np.float32)
    targets = np.load(dataset_dir / "data" / "targets.npy", allow_pickle=True).astype(np.float32)
    if inputs.ndim != 3 or targets.shape != inputs.shape:
        raise ValueError("responses.npy and targets.npy must share shape [row, neuron, token]")
    trial_map = json.loads((dataset_dir / "trial_dataset_map.json").read_text(encoding="utf-8"))

    trials: list[dict[str, Any]] = []
    for map_key, info in sorted(trial_map.items(), key=lambda item: int(item[1].get("trial_index", item[0]))):
        start, end = (int(value) for value in info["dataset_rows"].split(","))
        if not (0 <= start < end <= len(inputs)):
            raise ValueError(f"Trial {map_key} has invalid dataset rows {start},{end}")
        rows = set(range(start, end))
        if rows <= val_rows:
            split = "validation"
        elif rows.isdisjoint(val_rows):
            split = "train"
        else:
            raise ValueError(f"Trial {map_key} is split between train and validation rows")
        trials.append(
            {
                "split": split,
                "trial_map_key": str(map_key),
                "trial_id": str(info.get("trial_id", map_key)).rstrip("/"),
                "trial_index": int(info.get("trial_index", map_key)),
                "row_start": start,
                "row_end_exclusive": end,
                "inputs": inputs[start:end],
                "targets": targets[start:end],
            }
        )
    if not trials:
        raise ValueError("trial_dataset_map.json contains no trials")
    return trials


def predict_trial(model: torch.nn.Module, trial_inputs: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(trial_inputs), batch_size):
            x = torch.from_numpy(trial_inputs[start : start + batch_size]).to(device)
            padding_mask = torch.zeros(x.shape[:2], dtype=torch.bool, device=device)
            predictions.append(model.predict(x, padding_mask).detach().cpu().numpy())
    return np.concatenate(predictions, axis=0)


def write_results(output_dir: Path, records: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "split", "token", "threshold", "mae", "trial_map_key", "trial_id", "trial_index",
        "row_start", "row_end_exclusive", "row_count", "neuron_index", "neuron_id",
    ]
    with (output_dir / "outliers.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# Train vs validation outliers\n\n"
        "`outliers.csv` has one record per outlying `(split, trial, neuron, token)`. "
        "An outlier has trial-level MAE strictly above 0.3 for token 1 or 0.485 for token 2. "
        "`row_start:row_end_exclusive` is the complete dataset-row range used for that MAE.\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    run_dir = resolve_run_dir(args.run, args.runs_root)
    config = parse_neural_ae_experiment_config(run_dir / "config.toml")
    dataset_dir = resolve_dataset_dir(config.data.path, args.data_root)
    val_indices = [int(index) for index in json.loads((run_dir / "val_indices.json").read_text(encoding="utf-8"))]
    val_rows = set(val_indices)

    trials = load_trials(dataset_dir, val_rows)
    inputs_shape = trials[0]["inputs"].shape
    model_config = load_model_config(run_dir)
    model_kwargs = dict(model_config.get("kwargs", {}))
    model_kwargs.setdefault("num_tokens", int(inputs_shape[1]))
    model_kwargs.setdefault("token_dim", int(inputs_shape[2]))
    model = build_model_from_target(str(model_config["target"]), kwargs=model_kwargs)
    state_dict = torch.load(run_dir / "model.pt", map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict)
    requested_device = args.device or config.train.device
    device = torch.device("cpu" if requested_device.startswith("cuda") and not torch.cuda.is_available() else requested_device)
    model.eval().to(device)

    records: list[dict[str, Any]] = []
    counts: Counter[tuple[str, int]] = Counter()
    trial_counts: Counter[str] = Counter(trial["split"] for trial in trials)
    for trial in trials:
        prediction = predict_trial(model, trial["inputs"], device, args.batch_size)
        for token, threshold in OUTLIER_THRESHOLDS.items():
            if token == 1 and config.train.loss_weight_time == 0:
                continue  # Matches the existing high-MAE analysis: token 1 is not of interest.
            mae_by_neuron = np.mean(np.abs(prediction[:, :, token] - trial["targets"][:, :, token]), axis=0)
            for neuron_index in np.flatnonzero(mae_by_neuron > threshold):
                neuron_id = trial["targets"][0, neuron_index, 0]
                records.append(
                    {
                        "split": trial["split"], "token": token, "threshold": threshold,
                        "mae": float(mae_by_neuron[neuron_index]), "trial_map_key": trial["trial_map_key"],
                        "trial_id": trial["trial_id"], "trial_index": trial["trial_index"],
                        "row_start": trial["row_start"], "row_end_exclusive": trial["row_end_exclusive"],
                        "row_count": trial["row_end_exclusive"] - trial["row_start"],
                        "neuron_index": int(neuron_index), "neuron_id": int(neuron_id),
                    }
                )
                counts[(trial["split"], token)] += 1

    records.sort(key=lambda item: (item["split"], item["token"], item["trial_index"], item["neuron_index"]))
    output_dir = (args.output_dir or REPO_ROOT / "tests" / run_dir.name).resolve()
    summary = {
        "run": run_dir.name,
        "run_dir": str(run_dir),
        "dataset_dir": str(dataset_dir),
        "device": str(device),
        "definition": {"comparison": "mae > threshold", "thresholds": OUTLIER_THRESHOLDS},
        "trial_counts": dict(trial_counts),
        "outlier_counts": {
            split: {f"token_{token}": counts[(split, token)] for token in OUTLIER_THRESHOLDS}
            | {"total": sum(counts[(split, token)] for token in OUTLIER_THRESHOLDS)}
            for split in ("train", "validation")
        },
        "skipped_tokens": [1] if config.train.loss_weight_time == 0 else [],
        "record_count": len(records),
    }
    write_results(output_dir, records, summary)
    print(json.dumps(summary["outlier_counts"], indent=2))
    print(f"Saved {len(records)} outlier records to {output_dir}")


if __name__ == "__main__":
    main()
