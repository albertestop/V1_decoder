#!/usr/bin/env python3
"""Regenerate post-training diagnostics for one saved neural-AE run.

This reloads ``model.pt`` and the exact validation rows stored in
``val_indices.json``.  It does not train the model and does not alter the
checkpoint, history, summary, configuration, or split.  The generated files
are written back into the selected run directory, matching the final four
diagnostic calls in ``scripts/run_neural_ae_experiment.py``:

* ``val_sample.*.pt`` reconstruction tensors;
* reconstruction PNG plots (loss-history plots when ``history.json`` is available);
* ``validation_error_stats.json`` (and removed-neuron statistics when used);
* ``high_mae_trial_plots/`` and its ``plot_counts.txt``.

Example:
    /home/albertestop/miniconda3/envs/sci_albert/bin/python \
        tests/recompute_neural_ae_run_diagnostics.py --run run_193 --device cuda:1
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import sys
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from v1tovideo.neural_autoencoder.config_parser import parse_neural_ae_experiment_config
from v1tovideo.neural_autoencoder.data import build_dataset, collate_padded_trials
from v1tovideo.neural_autoencoder.models import build_model_from_target
from v1tovideo.neural_autoencoder.trainer_sc import (
    save_high_mae_trial_plots,
    save_reconstruction_artifacts,
    save_reconstruction_plots,
    save_validation_error_stats,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_RUNS_ROOT = REPO_ROOT / "outputs" / "neural_autoencoder"
DEFAULT_DATA_ROOT = Path("/home/albertestop/data/processed_data/tba")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate all post-training neural-AE diagnostics for one saved run."
    )
    parser.add_argument("--run", required=True, help="Run name (for example run_193) or run-directory path.")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Fallback processed-data root when the dataset path saved in config.toml is unavailable.",
    )
    parser.add_argument("--device", default=None, help="Model device; defaults to the saved run configuration.")
    parser.add_argument("--seed", type=int, default=0, help="Seed used for randomly selected reconstruction plots.")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_run_dir(run: str, runs_root: Path) -> Path:
    candidate = Path(run).expanduser()
    run_dir = candidate if candidate.is_dir() else runs_root / run
    run_dir = run_dir.resolve()
    required = ("config.toml", "model.pt", "val_indices.json")
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{run_dir} is missing required artifact(s): {', '.join(missing)}")
    return run_dir


def resolve_dataset_path(config: Any, data_root: Path) -> None:
    """Use a local dataset with the saved directory name when necessary."""
    if config.data.path is None:
        raise ValueError("The saved config does not define data.path")
    if config.data.path.is_dir():
        return
    fallback = data_root.expanduser() / config.data.path.name
    if not fallback.is_dir():
        raise FileNotFoundError(
            f"Dataset saved in config is unavailable ({config.data.path}); "
            f"fallback also does not exist: {fallback}"
        )
    config.data.path = fallback
    LOGGER.info("Using fallback dataset path: %s", fallback)


def load_model_config(run_dir: Path, fallback: dict[str, Any]) -> dict[str, Any]:
    """Prefer summary's resolved model config, which includes inferred dimensions."""
    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        summary = load_json(summary_path)
        saved = summary.get("model_config") if isinstance(summary, dict) else None
        if isinstance(saved, dict) and isinstance(saved.get("target"), str):
            return saved
    return fallback


def build_saved_split_loaders(config: Any, run_dir: Path) -> tuple[DataLoader[Any], DataLoader[Any], Any, dict[str, Any], np.ndarray]:
    """Rebuild the exact train/validation split saved for the run."""
    dataset = build_dataset(config.data)
    val_indices = [int(index) for index in load_json(run_dir / "val_indices.json")]
    if not val_indices or len(set(val_indices)) != len(val_indices):
        raise ValueError("Saved validation indices must be non-empty and unique")
    if min(val_indices) < 0 or max(val_indices) >= len(dataset):
        raise ValueError("Saved validation indices are outside the current dataset bounds")

    val_index_set = set(val_indices)
    train_indices = [index for index in range(len(dataset)) if index not in val_index_set]
    if not train_indices:
        raise ValueError("Saved validation split leaves no training samples")

    dataset_map = load_json(config.data.path / "trial_dataset_map.json")
    val_trial_indices: list[int] = []
    for trial_id, trial_info in dataset_map.items():
        start, end = (int(value) for value in trial_info["dataset_rows"].split(","))
        rows = set(range(start, end))
        if rows <= val_index_set:
            val_trial_indices.append(int(trial_id))
        elif not rows.isdisjoint(val_index_set):
            raise ValueError(f"Saved validation split contains only part of trial {trial_id}")
    if not val_trial_indices:
        raise ValueError("Saved validation indices do not contain a complete trial")

    collate_fn = partial(collate_padded_trials, pad_to_tokens=int(getattr(dataset, "max_tokens")))
    loader_options = {
        "batch_size": config.data.batch_size,
        "num_workers": config.data.num_workers,
        "pin_memory": config.data.pin_memory,
        "collate_fn": collate_fn,
    }
    train_loader = DataLoader(Subset(dataset, train_indices), shuffle=False, drop_last=False, **loader_options)
    val_loader = DataLoader(Subset(dataset, val_indices), shuffle=False, drop_last=False, **loader_options)
    return train_loader, val_loader, dataset, dataset_map, np.asarray(val_trial_indices, dtype=int)


def clear_existing_diagnostics(run_dir: Path) -> None:
    """Remove only artifacts regenerated by this script, never run metadata/checkpoints."""
    for path in run_dir.glob("*.png"):
        path.unlink()
    for path in run_dir.glob("val_sample.*.pt"):
        path.unlink()
    for name in ("validation_error_stats.json", "rem_neuron_validation_error_stats.json"):
        (run_dir / name).unlink(missing_ok=True)
    high_mae_dir = run_dir / "high_mae_trial_plots"
    if high_mae_dir.is_dir():
        shutil.rmtree(high_mae_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = parse_args()
    run_dir = resolve_run_dir(args.run, args.runs_root)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = parse_neural_ae_experiment_config(run_dir / "config.toml")
    resolve_dataset_path(config, args.data_root)
    train_loader, val_loader, dataset, dataset_map, val_trial_indices = build_saved_split_loaders(config, run_dir)

    model_config = load_model_config(run_dir, config.model)
    model_target = str(model_config["target"])
    model_kwargs = dict(model_config.get("kwargs", {}))
    # Older runs may not have summary.json, whose resolved config includes these dimensions.
    model_kwargs.setdefault("num_tokens", int(getattr(dataset, "max_tokens")))
    model_kwargs.setdefault("token_dim", int(getattr(dataset, "token_dim")))
    model = build_model_from_target(model_target, kwargs=model_kwargs)
    state_dict = torch.load(run_dir / "model.pt", map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    # torch.compile stores a checkpoint under _orig_mod.* keys; older BSC runs
    # such as run_281 used that format.
    if isinstance(state_dict, dict) and state_dict and all(
        str(key).startswith("_orig_mod.") for key in state_dict
    ):
        state_dict = {str(key).removeprefix("_orig_mod."): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)

    device = args.device or config.train.device
    LOGGER.info("Regenerating diagnostics for %s on %s", run_dir.name, device)
    clear_existing_diagnostics(run_dir)
    save_reconstruction_artifacts(model, next(iter(val_loader)), run_dir, device, prefix="val_sample")
    try:
        save_reconstruction_plots(
            model=model,
            output_dir=run_dir,
            dataset=dataset,
            dataset_map=dataset_map,
            val_map_idx=val_trial_indices,
            config=config.data,
            device=device,
        )
    except FileNotFoundError:
        if (run_dir / "history.json").is_file():
            raise
        LOGGER.warning(
            "Run has no history.json: generated validation reconstruction plots, but cannot "
            "recreate loss-history or training reconstruction plots."
        )
    save_validation_error_stats(model, val_loader, run_dir, device)
    result = save_high_mae_trial_plots(
        model=model,
        val_dataset=val_loader.dataset,
        train_dataset=train_loader.dataset,
        dataset_dir=config.data.path,
        output_dir=run_dir,
        device=device,
        weight_time=config.train.loss_weight_time,
    )
    LOGGER.info(
        "Finished %s | high-MAE plots: %s | output: %s",
        run_dir.name,
        result["plot_counts"],
        run_dir,
    )


if __name__ == "__main__":
    main()
