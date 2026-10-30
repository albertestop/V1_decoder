#!/usr/bin/env python3
"""Continue a saved neural-autoencoder run in place.

Edit ``RUN_DIR`` and ``ADDITIONAL_EPOCHS`` below, then run this file from the
repository root (or with an absolute path).  Command-line options are provided
as a convenience, but the constants make the intended run and epoch count
explicit in the script.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import shutil
import sys
import tempfile
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

from v1tovideo.neural_autoencoder import (
    build_model_from_target,
    evaluate_autoencoder,
    infer_batch_shape,
    save_checkpoint,
    train_autoencoder,
)
from v1tovideo.neural_autoencoder.config_parser import parse_neural_ae_experiment_config
from v1tovideo.neural_autoencoder.data import build_dataset, collate_padded_trials
from v1tovideo.neural_autoencoder.trainer_sc import (
    save_high_mae_trial_plots,
    save_reconstruction_artifacts,
    save_reconstruction_plots,
    save_validation_error_stats,
)

# Set these two values before running the script.
RUN_DIR = REPO_ROOT / "outputs" / "neural_autoencoder" / "run_X"
ADDITIONAL_EPOCHS = 100

LOGGER = logging.getLogger(__name__)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def load_saved_model_config(run_dir: Path, fallback: dict[str, Any]) -> dict[str, Any]:
    """Prefer the resolved model config saved in summary.json.

    The original training script adds inferred ``num_tokens`` and ``token_dim``
    to this config, so it is the most faithful description of the saved model.
    """
    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        summary = load_json(summary_path)
        saved = summary.get("model_config") if isinstance(summary, dict) else None
        if isinstance(saved, dict) and isinstance(saved.get("target"), str):
            return saved
    return fallback


def build_saved_split_loaders(config: Any, run_dir: Path) -> tuple[DataLoader[Any], DataLoader[Any], Any, dict[str, Any], np.ndarray]:
    """Rebuild the exact train/validation split recorded by the original run."""
    val_indices_path = run_dir / "val_indices.json"
    if not val_indices_path.is_file():
        raise FileNotFoundError(f"Missing saved validation indices: {val_indices_path}")

    dataset = build_dataset(config.data)
    val_indices = [int(index) for index in load_json(val_indices_path)]
    if not val_indices:
        raise ValueError(f"Saved validation indices are empty: {val_indices_path}")
    if len(set(val_indices)) != len(val_indices):
        raise ValueError(f"Saved validation indices contain duplicates: {val_indices_path}")
    if min(val_indices) < 0 or max(val_indices) >= len(dataset):
        raise ValueError(f"Saved validation indices are outside dataset bounds: {val_indices_path}")

    val_index_set = set(val_indices)
    train_indices = [index for index in range(len(dataset)) if index not in val_index_set]
    if not train_indices:
        raise ValueError("Saved validation split leaves no training samples")

    with (config.data.path / "trial_dataset_map.json").open("r", encoding="utf-8") as fp:
        dataset_map = json.load(fp)

    val_trial_indices: list[int] = []
    for trial_id, trial_info in dataset_map.items():
        start, end = (int(value) for value in trial_info["dataset_rows"].split(","))
        trial_rows = set(range(start, end))
        if trial_rows <= val_index_set:
            val_trial_indices.append(int(trial_id))
        elif not trial_rows.isdisjoint(val_index_set):
            raise ValueError(f"Saved split contains only part of trial {trial_id}")

    if not val_trial_indices:
        raise ValueError("Saved validation indices do not contain any complete trial")

    collate_fn = partial(collate_padded_trials, pad_to_tokens=int(getattr(dataset, "max_tokens")))
    loader_options = {
        "batch_size": config.data.batch_size,
        "num_workers": config.data.num_workers,
        "pin_memory": config.data.pin_memory,
        "collate_fn": collate_fn,
    }
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        shuffle=config.data.shuffle_train,
        drop_last=config.data.drop_last,
        **loader_options,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        shuffle=False,
        drop_last=False,
        **loader_options,
    )
    return train_loader, val_loader, dataset, dataset_map, np.asarray(val_trial_indices, dtype=int)


def clear_generated_outputs(run_dir: Path) -> None:
    """Remove only generated results; retain config.toml and val_indices.json."""
    for path in run_dir.glob("*.png"):
        path.unlink()
    for name in (
        "model.pt",
        "history.json",
        "summary.json",
        "kept_neuron_indices.json",
        "validation_error_stats.json",
        "rem_neuron_validation_error_stats.json",
    ):
        (run_dir / name).unlink(missing_ok=True)
    for path in run_dir.glob("val_sample.*.pt"):
        path.unlink()
    high_mae_dir = run_dir / "high_mae_trial_plots"
    if high_mae_dir.exists():
        shutil.rmtree(high_mae_dir)


def best_recorded_val_loss(history: list[dict[str, Any]], run_dir: Path) -> float:
    """Return the old model's best recorded validation loss."""
    losses = [
        float(row["val_loss"])
        for row in history
        if isinstance(row, dict)
        and "val_loss" in row
        and np.isfinite(float(row["val_loss"]))
    ]
    if losses:
        return min(losses)

    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        summary = load_json(summary_path)
        if isinstance(summary, dict) and "val_loss" in summary:
            loss = float(summary["val_loss"])
            if np.isfinite(loss):
                return loss
    raise ValueError(
        "Cannot determine the loaded model's validation loss from history.json or summary.json; "
        "refusing to risk replacing the existing best checkpoint."
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Continue a saved neural-autoencoder run in place.")
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR, help="Saved run directory to overwrite.")
    parser.add_argument("--epochs", type=int, default=ADDITIONAL_EPOCHS, help="Additional epochs to train.")
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    required = ("config.toml", "model.pt", "val_indices.json")
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{run_dir} is not a resumable run; missing: {', '.join(missing)}")

    config = parse_neural_ae_experiment_config(run_dir / "config.toml")
    train_loader, val_loader, dataset, dataset_map, val_trial_indices = build_saved_split_loaders(config, run_dir)
    model_config = load_saved_model_config(run_dir, config.model)
    model_target = str(model_config["target"])
    model_kwargs = dict(model_config.get("kwargs", {}))
    num_tokens, token_dim = infer_batch_shape(next(iter(train_loader)))
    model_kwargs.setdefault("num_tokens", num_tokens)
    model_kwargs.setdefault("token_dim", token_dim)
    model_config = {"target": model_target, "kwargs": model_kwargs}

    model = build_model_from_target(model_target, kwargs=model_kwargs)
    state_dict = torch.load(run_dir / "model.pt", map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict)
    LOGGER.info("Loaded %s from %s", model_target, run_dir / "model.pt")

    old_history: list[dict[str, Any]] = []
    history_path = run_dir / "history.json"
    if history_path.is_file():
        loaded_history = load_json(history_path)
        if not isinstance(loaded_history, list):
            raise ValueError(f"Expected a list in {history_path}")
        old_history = loaded_history
    previous_epoch = max((float(row.get("epoch", 0)) for row in old_history), default=0.0)
    old_best_val_loss = best_recorded_val_loss(old_history, run_dir)
    original_state_dict = copy.deepcopy(model.state_dict())
    total_epoch_budget = max(int(config.train.epochs), int(previous_epoch)) + args.epochs

    # train_autoencoder writes its best checkpoint during training.  Use a
    # temporary directory so an interrupted continuation cannot damage run_X.
    config.train.epochs = args.epochs
    config.train.early_stopping_patience_epochs = total_epoch_budget
    LOGGER.info(
        "Continuation early-stopping patience=%d epochs (ceil(%d / 10))",
        (total_epoch_budget + 9) // 10,
        total_epoch_budget,
    )
    with tempfile.TemporaryDirectory(prefix="neural_ae_continue_") as checkpoint_dir:
        new_history = train_autoencoder(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config.train,
            checkpoint_dir=Path(checkpoint_dir),
        )

    new_losses = [
        float(row["val_loss"])
        for row in new_history
        if "val_loss" in row and np.isfinite(float(row["val_loss"]))
    ]
    new_best_val_loss = min(new_losses, default=float("inf"))
    if new_best_val_loss >= old_best_val_loss:
        model.load_state_dict(original_state_dict)
        LOGGER.info(
            "Retained loaded model | old best val_loss=%.6f <= continuation best=%.6f",
            old_best_val_loss,
            new_best_val_loss,
        )
    else:
        LOGGER.info(
            "Continuation improved best val_loss | %.6f -> %.6f",
            old_best_val_loss,
            new_best_val_loss,
        )

    for row in new_history:
        row["epoch"] = previous_epoch + float(row["epoch"])
    history = old_history + new_history
    eval_metrics = evaluate_autoencoder(model=model, dataloader=val_loader, device=config.train.device)

    # All continuation training completed successfully: now replace the saved
    # model and every generated post-training artifact in the original run.
    clear_generated_outputs(run_dir)
    save_checkpoint(model, run_dir / "model.pt")
    with (run_dir / "history.json").open("w", encoding="utf-8") as fp:
        json.dump(history, fp, indent=2)

    latent_dim = int(model_kwargs["latent_dim"]) if "latent_dim" in model_kwargs else None
    summary: dict[str, Any] = {
        "dataset_shape": dataset.shape,
        "model_name": model_target,
        "model_config": model_config,
        "latent_dim": latent_dim,
        "train_loss": history[-1]["train_loss"] if history else float("nan"),
        "val_loss": history[-1]["val_loss"] if history else float("nan"),
        "val_mse": eval_metrics["mse"],
        "val_mae": eval_metrics["mae"],
        "continued_epochs": args.epochs,
        "total_epochs_recorded": len(history),
    }
    summary["compression_ratio"] = float((token_dim * num_tokens) / latent_dim) if latent_dim is not None else None
    with (run_dir / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    if hasattr(model, "keep"):
        with (run_dir / "kept_neuron_indices.json").open("w", encoding="utf-8") as fp:
            json.dump([int(index) for index in model.keep.detach().cpu().tolist()], fp, indent=2)

    save_reconstruction_artifacts(model, next(iter(val_loader)), run_dir, config.train.device, prefix="val_sample")
    save_reconstruction_plots(
        model=model,
        output_dir=run_dir,
        dataset=dataset,
        dataset_map=dataset_map,
        val_map_idx=val_trial_indices,
        config=config.data,
        device=config.train.device,
    )
    save_validation_error_stats(model, val_loader, run_dir, config.train.device)
    save_high_mae_trial_plots(
        model=model,
        val_dataset=val_loader.dataset,
        train_dataset=train_loader.dataset,
        dataset_dir=config.data.path,
        output_dir=run_dir,
        device=config.train.device,
        weight_time=config.train.loss_weight_time,
    )
    LOGGER.info("Continuation finished | output_dir=%s", run_dir)
    print(summary)


if __name__ == "__main__":
    main()
