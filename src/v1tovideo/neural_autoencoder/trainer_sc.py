from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from v1tovideo.neural_autoencoder.data import collate_padded_trials


def plot_neuron_data_evo(target_trial, recons_trial, neuron, token_idx, save_path):
    plt.figure(figsize=(8, 3))
    plt.plot(np.arange(len(target_trial[:, int(neuron), 0])), target_trial[:, int(neuron), token_idx], label="Target")
    plt.plot(np.arange(len(recons_trial[:, int(neuron), 0])), recons_trial[:, int(neuron), token_idx], label="Reconstructed")
    plt.xlabel("Cycle n")
    plt.ylabel("Token Value")
    plt.title(f"Token {token_idx} of neuron {int(target_trial[0, int(neuron), 0])} during trial: target vs reconstruction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_vol_error(error, token_idx, save_path):
    plt.figure(figsize=(16, 6))
    plt.bar(np.arange(len(error)), error, width=1)
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Neuron")
    plt.ylabel("Token Value")
    plt.title(f"Token {token_idx}  reconstruction error of each neuron on a single cycle n")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_reconstruction_artifacts(
    model: nn.Module,
    sample_batch: Any,
    output_dir: Path,
    device: str = "cuda",
    prefix: str = "sample",
) -> None:
    """Save input tensors, latent vectors, and reconstructions for inspection."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if device.startswith("cuda") and not torch.cuda.is_available():
        dev = torch.device("cpu")
    else: dev = torch.device(device)
    model.eval().to(dev)

    with torch.no_grad():
        if torch.is_tensor(sample_batch):
            x = sample_batch.to(dev)
            padding_mask = torch.zeros((x.shape[0], x.shape[1]), dtype=torch.bool, device=dev)
        else:
            x = sample_batch[0].to(dev)
            padding_mask = sample_batch[2].to(dev).bool()
        out = model(x, padding_mask=padding_mask)
        if not isinstance(out, (tuple, list)):
            raise ValueError("Model forward must return a tuple/list")
        if len(out) == 2:
            recon, latents = out
        elif len(out) == 4:
            id_output, time_pred, rec_pred, latents = out
            if model.outputs[0] == 'value':
                id_pred = id_output
            else:
                id_pred = id_output.argmax(dim=-1).to(dtype=x.dtype)
            recon = torch.stack(
                (id_pred, time_pred.squeeze(-1).to(dtype=x.dtype), rec_pred.squeeze(-1).to(dtype=x.dtype)),
                dim=-1,
            ).masked_fill(padding_mask.unsqueeze(-1), 0.0)
        else:
            raise ValueError(f"Unsupported model output tuple length: {len(out)}")

    torch.save(x.cpu(), output_dir / f"{prefix}.input.pt")
    torch.save(padding_mask.cpu(), output_dir / f"{prefix}.padding_mask.pt")
    torch.save(latents.cpu(), output_dir / f"{prefix}.latents.pt")
    torch.save(recon.cpu(), output_dir / f"{prefix}.reconstruction.pt")


def save_reconstruction_plots(
    model: nn.Module,
    output_dir: Path,
    dataset: Dataset,
    dataset_map: dict,
    val_map_idx: np.ndarray,
    config,
    device: str = "cuda",
) -> None:
    """Save before/after plots for selected tokens and parameter heatmaps."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if device.startswith("cuda") and not torch.cuda.is_available():
        dev = torch.device("cpu")
    else: dev = torch.device(device)
    model.eval().to(dev)
    keep = getattr(model, "keep", [])
    if torch.is_tensor(keep):
        keep = keep.detach().cpu().tolist()
    keep_ids = set(map(int, keep)) if len(keep) > 0 else set()

    plot_trial_idx = np.random.choice(val_map_idx)
    plot_rows_start, plot_rows_end = map(int, dataset_map[f"{plot_trial_idx}"]["dataset_rows"].split(","))
    plot_indices = set(range(plot_rows_start, plot_rows_end))
    plot_set = Subset(dataset, list(plot_indices))
    pad_to_tokens = int(getattr(dataset, "max_tokens"))
    collate_fn = partial(collate_padded_trials, pad_to_tokens=pad_to_tokens)
    plot_loader = DataLoader(
        plot_set,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last,
        collate_fn=collate_fn,
    )

    target_trial = []
    recons_trial = []
    for batch in plot_loader:
        with torch.no_grad():
            if torch.is_tensor(batch):
                x = batch.to(dev)
                target = x
                padding_mask = torch.zeros((x.shape[0], x.shape[1]), dtype=torch.bool, device=dev)
            else:
                x = batch[0].to(dev)
                target = batch[3].to(dev) if len(batch) >= 4 and torch.is_tensor(batch[3]) else x
                padding_mask = batch[2].to(dev).bool()
            out = model(x, padding_mask=padding_mask)
            if not isinstance(out, (tuple, list)):
                raise ValueError("Model forward must return a tuple/list")

            recon = model.predict(x, padding_mask)

        if x.ndim != 3 or x.shape[0] == 0:
            raise ValueError(f"Expected sample_batch shape [B, N, D], got {tuple(x.shape)}")

        valid_len = int((~padding_mask[0]).sum().item())
        target_values = target[0, :valid_len].detach().cpu().numpy()  # [P, T]
        reconstructed = recon[0, :valid_len].detach().cpu().numpy()  # [P, T]
        target_trial.append(target_values)
        recons_trial.append(reconstructed)

    target_trial = np.array(target_trial, dtype=object)
    recons_trial = np.array(recons_trial, dtype=object)

    vol_idx = np.random.randint(0, len(recons_trial))
    for token_idx in range(len(recons_trial[0, 0, :])):
        error = (
            recons_trial[vol_idx, :, token_idx]
            - target_trial[vol_idx, :, token_idx]
        )
        save_path = output_dir / f"vol_n_token_{token_idx}_val.png"
        plot_vol_error(error, token_idx, save_path)
        if keep_ids:
            keep_mask = np.isin(target_trial[vol_idx, :, 0], list(keep_ids))
            non_keep_idx = np.flatnonzero(~keep_mask)
            error = (
                recons_trial[vol_idx, non_keep_idx, token_idx]
                - target_trial[vol_idx, non_keep_idx, token_idx]
            )
            save_path = output_dir / f"rem_neuron_vol_n_token_{token_idx}_val.png"
            plot_vol_error(error, token_idx, save_path)

    neuron = np.random.randint(0, len(recons_trial[0, :, 0]))
    if keep_ids:
        keep_mask = np.isin(target_trial[vol_idx, :, 0], list(keep_ids))
        non_keep_idx = np.flatnonzero(~keep_mask)
        if len(non_keep_idx) > 0:
            rem_neuron_idx = np.random.choice(non_keep_idx)
    for token_idx in range(len(recons_trial[0, 0, :])):
        save_path = output_dir / f"neuron_n_token_{token_idx}_val.png"
        plot_neuron_data_evo(target_trial, recons_trial, neuron, token_idx, save_path)
        if keep_ids:
            if len(non_keep_idx) > 0:
                save_path = output_dir / f"rem_neuron_n_token_{token_idx}_val.png"
                plot_neuron_data_evo(target_trial, recons_trial, rem_neuron_idx, token_idx, save_path)

    with open(os.path.join(output_dir, 'history.json'), 'r') as file:
        history = json.load(file)
    epochs = [d["epoch"] for d in history]
    train_loss = [d["train_loss"] for d in history]
    val_loss = [d["val_loss"] for d in history]
    def smooth(values, window=10):
        if len(values) < window:
            return values
        values = np.asarray(values)
        kernel = np.ones(window) / window
        pad_left = window // 2
        pad_right = window - 1 - pad_left
        return np.convolve(np.pad(values, (pad_left, pad_right), mode="edge"), kernel, mode="valid")

    plt.figure()
    plt.plot(epochs, train_loss, color="C0", alpha=0.25, label="Train Loss")
    plt.plot(epochs, val_loss, color="C1", alpha=0.25, label="Validation Loss")
    plt.plot(epochs, smooth(train_loss), color="C0", label="Train Loss Smoothed")
    plt.plot(epochs, smooth(val_loss), color="C1", label="Validation Loss Smoothed")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, 'train_evo.png'))
    plt.close()

    component_losses = (
        ("id", "ID Loss"),
        ("time", "Time Loss"),
        ("rec", "Recording Loss"),
    )
    if all(f"{split}_loss_{name}" in history[0] for split in ("train", "val") for name, _ in component_losses):
        fig, axes = plt.subplots(len(component_losses), 1, figsize=(8, 9), sharex=True)
        for ax, (name, title) in zip(axes, component_losses):
            train_comp_loss = [d.get(f"train_loss_{name}", np.nan) for d in history]
            val_comp_loss = [d.get(f"val_loss_{name}", np.nan) for d in history]
            ax.plot(epochs, train_comp_loss, color="C0", alpha=0.25, label=f"Train {title}")
            ax.plot(epochs, val_comp_loss, color="C1", alpha=0.25, label=f"Validation {title}")
            ax.plot(epochs, smooth(train_comp_loss), color="C0", label=f"Train {title} Smoothed")
            ax.plot(epochs, smooth(val_comp_loss), color="C1", label=f"Validation {title} Smoothed")
            ax.set_ylabel("Loss")
            ax.set_title(title)
            ax.legend()
            ax.grid()
        axes[-1].set_xlabel("Epoch")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'train_comp_evo.png'))
        plt.close(fig)


    last_trial = list(dataset_map.keys())[-1]
    train_map_idx = np.arange(dataset_map[f"{last_trial}"]["trial_index"])
    plot_trial_idx = np.random.choice(train_map_idx)
    plot_rows_start, plot_rows_end = map(int, dataset_map[f"{plot_trial_idx}"]["dataset_rows"].split(","))
    plot_indices = set(range(plot_rows_start, plot_rows_end))
    plot_set = Subset(dataset, list(plot_indices))
    pad_to_tokens = int(getattr(dataset, "max_tokens"))
    collate_fn = partial(collate_padded_trials, pad_to_tokens=pad_to_tokens)
    plot_loader = DataLoader(
        plot_set,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last,
        collate_fn=collate_fn,
    )

    target_trial = []
    recons_trial = []
    for batch in plot_loader:
        with torch.no_grad():
            if torch.is_tensor(batch):
                x = batch.to(dev)
                target = x
                padding_mask = torch.zeros((x.shape[0], x.shape[1]), dtype=torch.bool, device=dev)
            else:
                x = batch[0].to(dev)
                target = batch[3].to(dev) if len(batch) >= 4 and torch.is_tensor(batch[3]) else x
                padding_mask = batch[2].to(dev).bool()
            out = model(x, padding_mask=padding_mask)
            if not isinstance(out, (tuple, list)):
                raise ValueError("Model forward must return a tuple/list")

            recon = model.predict(x, padding_mask)

        if x.ndim != 3 or x.shape[0] == 0:
            raise ValueError(f"Expected sample_batch shape [B, N, D], got {tuple(x.shape)}")

        valid_len = int((~padding_mask[0]).sum().item())
        target_values = target[0, :valid_len].detach().cpu().numpy()  # [P, T]
        reconstructed = recon[0, :valid_len].detach().cpu().numpy()  # [P, T]
        target_trial.append(target_values)
        recons_trial.append(reconstructed)

    target_trial = np.array(target_trial, dtype=object)
    recons_trial = np.array(recons_trial, dtype=object)

    vol_idx = np.random.randint(0, len(recons_trial))
    for token_idx in range(len(recons_trial[0, 0, :])):
        error = (
            recons_trial[vol_idx, :, token_idx]
            - target_trial[vol_idx, :, token_idx]
        )
        save_path = output_dir / f"vol_n_token_{token_idx}_tr.png"
        plot_vol_error(error, token_idx, save_path)
        if keep_ids:
            keep_mask = np.isin(target_trial[vol_idx, :, 0], list(keep_ids))
            non_keep_idx = np.flatnonzero(~keep_mask)
            error = (
                recons_trial[vol_idx, non_keep_idx, token_idx]
                - target_trial[vol_idx, non_keep_idx, token_idx]
            )
            save_path = output_dir / f"rem_neuron_vol_n_token_{token_idx}_tr.png"
            plot_vol_error(error, token_idx, save_path)

    neuron = np.random.randint(0, len(recons_trial[0, :, 0]))
    if keep_ids:
        keep_mask = np.isin(target_trial[vol_idx, :, 0], list(keep_ids))
        non_keep_idx = np.flatnonzero(~keep_mask)
        if len(non_keep_idx) > 0:
            rem_neuron_idx = np.random.choice(non_keep_idx)
    for token_idx in range(len(recons_trial[0, 0, :])):
        save_path = output_dir / f"neuron_n_token_{token_idx}_tr.png"
        plot_neuron_data_evo(target_trial, recons_trial, neuron, token_idx, save_path)
        if keep_ids:
            if len(non_keep_idx) > 0:
                save_path = output_dir / f"rem_neuron_n_token_{token_idx}_tr.png"
                plot_neuron_data_evo(target_trial, recons_trial, rem_neuron_idx, token_idx, save_path)


def save_validation_error_stats(model: Any, val_loader: Any, output_dir: Path, device: str) -> dict[str, Any]:
    if device.startswith("cuda") and not torch.cuda.is_available():
        dev = torch.device("cpu")
    else:
        dev = torch.device(device)
    model.eval().to(dev)
    has_keep = hasattr(model, "keep")
    keep = getattr(model, "keep", [])
    if torch.is_tensor(keep):
        keep = keep.detach().cpu().tolist()
    keep_ids = set(map(int, keep)) if len(keep) > 0 else set()
    keep_tensor = torch.tensor(list(keep_ids))

    def make_accumulators():
        return {1: [], 2: []}, {1: [], 2: []}, {1: [], 2: []}

    def add_errors(per_trial, all_errors, all_abs_errors, token, errs):
        if len(errs) == 0:
            return
        abs_errs = np.abs(errs)
        per_trial[token].append(
            {
                "pred_align_with_target": float(errs.mean()),
                "pred_std": float(errs.std()),
                "pred_distance_to_target": float(abs_errs.mean()),
                "distance_std": float(abs_errs.std()),
            }
        )
        all_errors[token].append(errs)
        all_abs_errors[token].append(abs_errs)

    def compute_stats(per_trial, all_errors, all_abs_errors):
        stats = {}
        for token in (1, 2):
            trial_means = [errs.mean() for errs in all_errors[token]]
            global_errs = np.concatenate(all_errors[token])
            global_abs_errs = np.concatenate(all_abs_errors[token])
            stats[f"token_{token}"] = {
                "pred_global_align_with_target": float(global_errs.mean()),
                "pred_global_std": float(global_errs.std()),
                "pred_global_distance_to_target": float(global_abs_errs.mean()),
                "global_distance_std": float(global_abs_errs.std()),
                "mean_align_dist": float(np.mean(np.abs(trial_means))),
                "align_dist_std": float(np.std(np.abs(trial_means))),
                "mean_align_error": float(np.mean(trial_means)),
                "align_error_std": float(np.std(trial_means)),
                "per_trial": per_trial[token],
            }
        return stats

    per_trial, all_errors, all_abs_errors = make_accumulators()
    rem_per_trial, rem_all_errors, rem_all_abs_errors = make_accumulators()
    with torch.no_grad():
        for batch in val_loader:
            x = batch[0].to(dev)
            padding_mask = batch[2].to(dev).bool()
            target = batch[3].to(dev) if len(batch) >= 4 and torch.is_tensor(batch[3]) else x
            pred = model.predict(x, padding_mask)
            valid = ~padding_mask
            for token in (1, 2):
                for i in range(x.shape[0]):
                    trial_valid = valid[i]
                    errs = (pred[i, trial_valid, token] - target[i, trial_valid, token]).detach().cpu().numpy()
                    add_errors(per_trial, all_errors, all_abs_errors, token, errs)
                    if has_keep:
                        non_keep = ~torch.isin(target[i, :, 0].detach().cpu(), keep_tensor)
                        rem_valid = trial_valid.detach().cpu() & non_keep
                        rem_errs = (pred[i, rem_valid.to(dev), token] - target[i, rem_valid.to(dev), token]).detach().cpu().numpy()
                        add_errors(rem_per_trial, rem_all_errors, rem_all_abs_errors, token, rem_errs)
    stats = compute_stats(per_trial, all_errors, all_abs_errors)
    with (output_dir / "validation_error_stats.json").open("w", encoding="utf-8") as fp:
        json.dump(stats, fp, indent=2)
    if has_keep:
        rem_stats = compute_stats(rem_per_trial, rem_all_errors, rem_all_abs_errors)
        with (output_dir / "rem_neuron_validation_error_stats.json").open("w", encoding="utf-8") as fp:
            json.dump(rem_stats, fp, indent=2)



def save_high_mae_trial_plots(
    model: nn.Module,
    val_dataset: Dataset,
    train_dataset: Dataset,
    dataset_dir: Path,
    output_dir: Path,
    device: str = "cuda",
    mae_threshold: float = 0.3,
    token_idxs: tuple[int, ...] = (1, 2),
    batch_size: int = 8,
    factor: int = 13,
    weight_time: float = 1.0,
) -> dict[str, Any]:
    """Save plots for validation-trial neurons whose token MAE is high.

    The saved arrays use ``[row, neuron, token]`` indexing. Rows are first
    assigned to the saved train/validation split, then grouped into complete
    trials using ``trial_dataset_map.json``. When ``weight_time`` is zero,
    token 1 is not part of the training objective, so its plot count is
    recorded as ``"NoI"`` (not of interest) without computing its plots.
    """
    out_dir = Path(output_dir) / "high_mae_trial_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    for token_idx in token_idxs:
        if token_idx == 1 and weight_time == 0:
            continue
        (out_dir / f"token_{int(token_idx)}").mkdir(parents=True, exist_ok=True)

    dev = torch.device("cpu" if device.startswith("cuda") and not torch.cuda.is_available() else device)
    model.eval().to(dev)

    dataset_inputs = np.load(Path(dataset_dir) / "data" / "responses.npy", allow_pickle=True).astype(np.float32)
    dataset_targets = np.load(Path(dataset_dir) / "data" / "targets.npy", allow_pickle=True).astype(np.float32)
    if dataset_inputs.ndim != 3 or dataset_targets.shape != dataset_inputs.shape:
        raise ValueError("responses.npy and targets.npy must have shape [row, neuron, token]")

    with (Path(dataset_dir) / "trial_dataset_map.json").open("r", encoding="utf-8") as fp:
        dataset_map = json.load(fp)
    with (Path(output_dir) / "val_indices.json").open("r", encoding="utf-8") as fp:
        val_rows = {int(row) for row in json.load(fp)}

    def trial_sort_key(item: tuple[str, dict[str, Any]]) -> int:
        trial_id, trial_info = item
        return int(trial_info.get("trial_index", trial_id))

    train_trials: list[tuple[str, np.ndarray, np.ndarray]] = []
    val_trials: list[tuple[str, np.ndarray, np.ndarray]] = []
    for trial_id, trial_info in sorted(dataset_map.items(), key=trial_sort_key):
        start, end = (int(value) for value in trial_info["dataset_rows"].split(","))
        row_indices = set(range(start, end))
        if row_indices <= val_rows:
            val_trials.append((trial_id, dataset_inputs[start:end], dataset_targets[start:end]))
        elif row_indices.isdisjoint(val_rows):
            train_trials.append((trial_id, dataset_inputs[start:end], dataset_targets[start:end]))
        else:
            raise ValueError(f"Trial {trial_id} is split between train and validation rows")

    if not train_trials or not val_trials:
        raise ValueError("Expected at least one complete train trial and one complete validation trial")
    train_inputs = np.stack([inputs for _, inputs, _ in train_trials])
    train_targets = np.stack([target for _, _, target in train_trials])
    val_trial_ids = [trial_id for trial_id, _, _ in val_trials]
    val_inputs = np.stack([inputs for _, inputs, _ in val_trials])
    val_targets = np.stack([target for _, _, target in val_trials])
    if (
        train_inputs.shape[1:] != val_inputs.shape[1:]
        or train_targets.shape[1:] != val_targets.shape[1:]
    ):
        raise ValueError("All trials must have the same [row, neuron, token] shape")

    def predict_trial(trial_inputs: np.ndarray) -> np.ndarray:
        predictions: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(trial_inputs), batch_size):
                x = torch.from_numpy(trial_inputs[start : start + batch_size]).to(dev)
                padding_mask = torch.zeros(x.shape[:2], dtype=torch.bool, device=dev)
                predictions.append(model.predict(x, padding_mask).detach().cpu().numpy())
        return np.concatenate(predictions, axis=0)

    plot_counts: dict[str, int | str] = {
        f"token_{int(token_idx)}": "NoI" if token_idx == 1 and weight_time == 0 else 0
        for token_idx in token_idxs
    }
    row_positions = np.arange(train_targets.shape[1])
    for token_idx in token_idxs:
        if not 0 <= token_idx < train_targets.shape[-1]:
            raise ValueError(f"Token index {token_idx} is outside the dataset token dimension")
        if token_idx == 1 and weight_time == 0:
            continue

        fig, ax = plt.subplots(figsize=(24, 24))
        for neuron_idx in range(np.random.randint(0, factor), train_targets.shape[2], factor):
            ax.plot(row_positions, train_targets[:, :, neuron_idx, token_idx].T, color='greenyellow', alpha=0.2, linewidth=0.5, label='_nolegend_')

        for trial_id, trial_inputs, trial_targets in zip(val_trial_ids, val_inputs, val_targets):
            trial_predictions = predict_trial(trial_inputs)
            mae_neurons, mae_neurons_mae = [], []
            for neuron_idx in range(train_targets.shape[2]):
                mae = float(np.mean(np.abs(
                    trial_predictions[:, neuron_idx, token_idx] - trial_targets[:, neuron_idx, token_idx]
                )))
                if mae <= mae_threshold:
                    continue
                plot_counts[f"token_{int(token_idx)}"] += 1
                mae_neurons.append(neuron_idx)
                mae_neurons_mae.append(mae)

            if len(mae_neurons) > 4:
                idx = np.random.choice(len(mae_neurons), size=4, replace=False)
                mae_neurons = [mae_neurons[i] for i in idx]
                mae_neurons_mae = [mae_neurons_mae[i] for i in idx]
            for neuron_idx, mae in zip(mae_neurons, mae_neurons_mae):
                train_lines = ax.plot(
                    row_positions, train_targets[:, :, neuron_idx, token_idx].T,
                    color='darkgrey', alpha=0.5, linewidth=0.7, label="Train target",
                )
                target_line, = ax.plot(
                    row_positions, trial_targets[:, neuron_idx, token_idx],
                    color='black', alpha=1.0, linewidth=1.0, label="Validation target",
                )
                prediction_line, = ax.plot(
                    row_positions, trial_predictions[:, neuron_idx, token_idx],
                    color='red', alpha=1.0, linewidth=1.0, label="Validation prediction",
                )
                ax.set_title(
                    f"Token {token_idx}, neuron {neuron_idx}, validation trial {trial_id} (MAE={mae:.4f})"
                )
                ax.set_xlabel("Row within trial")
                ax.set_ylabel("Token value")
                legend = ax.legend()
                fig.savefig(
                    out_dir / f"token_{token_idx}" / f"trial_{trial_id}_neuron_{neuron_idx}_mae_{mae:.3f}.png",
                    bbox_inches="tight",
                )
                legend.remove()
                target_line.remove()
                prediction_line.remove()
                for line in train_lines:
                    line.remove()
        plt.close(fig)

    with (out_dir / "plot_counts.txt").open("w", encoding="utf-8") as fp:
        for token_name, count in plot_counts.items():
            fp.write(f"{token_name}: {count}\n")
    return {
        "outliers": plot_counts,
        "plot_counts": plot_counts,
        "output_dir": str(out_dir),
    }
