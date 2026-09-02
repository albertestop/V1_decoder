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
        plt.figure(figsize=(16, 6))
        plt.scatter(np.arange(len(target_trial[vol_idx, :, 0])), target_trial[vol_idx, :, token_idx], label="Target", s=10)
        plt.scatter(np.arange(len(recons_trial[vol_idx, :, 0])), recons_trial[vol_idx, :, token_idx], label="Reconstructed", s=10)
        plt.xlabel("Neuron")
        plt.ylabel("Token Value")
        plt.title(f"Token {token_idx} value of each neuron on a single cycle: target vs reconstruction")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"vol_n_token_{token_idx}_val.png")
        plt.close()

    neuron = np.random.randint(0, len(recons_trial[0, :, 0]))
    for token_idx in range(len(recons_trial[0, 0, :])):
        plt.figure(figsize=(8, 3))
        plt.plot(np.arange(len(target_trial[:, int(neuron), 0])), target_trial[:, int(neuron), token_idx], label="Target")
        plt.plot(np.arange(len(recons_trial[:, int(neuron), 0])), recons_trial[:, int(neuron), token_idx], label="Reconstructed")
        plt.xlabel("Cycle n")
        plt.ylabel("Token Value")
        plt.title(f"Token {token_idx} of neuron {int(target_trial[0, int(neuron), 0])} during trial: target vs reconstruction")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"neuron_n_token_{token_idx}_val.png")
        plt.close()

    with open(os.path.join(output_dir, 'history.json'), 'r') as file:
        history = json.load(file)
    epochs = [d["epoch"] for d in history]
    train_loss = [d["train_loss"] for d in history]
    val_loss = [d["val_loss"] for d in history]

    plt.figure()
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
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
            ax.plot(epochs, train_comp_loss, label=f"Train {title}")
            ax.plot(epochs, val_comp_loss, label=f"Validation {title}")
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
        plt.figure(figsize=(16, 6))
        plt.scatter(np.arange(len(target_trial[vol_idx, :, 0])), target_trial[vol_idx, :, token_idx], label="Target", s=10)
        plt.scatter(np.arange(len(recons_trial[vol_idx, :, 0])), recons_trial[vol_idx, :, token_idx], label="Reconstructed", s=10)
        plt.xlabel("Neuron")
        plt.ylabel("Token Value")
        plt.title(f"Token {token_idx} value of each neuron on a single cycle: target vs reconstruction")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"vol_n_token_{token_idx}_tr.png")
        plt.close()

    neuron = np.random.randint(0, len(recons_trial[0, :, 0]))
    for token_idx in range(len(recons_trial[0, 0, :])):
        plt.figure(figsize=(8, 3))
        plt.plot(np.arange(len(target_trial[:, int(neuron), 0])), target_trial[:, int(neuron), token_idx], label="Target")
        plt.plot(np.arange(len(recons_trial[:, int(neuron), 0])), recons_trial[:, int(neuron), token_idx], label="Reconstructed")
        plt.xlabel("Cycle n")
        plt.ylabel("Token Value")
        plt.title(f"Token {token_idx} of neuron {int(target_trial[0, int(neuron), 0])} during trial: target vs reconstruction")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"neuron_n_token_{token_idx}_tr.png")
        plt.close()


def save_validation_error_stats(model: Any, val_loader: Any, output_dir: Path, device: str) -> dict[str, Any]:
    if device.startswith("cuda") and not torch.cuda.is_available():
        dev = torch.device("cpu")
    else:
        dev = torch.device(device)
    model.eval().to(dev)
    per_trial = {1: [], 2: []}
    all_errors = {1: [], 2: []}
    all_abs_errors = {1: [], 2: []}
    with torch.no_grad():
        for batch in val_loader:
            x = batch[0].to(dev)
            padding_mask = batch[2].to(dev).bool()
            target = batch[3].to(dev) if len(batch) >= 4 and torch.is_tensor(batch[3]) else x
            pred = model.predict(x, padding_mask)
            valid = ~padding_mask
            for token in (1, 2):
                for i in range(x.shape[0]):
                    errs = (pred[i, valid[i], token] - target[i, valid[i], token]).detach().cpu().numpy()
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
    with (output_dir / "validation_error_stats.json").open("w", encoding="utf-8") as fp:
        json.dump(stats, fp, indent=2)