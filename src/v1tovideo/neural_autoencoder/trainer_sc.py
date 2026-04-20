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
            id_logits, time_pred, rec_pred, latents = out
            id_pred = id_logits.argmax(dim=-1).to(dtype=x.dtype)
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

    original_trial = []
    recons_trial = []
    for batch in plot_loader:
        with torch.no_grad():
            if torch.is_tensor(batch):
                x = batch.to(dev)
                padding_mask = torch.zeros((x.shape[0], x.shape[1]), dtype=torch.bool, device=dev)
            else:
                x = batch[0].to(dev)
                padding_mask = batch[2].to(dev).bool()
            out = model(x, padding_mask=padding_mask)
            if not isinstance(out, (tuple, list)):
                raise ValueError("Model forward must return a tuple/list")

            recon = model.predict(x, padding_mask)

        if x.ndim != 3 or x.shape[0] == 0:
            raise ValueError(f"Expected sample_batch shape [B, N, D], got {tuple(x.shape)}")

        valid_len = int((~padding_mask[0]).sum().item())
        original = x[0, :valid_len].detach().cpu().numpy()  # [P, T]
        reconstructed = recon[0, :valid_len].detach().cpu().numpy()  # [P, T]
        original_trial.append(original)
        recons_trial.append(reconstructed)

    original_trial = np.array(original_trial, dtype=object)
    recons_trial = np.array(recons_trial, dtype=object)

    vol_idx = np.random.randint(0, len(recons_trial))
    for token_idx in range(len(recons_trial[0, 0, :])):
        plt.figure(figsize=(16, 6))
        plt.scatter(np.arange(len(original_trial[vol_idx, :, 0])), original_trial[vol_idx, :, token_idx], label="Original", s=10)
        plt.scatter(np.arange(len(recons_trial[vol_idx, :, 0])), recons_trial[vol_idx, :, token_idx], label="Recontructed", s=10)
        plt.xlabel("Neuron")
        plt.ylabel("Token Value")
        plt.title(f"Token {token_idx} value of each neuron on a single cycle: before vs after reconstruction")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"vol_n_token_{token_idx}.png")
        plt.close()

    neuron = np.random.randint(0, len(recons_trial[0, :, 0]))
    for token_idx in range(len(recons_trial[0, 0, :])):
        plt.figure(figsize=(8, 3))
        plt.plot(np.arange(len(original_trial[:, int(neuron), 0])), original_trial[:, int(neuron), token_idx], label="Original")
        plt.plot(np.arange(len(recons_trial[:, int(neuron), 0])), recons_trial[:, int(neuron), token_idx], label="Recontructed")
        plt.xlabel("Cycle n")
        plt.ylabel("Token Value")
        plt.title(f"Token {token_idx} of neuron {int(original_trial[0, int(neuron), 0])} during trial: before vs after reconstruction")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"neuron_n_token_{token_idx}.png")
        plt.close()

    with open(os.path.join(output_dir, 'history.json'), 'r') as file:
        history = json.load(file)
    epochs = [d["epoch"] for d in history]
    train_loss = [d["train_loss"] for d in history]
    val_loss = [d["val_loss"] for d in history]

    train_loss[0] = None

    plt.figure()
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, 'train_evo.png'))
