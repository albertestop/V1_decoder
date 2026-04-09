from __future__ import annotations

import logging
import time
import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
import pytorch_lightning as pl
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset

from v1tovideo.neural_autoencoder.data import collate_padded_trials

LOGGER = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    epochs: int = 25
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float | None = 1.0
    device: str = "cuda:1"
    loss_name: str = "masked_mse"
    poisson_log_input: bool = True
    poisson_full: bool = False
    poisson_eps: float = 1e-8


def _resolve_device(device: str) -> torch.device:
    if device.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def _lightning_trainer_kwargs(device: str) -> dict[str, Any]:
    resolved = _resolve_device(device)
    if resolved.type == "cuda":
        return {"accelerator": "gpu", "devices": 1}
    if resolved.type == "mps":
        return {"accelerator": "mps", "devices": 1}
    return {"accelerator": "cpu", "devices": 1}


class AutoencoderLightningModule(pl.LightningModule):
    def __init__(self, model: nn.Module, config: TrainConfig) -> None:
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=["model"])
        self.config = config
        self._loss_name = str(config.loss_name).strip().lower()
        supported_losses = {"masked_mse", "poisson_nll"}
        if self._loss_name not in supported_losses:
            raise ValueError(f"Unsupported loss_name '{config.loss_name}'. Supported: {sorted(supported_losses)}")
        self._poisson_nll = nn.PoissonNLLLoss(
            log_input=bool(config.poisson_log_input),
            full=bool(config.poisson_full),
            eps=float(config.poisson_eps),
            reduction="none",
        )

    def _unpack_batch(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.is_tensor(batch):
            x = batch
            padding_mask = torch.zeros((x.shape[0], x.shape[1]), dtype=torch.bool, device=x.device)
            return x, padding_mask
        if isinstance(batch, (tuple, list)) and len(batch) >= 3 and torch.is_tensor(batch[0]) and torch.is_tensor(batch[2]):
            x = batch[0]
            padding_mask = batch[2].bool()
            return x, padding_mask
        raise ValueError(f"Unsupported batch format: {type(batch)!r}")

    def _masked_mse(self, recon: torch.Tensor, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        valid = (~padding_mask).unsqueeze(-1).to(dtype=x.dtype)
        denom = valid.sum().clamp_min(1.0)
        return (((recon - x) ** 2) * valid).sum() / denom

    def _masked_mae(self, recon: torch.Tensor, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        valid = (~padding_mask).unsqueeze(-1).to(dtype=x.dtype)
        denom = valid.sum().clamp_min(1.0)
        return (torch.abs(recon - x) * valid).sum() / denom

    def _masked_poisson_nll(self, recon: torch.Tensor, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        valid = (~padding_mask).unsqueeze(-1).to(dtype=x.dtype)
        denom = valid.sum().clamp_min(1.0)
        per_element = self._poisson_nll(recon, x)
        return (per_element * valid).sum() / denom

    def _compute_loss(self, recon: torch.Tensor, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        if self._loss_name == "masked_mse":
            return self._masked_mse(recon, x, padding_mask)
        return self._masked_poisson_nll(recon, x, padding_mask)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, padding_mask = self._unpack_batch(batch)
        recon, _ = self.model(x, padding_mask=padding_mask)
        loss = self._compute_loss(recon, x, padding_mask)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=x.shape[0])
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, padding_mask = self._unpack_batch(batch)
        recon, _ = self.model(x, padding_mask=padding_mask)
        loss = self._compute_loss(recon, x, padding_mask)
        mse = self._masked_mse(recon, x, padding_mask)
        mae = self._masked_mae(recon, x, padding_mask)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=x.shape[0])
        self.log("val_mse", mse, on_step=False, on_epoch=True, batch_size=x.shape[0])
        self.log("val_mae", mae, on_step=False, on_epoch=True, batch_size=x.shape[0])
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )


class TrainHistoryCallback(pl.Callback):  # type: ignore[misc]
    def __init__(self) -> None:
        self.history: list[dict[str, float]] = []
        self._epoch_start_time = 0.0

    def on_train_epoch_start(self, trainer: Any, pl_module: Any) -> None:
        self._epoch_start_time = time.perf_counter()

    def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        metrics = trainer.callback_metrics
        row = {
            "epoch": float(trainer.current_epoch + 1),
            "train_loss": float(metrics["train_loss"].detach().cpu()) if "train_loss" in metrics else float("nan"),
            "val_loss": float(metrics["val_loss"].detach().cpu()) if "val_loss" in metrics else float("nan"),
            "epoch_time_sec": float(time.perf_counter() - self._epoch_start_time),
        }
        self.history.append(row)
        LOGGER.info(
            "Epoch %d/%d | train_loss=%.6f | val_loss=%.6f | epoch_time=%.2fs",
            trainer.current_epoch + 1,
            trainer.max_epochs,
            row["train_loss"],
            row["val_loss"],
            row["epoch_time_sec"],
        )


def train_autoencoder(
    model: nn.Module,
    train_loader: DataLoader[Any],
    val_loader: DataLoader[Any],
    config: TrainConfig,
) -> list[dict[str, float]]:
    """Train neural autoencoder with PyTorch Lightning and return epoch history."""
    lightning_model = AutoencoderLightningModule(model=model, config=config)
    history_callback = TrainHistoryCallback()
    train_start = time.perf_counter()
    trainer = pl.Trainer(
        max_epochs=config.epochs,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        gradient_clip_val=float(config.grad_clip_norm) if config.grad_clip_norm is not None else 0.0,
        enable_progress_bar=False,
        callbacks=[history_callback],
        **_lightning_trainer_kwargs(config.device),
    )
    trainer.fit(lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    LOGGER.info("Training completed | total_time=%.2fs", time.perf_counter() - train_start)
    return history_callback.history


def evaluate_autoencoder(
    model: nn.Module,
    dataloader: DataLoader[Any],
    device: str = "cuda",
) -> dict[str, float]:
    """Compute reconstruction metrics with PyTorch Lightning validation."""
    eval_config = TrainConfig(device=device)
    lightning_model = AutoencoderLightningModule(model=model, config=eval_config)
    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        enable_progress_bar=False,
        **_lightning_trainer_kwargs(device),
    )
    metrics = trainer.validate(lightning_model, dataloaders=dataloader, verbose=False)
    if not metrics:
        return {"mse": float("nan"), "mae": float("nan")}
    return {
        "mse": float(metrics[0].get("val_mse", float("nan"))),
        "mae": float(metrics[0].get("val_mae", float("nan"))),
    }


def save_checkpoint(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def save_reconstruction_artifacts(
    model: nn.Module,
    sample_batch: Any,
    output_dir: Path,
    device: str = "cuda",
    prefix: str = "sample",
) -> None:
    """Save input tensors, latent vectors, and reconstructions for inspection."""
    output_dir.mkdir(parents=True, exist_ok=True)

    dev = _resolve_device(device)
    model.eval().to(dev)

    with torch.no_grad():
        if torch.is_tensor(sample_batch):
            x = sample_batch.to(dev)
            padding_mask = torch.zeros((x.shape[0], x.shape[1]), dtype=torch.bool, device=dev)
        else:
            x = sample_batch[0].to(dev)
            padding_mask = sample_batch[2].to(dev).bool()
        recon, latents = model(x, padding_mask=padding_mask)

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
    dev = _resolve_device(device)
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
            recon, _ = model(x, padding_mask=padding_mask)

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
