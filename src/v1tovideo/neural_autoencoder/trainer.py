from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import pytorch_lightning as pl
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

LOGGER = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    epochs: int = 25
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float | None = 1.0
    device: str = "cuda"


def _resolve_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
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
        self.criterion = nn.MSELoss()

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x = batch
        recon, _ = self.model(x)
        loss = self.criterion(recon, x)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=x.shape[0])
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x = batch
        recon, _ = self.model(x)
        loss = self.criterion(recon, x)
        mse = torch.mean((recon - x) ** 2)
        mae = torch.mean(torch.abs(recon - x))
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
    train_loader: DataLoader[torch.Tensor],
    val_loader: DataLoader[torch.Tensor],
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
    dataloader: DataLoader[torch.Tensor],
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
    sample_batch: torch.Tensor,
    output_dir: Path,
    device: str = "cuda",
    prefix: str = "sample",
) -> None:
    """Save input tensors, latent vectors, and reconstructions for inspection."""
    output_dir.mkdir(parents=True, exist_ok=True)

    dev = _resolve_device(device)
    model.eval().to(dev)

    with torch.no_grad():
        x = sample_batch.to(dev)
        recon, latents = model(x)

    torch.save(x.cpu(), output_dir / f"{prefix}.input.pt")
    torch.save(latents.cpu(), output_dir / f"{prefix}.latents.pt")
    torch.save(recon.cpu(), output_dir / f"{prefix}.reconstruction.pt")


def save_reconstruction_plots(
    model: nn.Module,
    sample_batch: torch.Tensor,
    output_dir: Path,
    device: str = "cuda",
    prefix: str = "sample",
    num_neurons: int = 3,
) -> None:
    """Save before/after plots for selected tokens and parameter heatmaps."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dev = _resolve_device(device)
    model.eval().to(dev)

    with torch.no_grad():
        x = sample_batch.to(dev)
        recon, _ = model(x)

    if x.ndim != 3 or x.shape[0] == 0:
        raise ValueError(f"Expected sample_batch shape [B, P, T], got {tuple(x.shape)}")

    original = x[0].detach().cpu().numpy()  # [P, T]
    reconstructed = recon[0].detach().cpu().numpy()  # [P, T]
    param_axis = np.arange(original.shape[0], dtype=np.float32)

    num_tokens = int(original.shape[1])
    k = max(1, min(int(num_neurons), num_tokens))
    token_indices = np.linspace(0, num_tokens - 1, num=k, dtype=int).tolist()

    for token_idx in token_indices:
        plt.figure(figsize=(8, 3))
        plt.plot(param_axis, original[:, token_idx], label="Before")
        plt.plot(param_axis, reconstructed[:, token_idx], label="After")
        plt.xlabel("Parameter")
        plt.ylabel("Value")
        plt.title(f"Token {token_idx}: before vs after reconstruction")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{prefix}.token_{token_idx}.png")
        plt.close()

    stacked = np.concatenate([original, reconstructed], axis=0)
    vmin = 0
    vmax = float(original[original != 0].mean())

    plt.figure(figsize=(10, 5))
    plt.imshow(original, aspect="auto", vmin=vmin, vmax=vmax)
    plt.colorbar(label="Response Intensity")
    plt.xlabel("Token")
    plt.ylabel("Parameter")
    plt.title("Input tensor before reconstruction")
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}.population_before.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.imshow(reconstructed, aspect="auto", vmin=vmin, vmax=vmax)
    plt.colorbar(label="Response Intensity")
    plt.xlabel("Token")
    plt.ylabel("Parameter")
    plt.title("Input tensor after reconstruction")
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}.population_after.png")
    plt.close()
