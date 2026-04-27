from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import pytorch_lightning as pl
import torch.nn.functional as F
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
    device: str = "cuda:1"
    loss_name: str = "masked_mse"
    poisson_log_input: bool = True
    poisson_full: bool = False
    poisson_eps: float = 1e-8
    loss_weight_id: float = 1.0
    loss_weight_time: float = 1.0
    loss_weight_rec: float = 1.0


def _resolve_device(device: str) -> torch.device:
    if device.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def _lightning_trainer_kwargs(device: str) -> dict[str, Any]:
    resolved = _resolve_device(device)
    if resolved.type == "cuda":
        # Respect an explicit CUDA index from config (e.g. "cuda:1").
        if resolved.index is not None:
            return {"accelerator": "gpu", "devices": [resolved.index]}
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
        supported_losses = {"masked_mse", "poisson_nll", "combined"}
        if self._loss_name not in supported_losses:
            raise ValueError(f"Unsupported loss_name '{config.loss_name}'. Supported: {sorted(supported_losses)}")
        self._loss_weights = (
            float(config.loss_weight_id),
            float(config.loss_weight_time),
            float(config.loss_weight_rec),
        )
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

    def _masked_combined_loss(
        self,
        id_logits: torch.Tensor,
        time_pred: torch.Tensor,
        rec_pred: torch.Tensor,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        valid = ~padding_mask
        id_target = x[..., 0].long()
        time_target = x[..., 1]
        rec_target = x[..., 2]

        ignore_index = -100
        id_target_masked = id_target.masked_fill(~valid, ignore_index)
        loss_id = F.cross_entropy(
            id_logits.reshape(-1, id_logits.shape[-1]),
            id_target_masked.reshape(-1),
            ignore_index=ignore_index,
        )

        valid_float = valid.to(dtype=x.dtype)
        denom = valid_float.sum().clamp_min(1.0)
        loss_time = (((time_pred.squeeze(-1) - time_target) ** 2) * valid_float).sum() / denom
        loss_rec = (((rec_pred.squeeze(-1) - rec_target) ** 2) * valid_float).sum() / denom

        w_id, w_time, w_rec = self._loss_weights
        total = (w_id * loss_id) + (w_time * loss_time) + (w_rec * loss_rec)
        return total, {"loss_id": loss_id, "loss_time": loss_time, "loss_rec": loss_rec}

    def _forward_outputs(self, x: torch.Tensor, padding_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.model(x, padding_mask=padding_mask)
        if not isinstance(out, (tuple, list)):
            raise ValueError("Model forward must return a tuple/list")
        if len(out) == 2:
            recon, latents = out
            return {"recon": recon, "latents": latents}
        if len(out) == 4:
            id_logits, time_pred, rec_pred, latents = out
            recon = self.model.predict(x, padding_mask)
            return {
                "id_logits": id_logits,
                "time_pred": time_pred,
                "rec_pred": rec_pred,
                "recon": recon,
                "latents": latents,
            }
        raise ValueError(f"Unsupported model output tuple length: {len(out)}")

    def _compute_loss(self, outputs: torch.Tensor, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        if self._loss_name == "masked_mse":
            return self._masked_mse(outputs["recon"], x, padding_mask)
        if self._loss_name == "poisson_nll":
            return self._masked_poisson_nll(outputs["recon"], x, padding_mask)
        if self._loss_name == "combined":
            if not all(k in outputs for k in ("id_logits", "time_pred", "rec_pred")):
                raise ValueError("combined loss requires model outputs: id_logits, time_pred, rec_pred")
            loss, terms = self._masked_combined_loss(outputs["id_logits"], outputs["time_pred"], outputs["rec_pred"], x, padding_mask)
            self.log("loss_id", terms["loss_id"], on_step=False, on_epoch=True, batch_size=x.shape[0])
            self.log("loss_time", terms["loss_time"], on_step=False, on_epoch=True, batch_size=x.shape[0])
            self.log("loss_rec", terms["loss_rec"], on_step=False, on_epoch=True, batch_size=x.shape[0])
            return loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, padding_mask = self._unpack_batch(batch)
        outputs = self._forward_outputs(x, padding_mask)
        loss = self._compute_loss(outputs, x, padding_mask)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=x.shape[0])
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, padding_mask = self._unpack_batch(batch)
        outputs = self._forward_outputs(x, padding_mask)
        loss = self._compute_loss(outputs, x, padding_mask)
        mse = self._masked_mse(outputs["recon"], x, padding_mask)
        mae = self._masked_mae(outputs["recon"], x, padding_mask)
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
