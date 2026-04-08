from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from v1tovideo.config_utils import load_toml, resolve_repo_path
from v1tovideo.neural_autoencoder.data import NeuralDataConfig
from v1tovideo.neural_autoencoder.trainer import TrainConfig


@dataclass
class ExperimentConfig:
    data: NeuralDataConfig
    model: dict[str, Any]
    train: TrainConfig
    output_dir: Path


def parse_neural_ae_experiment_config(config_path: Path) -> ExperimentConfig:
    data = load_toml(config_path)

    data_cfg = data.get("data")
    custom_model_cfg = data.get("custom_model", {})
    built_in_model_cfg = data.get("built_in_model", {})
    train_cfg = data.get("train", {})
    output_cfg = data.get("output", {})

    if not isinstance(data_cfg, dict):
        raise ValueError("Config must define [data]")
    if not isinstance(custom_model_cfg, dict):
        raise ValueError("Config [custom_model] must be a table")
    if not isinstance(built_in_model_cfg, dict):
        raise ValueError("Config [built_in_model] must be a table")
    if not isinstance(train_cfg, dict):
        raise ValueError("Config [train] must be a table")
    if not isinstance(output_cfg, dict):
        raise ValueError("Config [output] must be a table")

    data_path = Path(str(data_cfg["data_root_path"])).expanduser() / Path(str(data_cfg["dataset_id"])).expanduser()

    data_config = NeuralDataConfig(
        path=data_path,
        npz_key=str(data_cfg["npz_key"]) if "npz_key" in data_cfg else None,
        batch_size=int(data_cfg.get("batch_size", 32)),
        val_split=float(data_cfg.get("val_split", 0.1)),
        shuffle_train=bool(data_cfg.get("shuffle_train", True)),
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        drop_last=bool(data_cfg.get("drop_last", False)),
    )

    architecture = str(data_cfg.get("architecture")).strip().lower()
    supported_architectures = {"mlp", "transformer", "custom"}
    if architecture not in supported_architectures:
        raise ValueError(
            f"Unsupported architecture '{architecture}'. Supported: {sorted(supported_architectures)}"
        )

    model_config: dict[str, Any]

    if architecture == "custom":
        model_target_raw = custom_model_cfg.get("target")
        model_target = str(model_target_raw).strip() if model_target_raw is not None else None
        model_kwargs = {k: v for k, v in custom_model_cfg.items() if k != "target"}

        if not model_target:
            raise ValueError("custom_model.target is required when architecture = 'custom'")
        model_config = {
            "architecture": "custom",
            "target": model_target,
            "kwargs": model_kwargs,
        }
    else:
        model_config = {
            "architecture": architecture,
            "latent_dim": int(built_in_model_cfg.get("latent_dim", 128)),
            "hidden_dim": int(built_in_model_cfg.get("hidden_dim", 256)),
            "num_layers": int(built_in_model_cfg.get("num_layers", 4)),
            "num_heads": int(built_in_model_cfg.get("num_heads", 8)),
            "dropout": float(built_in_model_cfg.get("dropout", 0.1)),
        }

    loss_cfg_raw = train_cfg.get("loss", {})
    if isinstance(loss_cfg_raw, str):
        loss_name = loss_cfg_raw
        loss_cfg: dict[str, Any] = {}
    elif isinstance(loss_cfg_raw, dict):
        loss_cfg = loss_cfg_raw
        loss_name = str(loss_cfg.get("name", "masked_mse"))
    else:
        raise ValueError("Config [train].loss must be a string or table")

    train_config = TrainConfig(
        epochs=int(train_cfg.get("epochs", 25)),
        learning_rate=float(train_cfg.get("learning_rate", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        grad_clip_norm=float(train_cfg.get("grad_clip_norm", 1.0)),
        device=str(train_cfg.get("device", "cuda")),
        loss_name=str(loss_name),
        poisson_log_input=bool(loss_cfg.get("poisson_log_input", train_cfg.get("poisson_log_input", True))),
        poisson_full=bool(loss_cfg.get("poisson_full", train_cfg.get("poisson_full", False))),
        poisson_eps=float(loss_cfg.get("poisson_eps", train_cfg.get("poisson_eps", 1e-8))),
    )

    output_dir = resolve_repo_path(output_cfg.get("dir", "outputs/neural_autoencoder"))

    return ExperimentConfig(
        data=data_config,
        model=model_config,
        train=train_config,
        output_dir=output_dir,
    )
