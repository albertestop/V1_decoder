from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from v1tovideo.config_utils import load_toml, resolve_maybe_repo_path, resolve_repo_path
from v1tovideo.neural_autoencoder.data import NeuralDataConfig
from v1tovideo.neural_autoencoder.trainer import TrainConfig


@dataclass
class ExperimentConfig:
    data: NeuralDataConfig
    model: dict[str, Any]
    expected_trace_shape: tuple[int, int] | None
    latent_dim: int | None
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

    data_source = str(data_cfg.get("source", "array")).strip().lower()
    sessions_raw = data_cfg.get("sessions")
    sessions: list[str] | None
    if sessions_raw is None:
        sessions = None
    else:
        if not isinstance(sessions_raw, list):
            raise ValueError("data.sessions must be a list of session names")
        sessions = [str(session) for session in sessions_raw]

    data_config = NeuralDataConfig(
        source=data_source,
        path=resolve_repo_path(data_cfg["path"]) if "path" in data_cfg else None,
        npz_key=str(data_cfg["npz_key"]) if "npz_key" in data_cfg else None,
        proc_session_root=resolve_maybe_repo_path(data_cfg["proc_session_root"])
        if "proc_session_root" in data_cfg
        else None,
        sessions=sessions,
        responses_subdir=str(data_cfg.get("responses_subdir", "data/responses")),
        file_pattern=str(data_cfg.get("file_pattern", "*.npy")),
        max_files_per_session=int(data_cfg["max_files_per_session"])
        if "max_files_per_session" in data_cfg
        else None,
        transpose_proc_session=bool(data_cfg.get("transpose_proc_session", True)),
        channel_mode=str(data_cfg.get("channel_mode", "error")),
        time_mode=str(data_cfg.get("time_mode", "error")),
        batch_size=int(data_cfg.get("batch_size", 32)),
        val_split=float(data_cfg.get("val_split", 0.1)),
        shuffle_train=bool(data_cfg.get("shuffle_train", True)),
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        drop_last=bool(data_cfg.get("drop_last", False)),
    )

    architecture = str(data.get("architecture", "transformer")).strip().lower()
    supported_architectures = {"mlp", "transformer", "custom"}
    if architecture not in supported_architectures:
        raise ValueError(
            f"Unsupported architecture '{architecture}'. Supported: {sorted(supported_architectures)}"
        )

    model_config: dict[str, Any]
    expected_trace_shape: tuple[int, int] | None = None
    latent_dim: int | None = None

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
        if "sequence_length" in model_kwargs and "num_channels" in model_kwargs:
            expected_trace_shape = (
                int(model_kwargs["sequence_length"]),
                int(model_kwargs["num_channels"]),
            )
        if "latent_dim" in model_kwargs:
            latent_dim = int(model_kwargs["latent_dim"])
    else:
        if custom_model_cfg:
            raise ValueError(
                "[custom_model] is only allowed when architecture = 'custom'. "
                f"Current architecture is '{architecture}'."
            )
        model_config = {
            "architecture": architecture,
            "sequence_length": int(built_in_model_cfg["sequence_length"]),
            "num_channels": int(built_in_model_cfg["num_channels"]),
            "latent_dim": int(built_in_model_cfg.get("latent_dim", 128)),
            "hidden_dim": int(built_in_model_cfg.get("hidden_dim", 256)),
            "num_layers": int(built_in_model_cfg.get("num_layers", 4)),
            "num_heads": int(built_in_model_cfg.get("num_heads", 8)),
            "dropout": float(built_in_model_cfg.get("dropout", 0.1)),
        }
        expected_trace_shape = (int(model_config["sequence_length"]), int(model_config["num_channels"]))
        latent_dim = int(model_config["latent_dim"])

    train_config = TrainConfig(
        epochs=int(train_cfg.get("epochs", 25)),
        learning_rate=float(train_cfg.get("learning_rate", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        grad_clip_norm=float(train_cfg.get("grad_clip_norm", 1.0)),
        device=str(train_cfg.get("device", "cuda")),
    )

    output_dir = resolve_repo_path(output_cfg.get("dir", "outputs/neural_autoencoder"))

    return ExperimentConfig(
        data=data_config,
        model=model_config,
        expected_trace_shape=expected_trace_shape,
        latent_dim=latent_dim,
        train=train_config,
        output_dir=output_dir,
    )
