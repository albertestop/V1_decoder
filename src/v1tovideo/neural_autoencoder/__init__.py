"""Neural activity autoencoder components."""

from .data import NeuralDataConfig, NeuralTraceDataset, build_dataloaders
from .models import BaseNeuralAutoencoder, ModelSpec, build_model, build_model_from_target
from .synthetic import (
    SyntheticFactorDatasetConfig,
    generate_factor_dataset,
    save_factor_dataset,
)
from .trainer import (
    EvalSummary,
    TrainConfig,
    evaluate_autoencoder,
    save_checkpoint,
    save_reconstruction_plots,
    save_reconstruction_artifacts,
    train_autoencoder,
)

__all__ = [
    "BaseNeuralAutoencoder",
    "EvalSummary",
    "ModelSpec",
    "NeuralDataConfig",
    "NeuralTraceDataset",
    "SyntheticFactorDatasetConfig",
    "TrainConfig",
    "build_dataloaders",
    "build_model",
    "build_model_from_target",
    "evaluate_autoencoder",
    "generate_factor_dataset",
    "save_factor_dataset",
    "save_checkpoint",
    "save_reconstruction_plots",
    "save_reconstruction_artifacts",
    "train_autoencoder",
]
