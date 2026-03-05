from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SyntheticFactorDatasetConfig:
    """Configuration for synthetic neural traces generated from latent factors."""

    n_samples: int = 336
    sequence_length: int = 300
    n_neurons: int = 5443
    n_factors: int = 16
    factor_scale: float = 1.0
    noise_std: float = 0.05
    baseline_std: float = 0.0
    seed: int = 0


def generate_factor_dataset(config: SyntheticFactorDatasetConfig) -> np.ndarray:
    """Generate [N, T, C] synthetic traces from low-rank K-factor dynamics."""
    if config.n_samples <= 0 or config.sequence_length <= 0 or config.n_neurons <= 0:
        raise ValueError("n_samples, sequence_length, and n_neurons must be positive")
    if config.n_factors <= 0:
        raise ValueError("n_factors must be positive")
    if config.noise_std < 0 or config.baseline_std < 0:
        raise ValueError("noise_std and baseline_std must be non-negative")

    rng = np.random.default_rng(config.seed)

    # Neuron-specific factor loadings: [C, K]
    loadings = rng.normal(0.0, 1.0, size=(config.n_neurons, config.n_factors)).astype(np.float32)

    # Per-sample latent factor traces: [N, T, K]
    factors = rng.normal(
        0.0,
        config.factor_scale,
        size=(config.n_samples, config.sequence_length, config.n_factors),
    ).astype(np.float32)

    # Low-rank neural activity: [N, T, C]
    traces = np.einsum("ntk,ck->ntc", factors, loadings, optimize=True).astype(np.float32)

    if config.baseline_std > 0:
        baseline = rng.normal(0.0, config.baseline_std, size=(config.n_samples, 1, config.n_neurons))
        traces += baseline.astype(np.float32)

    if config.noise_std > 0:
        noise = rng.normal(0.0, config.noise_std, size=traces.shape).astype(np.float32)
        traces += noise

    return traces


def save_factor_dataset(config: SyntheticFactorDatasetConfig, output_path: Path) -> Path:
    """Generate and save dataset as .npy, returning output path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    traces = generate_factor_dataset(config)
    np.save(output_path, traces)
    return output_path
