from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split


@dataclass
class NeuralDataConfig:
    """Configuration for loading tokenized neural tensors."""

    path: Path | None = None
    npz_key: str | None = None

    batch_size: int = 32
    val_split: float = 0.1
    shuffle_train: bool = True
    num_workers: int = 0
    pin_memory: bool = True
    drop_last: bool = False


def infer_batch_shape(batch: Any) -> tuple[int, int]:
    """Extract `[num_tokens, token_dim]` from a batched tensor `[B, N, D]`."""
    if not torch.is_tensor(batch):
        raise ValueError(f"Expected tensor batch, got {type(batch)!r}")
    if batch.ndim != 3:
        raise ValueError(f"Expected training batch shape [B, N, D], got {tuple(batch.shape)}")
    return int(batch.shape[1]), int(batch.shape[2])


class NeuralTraceDataset(Dataset[torch.Tensor]):
    """Dataset wrapper for in-memory tensors with shape [num_trials, num_tokens, token_dim]."""

    def __init__(self, traces: torch.Tensor) -> None:
        if traces.ndim != 3:
            raise ValueError(
                "Expected tensors with shape [num_trials, num_tokens, token_dim], "
                f"got {tuple(traces.shape)}"
            )
        self._traces = traces.float().contiguous()

    @classmethod
    def from_file(cls, path: Path, npz_key: str | None = None) -> "NeuralTraceDataset":
        if not path.exists():
            raise FileNotFoundError(f"Neural trace file not found: {path}")

        if path.suffix == ".npy":
            data = np.load(path)
        elif path.suffix == ".npz":
            npz_data = np.load(path)
            if npz_key:
                if npz_key not in npz_data:
                    raise ValueError(f"npz_key '{npz_key}' not found in {path}")
                data = npz_data[npz_key]
            else:
                first_key = next(iter(npz_data.files), None)
                if first_key is None:
                    raise ValueError(f"No arrays found in {path}")
                data = npz_data[first_key]
        else:
            raise ValueError("Unsupported neural tensor format. Use .npy or .npz")

        if data.ndim != 3:
            raise ValueError(
                "Loaded tensors must have shape [num_trials, num_tokens, token_dim], "
                f"got {data.shape}"
            )

        return cls(torch.from_numpy(data))

    @property
    def shape(self) -> tuple[int, int, int]:
        n, num_tokens, token_dim = self._traces.shape
        return int(n), int(num_tokens), int(token_dim)

    def __len__(self) -> int:
        return int(self._traces.shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self._traces[idx]


def build_dataset(config: NeuralDataConfig) -> Dataset[torch.Tensor]:
    """Build one dataset from config."""
    if config.path is None:
        raise ValueError("data.path is required")
    return NeuralTraceDataset.from_file(config.path, npz_key=config.npz_key)


def build_dataloaders(
    config: NeuralDataConfig,
) -> tuple[DataLoader[torch.Tensor], DataLoader[torch.Tensor], Dataset[torch.Tensor]]:
    """Build train/validation dataloaders from configured neural data source."""
    dataset = build_dataset(config)

    val_split = float(config.val_split)
    if not 0.0 < val_split < 1.0:
        raise ValueError("data.val_split must be in (0, 1)")

    val_size = max(1, int(round(len(dataset) * val_split)))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError("Validation split is too large for the dataset size")

    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=config.shuffle_train,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, dataset
