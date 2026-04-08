from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset


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
    """Extract `[num_tokens, token_dim]` from `[B, N, D]` padded batch formats."""
    tensor_batch: torch.Tensor
    if torch.is_tensor(batch):
        tensor_batch = batch
    elif isinstance(batch, (tuple, list)) and len(batch) > 0 and torch.is_tensor(batch[0]):
        tensor_batch = batch[0]
    else:
        raise ValueError(f"Expected tensor batch or tuple/list with tensor first element, got {type(batch)!r}")

    if tensor_batch.ndim != 3:
        raise ValueError(f"Expected training batch shape [B, N, D], got {tuple(tensor_batch.shape)}")
    return int(tensor_batch.shape[1]), int(tensor_batch.shape[2])


class NeuralTraceDataset(Dataset[torch.Tensor]):
    """Dataset wrapper for per-trial tensors with shape [num_tokens_i, token_dim]."""

    def __init__(self, traces: list[torch.Tensor]) -> None:
        if not traces:
            raise ValueError("Expected at least one trial tensor")

        cleaned: list[torch.Tensor] = []
        token_dim: int | None = None
        max_tokens = 0
        for idx, trial in enumerate(traces):
            if not torch.is_tensor(trial):
                raise ValueError(f"Trial {idx} must be a tensor, got {type(trial)!r}")
            if trial.ndim != 2:
                raise ValueError(
                    "Each trial must have shape [num_tokens_i, token_dim], "
                    f"got {tuple(trial.shape)} at index {idx}"
                )
            if token_dim is None:
                token_dim = int(trial.shape[1])
            elif int(trial.shape[1]) != token_dim:
                raise ValueError(
                    f"All trials must share token_dim={token_dim}, got {int(trial.shape[1])} at index {idx}"
                )

            cleaned_trial = trial.float().contiguous()
            cleaned.append(cleaned_trial)
            max_tokens = max(max_tokens, int(cleaned_trial.shape[0]))

        self._traces = cleaned
        self.token_dim = int(token_dim)
        self.max_tokens = int(max_tokens)

    @classmethod
    def from_file(cls, path: Path, npz_key: str | None = None) -> "NeuralTraceDataset":
        if not path.exists():
            raise FileNotFoundError(f"Neural trace file not found: {path}")

        if path.suffix == ".npy":
            data = np.load(path, allow_pickle=True)
        elif path.suffix == ".npz":
            npz_data = np.load(path, allow_pickle=True)
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

        data = data.astype(np.float32)
        
        trials: list[torch.Tensor] = []
        if isinstance(data, np.ndarray) and data.ndim == 3:
            trials = [torch.from_numpy(np.array(data[i])) for i in range(int(data.shape[0]))]
        elif isinstance(data, np.ndarray) and data.ndim == 2:
            trials = [torch.from_numpy(data)]
        elif isinstance(data, np.ndarray) and data.dtype == object:
            for i, item in enumerate(data.tolist()):
                arr = np.asarray(item, dtype=np.float32)
                if arr.ndim != 2:
                    raise ValueError(f"Object trial at index {i} must be 2D, got shape {arr.shape}")
                trials.append(torch.from_numpy(arr))
        else:
            raise ValueError(
                "Loaded tensors must be either a 3D array [num_trials, num_tokens, token_dim] "
                "or an object array/list of 2D trials [num_tokens_i, token_dim]."
            )

        return cls(trials)

    @property
    def shape(self) -> tuple[int, int, int]:
        return int(len(self._traces)), int(self.max_tokens), int(self.token_dim)

    def __len__(self) -> int:
        return int(len(self._traces))

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self._traces[idx]


def collate_padded_trials(
    batch: list[torch.Tensor],
    pad_to_tokens: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad trial tensors and return (x, lengths, padding_mask)."""
    if not batch:
        raise ValueError("Empty batch")
    lengths = torch.tensor([int(x.shape[0]) for x in batch], dtype=torch.long)
    padded = pad_sequence(batch, batch_first=True, padding_value=0.0)  # [B, N_max, D]
    if pad_to_tokens is not None and int(padded.shape[1]) < int(pad_to_tokens):
        pad_steps = int(pad_to_tokens) - int(padded.shape[1])
        padded = F.pad(padded, (0, 0, 0, pad_steps, 0, 0))
    steps = torch.arange(int(padded.shape[1]), dtype=torch.long).unsqueeze(0)
    padding_mask = steps >= lengths.unsqueeze(1)  # [B, N_max], True for padding positions
    return padded, lengths, padding_mask


def build_dataset(config: NeuralDataConfig) -> Dataset[torch.Tensor]:
    """Build one dataset from config."""
    data_path = config.path / Path('data/responses.npy').expanduser()
    if data_path is None:
        raise ValueError("data.path is required")
    return NeuralTraceDataset.from_file(data_path, npz_key=config.npz_key)


def build_dataloaders(
    config: NeuralDataConfig,
) -> tuple[DataLoader[Any], DataLoader[Any], Dataset[torch.Tensor]]:
    """ 
        Build train/validation dataloaders from configured neural data source.
        Validation split is the data of n random trials.
    """
    dataset = build_dataset(config)
    with open(config.path / Path('trial_dataset_map.json').expanduser(), "r") as f:
        data_map = json.load(f)
    n_exp_trials = len(data_map)

    val_split = float(config.val_split)
    if not 0.0 < val_split < 1.0:
        raise ValueError("data.val_split must be in (0, 1)")

    val_n_exp_trials = int(round(n_exp_trials * val_split))
    val_exp_trials_idx = np.random.randint(0, n_exp_trials, val_n_exp_trials)
    mask = np.full(len(dataset), False)
    for exp_trial in val_exp_trials_idx:
        start, end = map(int, data_map[f"{exp_trial}"]["dataset_rows"].split(","))
        mask[start:end] = True
    val_size = max(1, int(sum(mask)))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError("Validation split is too large for the dataset size")

    val_indices = torch.where(torch.from_numpy(mask))[0]
    train_indices = torch.where(~torch.from_numpy(mask))[0]
    val_set = Subset(dataset, val_indices)
    train_set = Subset(dataset, train_indices)
    pad_to_tokens = int(getattr(dataset, "max_tokens"))

    collate_fn = partial(collate_padded_trials, pad_to_tokens=pad_to_tokens)

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=config.shuffle_train,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, dataset, data_map, val_exp_trials_idx
