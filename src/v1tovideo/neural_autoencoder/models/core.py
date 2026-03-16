from __future__ import annotations

from typing import Any

import torch
from torch import nn


class BaseNeuralAutoencoder(nn.Module):
    """Base interface for neural autoencoders."""

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


class MLPNeuralAutoencoder(BaseNeuralAutoencoder):
    """Simple baseline autoencoder that flattens the trace sequence."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        sequence_length = int(config["sequence_length"])
        num_channels = int(config["num_channels"])
        latent_dim = int(config.get("latent_dim", 128))
        hidden_dim = int(config.get("hidden_dim", 256))
        input_dim = sequence_length * num_channels

        self._sequence_length = sequence_length
        self._num_channels = num_channels

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(x.shape[0], -1)
        return self.encoder(flat)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        flat = self.decoder(z)
        return flat.reshape(z.shape[0], self._sequence_length, self._num_channels)


class TransformerNeuralAutoencoder(BaseNeuralAutoencoder):
    """Transformer skeleton with a learned latent bottleneck token set."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.sequence_length = int(config["sequence_length"])
        self.num_channels = int(config["num_channels"])
        latent_dim = int(config.get("latent_dim", 128))
        hidden_dim = int(config.get("hidden_dim", 256))
        num_layers = int(config.get("num_layers", 4))
        num_heads = int(config.get("num_heads", 8))
        dropout = float(config.get("dropout", 0.1))

        self.input_proj = nn.Linear(self.num_channels, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.sequence_length, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.to_latent = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(hidden_dim, self.num_channels)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = h + self.pos_embed[:, : x.shape[1], :]
        h = self.encoder(h)
        pooled = h.mean(dim=1)
        return self.to_latent(pooled)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        seed = self.from_latent(z).unsqueeze(1).repeat(1, self.sequence_length, 1)
        seed = seed + self.pos_embed[:, : self.sequence_length, :]
        decoded = self.decoder(seed)
        return self.output_proj(decoded)


def build_model(config: dict[str, Any]) -> BaseNeuralAutoencoder:
    """Construct a neural autoencoder from a raw config dictionary."""
    architecture = str(config["architecture"]).lower().strip()
    if architecture == "mlp":
        return MLPNeuralAutoencoder(config)
    if architecture == "transformer":
        return TransformerNeuralAutoencoder(config)

    supported = ["mlp", "transformer"]
    raise ValueError(f"Unsupported architecture '{config['architecture']}'. Supported: {supported}")
