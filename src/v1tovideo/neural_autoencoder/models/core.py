from __future__ import annotations

from typing import Any

import torch
from torch import nn


class BaseNeuralAutoencoder(nn.Module):
    """Base interface for neural autoencoders."""

    def encode(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError

    def decode(
        self,
        z: torch.Tensor,
        num_tokens: int | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x, padding_mask=padding_mask)
        recon = self.decode(z, num_tokens=int(x.shape[1]), padding_mask=padding_mask)
        return recon, z


class MLPNeuralAutoencoder(BaseNeuralAutoencoder):
    """Simple baseline autoencoder that flattens `[B, num_tokens, token_dim]`."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        token_dim = int(config["token_dim"])
        num_tokens = int(config["num_tokens"])
        latent_dim = int(config.get("latent_dim", 128))
        hidden_dim = int(config.get("hidden_dim", 256))
        input_dim = token_dim * num_tokens

        self._token_dim = token_dim
        self._num_tokens = num_tokens

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

    def encode(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape [B, N, D], got {tuple(x.shape)}")
        if x.shape[1] != self._num_tokens or x.shape[2] != self._token_dim:
            raise ValueError(
                f"MLPNeuralAutoencoder expects [B, {self._num_tokens}, {self._token_dim}], got {tuple(x.shape)}"
            )
        flat = x.reshape(x.shape[0], -1)
        return self.encoder(flat)

    def decode(
        self,
        z: torch.Tensor,
        num_tokens: int | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        flat = self.decoder(z)
        return flat.reshape(z.shape[0], self._num_tokens, self._token_dim)


class TransformerNeuralAutoencoder(BaseNeuralAutoencoder):
    """Transformer autoencoder for `[B, num_tokens, token_dim]` inputs."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.token_dim = int(config["token_dim"])
        self.num_tokens = int(config["num_tokens"]) if config.get("num_tokens") is not None else None
        latent_dim = int(config.get("latent_dim", 128))
        hidden_dim = int(config.get("hidden_dim", 256))
        num_layers = int(config.get("num_layers", 4))
        num_heads = int(config.get("num_heads", 8))
        dropout = float(config.get("dropout", 0.1))
        self._last_num_tokens: int | None = None

        self.input_proj = nn.Linear(self.token_dim, hidden_dim)

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
        self.output_proj = nn.Linear(hidden_dim, self.token_dim)

    def _position_encoding(self, num_tokens: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        hidden_dim = int(self.input_proj.out_features)
        position = torch.arange(num_tokens, device=device, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2, device=device, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0, device=device)) / hidden_dim)
        )
        pe = torch.zeros(1, num_tokens, hidden_dim, device=device, dtype=torch.float32)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term[: pe[:, :, 1::2].shape[-1]])
        return pe.to(dtype=dtype)

    def encode(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape [B, N, D], got {tuple(x.shape)}")
        if x.shape[2] != self.token_dim:
            raise ValueError(f"Expected token_dim={self.token_dim}, got input shape {tuple(x.shape)}")
        if self.num_tokens is not None and x.shape[1] != self.num_tokens:
            raise ValueError(f"Expected num_tokens={self.num_tokens}, got input shape {tuple(x.shape)}")

        self._last_num_tokens = int(x.shape[1])
        h = self.input_proj(x)
        h = h + self._position_encoding(x.shape[1], h.device, h.dtype)
        h = self.encoder(h, src_key_padding_mask=padding_mask)
        if padding_mask is None:
            pooled = h.mean(dim=1)
        else:
            valid = (~padding_mask).unsqueeze(-1).to(dtype=h.dtype)
            pooled = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return self.to_latent(pooled)

    def decode(
        self,
        z: torch.Tensor,
        num_tokens: int | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        target_tokens = int(num_tokens) if num_tokens is not None else self._last_num_tokens
        if target_tokens is None:
            if self.num_tokens is None:
                raise RuntimeError("num_tokens is unknown; call encode() before decode().")
            target_tokens = self.num_tokens
        seed = self.from_latent(z).unsqueeze(1).repeat(1, target_tokens, 1)
        seed = seed + self._position_encoding(target_tokens, seed.device, seed.dtype)
        decoded = self.decoder(seed, src_key_padding_mask=padding_mask)
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
