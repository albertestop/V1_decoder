from __future__ import annotations

import torch
from torch import nn


class TAE(nn.Module):
    """Starter template for custom neural autoencoder experiments.

    Expected input shape: [batch, num_tokens, token_dim]
    Forward return contract: (reconstruction, latents)
    """

    def __init__(
        self,
        token_dim: int,
        latent_dim: int,
        input_dim: int,
        nhead: int = 4,
        num_layers: int = 2,
        num_tokens: int | None = None,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.num_tokens = int(num_tokens) if num_tokens is not None else None
        self.latent_dim = int(latent_dim)
        self.input_dim = int(input_dim)
        self._last_num_tokens: int | None = None

        self.input_proj = nn.Linear(token_dim, input_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.to_latent = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, latent_dim),
        )
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, input_dim),
        )

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            batch_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(input_dim, token_dim)

    def _position_encoding(self, num_tokens: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        position = torch.arange(num_tokens, device=device, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.input_dim, 2, device=device, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0, device=device)) / self.input_dim)
        )
        pe = torch.zeros(1, num_tokens, self.input_dim, device=device, dtype=torch.float32)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term[: pe[:, :, 1::2].shape[-1]])
        return pe.to(dtype=dtype)

    def encode(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape [B, N, D], got {tuple(x.shape)}")
        if int(x.shape[2]) != self.token_dim:
            raise ValueError(
                f"Expected token_dim={self.token_dim}, got {tuple(x.shape)}"
            )
        if self.num_tokens is not None and int(x.shape[1]) != self.num_tokens:
            raise ValueError(f"Expected num_tokens={self.num_tokens}, got {tuple(x.shape)}")

        if padding_mask is None:
            padding_mask = torch.zeros((x.shape[0], x.shape[1]), dtype=torch.bool, device=x.device)
        self._last_num_tokens = int(x.shape[1])
        
        x = self.input_proj(x)
        x = x + self._position_encoding(x.shape[1], x.device, x.dtype)
        x = self.encoder(x, src_key_padding_mask=padding_mask)

        valid = (~padding_mask).unsqueeze(-1).to(dtype=x.dtype)
        pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        z = self.to_latent(pooled)
        return z

    def decode(self, z: torch.Tensor, num_tokens: int | None = None, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        target_tokens = int(num_tokens) if num_tokens is not None else self._last_num_tokens
        if target_tokens is None:
            if self.num_tokens is None:
                raise RuntimeError("num_tokens is unknown; call encode() before decode().")
            target_tokens = self.num_tokens

        x = self.from_latent(z).unsqueeze(1).repeat(1, target_tokens, 1)
        x = x + self._position_encoding(target_tokens, x.device, x.dtype)

        x = self.decoder(x, src_key_padding_mask=padding_mask)
        x = self.output_proj(x)
        return x

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latents = self.encode(x, padding_mask=padding_mask)
        reconstruction = self.decode(latents, num_tokens=x.shape[1], padding_mask=padding_mask)
        return reconstruction, latents
