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
        num_tokens: int,
        latent_dim: int,
        input_dim: int,
        nhead: int = 4,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.num_tokens = int(num_tokens)
        self.latent_dim = int(latent_dim)
        self.input_dim = int(input_dim)

        self.input_proj = nn.Linear(token_dim, input_dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, input_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.to_latent = nn.Linear(input_dim * num_tokens, latent_dim)
        self.from_latent = nn.Linear(latent_dim, input_dim * num_tokens)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            batch_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(input_dim, token_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape [B, N, D], got {tuple(x.shape)}")
        if int(x.shape[1]) != self.num_tokens or int(x.shape[2]) != self.token_dim:
            raise ValueError(
                f"Expected [B, {self.num_tokens}, {self.token_dim}], got {tuple(x.shape)}"
            )

        x = self.input_proj(x)
        x = x + self.pos_embedding
        x = self.encoder(x)

        x = x.flatten(1)
        z = self.to_latent(x)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.from_latent(z)
        x = x.view(z.shape[0], self.num_tokens, self.input_dim)

        x = self.decoder(x)
        x = self.output_proj(x)
        return x

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latents = self.encode(x)
        reconstruction = self.decode(latents)
        return reconstruction, latents
