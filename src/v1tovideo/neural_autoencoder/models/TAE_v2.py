from __future__ import annotations

import torch
from torch import nn


class TAE_v2(nn.Module):
    """Starter template for custom neural autoencoder experiments.

    Expected input shape: [batch, num_tokens, token_dim]
    Forward return contract: (reconstruction, latents)
    """

    def __init__(
        self,
        token_dim: int,
        latent_dim: int,
        input_dim: int,
        latent_num_tokens: int,
        nhead: int = 4,
        num_layers: int = 2,
        num_tokens: int | None = None,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.num_tokens = int(num_tokens) if num_tokens is not None else None
        self.latent_dim = int(latent_dim)
        self.input_dim = int(input_dim)
        self.laten_num_tokens = int(latent_num_tokens)

        self._last_num_tokens: int | None = None

        self.id_embedding = nn.Embedding(num_tokens, input_dim)

        self.input_proj = nn.Linear(token_dim - 1, input_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pool_queries = nn.Parameter(torch.randn(1, latent_num_tokens, input_dim))
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=nhead,
            batch_first=True,
        )

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

        self.output_id = nn.Linear(input_dim, 1)

        self.output_other = nn.Linear(input_dim, token_dim - 1)

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
        
        id = x[..., 0].long()
        time = x[..., 1].unsqueeze(-1)
        recording = x[..., 2].unsqueeze(-1)

        id_emb = self.id_embedding(id)
        other = torch.cat([time, recording], dim=-1)
        other = self.input_proj(other)
        x = id_emb + other

        x = self.encoder(x, src_key_padding_mask=padding_mask)

        queries = self.pool_queries.repeat(x.shape[0], 1, 1)

        pooled, _ = self.pool_attn(
            query=queries,
            key=x,
            value=x,
            key_padding_mask=padding_mask,
        )

        z = self.to_latent(pooled)
        return z

    def decode(
        self,
        z: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        target_tokens = self.num_tokens

        B, M, _ = z.shape

        memory = self.from_latent(z)

        queries = torch.zeros(B, target_tokens, self.input_dim, device=z.device)

        x, _ = self.pool_attn(
            query=queries,
            key=memory,
            value=memory,
        )

        x = self.decoder(x, src_key_padding_mask=padding_mask)

        id_logits = self.output_id(x)
        other = self.output_other(x)

        return torch.cat([id_logits, other], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latents = self.encode(x, padding_mask=padding_mask)
        reconstruction = self.decode(latents, padding_mask=padding_mask)
        return reconstruction, latents
