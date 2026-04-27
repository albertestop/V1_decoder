from __future__ import annotations

import torch
from torch import nn
import numpy as np


class TAE_v1(nn.Module):
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
        self.time_proj = nn.Linear(1, input_dim)
        self.rec_proj = nn.Linear(1, input_dim)

        self.fusion_proj = nn.Sequential(
            nn.LayerNorm(3 * input_dim),
            nn.Linear(3 * input_dim, input_dim),
            # nn.GELU(),
            # nn.Linear(d_model, d_model),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # self.pool_queries = nn.Parameter(torch.randn(1, latent_num_tokens, input_dim))
        # self.pool_attn = nn.MultiheadAttention(
        #     embed_dim=input_dim,
        #     num_heads=nhead,
        #     batch_first=True,
        # )

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

        self.id_head = nn.Linear(latent_dim, num_tokens)
        self.time_head = nn.Linear(latent_dim, 1)       
        self.rec_head = nn.Linear(latent_dim, 1)        

    def encode_sc(self, x, padding_mask):
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
            
    def encode(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        self.encode_sc(x, padding_mask)

        self._last_num_tokens = int(x.shape[1])

        id = x[..., 0].long()
        time = x[..., 1].unsqueeze(-1)
        recording = x[..., 2].unsqueeze(-1)


        id_emb = self.id_embedding(id)  # A dictionary where each token has a trainable vector to identify it
        t_proj = self.time_proj(time)   # Project them into the same embedding space
        rec_proj = self.rec_proj(recording) # You want each token to become a single vector that encodes:what (id)when (time)value (recording)

        x = torch.cat([id_emb, t_proj, rec_proj], dim=-1)
        x = self.fusion_proj(x)

        x = self.encoder(x, src_key_padding_mask=padding_mask)
        # queries = self.pool_queries.repeat(x.shape[0], 1, 1)

        # pooled, _ = self.pool_attn(
        #     query=queries,
        #     key=x,
        #     value=x,
        #     key_padding_mask=padding_mask,
        # )

        z = self.to_latent(x)
        return z

    def decode(
        self,
        z: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        x = self.from_latent(z)

        x = self.decoder(x, src_key_padding_mask=padding_mask)

        id_logits = self.id_head(x)         # classification over IDs
        time_pred = self.time_head(x)       # regression
        rec_pred = self.rec_head(x)         # regression

        return id_logits, time_pred, rec_pred

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latents = self.encode(x, padding_mask=padding_mask)
        id_logits, time_pred, rec_pred = self.decode(latents, padding_mask=padding_mask)
        return id_logits, time_pred, rec_pred, latents

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        dtype = x.dtype
        out = self(x)
        id_logits, time_pred, rec_pred, _ = out

        id_pred = id_logits.argmax(dim=-1).to(dtype=dtype)
        preds = torch.stack(
            (id_pred, time_pred.squeeze(-1).to(dtype=dtype), rec_pred.squeeze(-1).to(dtype=dtype)),
            dim=-1,
        )
        return preds.masked_fill(padding_mask.unsqueeze(-1), 0.0)