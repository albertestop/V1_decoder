from __future__ import annotations

import torch
from torch import nn
import numpy as np

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, nhead):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=nhead,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(dim)

        self.ff = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

        self.norm2 = nn.LayerNorm(dim)

    def forward(self, query, key_value, key_padding_mask=None):

        attn_out, _ = self.attn(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=key_padding_mask,
        )

        x = self.norm1(query + attn_out)

        ff_out = self.ff(x)

        x = self.norm2(x + ff_out)

        return x


class TAE_v4(nn.Module):
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

        self.id_embedding = nn.Embedding(num_tokens, input_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(1, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, input_dim),
        )
        self.rec_proj = nn.Sequential(
            nn.Linear(1, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, input_dim),
        )

        self.fusion_proj = nn.Sequential(
            nn.LayerNorm(3 * input_dim),
            nn.Linear(3 * input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, input_dim),
        )

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

        self.latent_array = nn.Parameter(
            torch.randn(1, latent_num_tokens, latent_dim)
        )

        self.encoder_to_latent = CrossAttentionBlock(
            dim=latent_dim,
            nhead=nhead,
        )

        latent_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=nhead,
            batch_first=True,
            norm_first=True,
        )
        self.latent_transformer = nn.TransformerEncoder(latent_layer, num_layers=num_layers)

        self.output_queries = nn.Parameter(
            torch.randn(1, num_tokens, latent_dim)
        )

        self.latent_to_output = CrossAttentionBlock(
            dim=latent_dim,
            nhead=nhead,
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

        self.id_head = nn.Linear(input_dim, num_tokens)
        self.time_head = nn.Linear(input_dim, 1)       
        self.rec_head = nn.Linear(input_dim, 1)        

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
            raise ValueError(f"Got None for padding_mask")
            
    def encode(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        self.encode_sc(x, padding_mask)

        id = x[..., 0].long()
        time = x[..., 1].unsqueeze(-1)
        recording = x[..., 2].unsqueeze(-1)


        id_emb = self.id_embedding(id)  # A dictionary where each token has a trainable vector to identify it
        t_proj = self.time_proj(time)   # Project them into the same embedding space
        rec_proj = self.rec_proj(recording) # You want each token to become a single vector that encodes:what (id)when (time)value (recording)

        x = torch.cat([id_emb, t_proj, rec_proj], dim=-1)
        x = self.fusion_proj(x)

        x = self.encoder(x, src_key_padding_mask=padding_mask)

        x = self.to_latent(x)

        B = x.shape[0]
        latent_queries = self.latent_array.expand(B, -1, -1)

        x = self.encoder_to_latent(
            query=latent_queries,
            key_value=x,
            key_padding_mask=padding_mask,
        )

        z = self.latent_transformer(x)

        return z

    def decode(
        self,
        z: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        B = z.shape[0]

        output_queries = self.output_queries.expand(B, -1, -1)

        x = self.latent_to_output(
            query=output_queries,
            key_value=z,
        )

        x = self.from_latent(x)

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