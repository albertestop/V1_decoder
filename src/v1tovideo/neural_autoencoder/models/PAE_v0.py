from __future__ import annotations

import torch
from torch import nn


class PAE_v0(nn.Module):
    """
    Perceiver autoencoder:
    many input tokens -> cross-attention -> few latent tokens -> latent self-attention
    -> cross-attention back to output tokens.
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
        if num_tokens is None:
            raise ValueError("PAE_v0 needs num_tokens to build id embeddings and output queries")

        self.token_dim = int(token_dim)
        self.num_tokens = int(num_tokens)
        self.latent_dim = int(latent_dim)
        self.input_dim = int(input_dim)
        self.latent_num_tokens = int(latent_num_tokens)

        self._last_num_tokens: int | None = None

        self.id_embedding = nn.Embedding(self.num_tokens, input_dim)
        self.time_proj = nn.Linear(1, input_dim)
        self.rec_proj = nn.Linear(1, input_dim)

        self.fusion_proj = nn.Sequential(
            nn.LayerNorm(3 * input_dim),
            nn.Linear(3 * input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, input_dim),
        )

        self.input_to_latent = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, latent_dim),
        )

        self.latent_queries = nn.Parameter(
            torch.randn(1, self.latent_num_tokens, self.latent_dim) * 0.02
        )
        self.encoder_cross_attn = nn.MultiheadAttention(
            latent_dim,
            nhead,
            batch_first=True,
        )

        latent_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=nhead,
            batch_first=True,
        )
        self.latent_encoder = nn.TransformerEncoder(latent_layer, num_layers=num_layers)

        self.output_queries = nn.Parameter(
            torch.randn(1, self.num_tokens, self.latent_dim) * 0.02
        )
        self.decoder_cross_attn = nn.MultiheadAttention(
            latent_dim,
            nhead,
            batch_first=True,
        )

        self.latent_to_input = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, input_dim),
        )

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)

        self.id_head = nn.Linear(input_dim, self.num_tokens)
        self.time_head = nn.Linear(input_dim, 1)
        self.rec_head = nn.Linear(input_dim, 1)

    def encode_sc(self, x: torch.Tensor, padding_mask: torch.Tensor | None) -> None:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape [B, N, D], got {tuple(x.shape)}")
        if int(x.shape[2]) != self.token_dim:
            raise ValueError(f"Expected token_dim={self.token_dim}, got {tuple(x.shape)}")
        if int(x.shape[1]) != self.num_tokens:
            raise ValueError(f"Expected num_tokens={self.num_tokens}, got {tuple(x.shape)}")
        if padding_mask is None:
            raise ValueError("Got None for padding_mask")


    def encode(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.encode_sc(x, padding_mask)
        self._last_num_tokens = int(x.shape[1])

        ids = x[..., 0].long()
        time = x[..., 1].unsqueeze(-1)
        recording = x[..., 2].unsqueeze(-1)

        id_emb = self.id_embedding(ids)
        time_emb = self.time_proj(time)
        rec_emb = self.rec_proj(recording)

        x = torch.cat([id_emb, time_emb, rec_emb], dim=-1)
        x = self.fusion_proj(x)
        x = self.input_to_latent(x)

        q = self.latent_queries.expand(x.size(0), -1, -1)
        z, _ = self.encoder_cross_attn(
            query=q,
            key=x,
            value=x,
            key_padding_mask=padding_mask,
        )
        return self.latent_encoder(z)

    def decode(
        self,
        z: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.output_queries.expand(z.size(0), -1, -1)
        x, _ = self.decoder_cross_attn(query=q, key=z, value=z)

        x = self.latent_to_input(x)
        x = self.decoder(x, src_key_padding_mask=padding_mask)

        id_logits = self.id_head(x)
        time_pred = self.time_head(x)
        rec_pred = self.rec_head(x)

        return id_logits, time_pred, rec_pred

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        latents = self.encode(x, padding_mask=padding_mask)
        id_logits, time_pred, rec_pred = self.decode(latents, padding_mask=padding_mask)
        return id_logits, time_pred, rec_pred, latents

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        dtype = x.dtype
        id_logits, time_pred, rec_pred, _ = self(x, padding_mask=padding_mask)

        id_pred = id_logits.argmax(dim=-1).to(dtype=dtype)
        preds = torch.stack(
            (
                id_pred,
                time_pred.squeeze(-1).to(dtype=dtype),
                rec_pred.squeeze(-1).to(dtype=dtype),
            ),
            dim=-1,
        )
        if was_training:
            self.train()
        return preds.masked_fill(padding_mask.unsqueeze(-1), 0.0)
