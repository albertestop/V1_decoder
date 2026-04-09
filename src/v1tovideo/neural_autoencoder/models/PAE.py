from __future__ import annotations

import torch
from torch import nn


class _CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, nhead: int, dropout: float) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(
        self,
        queries: torch.Tensor,
        kv: torch.Tensor,
        kv_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self.norm_q(queries)
        k = self.norm_kv(kv)
        v = k
        attn_out, _ = self.attn(q, k, v, key_padding_mask=kv_padding_mask, need_weights=False)
        x = queries + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.norm_ffn(x)))
        return x


class _SelfAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, nhead: int, dropout: float) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm_attn(x)
        attn_out, _ = self.attn(h, h, h, key_padding_mask=padding_mask, need_weights=False)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.norm_ffn(x)))
        return x


class PAE(nn.Module):
    """Perceiver-style autoencoder for set-like `[batch, num_tokens, token_dim]` inputs.

    This model does not use temporal position encodings; it treats tokens as a collection
    and compresses them through a latent bottleneck with cross-attention.
    """

    def __init__(
        self,
        token_dim: int,
        latent_dim: int,
        input_dim: int,
        nhead: int = 4,
        num_layers: int = 2,
        num_tokens: int | None = None,
        num_latents: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.num_tokens = int(num_tokens) if num_tokens is not None else None
        self.latent_dim = int(latent_dim)
        self.input_dim = int(input_dim)
        self.num_latents = int(num_latents)
        self._last_num_tokens: int | None = None

        self.input_proj = nn.Linear(self.token_dim, self.input_dim)

        self.encoder_latents = nn.Parameter(torch.randn(self.num_latents, self.input_dim) * 0.02)
        self.encoder_cross = _CrossAttentionBlock(self.input_dim, nhead=nhead, dropout=dropout)
        self.encoder_self = nn.ModuleList(
            [_SelfAttentionBlock(self.input_dim, nhead=nhead, dropout=dropout) for _ in range(num_layers)]
        )

        self.to_latent = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.latent_dim),
        )
        self.from_latent = nn.Sequential(
            nn.Linear(self.latent_dim, self.input_dim),
            nn.GELU(),
            nn.Linear(self.input_dim, self.input_dim),
        )

        self.decoder_latents = nn.Parameter(torch.randn(self.num_latents, self.input_dim) * 0.02)
        self.decoder_self = nn.ModuleList(
            [_SelfAttentionBlock(self.input_dim, nhead=nhead, dropout=dropout) for _ in range(num_layers)]
        )
        self.decoder_cross = _CrossAttentionBlock(self.input_dim, nhead=nhead, dropout=dropout)

        # Token-query generator from normalized token indices (identity, not temporal encoding).
        self.query_index_mlp = nn.Sequential(
            nn.Linear(1, self.input_dim),
            nn.GELU(),
            nn.Linear(self.input_dim, self.input_dim),
        )
        self.output_proj = nn.Linear(self.input_dim, self.token_dim)

    def _make_output_queries(
        self,
        batch_size: int,
        target_tokens: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if target_tokens <= 0:
            raise ValueError(f"target_tokens must be positive, got {target_tokens}")
        if target_tokens == 1:
            idx = torch.zeros(1, device=device, dtype=torch.float32)
        else:
            idx = torch.linspace(0.0, 1.0, target_tokens, device=device, dtype=torch.float32)
        idx = idx.view(1, target_tokens, 1).expand(batch_size, -1, -1)
        return self.query_index_mlp(idx).to(dtype=dtype)

    def encode(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape [B, N, D], got {tuple(x.shape)}")
        if int(x.shape[2]) != self.token_dim:
            raise ValueError(f"Expected token_dim={self.token_dim}, got {tuple(x.shape)}")
        if self.num_tokens is not None and int(x.shape[1]) != self.num_tokens:
            raise ValueError(f"Expected num_tokens={self.num_tokens}, got {tuple(x.shape)}")

        batch_size = int(x.shape[0])
        self._last_num_tokens = int(x.shape[1])
        h = self.input_proj(x)

        latents = self.encoder_latents.unsqueeze(0).expand(batch_size, -1, -1)
        latents = self.encoder_cross(latents, h, kv_padding_mask=padding_mask)
        for block in self.encoder_self:
            latents = block(latents)

        pooled = latents.mean(dim=1)
        z = self.to_latent(pooled)
        return z

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

        batch_size = int(z.shape[0])
        latent_seed = self.from_latent(z).unsqueeze(1)
        latents = self.decoder_latents.unsqueeze(0).expand(batch_size, -1, -1) + latent_seed
        for block in self.decoder_self:
            latents = block(latents)

        queries = self._make_output_queries(
            batch_size=batch_size,
            target_tokens=target_tokens,
            device=z.device,
            dtype=latents.dtype,
        )
        decoded = self.decoder_cross(queries, latents)
        out = self.output_proj(decoded)

        # Keep padded tokens neutral if a padding mask is provided by the dataloader.
        if padding_mask is not None:
            out = out.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return out

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latents = self.encode(x, padding_mask=padding_mask)
        reconstruction = self.decode(latents, num_tokens=x.shape[1], padding_mask=padding_mask)
        return reconstruction, latents
