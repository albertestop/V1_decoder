from __future__ import annotations

import torch
from torch import nn


class TAE_v1_00(nn.Module):
    """

        Like TAE_v1 with token compression with no ID prediction:
        We reorder the arrays so that cells are ordered always from 0 to 5443.
        SET ID WEIGHT TO 0!
        We compress the token n. 
        We add them back by generating random parameter array, 
        transforming it with some transformer layers, and finally
        concatenating them with the original array.

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
        self.outputs = ['value', 'value', 'value']
        self.token_dim = int(token_dim)
        self.num_tokens = int(num_tokens) if num_tokens is not None else None
        self.latent_dim = int(latent_dim)
        self.input_dim = int(input_dim)
        self.latent_num_tokens = int(latent_num_tokens)

        self._last_num_tokens: int | None = None

        self.id_embedding = nn.Embedding(num_tokens, input_dim)
        self.time_proj = nn.Linear(1, input_dim)
        self.rec_proj = nn.Linear(1, input_dim)

        self.fusion_proj = nn.Sequential(
            nn.LayerNorm(2 * input_dim),
            nn.Linear(2 * input_dim, input_dim),
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
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, input_dim),
        )

        self.readd_tokens = nn.Parameter(torch.randn(1, self.num_tokens - self.latent_num_tokens, self.latent_dim) * 0.02)

        readdition_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=nhead,
            batch_first=True,
        )
        self.readd_transform = nn.TransformerEncoder(readdition_layer, num_layers=num_layers)

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

        self._last_num_tokens = int(x.shape[1])

        time = x[..., 1].unsqueeze(-1)
        recording = x[..., 2].unsqueeze(-1)


        t_proj = self.time_proj(time)   # Project them into the same embedding space
        rec_proj = self.rec_proj(recording) # You want each token to become a single vector that encodes:what (id)when (time)value (recording)

        x = torch.cat([t_proj, rec_proj], dim=-1)
        x = self.fusion_proj(x)

        x = self.encoder(x, src_key_padding_mask=padding_mask)

        z = self.to_latent(x)

        z = z[:, :self.latent_num_tokens, :]

        return z

    def decode(
        self,
        z: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        readdition_init = self.readd_tokens.expand(z.shape[0], -1, -1)
        readdition = self.readd_transform(readdition_init)
        x = torch.concatenate([z, readdition], axis=1)

        x = self.from_latent(x)

        x = self.decoder(x, src_key_padding_mask=padding_mask)

        time_pred = self.time_head(x)       # regression
        rec_pred = self.rec_head(x)         # regression

        return time_pred, rec_pred

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        indices = torch.argsort(x[:, :, 0], dim=1)
        x = x.gather(1, indices[:, :, None].expand(-1, -1, x.shape[-1]))
        if padding_mask is not None:
            padding_mask = padding_mask.gather(1, indices)
        ids = x[:, :, 0]
        latents = self.encode(x, padding_mask=padding_mask)
        time_pred, rec_pred = self.decode(latents, padding_mask=padding_mask)
        return ids, time_pred, rec_pred, latents

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        was_training = self.training
        self.eval()
        dtype = x.dtype
        out = self(x, padding_mask=padding_mask)
        ids, time_pred, rec_pred, _ = out

        preds = torch.stack(
            (ids.to(dtype=dtype), time_pred.squeeze(-1).to(dtype=dtype), rec_pred.squeeze(-1).to(dtype=dtype)),
            dim=-1,
        )
        if was_training:
            self.train()
        return preds.masked_fill(padding_mask.unsqueeze(-1), 0.0)
