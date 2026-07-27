from __future__ import annotations

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
