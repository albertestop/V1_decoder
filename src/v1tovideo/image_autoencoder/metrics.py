from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim


def _spectrum_similarity(orig: np.ndarray, recon: np.ndarray, cutoff: float = 0.25, eps: float = 1e-8) -> tuple[float, float]:
    h, w = orig.shape
    cy, cx = h // 2, w // 2
    r = int(min(cy, cx) * cutoff)

    f1 = np.fft.fftshift(np.fft.fft2(orig))
    f2 = np.fft.fftshift(np.fft.fft2(recon))

    mag1, mag2 = np.abs(f1), np.abs(f2)

    y, x = np.ogrid[:h, :w]
    mask_low = (x - cx) ** 2 + (y - cy) ** 2 <= r**2
    mask_high = ~mask_low

    diff_low = np.linalg.norm(mag1[mask_low] - mag2[mask_low])
    norm_low = np.linalg.norm(mag1[mask_low])
    low = 1 - diff_low / (norm_low + eps)

    diff_high = np.linalg.norm(mag1[mask_high] - mag2[mask_high])
    norm_high = np.linalg.norm(mag1[mask_high])
    high = 1 - diff_high / (norm_high + eps)

    return float(low), float(high)


def _rgb_to_gray(img: torch.Tensor) -> torch.Tensor:
    # expects channel-first RGB tensor in [-1, 1]
    img_01 = img.add(1).div(2)
    r, g, b = img_01[0], img_01[1], img_01[2]
    return 0.299 * r + 0.587 * g + 0.114 * b


def grayscale_reconstruction_metrics(
    original_rgb: torch.Tensor,
    reconstruction_rgb: torch.Tensor,
    compression_ratio: float,
) -> dict[str, float]:
    original = _rgb_to_gray(original_rgb)
    reconstruction = _rgb_to_gray(reconstruction_rgb)

    mse = float(F.mse_loss(reconstruction, original).item())
    mae = float(F.l1_loss(reconstruction, original).item())

    original_np = original.cpu().detach().numpy()
    reconstruction_np = reconstruction.cpu().detach().numpy()

    low, high = _spectrum_similarity(original_np, reconstruction_np)
    ssim_score, _ = ssim(
        original_np,
        reconstruction_np,
        full=True,
        data_range=np.float32(reconstruction_np.max() - reconstruction_np.min()),
    )

    return {
        "mse": mse,
        "mae": mae,
        "low_freq_accuracy": float(low),
        "high_freq_accuracy": float(high),
        "ssim": float(ssim_score),
        "compression_ratio": float(compression_ratio),
    }
