from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
from diffusers import models
from PIL import Image
from torchvision import transforms

from .metrics import grayscale_reconstruction_metrics


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_sd3_vae(device: str | None = None) -> models.AutoencoderKL:
    """Load the Stable Diffusion 3 VAE used as image compressor/decompressor."""
    use_device = device or get_device()
    hf_token = os.getenv("HF_TOKEN")

    vae = models.AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        subfolder="vae",
        token=hf_token,
    ).to(use_device).eval()
    return vae


def _preprocess_rgb(height: int, width: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((height, width), interpolation=Image.BICUBIC, antialias=True),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )


def _load_image_as_rgb(image_path: Path, target_height: int, target_width: int) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    return _preprocess_rgb(target_height, target_width)(image).unsqueeze(0)


def encode_decode_image(
    image_path: Path,
    output_dir: Path,
    vae: models.AutoencoderKL,
    target_height: int = 144,
    target_width: int = 256,
    save_prefix: str = "sample",
) -> dict[str, float | torch.Tensor]:
    """Encode and decode one image through SD3 VAE and save artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    device = next(vae.parameters()).device

    original = _load_image_as_rgb(image_path, target_height, target_width).to(device)

    with torch.no_grad():
        posterior = vae.encode(original)
        latents = posterior.latent_dist.sample() * vae.config.scaling_factor

    latents_path = output_dir / f"{save_prefix}.latents.pt"
    torch.save(latents.half().cpu(), latents_path)

    decoded_latents = torch.load(latents_path, map_location=device)
    decoded_latents = decoded_latents.float() / vae.config.scaling_factor

    with torch.no_grad():
        reconstruction = vae.decode(decoded_latents).sample.clamp(-1, 1)

    reconstruction_img = reconstruction.add(1).div(2)
    reconstructed_path = output_dir / f"{save_prefix}.reconstruction.png"
    transforms.ToPILImage()(reconstruction_img[0].cpu()).save(reconstructed_path)

    compression_ratio = (
        (original.shape[2] * original.shape[3])
        / (latents.shape[1] * latents.shape[2] * latents.shape[3])
    )

    metrics = grayscale_reconstruction_metrics(original[0], reconstruction[0], compression_ratio)

    return {
        "original": original[0].detach().cpu(),
        "reconstruction": reconstruction[0].detach().cpu(),
        "compression_ratio": float(compression_ratio),
        **metrics,
    }


def _list_frame_paths(frames_root: Path) -> list[Path]:
    frame_paths: list[Path] = []
    for trial_dir in frames_root.iterdir():
        if not trial_dir.is_dir():
            continue
        frame_paths.extend([p for p in trial_dir.iterdir() if p.is_file()])
    return frame_paths


def evaluate_random_frames(
    frames_root: Path,
    num_samples: int,
    output_dir: Path,
    seed: int = 0,
    target_height: int = 144,
    target_width: int = 256,
) -> dict[str, float]:
    """Run reconstruction stats on random frame samples from a dataset tree."""
    random.seed(seed)
    np.random.seed(seed)

    frame_paths = _list_frame_paths(frames_root)
    if not frame_paths:
        raise ValueError(f"No frame files found under {frames_root}")

    vae = load_sd3_vae()
    stats: list[dict[str, float | torch.Tensor]] = []

    for idx in range(min(num_samples, len(frame_paths))):
        frame_path = random.choice(frame_paths)
        result = encode_decode_image(
            image_path=frame_path,
            output_dir=output_dir,
            vae=vae,
            target_height=target_height,
            target_width=target_width,
            save_prefix=f"sample_{idx}",
        )
        stats.append(result)

    valid = [s for s in stats if float(s["high_freq_accuracy"]) > 0]
    if not valid:
        raise ValueError("No valid samples with positive high-frequency accuracy.")

    keys = ["mse", "mae", "low_freq_accuracy", "high_freq_accuracy", "ssim", "compression_ratio"]
    summary = {k: float(np.mean([float(sample[k]) for sample in valid])) for k in keys}

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_dir / "mean_stats.txt", np.array([summary[k] for k in keys], dtype=np.float32))
    return summary
