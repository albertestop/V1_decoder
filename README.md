# V1-to-Video Reconstruction

This repository contains the initial implementation for a 3-stage pipeline to reconstruct viewed video from V1 neural recordings:

1. Neural autoencoder: compress/decompress neural activity signals.
2. Image autoencoder: compress/decompress video frames.
3. Latent mapper: map neural latents to image latents.

Only stage 2 (image autoencoder) is currently implemented with working code, using the Stable Diffusion 3 VAE from `diffusers`.

## Current status

- Implemented:
  - SD3 VAE image compression/decompression module.
  - Reconstruction quality metrics (MSE, MAE, frequency-domain scores, SSIM, compression ratio).
  - Single-image and batch evaluation scripts.
- Planned:
  - Transformer-based neural autoencoder.
  - Transformer-based neural-latent -> image-latent mapper.

## Project structure

- `src/v1tovideo/image_autoencoder/`: implemented image compression module.
- `src/v1tovideo/neural_autoencoder/`: placeholder package for neural compression.
- `src/v1tovideo/latent_mapper/`: placeholder package for latent mapping model.
- `scripts/`: runnable entrypoints.
- `docs/`: project documentation and roadmap.
- `legacy/`: original prototype files retained for reference.
- `assets/`: tracked sample images/results from the original prototype.
- `data/`: local datasets (ignored by git).
- `outputs/`: generated artifacts (ignored by git).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

If your Hugging Face access requires authentication, export:

```bash
export HF_TOKEN=<your_token>
```

## Run examples

Single image:

```bash
python scripts/run_image_vae_single.py
```

Batch frame evaluation:

```bash
python scripts/run_image_vae_batch.py
```

Custom config path:

```bash
python scripts/run_image_vae_single.py --config scripts/configs/image_vae_single.toml
python scripts/run_image_vae_batch.py --config scripts/configs/image_vae_batch.toml
```

## Notes

- Edit `scripts/configs/image_vae_single.toml` and `scripts/configs/image_vae_batch.toml` to set input/output paths and runtime parameters.
- The current image module resizes frames to `144x256` before encoding. This matches your prototype and can be changed in the config files.
- `legacy/` preserves the original scripts.
- `assets/` preserves representative sample inputs and legacy outputs.
