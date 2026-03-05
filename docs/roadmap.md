# Roadmap

## Stage 1: Neural autoencoder

Goal: encode V1 neural activity into a compact latent representation.

- Define neural dataset interface (`subject`, `session`, `trial`, `frame` indexing).
- Implement baseline transformer encoder/decoder.
- Add training script with reconstruction loss + temporal consistency term.

## Stage 2: Image autoencoder (current)

Goal: use a fixed image latent space for supervision.

- Current implementation uses SD3 VAE as a frozen image compressor.
- Next improvements:
  - deterministic dataset splits.
  - configurable metrics logging.
  - optional latent caching for faster mapper training.

## Stage 3: Neural-to-image latent mapper

Goal: map neural latent sequence to image latent sequence.

- Start with transformer encoder-decoder or Perceiver-style cross-attention.
- Train on paired `(neural_latents, image_latents)`.
- Decode predicted image latents via frozen SD3 VAE decoder.

## Stage 4: Full video reconstruction

- Reconstruct full clips from neural activity sequences.
- Evaluate with image metrics, temporal metrics, and perceptual quality metrics.
