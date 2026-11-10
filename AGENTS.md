# Project context and working conventions

## Purpose

This repository develops a machine-learning pipeline for reconstructing images
from neural recordings. The intended final architecture has three stages:

```text
neural recordings
  -> neural autoencoder -> neural latents
  -> neural-to-image mapper -> image latents
  -> image decoder -> reconstructed images
```

1. **Neural autoencoder (active development):** compress and reconstruct neural
   recordings with minimal loss.
2. **Image compressor (active development):** encode images as compact latent
   representations and decode them back to images.
3. **Latent mapper (future work):** map neural latents to image latents, enabling
   the complete pipeline above.

Consult this file before answering or making changes in this repository. Treat
the paths and conventions below as the default project context unless the user
provides newer, more specific instructions.

## Data conventions

- Processed training datasets live under
  `/home/albertestop/data/processed_data/tba`.
- For the image-compressor stage, use a simplifying subset of video frames from
  `data/video_frames` in this repository.
- Neural recordings use shape **`(instance, neuron, token)`**:

  - `N` instances, each containing the same `M` neurons.
  - Each neuron has three tokens:
    - token `0`: neuron ID, from `0` to `M - 1`;
    - token `1`: recording time;
    - token `2`: recorded spike amplitude.
- Dataset inputs depend on the task: commonly `responses.npy`, `targets.npy`,
  and related dataset information.
- Dataset instances are associated with experimental trials through
  `dataset_name/trial_dataset_map.json` below the processed-data root. Example:

  ```json
  {
    "0": {
      "trial_id": "00438/",
      "trial_index": 0,
      "dataset_rows": "0,94"
    }
  }
  ```

  Here trial index `0`, ID `00438`, occupies dataset rows `0` through `93`
  (the end is exclusive), in order.
- Important per-dataset arrays:

  - `frames.npy`: image-frame reconstruction targets for the final model;
  - `frames_t.npy`: frame timeline;
  - `responses.npy`: neural-recording inputs;
  - `targets.npy`: a version of the neural responses used to train the neural
    autoencoder.

## Neural-autoencoder workflow

- Configuration files: `scripts/configs/`.
- Local training entry point: `scripts/run_neural_ae_experiment.py`.
- Training configuration: `scripts/configs/neural_ae_experiment.toml`.
- Cluster-ready counterpart: `scripts/BSC/run_neural_ae_experiment.py`.
  These scripts must remain functionally aligned: when changing one, implement
  the analogous change in the other unless there is a documented cluster-only
  reason not to.
- BSC cluster configuration: `scripts/BSC/bsc_neural_ae.conf`.
- BSC training launcher: `scripts/BSC/train_ae.py`.
- BSC results retrieval: `scripts/BSC/bsc_neural_ae_down_results.py`.

## Neural-autoencoder outputs

Each run is saved at `outputs/neural_autoencoder/run_X/`. Key artifacts include:

- the training configuration;
- training history (evolving metrics/parameters);
- a summary, including such details as the input dataset shape;
- validation indices (`val_indices`) used for that run;
- additional diagnostic artifacts.

The primary consolidated inspection utility is
`tests/summarize_neural_autoencoder_run.py`; use it when evaluating or reporting
on a neural-autoencoder run.

## Change discipline

- Preserve the alignment of local and BSC neural-autoencoder workflows.
- When interpreting results, identify the exact `run_X`, configuration, data
  split, and relevant input array.
- Do not assume trial row boundaries are inclusive: `"0,94"` denotes rows
  `0:94`.
- The neural-to-image latent mapper is not yet implemented; do not describe it
  as an existing trained component unless the repository state later shows one.
