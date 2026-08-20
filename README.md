# V1-to-Video Transformer Architecture

This project trains autoencoder models for V1 neural activity as part of a video reconstruction pipeline. The current active workflow focuses on the neural autoencoder experiments.

## Current Execution Flow

The project is currently executed through the scripts in `scripts/BSC/`.

Main launcher:

```bash
python scripts/BSC/bsc_neural_ae.py
```

This script:

- reads `scripts/BSC/bsc_neural_ae.conf`;
- syncs the repository and dataset to BSC;
- creates `scripts/BSC/generated_remote_config.toml`;
- submits the BSC job with `scripts/BSC/run_neural_ae_remote.sh`.

On BSC, the submitted job runs:

```bash
srun python train_ae.py
```

`scripts/BSC/train_ae.py` then executes:

```bash
python scripts/BSC/run_neural_ae_experiment.py --config scripts/BSC/generated_remote_config.toml
```

`scripts/BSC/run_neural_ae_experiment.py` handles the full model training flow: loading the dataset, building the model, training, evaluation, checkpoint saving, metrics, summaries, and reconstruction plots.

## Important Files

- `scripts/BSC/bsc_neural_ae.py`: local launcher for BSC runs.
- `scripts/BSC/bsc_neural_ae.conf`: BSC SSH, remote path, and config settings.
- `scripts/BSC/run_neural_ae_remote.sh`: Slurm job script.
- `scripts/BSC/train_ae.py`: remote wrapper that starts the training entrypoint.
- `scripts/BSC/run_neural_ae_experiment.py`: main neural autoencoder training script.
- `scripts/configs/neural_ae_experiment.toml`: base experiment configuration.
- `src/v1tovideo/neural_autoencoder/`: data loading, models, training, evaluation, and plotting code.

## Setup

Install the project locally:

```bash
pip install -e .
```

For the BSC environment, use the requirements file in:

```bash
scripts/BSC/requirements_sci_albert_full.txt
```

## Configuration

Edit the base experiment config before launching:

```bash
scripts/configs/neural_ae_experiment.toml
```

The BSC launcher rewrites local paths for the remote machine and stores the generated config at:

```bash
scripts/BSC/generated_remote_config.toml
```

## Results

Training outputs are saved under the configured output directory, usually:

```bash
outputs/neural_autoencoder/default_run
```

To download finished BSC runs:

```bash
python scripts/BSC/bsc_neural_ae_down_results.py
```

Downloaded runs are stored locally under:

```bash
outputs/neural_autoencoder/
```

## Legacy and Other Scripts

Older local scripts and image autoencoder utilities are still present in `scripts/` and `src/v1tovideo/image_autoencoder/`, but the current project execution path is the BSC neural autoencoder workflow described above.
