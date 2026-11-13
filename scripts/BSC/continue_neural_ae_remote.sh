#!/bin/bash
#SBATCH --job-name=continue_ae
#SBATCH --account=uab108
#SBATCH --qos=acc_resb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%j.out

set -euo pipefail

: "${NEURAL_AE_RUN_DIR:?NEURAL_AE_RUN_DIR must be set by the watchdog}"
: "${NEURAL_AE_ADDITIONAL_EPOCHS:?NEURAL_AE_ADDITIONAL_EPOCHS must be set by the watchdog}"

module load miniforge
source activate sci_albert

srun python scripts/continue_neural_ae_training.py --run-dir "$NEURAL_AE_RUN_DIR" --epochs "$NEURAL_AE_ADDITIONAL_EPOCHS"
