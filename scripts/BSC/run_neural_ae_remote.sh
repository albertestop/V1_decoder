#!/bin/bash
#SBATCH --job-name=train_ae
#SBATCH --account=uab108
#SBATCH --qos=acc_resb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --output=%x.out

module load miniforge
source activate sci_albert

srun python train_ae.py
