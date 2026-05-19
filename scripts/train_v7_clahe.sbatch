#!/bin/bash
#SBATCH --job-name=cs156b_train_v7_clahe
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=18:00:00
#SBATCH --output=logs/train_v7_clahe_%j.out
#SBATCH --error=logs/train_v7_clahe_%j.err
#SBATCH --mail-user=cewillia@caltech.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Same recipe as v4 (DenseNet121, 384px, 10 epochs, no imputation) but
# with CLAHE preprocessing turned on -- isolates CLAHE as the only variable.

set -e
mkdir -p logs

source /resnick/groups/CS156b/from_central/2026/JSC/miniconda/etc/profile.d/conda.sh
conda activate cs156b

cd "$SLURM_SUBMIT_DIR"

nvidia-smi

export CS156B_BACKBONE=densenet121
export CS156B_RUN_NAME=v7_densenet121_clahe
export CS156B_USE_CLAHE=1

python -u src/train.py
