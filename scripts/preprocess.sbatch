#!/bin/bash
#SBATCH --job-name=cs156b_preprocess
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/preprocess_%j.out
#SBATCH --error=logs/preprocess_%j.err
#SBATCH --mail-user=cewillia@caltech.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Pre-resize every JPEG to 320x320 PNG

set -e
mkdir -p logs

source /resnick/groups/CS156b/from_central/2026/JSC/env/etc/profile.d/conda.sh
conda activate cs156b

cd "$SLURM_SUBMIT_DIR"

python -u src/preprocess_all.py
