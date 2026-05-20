#!/bin/bash
#SBATCH --job-name=cs156b_predict_v8_pseudo
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/predict_v8_pseudo_%j.out
#SBATCH --error=logs/predict_v8_pseudo_%j.err
#SBATCH --mail-user=cewillia@caltech.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -e
mkdir -p logs

source /resnick/groups/CS156b/from_central/2026/JSC/miniconda/etc/profile.d/conda.sh
conda activate cs156b

cd "$SLURM_SUBMIT_DIR"

nvidia-smi

export CS156B_RUN_NAME=v8_pseudo_full_data

python -u src/predict.py
