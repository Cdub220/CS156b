#!/bin/bash
#SBATCH --job-name=cs156b_train
#SBATCH --partition=gpu
#SBATCH --array=0-17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/train_%A_%a.out
#SBATCH --error=logs/train_%A_%a.err
#SBATCH --mail-user=cewillia@caltech.edu
#SBATCH --mail-type=END,FAIL

# SLURM array: 18 tasks, one per (view, pathology) model.
# Each task gets its own GPU and trains one model. They run in parallel
# as nodes free up. train.py reads SLURM_ARRAY_TASK_ID to know which.
#
# Per-task walltime is 6h, plenty for one model on half the data.
# %A = parent job id, %a = array task id, so logs don't collide.
# BEGIN emails turned off (you'd get 18 of them).

set -e
mkdir -p logs

source /resnick/groups/CS156b/from_central/2026/JSC/miniconda/etc/profile.d/conda.sh
conda activate cs156b

cd "$SLURM_SUBMIT_DIR"

nvidia-smi

python -u src/train.py
