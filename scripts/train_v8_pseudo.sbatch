#!/bin/bash
#SBATCH --job-name=cs156b_train_v8_pseudo
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=18:00:00
#SBATCH --output=logs/train_v8_pseudo_%j.out
#SBATCH --error=logs/train_v8_pseudo_%j.err
#SBATCH --mail-user=cewillia@caltech.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# v8: DenseNet121 + pseudo-labels from current best ensemble + train on 100% data.
# Same recipe as v4 except (a) pseudo-labels appended to training set,
# and (b) no patient val split -- final-epoch model is the checkpoint.

set -e
mkdir -p logs

source /resnick/groups/CS156b/from_central/2026/JSC/miniconda/etc/profile.d/conda.sh
conda activate cs156b

cd "$SLURM_SUBMIT_DIR"

nvidia-smi

export CS156B_BACKBONE=densenet121
export CS156B_RUN_NAME=v8_pseudo_full_data
export CS156B_USE_CLAHE=0
export CS156B_PSEUDO_LABELS_CSV=/resnick/groups/CS156b/from_central/2026/JSC/pseudo_labels.csv
export CS156B_USE_VAL_SPLIT=0

python -u src/train.py
