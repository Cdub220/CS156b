#!/bin/bash
#SBATCH --job-name=cs156b_sol_v6
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/predict_solution_v6_%j.out
#SBATCH --error=logs/predict_solution_v6_%j.err
#SBATCH --mail-user=cewillia@caltech.edu
#SBATCH --mail-type=END,FAIL

set -e
mkdir -p logs

source /resnick/groups/CS156b/from_central/2026/JSC/miniconda/etc/profile.d/conda.sh
conda activate cs156b

cd "$SLURM_SUBMIT_DIR"

nvidia-smi

export CS156B_RUN_NAME=v6_densenet169
export CS156B_TEST_CSV=/resnick/groups/CS156b/from_central/data/student_labels/solution_ids.csv
export CS156B_OUT_FILENAME=solution_predictions.csv

python -u src/predict.py
