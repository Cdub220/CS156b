#!/bin/bash
# Train v8 (DenseNet121 + pseudo-labels + 100% data) and chain predict.
# Make sure /resnick/groups/CS156b/from_central/2026/JSC/pseudo_labels.csv
# exists before submitting -- it's the teacher ensemble's predictions.

set -e

TRAIN_ID=$(sbatch --parsable scripts/train_v8_pseudo.sbatch)
echo "v8 train:   $TRAIN_ID"

PRED_ID=$(sbatch --parsable --dependency=afterok:$TRAIN_ID scripts/predict_v8_pseudo.sbatch)
echo "v8 predict: $PRED_ID  (waits on $TRAIN_ID)"

echo
echo "Watch with:  squeue -u \$USER"
