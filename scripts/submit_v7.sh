#!/bin/bash
# Train v7 (DenseNet121 + CLAHE) and chain predict after.

set -e

TRAIN_ID=$(sbatch --parsable scripts/train_v7_clahe.sbatch)
echo "v7 train:   $TRAIN_ID"

PRED_ID=$(sbatch --parsable --dependency=afterok:$TRAIN_ID scripts/predict_v7_clahe.sbatch)
echo "v7 predict: $PRED_ID  (waits on $TRAIN_ID)"

echo
echo "Watch with:  squeue -u \$USER"
