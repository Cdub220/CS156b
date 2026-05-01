#!/bin/bash


set -e

PRE_ID=$(sbatch --parsable scripts/preprocess.sbatch)
echo "preprocess: $PRE_ID"

TRAIN_ID=$(sbatch --parsable --dependency=afterok:$PRE_ID scripts/train.sbatch)
echo "train:      $TRAIN_ID  (waits on $PRE_ID)"

PRED_ID=$(sbatch --parsable --dependency=afterok:$TRAIN_ID scripts/predict.sbatch)
echo "predict:    $PRED_ID  (waits on $TRAIN_ID)"

echo
echo "Watch with:  squeue -u \$USER"
