#!/bin/bash
# Submit train + predict, chained.
# The preprocessing cache is already built and reused across runs, so we
# skip preprocess by default. Uncomment the block below if you ever need
# to rebuild the cache (e.g. changed image size, lost the cache dir).

set -e

# --- Optional: rebuild the image cache ---
# PRE_ID=$(sbatch --parsable scripts/preprocess.sbatch)
# echo "preprocess: $PRE_ID"
# TRAIN_DEP="--dependency=afterok:$PRE_ID"

TRAIN_DEP=""

TRAIN_ID=$(sbatch --parsable $TRAIN_DEP scripts/train.sbatch)
echo "train:      $TRAIN_ID"

PRED_ID=$(sbatch --parsable --dependency=afterok:$TRAIN_ID scripts/predict.sbatch)
echo "predict:    $PRED_ID  (waits on $TRAIN_ID)"

echo
echo "Watch with:  squeue -u \$USER"
