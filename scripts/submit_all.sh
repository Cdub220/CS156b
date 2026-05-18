#!/bin/bash
# Train (SLURM array of 18 tasks, one per model) then predict.
#
# afterok on an array job ID waits for ALL 18 array tasks to succeed
# before predict starts. If any single task fails, predict is auto-cancelled.
#
# train.py is resumable: each task skips its (view, pathology) if a best.pt
# already exists. So you can resubmit only the failed tasks:
#   sbatch --array=5,12 scripts/train.sbatch

set -e

TRAIN_ID=$(sbatch --parsable scripts/train.sbatch)
echo "train:    $TRAIN_ID  (array of 18 tasks)"

PRED_ID=$(sbatch --parsable --dependency=afterok:$TRAIN_ID scripts/predict.sbatch)
echo "predict:  $PRED_ID  (waits on $TRAIN_ID)"

echo
echo "Watch with:  squeue -u \$USER"
