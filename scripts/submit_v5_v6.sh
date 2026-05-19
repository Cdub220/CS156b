#!/bin/bash
# Submit two training jobs (ResNet50 + DenseNet169) in parallel, each
# chained to its own predict job. The two trainings run independently
# on different GPUs at the same time.

set -e

# ResNet50 train -> predict
TRAIN_R50=$(sbatch --parsable scripts/train_resnet50.sbatch)
PRED_R50=$(sbatch --parsable --dependency=afterok:$TRAIN_R50 scripts/predict_resnet50.sbatch)
echo "resnet50    train=$TRAIN_R50  predict=$PRED_R50"

# DenseNet169 train -> predict
TRAIN_D169=$(sbatch --parsable scripts/train_densenet169.sbatch)
PRED_D169=$(sbatch --parsable --dependency=afterok:$TRAIN_D169 scripts/predict_densenet169.sbatch)
echo "densenet169 train=$TRAIN_D169  predict=$PRED_D169"

echo
echo "Watch with:  squeue -u \$USER"
