#!/bin/bash
# Final-submission pipeline:
#   1. Preprocess the solution images into the cache (skips already-cached).
#   2. Run inference for each of the 5 ensemble models against solution_ids.csv.
#
# All 5 predict jobs depend on preprocess via afterok, and they run on
# separate GPUs in parallel. Output filenames are "solution_predictions.csv"
# in each model's output directory, so test predictions aren't overwritten.

set -e

# Step 1: make sure solution images are cached.
PRE_ID=$(sbatch --parsable scripts/preprocess.sbatch)
echo "preprocess:        $PRE_ID"

# Step 2: predict with each model, waiting on preprocess.
for V in v1 v4 v5 v6 v7; do
    JID=$(sbatch --parsable --dependency=afterok:$PRE_ID scripts/predict_solution_${V}.sbatch)
    echo "predict $V:        $JID  (waits on $PRE_ID)"
done

echo
echo "Watch with:  squeue -u \$USER"
echo
echo "When all 5 finish, pull these to your Mac and ensemble:"
echo "  outputs/densenet121/solution_predictions.csv             (v1)"
echo "  outputs/v4_multitask_no_imputation/solution_predictions.csv"
echo "  outputs/v5_resnet50/solution_predictions.csv"
echo "  outputs/v6_densenet169/solution_predictions.csv"
echo "  outputs/v7_densenet121_clahe/solution_predictions.csv"
