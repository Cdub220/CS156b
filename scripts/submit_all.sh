#!/bin/bash
# Build the test-image cache (preprocess) then run inference on the test
# set (predict). Training is skipped -- best.pt from the previous run is
# reused. preprocess is idempotent: already-cached images are skipped, so
# only the new test images get processed.

set -e

PRE_ID=$(sbatch --parsable scripts/preprocess.sbatch)
echo "preprocess: $PRE_ID"

PRED_ID=$(sbatch --parsable --dependency=afterok:$PRE_ID scripts/predict.sbatch)
echo "predict:    $PRED_ID  (waits on $PRE_ID)"

echo
echo "Watch with:  squeue -u \$USER"
