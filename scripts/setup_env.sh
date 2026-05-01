#!/bin/bash
#   bash scripts/setup_env.sh

set -e

CONDA_ROOT=/resnick/groups/CS156b/from_central/2026/JSC/miniconda


source "$CONDA_ROOT/etc/profile.d/conda.sh"

ENV_NAME=cs156b

conda create -y -n "$ENV_NAME" python=3.10
conda activate "$ENV_NAME"

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Everything else.
pip install -r requirements.txt

echo
echo "Done. To use the env later:"
echo "  source $CONDA_ROOT/etc/profile.d/conda.sh"
echo "  conda activate $ENV_NAME"
