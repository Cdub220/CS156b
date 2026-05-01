#!/bin/bash
#   bash scripts/setup_env.sh

set -e

CONDA_ROOT=/resnick/groups/CS156b/from_central/2026/JSC/miniconda

# If you haven't actually run the miniconda installer yet, do this first:
#   bash "$CONDA_ROOT/Miniconda3-latest-Linux-x86_64.sh" -b -p "$CONDA_ROOT/miniconda"
# (and then change the source line below to "$CONDA_ROOT/miniconda/etc/profile.d/conda.sh")

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
