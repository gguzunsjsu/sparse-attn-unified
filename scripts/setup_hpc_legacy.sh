#!/usr/bin/env bash
# Fallback setup using legacy Miniconda (GLIBC 2.17 compatible installer).
# Use this if micromamba (scripts/setup_hpc.sh) is unavailable.
#
# Usage:
#   bash scripts/setup_hpc_legacy.sh

set -euo pipefail

ENV_NAME="${ENV_NAME:-ssa-h100}"
MINICONDA="$HOME/miniconda3"
INSTALLER="Miniconda3-py39_4.12.0-Linux-x86_64.sh"
INSTALLER_URL="https://repo.anaconda.com/miniconda/${INSTALLER}"

unset PYTHONPATH
export PYTHONNOUSERSITE=1

if [[ ! -x "$MINICONDA/bin/conda" ]]; then
    echo "Installing Miniconda 4.12.0 (compatible with GLIBC 2.17)..."
    wget -O "/tmp/${INSTALLER}" "$INSTALLER_URL"
    bash "/tmp/${INSTALLER}" -b -p "$MINICONDA"
fi

source "$MINICONDA/etc/profile.d/conda.sh"

if ! conda env list | grep -q "^${ENV_NAME} "; then
    conda create -n "$ENV_NAME" python=3.10 -y
fi

conda activate "$ENV_NAME"

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python -m pip install -e "$PROJECT_ROOT"
python -m pip install transformers datasets accelerate einops pyyaml tqdm pytest

echo "Done. Run: conda activate $ENV_NAME"
