#!/usr/bin/env bash
# Setup script for SJSU CoE HPC (older GLIBC 2.17 login nodes).
#
# Fixes two common failures:
#   1. Miniconda3-latest requires GLIBC >= 2.28
#   2. module-load system pip hits PermissionError on /opt/ohpc/...
#
# Usage (on login node):
#   cd ~/sparse-attn-unified
#   bash scripts/setup_hpc.sh

set -euo pipefail

ENV_NAME="${ENV_NAME:-ssa-h100}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAMBA_ROOT="${MAMBA_ROOT:-$HOME/micromamba}"

echo "=== sparse-attn-unified HPC setup ==="
echo "Project:  $PROJECT_ROOT"
echo "Env:      $ENV_NAME"
echo "GLIBC:    $(ldd --version 2>/dev/null | head -1 || echo 'unknown')"

# Never use the module system Python/pip for installs.
unset PYTHONPATH
export PYTHONNOUSERSITE=1

install_micromamba() {
    if [[ -x "$MAMBA_ROOT/bin/micromamba" ]]; then
        echo "micromamba already installed at $MAMBA_ROOT"
        return
    fi

    echo "Installing micromamba (static binary, works on GLIBC 2.17)..."
    mkdir -p "$MAMBA_ROOT"
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
        | tar -xj -C "$MAMBA_ROOT" bin/micromamba

    "$MAMBA_ROOT/bin/micromamba" shell init -s bash -p "$MAMBA_ROOT" >/dev/null
}

eval "$("$MAMBA_ROOT/bin/micromamba" shell hook -s bash)"
install_micromamba

if ! micromamba env list | grep -q "^${ENV_NAME} "; then
    echo "Creating conda env: $ENV_NAME (python 3.10)..."
    micromamba create -n "$ENV_NAME" -y -c conda-forge \
        python=3.10 pip setuptools wheel
fi

micromamba activate "$ENV_NAME"

# PyTorch + CUDA — use conda-forge/pytorch channel (no system pip needed).
# Adjust cuda version after: module avail cuda  (on a GPU node)
if ! python -c "import torch" 2>/dev/null; then
    echo "Installing PyTorch (CUDA 12.1 build)..."
    micromamba install -y -c pytorch -c nvidia \
        pytorch pytorch-cuda=12.1
fi

echo "Installing project dependencies..."
python -m pip install --upgrade pip
python -m pip install -e "$PROJECT_ROOT"
python -m pip install transformers datasets accelerate einops pyyaml tqdm pytest huggingface_hub

echo ""
echo "=== Setup complete ==="
echo "Verify with:"
echo "  micromamba activate $ENV_NAME"
echo "  python -c \"import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())\""
echo ""
echo "On GPU node:"
echo "  module load cuda"
echo "  python $PROJECT_ROOT/scripts/train_llama1b_ssa.py --smoke-test --from-scratch"
echo ""
echo "Add to ~/.bashrc (once):"
echo "  eval \"\$($MAMBA_ROOT/bin/micromamba shell hook -s bash)\""
