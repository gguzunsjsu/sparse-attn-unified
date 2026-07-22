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
echo "Mamba:    $MAMBA_ROOT"
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

install_pytorch() {
    echo "Installing PyTorch..."
    if micromamba install -y -c pytorch -c nvidia pytorch pytorch-cuda=12.1; then
        return 0
    fi
    echo "Conda PyTorch install failed — trying pip wheel (cu121)..."
    python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
}

if ! python -c "import torch" 2>/dev/null; then
    install_pytorch
fi

if ! python -c "import torch" 2>/dev/null; then
    echo ""
    echo "FATAL: torch still not importable after install."
    echo "Try manually inside the env:"
    echo "  python -m pip install torch --index-url https://download.pytorch.org/whl/cu121"
    exit 1
fi

echo "Installing project dependencies..."
python -m pip install --upgrade pip
python -m pip install -e "$PROJECT_ROOT"
python -m pip install transformers datasets accelerate einops pyyaml tqdm pytest huggingface_hub

PYTHON_BIN="$MAMBA_ROOT/envs/$ENV_NAME/bin/python"
echo ""
echo "=== Setup complete ==="
echo "Python:  $PYTHON_BIN"
"$PYTHON_BIN" -c "import torch; print('Torch:', torch.__version__)"
echo ""
echo "On GPU node (new shell — must activate env every time):"
echo "  source $PROJECT_ROOT/scripts/activate_env.sh"
echo "  module load cuda"
echo "  bash $PROJECT_ROOT/scripts/run_smoke_test.sh"
echo ""
echo "Or call python directly (no activate needed):"
echo "  $PYTHON_BIN $PROJECT_ROOT/scripts/train_llama1b_ssa.py --smoke-test --from-scratch"
echo ""
echo "Add to ~/.bashrc (once):"
echo "  eval \"\$($MAMBA_ROOT/bin/micromamba shell hook -s bash)\""
