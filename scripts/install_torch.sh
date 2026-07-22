#!/usr/bin/env bash
# Install PyTorch into the ssa-h100 env. Safe to run on login or GPU node.
#
# Usage:
#   bash scripts/install_torch.sh

set -euo pipefail

ENV_NAME="${ENV_NAME:-ssa-h100}"
MAMBA_ROOT="${MAMBA_ROOT:-$HOME/micromamba}"

unset PYTHONPATH
export PYTHONNOUSERSITE=1

activate_env() {
    if [[ -x "$MAMBA_ROOT/bin/micromamba" ]]; then
        eval "$("$MAMBA_ROOT/bin/micromamba" shell hook -s bash)"
        micromamba activate "$ENV_NAME"
        return 0
    fi
    if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
        conda activate "$ENV_NAME"
        return 0
    fi
    echo "ERROR: No micromamba/miniconda found."
    echo "Run on login node: bash scripts/setup_hpc.sh"
    exit 1
}

if [[ -x "$MAMBA_ROOT/bin/micromamba" ]]; then
    if ! "$MAMBA_ROOT/bin/micromamba" env list | grep -q "^${ENV_NAME} "; then
        echo "Env '$ENV_NAME' not found. Run: bash scripts/setup_hpc.sh"
        exit 1
    fi
fi

activate_env

echo "Python: $(which python)"
echo "Installing PyTorch into env: $ENV_NAME"

if python -c "import torch; print('torch already installed:', torch.__version__)" 2>/dev/null; then
    exit 0
fi

# Try conda first (works on login nodes)
if command -v micromamba &>/dev/null; then
    echo "Trying: micromamba install pytorch pytorch-cuda=12.1 ..."
    if micromamba install -y -c pytorch -c nvidia pytorch pytorch-cuda=12.1 2>&1; then
        python -c "import torch; print('OK:', torch.__version__)"
        exit 0
    fi
    echo "Conda install failed, falling back to pip..."
fi

# Pip wheel — works on GPU nodes too (no conda solver needed)
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121

python -c "import torch; print('OK:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
echo "Done."
