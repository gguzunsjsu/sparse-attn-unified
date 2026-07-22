#!/usr/bin/env bash
# Install PyTorch into the ssa-h100 env. Safe to run on login or GPU node.
#
# Usage:
#   bash scripts/install_torch.sh

set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=hpc_common.sh
source "$_SCRIPT_DIR/hpc_common.sh"

if ! activate_env "$ENV_NAME"; then
    echo "ERROR: env '$ENV_NAME' not found."
    print_hpc_diagnostics
    echo "Run on login node: bash scripts/setup_hpc.sh"
    exit 1
fi

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
