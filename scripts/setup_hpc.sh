#!/usr/bin/env bash
# Setup script for SJSU CoE HPC (older GLIBC 2.17 login nodes).
#
# Usage (on login node):
#   cd ~/sparse-attn-unified
#   bash scripts/setup_hpc.sh

set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$_SCRIPT_DIR/.." && pwd)"
# shellcheck source=hpc_common.sh
source "$_SCRIPT_DIR/hpc_common.sh"

echo "=== sparse-attn-unified HPC setup ==="
echo "Project:  $PROJECT_ROOT"
echo "Env:      $ENV_NAME"
echo "HPC_HOME: $HPC_HOME"
echo "Mamba:    $MAMBA_ROOT"
echo "GLIBC:    $(ldd --version 2>/dev/null | head -1 || echo 'unknown')"

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

if [[ ! -x "$MAMBA_ROOT/bin/micromamba" ]]; then
    install_micromamba
fi

eval "$("$MAMBA_ROOT/bin/micromamba" shell hook -s bash)"

if ! env_exists "$ENV_NAME"; then
    echo "Creating conda env: $ENV_NAME (python 3.10)..."
    micromamba create -n "$ENV_NAME" -y -c conda-forge \
        python=3.10 pip setuptools wheel
fi

if ! env_exists "$ENV_NAME"; then
    echo "FATAL: env '$ENV_NAME' was not created at ${MAMBA_ROOT}/envs/${ENV_NAME}"
    exit 1
fi

activate_env "$ENV_NAME"

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
    echo "Try: bash scripts/install_torch.sh"
    exit 1
fi

echo "Installing project dependencies..."
python -m pip install --upgrade pip
python -m pip install -e "$PROJECT_ROOT"
python -m pip install transformers datasets accelerate einops pyyaml tqdm pytest huggingface_hub

PYTHON_BIN="$(find_env_python "$ENV_NAME")"
echo ""
echo "=== Setup complete ==="
echo "Python:  $PYTHON_BIN"
"$PYTHON_BIN" -c "import torch; print('Torch:', torch.__version__)"
echo ""
echo "Verify anywhere (login or GPU node):"
echo "  bash $PROJECT_ROOT/scripts/doctor_hpc.sh"
echo ""
echo "Smoke test on GPU node:"
echo "  bash $PROJECT_ROOT/scripts/run_smoke_test.sh"
echo ""
echo "Add to ~/.bashrc (once):"
echo "  eval \"\$($MAMBA_ROOT/bin/micromamba shell hook -s bash)\""
