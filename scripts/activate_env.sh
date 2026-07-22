#!/usr/bin/env bash
# Activate the ssa-h100 env on any node (login or GPU).
# Usage: source scripts/activate_env.sh

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=hpc_common.sh
source "$_SCRIPT_DIR/hpc_common.sh"

if ! activate_env "$ENV_NAME"; then
    echo "ERROR: env '$ENV_NAME' not found."
    echo ""
    print_hpc_diagnostics
    echo ""
    echo "Fix on login node:"
    echo "  bash scripts/setup_hpc.sh"
    return 1 2>/dev/null || exit 1
fi

if ! python -c "import torch" 2>/dev/null; then
    echo "ERROR: torch not installed in env '$ENV_NAME'"
    echo "Fix: bash scripts/install_torch.sh"
    return 1 2>/dev/null || exit 1
fi

echo "Active env: $ENV_NAME"
echo "Python:     $(which python)"
python -c "import torch; print('Torch:', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
