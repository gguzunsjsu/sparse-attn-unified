#!/usr/bin/env bash
# Install project + runtime deps into the active env (login or GPU node).
#
# Usage:
#   source scripts/activate_env.sh
#   bash scripts/install_project_deps.sh
#
# Or standalone:
#   bash scripts/install_project_deps.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# shellcheck source=hpc_common.sh
source "$PROJECT_ROOT/scripts/hpc_common.sh"

if ! activate_env "$ENV_NAME" 2>/dev/null; then
    echo "ERROR: Could not activate env '$ENV_NAME'"
    print_hpc_diagnostics
    exit 1
fi

echo "Installing into: $(which python)"
python -m pip install --upgrade pip setuptools wheel

# Core runtime deps (numpy often missing when torch installed via pip alone)
python -m pip install "numpy>=1.26.0" einops pyyaml tqdm datasets

# Editable install registers the sparse_attn package
python -m pip install -e "$PROJECT_ROOT"

# Verify
python -c "import numpy; import sparse_attn; print('OK: numpy', numpy.__version__, '| sparse_attn', sparse_attn.__version__)"

echo "Project deps installed."
