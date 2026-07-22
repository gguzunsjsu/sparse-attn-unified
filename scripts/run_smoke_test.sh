#!/usr/bin/env bash
# Smoke test wrapper — activates env, loads CUDA, verifies torch, then runs.
#
# Usage (on GPU node):
#   bash scripts/run_smoke_test.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

module purge
module load cuda 2>/dev/null || true

# shellcheck source=/dev/null
source "$PROJECT_ROOT/scripts/activate_env.sh"

# Ensure sparse_attn + numpy are installed (common gap after install_torch.sh only)
if ! python -c "import sparse_attn" 2>/dev/null; then
    echo "Installing project dependencies..."
    bash "$PROJECT_ROOT/scripts/install_project_deps.sh"
fi

if ! python -c "import numpy" 2>/dev/null; then
    python -m pip install "numpy>=1.26.0"
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/train_llama1b_ssa.py \
  --smoke-test \
  --from-scratch \
  --seq-length 512 \
  --batch-size 1 \
  "$@"
