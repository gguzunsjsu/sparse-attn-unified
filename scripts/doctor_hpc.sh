#!/usr/bin/env bash
# Print environment diagnostics for HPC troubleshooting.
#
# Usage:
#   bash scripts/doctor_hpc.sh

set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$_SCRIPT_DIR/.." && pwd)"

# shellcheck source=hpc_common.sh
source "$_SCRIPT_DIR/hpc_common.sh"

print_hpc_diagnostics

echo ""
echo "=== Quick checks ==="

py="$(find_env_python "$ENV_NAME" 2>/dev/null || true)"
if [[ -n "$py" ]]; then
    echo "[OK] Env python exists: $py"
    if "$py" -c "import torch; print('[OK] torch', torch.__version__)" 2>/dev/null; then
        "$py" -c "import torch; print('[OK] CUDA available:', torch.cuda.is_available())"
    else
        echo "[FAIL] torch not importable — run: bash scripts/install_torch.sh"
    fi
else
    echo "[FAIL] Env '$ENV_NAME' not found — run on login node: bash scripts/setup_hpc.sh"
fi

if [[ -d "$PROJECT_ROOT/sparse_attn" ]]; then
    echo "[OK] Project root: $PROJECT_ROOT"
else
    echo "[FAIL] sparse_attn package not found under $PROJECT_ROOT"
fi

module load cuda 2>/dev/null && echo "[OK] cuda module loaded" || echo "[WARN] cuda module not loaded"
