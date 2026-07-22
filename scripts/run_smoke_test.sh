#!/usr/bin/env bash
# Smoke test wrapper — activates env, loads CUDA, verifies torch, then runs.
#
# Usage (on GPU node):
#   bash scripts/run_smoke_test.sh
#
# Or from login node:
#   srun -p gpu --gres=gpu:1 --cpus-per-task=4 --mem=64G --time=00:30:00 \
#     bash scripts/run_smoke_test.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

module purge
module load cuda 2>/dev/null || true

# shellcheck source=/dev/null
source "$PROJECT_ROOT/scripts/activate_env.sh"

python scripts/train_llama1b_ssa.py --smoke-test --from-scratch "$@"
