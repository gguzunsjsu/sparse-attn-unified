#!/usr/bin/env bash
# Run on the LOGIN NODE (has internet) before submitting GPU jobs.
#
# Usage:
#   cd /scratch/rnd-guzun/sparse-attn-unified
#   source scripts/activate_env.sh
#   huggingface-cli login          # once, for gated Llama 3.2 1B
#   bash scripts/prefetch_offline_assets.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# shellcheck source=/dev/null
source "$PROJECT_ROOT/scripts/activate_env.sh"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
# Keep HF hub cache on home; also store explicit local copies under project cache/
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PROJECT_ROOT/cache/huggingface/datasets}"

python scripts/prefetch_offline_assets.py \
  --project-root "$PROJECT_ROOT" \
  --seq-length 4096 \
  --num-sequences 170000 \
  "$@"

echo ""
echo "Done. Submit training with:"
echo "  cd $PROJECT_ROOT"
echo "  sbatch scripts/slurm/train_llama1b_h100.slurm"
