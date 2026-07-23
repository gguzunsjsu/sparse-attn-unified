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
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PROJECT_ROOT/cache/huggingface/datasets}"

# Auth check before multi-GB download
if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  if ! huggingface-cli whoami &>/dev/null; then
    echo "ERROR: Not logged in to HuggingFace."
    echo ""
    echo "  1. Accept license: https://huggingface.co/meta-llama/Llama-3.2-1B"
    echo "  2. Create token:   https://huggingface.co/settings/tokens"
    echo "  3. Login:          huggingface-cli login"
    echo "     OR:             export HF_TOKEN=hf_xxxx"
    echo "  4. Verify:         huggingface-cli whoami"
    exit 1
  fi
  echo "HuggingFace: $(huggingface-cli whoami 2>/dev/null | head -1)"
fi

python scripts/prefetch_offline_assets.py \
  --project-root "$PROJECT_ROOT" \
  --seq-length 4096 \
  --num-sequences 85000 \
  "$@"

echo ""
echo "Done. Submit training with:"
echo "  cd $PROJECT_ROOT"
echo "  sbatch scripts/slurm/train_llama1b_h100.slurm"
