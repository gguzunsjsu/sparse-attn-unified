#!/usr/bin/env bash
# Run on the LOGIN NODE (has internet) before submitting GPU jobs.
#
# Downloads the model + caches raw dataset text locally. Tokenization runs
# offline inside the SLURM training job (128 GB compute node).
#
# Usage:
#   cd /scratch/rnd-guzun/sparse-attn-unified
#   source scripts/activate_env.sh
#   hf auth login                  # once, for gated Llama 3.2 1B
#   bash scripts/prefetch_offline_assets.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# shellcheck source=/dev/null
source "$PROJECT_ROOT/scripts/activate_env.sh"

# shellcheck source=/dev/null
source "$PROJECT_ROOT/scripts/hf_helpers.sh"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PROJECT_ROOT/cache/huggingface/datasets}"
# Avoid hf-xet "Unable to parse string as hex hash value" on large Llama weights.
export HF_HUB_DISABLE_XET=1

# Auth check before multi-GB download
if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  if ! hf_whoami &>/dev/null; then
    echo "ERROR: Not logged in to HuggingFace."
    echo ""
    hf_login_hint
    exit 1
  fi
  echo "HuggingFace: $(hf_whoami 2>/dev/null | head -1)"
fi

python scripts/prefetch_offline_assets.py \
  --project-root "$PROJECT_ROOT" \
  --seq-length 4096 \
  --num-sequences 85000 \
  --skip-data \
  --cache-raw \
  "$@"

echo ""
echo "Done. Submit training (tokenizes offline on the GPU node if needed):"
echo "  cd $PROJECT_ROOT"
echo "  mkdir -p logs"
echo "  sbatch scripts/slurm/train_llama1b_h100.slurm"
