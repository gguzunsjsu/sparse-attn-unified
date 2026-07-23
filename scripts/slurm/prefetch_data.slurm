#!/bin/bash
#SBATCH --job-name=prefetch-data
#SBATCH --output=logs/prefetch-data-%j.out
#SBATCH --error=logs/prefetch-data-%j.err
#SBATCH --partition=nsfql
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gheorghi.guzun@sjsu.edu

# Tokenize training data on a compute node with more RAM than the login node.
# Requires outbound network on the allocated node (FineWeb streaming).
#
# Usage (model must already be cached on login node):
#   cd /scratch/rnd-guzun/sparse-attn-unified
#   mkdir -p logs
#   sbatch scripts/slurm/prefetch_data.slurm

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$HOME/sparse-attn-unified}}"
cd "$PROJECT_ROOT"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "Node: $(hostname)  Date: $(date)"

MODEL_DIR="$PROJECT_ROOT/cache/models/Llama-3.2-1B"
if [[ ! -d "$MODEL_DIR" ]]; then
    echo "ERROR: Model not cached: $MODEL_DIR"
    echo "Run on login node first: bash scripts/prefetch_offline_assets.sh --skip-data"
    exit 1
fi

# shellcheck source=/dev/null
source "$PROJECT_ROOT/scripts/activate_env.sh"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PROJECT_ROOT/cache/huggingface/datasets}"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false

python scripts/prefetch_offline_assets.py \
  --project-root "$PROJECT_ROOT" \
  --skip-model \
  --seq-length 4096 \
  --num-sequences 85000

echo "Prefetch data finished at $(date)"
