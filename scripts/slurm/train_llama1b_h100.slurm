#!/bin/bash
#SBATCH --job-name=ssa-llama1b
#SBATCH --output=logs/ssa-llama1b-%j.out
#SBATCH --error=logs/ssa-llama1b-%j.err
#SBATCH --partition=nsfql
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=gheorghi.guzun@sjsu.edu

# SJSU CoE HPC — single H100 (nodes g2, g6), 256 GB RAM
# Docs: https://www.sjsu.edu/cmpe/resources/hpc.php
#
# Request H100 explicitly (optional — omit to accept any GPU):
#   #SBATCH --constraint=h100
#
# Usage (run from the project directory):
#   cd /scratch/rnd-guzun/sparse-attn-unified
#   mkdir -p logs
#   sbatch scripts/slurm/train_llama1b_h100.slurm
#
# Or override path explicitly:
#   PROJECT_ROOT=/path/to/sparse-attn-unified sbatch scripts/slurm/train_llama1b_h100.slurm

set -euo pipefail

# SLURM copies this script to /var/spool/... — use submit directory, not script location.
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$HOME/sparse-attn-unified}}"
if [[ ! -d "$PROJECT_ROOT/sparse_attn" ]]; then
    echo "ERROR: PROJECT_ROOT invalid: $PROJECT_ROOT"
    echo "Submit from the project directory:"
    echo "  cd /scratch/rnd-guzun/sparse-attn-unified"
    echo "  sbatch scripts/slurm/train_llama1b_h100.slurm"
    exit 1
fi
cd "$PROJECT_ROOT"
echo "PROJECT_ROOT=$PROJECT_ROOT"

module purge
# shellcheck source=/dev/null
source "$PROJECT_ROOT/scripts/load_cuda.sh"
load_cuda_module

# Do NOT module load python3 — use micromamba/conda env only.
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

# Require offline assets (download on login node first)
MODEL_DIR="$PROJECT_ROOT/cache/models/Llama-3.2-1B"
DATA_BIN="$PROJECT_ROOT/cache/data/train_4096.bin"
if [[ ! -d "$MODEL_DIR" ]]; then
    echo "ERROR: Model not prefetched: $MODEL_DIR"
    echo "Run on login node: bash scripts/prefetch_offline_assets.sh"
    exit 1
fi
if [[ ! -f "$DATA_BIN" ]]; then
    echo "ERROR: Training data not prefetched: $DATA_BIN"
    echo "Run on login node: bash scripts/prefetch_offline_assets.sh"
    exit 1
fi
echo "Offline assets OK: model + data"

# Activate env via shared helpers (handles empty $HOME on compute nodes)
# shellcheck source=/dev/null
source "$PROJECT_ROOT/scripts/activate_env.sh"

mkdir -p logs outputs

python scripts/train_llama1b_ssa.py \
  --sparse-backend socket \
  --seq-length 4096 \
  --batch-size 1 \
  --grad-accum 16 \
  --max-steps 10000 \
  --lr 3e-4 \
  --local-model-dir "$MODEL_DIR" \
  --local-data-bin "$DATA_BIN" \
  --output-dir "$PROJECT_ROOT/outputs/llama1b-ssa-socket-${SLURM_JOB_ID}"

echo "Job finished at $(date)"
