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
# Prerequisite (login node, once):
#   bash scripts/prefetch_offline_assets.sh
# That caches the model + parquet shards. This job builds JSONL, tokenizes, trains.
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
export PYTHONPATH="$PROJECT_ROOT"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PROJECT_ROOT/cache/huggingface/datasets}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

SEQ_LENGTH=2048
TOKENIZE_SEQ=4096
NUM_SEQUENCES=85000
MODEL_DIR="$PROJECT_ROOT/cache/models/Llama-3.2-1B"
PARQUET_DIR="$PROJECT_ROOT/cache/datasets/fineweb_parquet"
RAW_DATA="$PROJECT_ROOT/cache/datasets/fineweb_raw.jsonl"
# Tokenize at 4096 once; training slices to SEQ_LENGTH (2048).
DATA_BIN="$PROJECT_ROOT/cache/data/train_${TOKENIZE_SEQ}.bin"
MAX_STEPS=2000

if [[ ! -d "$MODEL_DIR" ]]; then
    echo "ERROR: Model not prefetched: $MODEL_DIR"
    echo "Run on login node: bash scripts/prefetch_offline_assets.sh"
    exit 1
fi
if [[ -z "$(find "$PARQUET_DIR" -name '*.parquet' -print -quit 2>/dev/null)" ]]; then
    echo "ERROR: Parquet shards not cached: $PARQUET_DIR"
    echo "Run on login node: bash scripts/prefetch_offline_assets.sh"
    exit 1
fi
echo "Offline assets OK: model + parquet shards"

# Activate env via shared helpers (handles empty $HOME on compute nodes)
# shellcheck source=/dev/null
source "$PROJECT_ROOT/scripts/activate_env.sh"

mkdir -p logs outputs cache/data cache/datasets

echo "Ensuring raw JSONL has enough docs (seq=${TOKENIZE_SEQ} for tokenization)"
python scripts/prefetch_offline_assets.py \
  --project-root "$PROJECT_ROOT" \
  --skip-model \
  --skip-raw-cache \
  --build-jsonl-only \
  --offline \
  --seq-length "$TOKENIZE_SEQ" \
  --num-sequences "$NUM_SEQUENCES"

if [[ ! -f "$DATA_BIN" ]]; then
    echo "Tokenizing offline on compute node -> $DATA_BIN (seq=${TOKENIZE_SEQ})"
    python scripts/prefetch_offline_assets.py \
      --project-root "$PROJECT_ROOT" \
      --skip-model \
      --skip-raw-cache \
      --tokenize-only \
      --offline \
      --seq-length "$TOKENIZE_SEQ" \
      --num-sequences "$NUM_SEQUENCES" \
      --data-bin "$DATA_BIN"
    echo "Tokenization complete."
else
    echo "Validating cached tokenized data: $DATA_BIN"
    python scripts/prefetch_offline_assets.py \
      --project-root "$PROJECT_ROOT" \
      --validate-only \
      --offline \
      --seq-length "$TOKENIZE_SEQ" \
      --num-sequences "$NUM_SEQUENCES" \
      --data-bin "$DATA_BIN"
fi

python -m pip install -e "$PROJECT_ROOT" --no-deps -q
echo "sparse_attn import path:"
python -c "import sparse_attn.models.llama_ssa as m; print(m.__file__)"

python scripts/train_llama1b_ssa.py \
  --sparse-backend socket \
  --seq-length "$SEQ_LENGTH" \
  --batch-size 1 \
  --grad-accum 16 \
  --max-steps "$MAX_STEPS" \
  --lr 3e-4 \
  --checkpoint-fa \
  --socket-train-l 8 \
  --local-model-dir "$MODEL_DIR" \
  --local-data-bin "$DATA_BIN" \
  --output-dir "$PROJECT_ROOT/outputs/llama1b-ssa-socket-${SLURM_JOB_ID}"

echo "Job finished at $(date)"
