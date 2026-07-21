#!/bin/bash
#SBATCH --job-name=ssa-llama1b
#SBATCH --output=logs/ssa-llama1b-%j.out
#SBATCH --error=logs/ssa-llama1b-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=YOUR_SJSU_ID@sjsu.edu

# SJSU CoE HPC — single H100 (nodes g2, g6), 256 GB RAM
# Docs: https://www.sjsu.edu/cmpe/resources/hpc.php
#
# Request H100 explicitly (optional — omit to accept any GPU):
#   #SBATCH --constraint=h100
#
# Usage:
#   mkdir -p logs
#   sbatch scripts/slurm/train_llama1b_h100.slurm

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/sparse-attn-unified}"
cd "$PROJECT_ROOT"

module purge
module load python3
module load cuda

# Conda env (create once on login node — see README)
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate ssa-h100

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

mkdir -p logs outputs

python scripts/train_llama1b_ssa.py \
  --sparse-backend socket \
  --seq-length 4096 \
  --batch-size 2 \
  --grad-accum 8 \
  --max-steps 10000 \
  --lr 3e-4 \
  --base-model meta-llama/Llama-3.2-1B \
  --output-dir "$PROJECT_ROOT/outputs/llama1b-ssa-socket-${SLURM_JOB_ID}"

echo "Job finished at $(date)"
