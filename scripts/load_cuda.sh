#!/usr/bin/env bash
# Load a CUDA module on SJSU / Lmod HPC systems.
# SJSU does not provide a bare "cuda" module — use cuda/VERSION.
#
# Usage:
#   source scripts/load_cuda.sh
#   load_cuda_module

load_cuda_module() {
    if [[ -n "${CUDA_MODULE_LOADED:-}" ]]; then
        return 0
    fi

    # User override: export CUDA_MODULE=cuda/12.1 before submitting job
    if [[ -n "${CUDA_MODULE:-}" ]]; then
        if module load "$CUDA_MODULE"; then
            echo "Loaded CUDA module: $CUDA_MODULE"
            export CUDA_MODULE_LOADED=1
            return 0
        fi
        echo "ERROR: Failed to load CUDA_MODULE=$CUDA_MODULE"
        return 1
    fi

    local candidates=(
        cuda/12.4
        cuda/12.3
        cuda/12.2
        cuda/12.1
        cuda/12.0
        cuda/11.8
        cuda/11.7
        cuda
    )

    local mod
    for mod in "${candidates[@]}"; do
        if module load "$mod" 2>/dev/null; then
            echo "Loaded CUDA module: $mod"
            export CUDA_MODULE_LOADED=1
            return 0
        fi
    done

    # Stale Lmod cache on some nodes
    if module --ignore_cache load cuda/12.1 2>/dev/null; then
        echo "Loaded CUDA module: cuda/12.1 (ignore_cache)"
        export CUDA_MODULE_LOADED=1
        return 0
    fi

    echo "WARN: No CUDA module found."
    echo "  Run on a GPU node: module avail cuda 2>&1 | head -30"
    echo "  Then submit with:  CUDA_MODULE=cuda/X.Y sbatch scripts/slurm/train_llama1b_h100.slurm"
    echo "  Continuing — PyTorch cu121 wheels may still detect the GPU via driver."
    return 0
}

# Allow: bash scripts/load_cuda.sh  (prints available modules hint)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Available CUDA modules (if any):"
    module avail cuda 2>&1 | head -30 || true
    echo ""
    load_cuda_module
fi
