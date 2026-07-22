#!/usr/bin/env bash
# Activate the ssa-h100 env on any node (login or GPU).
# Usage: source scripts/activate_env.sh

MAMBA_ROOT="${MAMBA_ROOT:-$HOME/micromamba}"
ENV_NAME="${ENV_NAME:-ssa-h100}"

unset PYTHONPATH
export PYTHONNOUSERSITE=1

_activate_micromamba() {
    eval "$("$MAMBA_ROOT/bin/micromamba" shell hook -s bash)"
    micromamba activate "$ENV_NAME"
}

_activate_conda() {
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
}

if [[ -x "$MAMBA_ROOT/bin/micromamba" ]]; then
    if "$MAMBA_ROOT/bin/micromamba" env list | grep -q "^${ENV_NAME} "; then
        _activate_micromamba
    elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]] \
         && conda env list 2>/dev/null | grep -q "^${ENV_NAME} "; then
        _activate_conda
    else
        echo "ERROR: env '$ENV_NAME' not found."
        echo "Run on login node: bash scripts/setup_hpc.sh"
        return 1 2>/dev/null || exit 1
    fi
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    _activate_conda
else
    echo "ERROR: No conda/micromamba at $MAMBA_ROOT or ~/miniconda3"
    echo "Run on login node: bash scripts/setup_hpc.sh"
    return 1 2>/dev/null || exit 1
fi

if ! python -c "import torch" 2>/dev/null; then
    echo "ERROR: torch not installed in env '$ENV_NAME'"
    echo "Fix: bash scripts/install_torch.sh"
    return 1 2>/dev/null || exit 1
fi

echo "Active env: $ENV_NAME"
echo "Python:     $(which python)"
python -c "import torch; print('Torch:', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
