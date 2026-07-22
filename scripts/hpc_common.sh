#!/usr/bin/env bash
# Shared helpers for SJSU HPC scripts.
# Source from other scripts: source "$(dirname "$0")/hpc_common.sh"

ENV_NAME="${ENV_NAME:-ssa-h100}"

# Compute nodes sometimes have empty/wrong $HOME ("I have no name!" sessions).
resolve_hpc_home() {
    if [[ -n "${HOME:-}" && "$HOME" != "/" && -d "$HOME" ]]; then
        echo "$HOME"
        return
    fi
    local user
    user="$(id -un 2>/dev/null || true)"
    if [[ -n "$user" && -d "/home/$user" ]]; then
        echo "/home/$user"
        return
    fi
    if [[ -n "$user" ]]; then
        getent passwd "$user" 2>/dev/null | cut -d: -f6
    fi
}

HPC_HOME="$(resolve_hpc_home)"
MAMBA_ROOT="${MAMBA_ROOT:-${HPC_HOME}/micromamba}"

# Return 0 if env exists (by filesystem, not grep on env list output).
env_exists() {
    local name="${1:-$ENV_NAME}"
    find_env_python "$name" >/dev/null 2>&1
}

# Print absolute path to env python, or return 1.
find_env_python() {
    local name="${1:-$ENV_NAME}"
    local home="${HPC_HOME:-$(resolve_hpc_home)}"
    local candidates=(
        "${MAMBA_ROOT}/envs/${name}/bin/python"
        "${home}/micromamba/envs/${name}/bin/python"
        "${home}/miniconda3/envs/${name}/bin/python"
        "${home}/anaconda3/envs/${name}/bin/python"
    )
    local p
    for p in "${candidates[@]}"; do
        if [[ -x "$p" ]]; then
            echo "$p"
            return 0
        fi
    done
    return 1
}

# Activate env by PATH (most reliable on GPU nodes).
activate_env_by_path() {
    local name="${1:-$ENV_NAME}"
    local py
    py="$(find_env_python "$name")" || return 1
    local bindir
    bindir="$(dirname "$py")"
    export PATH="${bindir}:$PATH"
    unset PYTHONPATH
    export PYTHONNOUSERSITE=1
    hash -r 2>/dev/null || true
    return 0
}

# Try micromamba/conda activate; fall back to PATH.
activate_env() {
    local name="${1:-$ENV_NAME}"

    if activate_env_by_path "$name"; then
        # Optional: also run shell hook if available (for conda env vars)
        if [[ -x "${MAMBA_ROOT}/bin/micromamba" ]]; then
            eval "$("${MAMBA_ROOT}/bin/micromamba" shell hook -s bash)" 2>/dev/null || true
            micromamba activate "$name" 2>/dev/null || true
        elif [[ -f "${HPC_HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
            # shellcheck source=/dev/null
            source "${HPC_HOME}/miniconda3/etc/profile.d/conda.sh"
            conda activate "$name" 2>/dev/null || true
        fi
        return 0
    fi
    return 1
}

print_hpc_diagnostics() {
    echo "=== HPC diagnostics ==="
    echo "USER:       ${USER:-unset}"
    echo "HOME:       ${HOME:-unset}"
    echo "HPC_HOME:   ${HPC_HOME:-unset}"
    echo "MAMBA_ROOT: ${MAMBA_ROOT:-unset}"
    echo "PWD:        $(pwd)"
    echo ""
    echo "Looking for env: ${ENV_NAME}"
    local py
    if py="$(find_env_python "$ENV_NAME" 2>/dev/null)"; then
        echo "Found python: $py"
    else
        echo "Env python NOT found. Checked:"
        echo "  ${MAMBA_ROOT}/envs/${ENV_NAME}/bin/python"
        echo "  ${HPC_HOME}/micromamba/envs/${ENV_NAME}/bin/python"
        echo "  ${HPC_HOME}/miniconda3/envs/${ENV_NAME}/bin/python"
    fi
    if [[ -x "${MAMBA_ROOT}/bin/micromamba" ]]; then
        echo ""
        echo "micromamba env list:"
        "${MAMBA_ROOT}/bin/micromamba" env list 2>/dev/null || true
    fi
    if [[ -f "${HPC_HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
        echo ""
        echo "conda env list:"
        # shellcheck source=/dev/null
        source "${HPC_HOME}/miniconda3/etc/profile.d/conda.sh"
        conda env list 2>/dev/null || true
    fi
}
