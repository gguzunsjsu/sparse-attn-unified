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
    if [[ -n "$user" && -d "/fs/atipa/home/$user" ]]; then
        echo "/fs/atipa/home/$user"
        return
    fi
    if [[ -n "$user" ]]; then
        getent passwd "$user" 2>/dev/null | cut -d: -f6
    fi
}

HPC_HOME="$(resolve_hpc_home)"
MAMBA_ROOT="${MAMBA_ROOT:-${HPC_HOME}/micromamba}"

# Find micromamba/mamba/conda executables (SJSU may use miniforge or .local/share/mamba).
find_mamba_bin() {
    local home="${HPC_HOME:-$(resolve_hpc_home)}"
    local user="${USER:-}"
    local candidates=(
        "${MAMBA_ROOT}/bin/micromamba"
        "${home}/micromamba/bin/micromamba"
        "${home}/miniforge3/bin/micromamba"
        "${home}/miniforge3/bin/mamba"
        "${home}/.local/share/mamba/bin/micromamba"
        "/fs/atipa/home/${user}/.local/share/mamba/bin/micromamba"
    )
    local c
    for c in "${candidates[@]}"; do
        if [[ -x "$c" ]]; then
            echo "$c"
            return 0
        fi
    done
    if command -v micromamba &>/dev/null; then
        command -v micromamba
        return 0
    fi
    return 1
}

# Parse `micromamba env list` / `conda env list` for a named env prefix.
_env_prefix_from_tool() {
    local tool_bin="$1"
    local name="$2"
    local line prefix

    while IFS= read -r line; do
        # Match lines where first field is the env name (ignore header/separators)
        if [[ "$line" =~ ^[[:space:]]*${name}[[:space:]] ]]; then
            prefix="$(echo "$line" | awk '{print $NF}')"
            if [[ -n "$prefix" && -d "$prefix" ]]; then
                echo "$prefix"
                return 0
            fi
        fi
    done < <("$tool_bin" env list 2>/dev/null)

    return 1
}

# Return 0 if env exists.
env_exists() {
    local name="${1:-$ENV_NAME}"
    find_env_python "$name" >/dev/null 2>&1
}

# Print absolute path to env python, or return 1.
find_env_python() {
    local name="${1:-$ENV_NAME}"
    local home="${HPC_HOME:-$(resolve_hpc_home)}"
    local user="${USER:-}"
    local prefix py mamba_bin

    # 1. Ask micromamba/mamba where the env lives (handles .local/share/mamba, miniforge, etc.)
    mamba_bin="$(find_mamba_bin 2>/dev/null || true)"
    if [[ -n "$mamba_bin" ]]; then
        prefix="$(_env_prefix_from_tool "$mamba_bin" "$name" || true)"
        py="${prefix}/bin/python"
        if [[ -x "$py" ]]; then
            echo "$py"
            return 0
        fi
    fi

    # 2. Ask conda if available
    if [[ -f "${home}/miniconda3/etc/profile.d/conda.sh" ]]; then
        # shellcheck source=/dev/null
        source "${home}/miniconda3/etc/profile.d/conda.sh"
        prefix="$(_env_prefix_from_tool conda "$name" || true)"
        py="${prefix}/bin/python"
        if [[ -x "$py" ]]; then
            echo "$py"
            return 0
        fi
    fi
    if [[ -f "${home}/miniforge3/etc/profile.d/conda.sh" ]]; then
        # shellcheck source=/dev/null
        source "${home}/miniforge3/etc/profile.d/conda.sh"
        prefix="$(_env_prefix_from_tool conda "$name" || true)"
        py="${prefix}/bin/python"
        if [[ -x "$py" ]]; then
            echo "$py"
            return 0
        fi
    fi

    # 3. Filesystem fallbacks (common HPC layouts)
    local candidates=(
        "${home}/.local/share/mamba/envs/${name}/bin/python"
        "/fs/atipa/home/${user}/.local/share/mamba/envs/${name}/bin/python"
        "${home}/miniforge3/envs/${name}/bin/python"
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
    local py bindir
    py="$(find_env_python "$name")" || return 1
    bindir="$(dirname "$py")"
    export PATH="${bindir}:$PATH"
    unset PYTHONPATH
    export PYTHONNOUSERSITE=1
    hash -r 2>/dev/null || true
    return 0
}

# Activate env: PATH first, then optional shell hook.
activate_env() {
    local name="${1:-$ENV_NAME}"
    local mamba_bin

    if ! activate_env_by_path "$name"; then
        return 1
    fi

    mamba_bin="$(find_mamba_bin 2>/dev/null || true)"
    if [[ -n "$mamba_bin" ]]; then
        eval "$("$mamba_bin" shell hook -s bash)" 2>/dev/null || true
        micromamba activate "$name" 2>/dev/null || true
    elif [[ -f "${HPC_HOME}/miniforge3/etc/profile.d/conda.sh" ]]; then
        # shellcheck source=/dev/null
        source "${HPC_HOME}/miniforge3/etc/profile.d/conda.sh"
        conda activate "$name" 2>/dev/null || true
    elif [[ -f "${HPC_HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
        # shellcheck source=/dev/null
        source "${HPC_HOME}/miniconda3/etc/profile.d/conda.sh"
        conda activate "$name" 2>/dev/null || true
    fi
    return 0
}

print_hpc_diagnostics() {
    local py mamba_bin
    echo "=== HPC diagnostics ==="
    echo "USER:       ${USER:-unset}"
    echo "HOME:       ${HOME:-unset}"
    echo "HPC_HOME:   ${HPC_HOME:-unset}"
    echo "MAMBA_ROOT: ${MAMBA_ROOT:-unset}"
    echo "PWD:        $(pwd)"
    echo ""
    echo "Looking for env: ${ENV_NAME}"

    mamba_bin="$(find_mamba_bin 2>/dev/null || true)"
    if [[ -n "$mamba_bin" ]]; then
        echo "micromamba: $mamba_bin"
    fi

    if py="$(find_env_python "$ENV_NAME" 2>/dev/null)"; then
        echo "Found python: $py"
    else
        echo "Env python NOT found via discovery."
        echo "Known fallbacks include:"
        echo "  ${HPC_HOME}/.local/share/mamba/envs/${ENV_NAME}/bin/python"
        echo "  ${HPC_HOME}/miniforge3/envs/${ENV_NAME}/bin/python"
        echo "  ${MAMBA_ROOT}/envs/${ENV_NAME}/bin/python"
    fi

    if [[ -n "$mamba_bin" ]]; then
        echo ""
        echo "micromamba env list:"
        "$mamba_bin" env list 2>/dev/null || true
    fi
}
