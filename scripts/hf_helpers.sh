#!/usr/bin/env bash
# HuggingFace CLI helpers (supports new `hf` and legacy `huggingface-cli`).

hf_whoami() {
    if command -v hf &>/dev/null; then
        hf auth whoami "$@" 2>/dev/null
        return $?
    fi
    if command -v huggingface-cli &>/dev/null; then
        huggingface-cli whoami "$@" 2>/dev/null
        return $?
    fi
    return 1
}

hf_login_hint() {
    cat <<'EOF'
  1. Accept license: https://huggingface.co/meta-llama/Llama-3.2-1B
  2. Create token:   https://huggingface.co/settings/tokens
  3. Login:          hf auth login
     OR:             export HF_TOKEN=hf_xxxx
  4. Verify:         hf auth whoami
EOF
}
