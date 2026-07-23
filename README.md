# Sparse Attention Unified

SSA (Sparse Sparse Attention) training framework with pluggable **SOCKET** and **Soft-SAAP** sparse backends, targeting **Llama 3.2 1B** on a **single H100** at [SJSU CoE HPC](https://www.sjsu.edu/cmpe/resources/hpc.php).

## Architecture

```
Input Tokens
     │
     ├─► Full Attention (FA) ──► O_FA ──┐
     │                                   ├─► L_align = Σ D(O_FA, O_SA)
     └─► Sparse Backend (SA) ──► O_SA ──┘
              ├── socket  (soft-LSH)
              └── saap    (Gumbel-Softmax clusters)
```

During training, both streams run every layer; the primary output is randomly FA or SA (`p=0.5`). At inference, only the sparse backend runs.

## H100 / SJSU HPC Quick Start

### 1. Clone on login node

```bash
ssh YOUR_ID@coe-hpc.sjsu.edu
git clone https://github.com/gguzunsjsu/sparse-attn-unified.git ~/sparse-attn-unified
cd ~/sparse-attn-unified
```

### 2. Create environment (once)

SJSU login nodes use **GLIBC 2.17**. Do **not** use `Miniconda3-latest` (requires GLIBC ≥ 2.28) and do **not** run `pip install` with `module load python3` (system pip is read-only and will fail with `PermissionError`).

**Recommended — micromamba (works on GLIBC 2.17):**

```bash
cd ~/sparse-attn-unified
bash scripts/setup_hpc.sh
```

This creates env `ssa-h100` and installs PyTorch + project deps. On SJSU HPC the env may live under `~/.local/share/mamba/envs/` or `~/miniforge3/envs/` rather than `~/micromamba/` — the scripts auto-detect this.

Add to `~/.bashrc` (once, after setup — use whichever path exists on your account):

```bash
eval "$($HOME/miniforge3/bin/micromamba shell hook -s bash)" 2>/dev/null || true
eval "$($HOME/.local/share/mamba/bin/micromamba shell hook -s bash)" 2>/dev/null || true
eval "$($HOME/micromamba/bin/micromamba shell hook -s bash)" 2>/dev/null || true
```

**Alternative — legacy Miniconda 4.12.0:**

```bash
bash scripts/setup_hpc_legacy.sh
```

<details>
<summary>Troubleshooting HPC setup</summary>

| Error | Cause | Fix |
|-------|-------|-----|
| `Installer requires GLIBC >=2.28, but system has 2.17` | Latest Miniconda too new for login node OS | Use `bash scripts/setup_hpc.sh` (micromamba) or `setup_hpc_legacy.sh` |
| `PermissionError: ... /opt/ohpc/.../site-packages/...` | Using system `pip` from `module load python3` | Never install with system pip; activate `ssa-h100` first, use `python -m pip` |
| `Defaulting to user installation because normal site-packages is not writeable` | Same as above — wrong Python | Run `which python` — must point to `~/micromamba/envs/ssa-h100/` or `~/miniconda3/envs/ssa-h100/` |
| `ModuleNotFoundError: No module named 'sparse_attn'` | Project not installed in env | `bash scripts/install_project_deps.sh` |
| `CUDA out of memory` during smoke test | Smoke test used seq=4096 + dual SSA streams | `git pull` and rerun `bash scripts/run_smoke_test.sh` (uses seq=512, batch=1) |
| `CUDA out of memory` during training | SSA runs FA + SA each layer | Use batch=1; ensure latest code (fixed sparse_attention memory bug) |
| `ERROR: torch not installed in env 'ssa-h100'` | Env created but PyTorch install failed/skipped | `bash scripts/install_torch.sh` |
| `ModuleNotFoundError: No module named 'torch'` | Env not activated on GPU node (fresh `srun` shell) | `source scripts/activate_env.sh` then retry, OR use `bash scripts/run_smoke_test.sh` |
| `401 Unauthorized` / `GatedRepoError` on prefetch | Not logged in or license not accepted | Accept license at huggingface.co/meta-llama/Llama-3.2-1B, then `hf auth login` |
| `Network is unreachable` / HuggingFace download on GPU node | GPU nodes have no internet | Run `bash scripts/prefetch_offline_assets.sh` on **login node** first |
| `module(s) are unknown: "cuda"` | SJSU uses versioned modules (`cuda/12.1`), not bare `cuda` | Run `module avail cuda`; then `CUDA_MODULE=cuda/X.Y sbatch ...` |
| `torch.cuda.is_available()` is False on GPU node | CUDA module not loaded | `bash scripts/load_cuda.sh` or set `CUDA_MODULE=cuda/12.1` |
| `module avail` shows no cuda 12.x | Older CUDA module | Run `module avail cuda`; if only 11.x is available, reinstall torch for cu118: `pip install torch --index-url https://download.pytorch.org/whl/cu118` |

Verify your env:

```bash
micromamba activate ssa-h100   # or: conda activate ssa-h100
which python                   # should NOT be /opt/ohpc/...
which pip
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

</details>

### 3. HuggingFace access

Llama 3.2 1B is gated. **Activate your env first**, then use the `hf` CLI (`pip install huggingface_hub` provides it).

```bash
micromamba activate ssa-h100   # or: conda activate ssa-h100

python -m pip install huggingface_hub

# 1. Accept Meta Llama license in browser (required once per HF account):
#    https://huggingface.co/meta-llama/Llama-3.2-1B  → "Agree and access repository"

# 2. Create a READ token: https://huggingface.co/settings/tokens

# 3. Login on login node (pick ONE):
hf auth login
# OR paste token non-interactively:
# export HF_TOKEN=hf_xxxxxxxx

export HF_HOME=$HOME/.cache/huggingface

# 4. Verify before prefetch:
hf auth whoami
```

### 3b. Prefetch assets for offline GPU nodes (required)

**GPU nodes have no internet.** On the **login node**, cache the model and download FineWeb parquet shards (needs internet). JSONL build and tokenization run offline on the **GPU compute node** inside the SLURM job:

```bash
cd /scratch/rnd-guzun/sparse-attn-unified
source scripts/activate_env.sh
hf auth login                    # once, for gated Llama 3.2 1B
bash scripts/prefetch_offline_assets.sh
```

This creates (under your project on scratch):

| Path | Contents |
|------|----------|
| `cache/models/Llama-3.2-1B/` | Full model + tokenizer (~2.5 GB) |
| `cache/datasets/fineweb_parquet/` | ~12 parquet shards (login node download) |
| `cache/datasets/fineweb_raw.jsonl` | Built offline on GPU node (~96k docs for 40k seq @ 2048) |
| `cache/data/train_2048.bin` | Built offline on GPU node (~0.33 GB) |

The SLURM job builds JSONL (if needed), tokenizes, then trains. Re-runs skip steps when outputs already exist.

Verify after prefetch:

```bash
ls -lh cache/models/Llama-3.2-1B/config.json
ls -lh cache/datasets/fineweb_parquet/*.parquet 2>/dev/null | head
```

If you still get `command not found`, check you're in the right env:

```bash
which hf    # should be in your conda env
hf auth whoami
```

### 4. Smoke test (interactive H100)

**First, diagnose your env (works on login or GPU node):**

```bash
bash scripts/doctor_hpc.sh
```

Each new GPU shell starts **without** your conda env. You must activate it every time.

**Recommended — use the wrapper script:**

```bash
srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=128G --time=01:00:00 --pty /bin/bash
cd /scratch/rnd-guzun/sparse-attn-unified   # or your clone path
bash scripts/install_project_deps.sh        # once per env
bash scripts/run_smoke_test.sh
```

**Manual steps:**

```bash
srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=128G --time=01:00:00 --pty /bin/bash

cd ~/sparse-attn-unified
source scripts/activate_env.sh    # required — loads micromamba + ssa-h100
module load cuda

python scripts/train_llama1b_ssa.py --smoke-test --from-scratch
pytest tests/ -v
```

**Without activate** (direct path — always works):

```bash
~/micromamba/envs/ssa-h100/bin/python scripts/train_llama1b_ssa.py --smoke-test --from-scratch
```

If `activate_env.sh` reports torch missing, install it:

```bash
bash scripts/install_torch.sh
bash scripts/run_smoke_test.sh
```

Or re-run full setup on the **login node**:

```bash
bash scripts/setup_hpc.sh
```

### 5. Submit training job

**Important:** run `sbatch` from the project directory (not `$HOME` unless the repo is cloned there).

```bash
cd /scratch/rnd-guzun/sparse-attn-unified   # your clone path
git pull
mkdir -p logs
# Edit mail-user in scripts/slurm/train_llama1b_h100.slurm
# Optional: set CUDA module if auto-detect fails (check with: module avail cuda)
# export CUDA_MODULE=cuda/12.1
sbatch scripts/slurm/train_llama1b_h100.slurm
```

Monitor: `tail -f logs/ssa-llama1b-<jobid>.out`

## Memory Budget (1× H100 80 GB)

| Setting | Value | Rationale |
|---------|-------|-----------|
| Model | Llama 3.2 1B (~2 GB bf16) | SSA official baseline |
| Seq length | 4096 | Fits dual-stream + alignment |
| Batch size | 1 | Safer for dual-stream SSA at seq 4096 |
| Grad accum | 16 | Effective batch = 16 |
| SOCKET `train_l` | 16 | Full `bucket_l=60` at inference |
| FA checkpointing | on | Cuts FA activation VRAM |

To push to 8192 context, drop batch to 1 and increase grad accum to 16.

## Configurations

Default H100 config: `configs/llama1b_h100_socket.yaml`

Switch sparse backend:

```bash
python scripts/train_llama1b_ssa.py --sparse-backend saap --from-scratch --smoke-test
python scripts/train_llama1b_ssa.py --sparse-backend socket --base-model meta-llama/Llama-3.2-1B
```

## Project Layout

```
sparse_attn/
├── backends/
│   ├── full.py           # FA stream (FlashAttention / SDPA)
│   ├── socket/           # Soft-LSH collision kernel
│   └── saap/             # Differentiable asymmetric partitioning
├── layers/ssa_attention.py
├── models/llama_ssa.py   # Llama 1B + SSA blocks
└── ssa/alignment.py
scripts/
├── setup_hpc.sh          # Recommended SJSU HPC setup (micromamba)
├── setup_hpc_legacy.sh   # Fallback for GLIBC 2.17
├── activate_env.sh       # Source on every new shell (login or GPU)
├── doctor_hpc.sh         # Diagnose env / torch / paths
├── install_torch.sh      # Install PyTorch if missing
├── install_project_deps.sh     # pip install -e . + numpy etc.
├── prefetch_offline_assets.sh  # Login node: download model + data
├── prefetch_offline_assets.py
├── run_smoke_test.sh     # GPU smoke test wrapper
├── train_llama1b_ssa.py
└── slurm/train_llama1b_h100.slurm
```

## Inference

After training, run with sparse-only path:

```python
model.eval()
out = model(input_ids, training=False, inference_mode="sparse")
```

For SOCKET production decode kernels, vendor `amarka8/SOCKET` CUDA/Triton backends separately — this repo implements the **training-time** PyTorch masker compatible with SSA alignment.

## References

- [SSA (ICML 2026)](https://arxiv.org/abs/2511.20102) — [zhenyi4/ssa](https://github.com/zhenyi4/ssa)
- [SOCKET](https://arxiv.org/abs/2602.06283) — [amarka8/SOCKET](https://github.com/amarka8/SOCKET)
- [SAAP](https://arxiv.org/abs/2502.08246) — Inference-time asymmetric partitioning

## License

MIT
