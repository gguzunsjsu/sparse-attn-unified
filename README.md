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
git clone <your-repo-url> ~/sparse-attn-unified
cd ~/sparse-attn-unified
```

### 2. Create conda environment (once)

```bash
module load python3 cuda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
source $HOME/miniconda3/etc/profile.d/conda.sh

conda create -n ssa-h100 python=3.10 -y
conda activate ssa-h100

pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e .
pip install transformers datasets accelerate einops pyyaml tqdm pytest
```

> **Note:** SJSU HPC uses `module load cuda` — check available versions with `module avail cuda`. Match your PyTorch CUDA wheel accordingly.

### 3. HuggingFace access

Llama 3.2 1B is gated:

```bash
huggingface-cli login
export HF_HOME=$HOME/.cache/huggingface
```

### 4. Smoke test (interactive H100)

```bash
srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=128G --time=01:00:00 --pty /bin/bash
module load python3 cuda
conda activate ssa-h100
cd ~/sparse-attn-unified

python scripts/train_llama1b_ssa.py --smoke-test --from-scratch
pytest tests/ -v
```

### 5. Submit training job

```bash
mkdir -p logs
# Edit mail-user in scripts/slurm/train_llama1b_h100.slurm
sbatch scripts/slurm/train_llama1b_h100.slurm
```

## Memory Budget (1× H100 80 GB)

| Setting | Value | Rationale |
|---------|-------|-----------|
| Model | Llama 3.2 1B (~2 GB bf16) | SSA official baseline |
| Seq length | 4096 | Fits dual-stream + alignment |
| Batch size | 2 | Conservative for SA masks |
| Grad accum | 8 | Effective batch = 16 |
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
