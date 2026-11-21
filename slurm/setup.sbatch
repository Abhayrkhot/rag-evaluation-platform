#!/usr/bin/env bash
#SBATCH --job-name=rag-setup
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/%u/rag/logs/%x-%j.out
# One-time (CPU node): venv on top of the vLLM image and model downloads into the scratch cache.
# Compute nodes reach PyPI and Hugging Face through the cluster proxy.
#   cd /scratch/$USER/rag/src/<sha> && sbatch slurm/setup.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export RAG_ONLINE=1
source slurm/env.sh
# Reuse model snapshots already downloaded by other projects instead of fetching them again.
for d in /scratch/$USER/icp/hf/hub/models--openai--gpt-oss-120b /scratch/$USER/infersched/hf-home/hub/models--Qwen--Qwen3-8B; do
  [ -e "$HF_HOME/hub/$(basename "$d")" ] || ln -s "$d" "$HF_HOME/hub/$(basename "$d")"
done
apptainer exec --bind "/scratch/$USER" "$RAG_SIF" bash -c '
  set -e
  [ -d "$RAG_ROOT/venv" ] || python3 -m venv --system-site-packages "$RAG_ROOT/venv"
  source "$RAG_ROOT/venv/bin/activate"
  pip freeze --exclude-editable | grep -v " @ " > /tmp/rag-base.txt
  pip install -q -c /tmp/rag-base.txt -r slurm/requirements-explorer.txt
  pip install -q --no-deps -e .
'
in_image python3 - <<'PY'
import importlib.metadata as m
print({p: m.version(p) for p in ["torch", "transformers", "vllm", "bm25s", "faiss-cpu", "langgraph", "fastapi"]})
from huggingface_hub import snapshot_download
pats = ["*.json", "*.safetensors", "*.txt", "*.model", "*.py", "*.jinja", "tokenizer*", "spiece.model"]
for repo in [
    "Qwen/Qwen3-8B", "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-Embedding-0.6B", "BAAI/bge-base-en-v1.5", "BAAI/bge-large-en-v1.5",
    "BAAI/bge-reranker-v2-m3", "Alibaba-NLP/gte-reranker-modernbert-base", "cross-encoder/ms-marco-MiniLM-L6-v2",
    "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
    "google/t5_xxl_true_nli_mixture",
    "openai/gpt-oss-120b",
]:
    p = snapshot_download(repo, allow_patterns=pats + (["*.bin"] if repo.startswith("google/") else []))
    print("ok", repo, p)
PY
