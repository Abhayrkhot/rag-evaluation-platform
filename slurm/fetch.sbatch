#!/usr/bin/env bash
#SBATCH --job-name=rag-fetch
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/%u/rag/logs/%x-%j.out
# Download model snapshots into the project cache:  sbatch slurm/fetch.sbatch Qwen/Qwen3-8B [...]
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export RAG_ONLINE=1
source slurm/env.sh
in_image python3 - "$@" <<'PY'
import sys, os
from huggingface_hub import snapshot_download
for repo in sys.argv[1:]:
    p = snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "tokenizer*"])
    missing = [f for f in os.listdir(p) if not os.path.exists(os.path.join(p, f))]
    print("ok", repo, p, "missing:", missing)
PY
