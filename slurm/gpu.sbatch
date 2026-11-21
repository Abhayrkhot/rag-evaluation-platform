#!/usr/bin/env bash
#SBATCH --job-name=rag-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/%u/rag/logs/%x-%j.out
# Run one stage script on a GPU node after the unit tests pass there:
#   sbatch slurm/gpu.sbatch scripts/m1_retrieval.sh
#   sbatch --gres=gpu:h200:2 slurm/gpu.sbatch scripts/m2_generate.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/env.sh
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
in_image python3 -m pytest -q -p no:cacheprovider tests
export -f in_image wait_http
bash "$@"
