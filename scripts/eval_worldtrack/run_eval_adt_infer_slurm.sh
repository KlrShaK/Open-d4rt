#!/usr/bin/env bash
#SBATCH --job-name=d4rt-adt-infer
#SBATCH --partition=gpupr.4h
#SBATCH --account=es_schin
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --time=00:30:00
#SBATCH --output=logs/eval/adt_infer_%j.out

set -euo pipefail

REPO_ROOT=/cluster/work/igp_psr/spanwar/Open-d4rt
cd "$REPO_ROOT"

# Activate the d4rt Python venv (Python 3.10 + Torch 2.6)
source /cluster/work/igp_psr/spanwar/envs/d4rt/bin/activate

nvidia-smi --query-gpu=name,memory.total --format=csv || true

# Run D4RT inference/eval on 3 WorldTrack adt_mini scenes (README Evaluation section)
SUBSETS=adt_mini \
LIMIT_SEQS=3 \
NUM_FRAMES=64 \
QUERY_CHUNK_SIZE=4096 \
OUTPUT_DIR=tmp/eval_adt_infer \
bash run_eval_worldtrack.sh
