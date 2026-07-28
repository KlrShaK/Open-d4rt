#!/usr/bin/env bash
#SBATCH --job-name=d4rt-demo-multi
#SBATCH --partition=gpupr.4h
#SBATCH --account=es_schin
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --time=00:40:00
#SBATCH --array=0-11
#SBATCH --exclude=eu-lo-g2-028
#SBATCH --output=logs/demo/demo_multi_%A_%a.out

set -euo pipefail

REPO_ROOT=/cluster/work/igp_psr/spanwar/Open-d4rt
cd "$REPO_ROOT"

# Activate the d4rt Python venv (Python 3.10 + Torch 2.6)
source /cluster/work/igp_psr/spanwar/envs/d4rt/bin/activate

LIST="${LIST:-scripts/eval_worldtrack/demo_build_list.txt}"
mapfile -t LINES < "$LIST"
LINE="${LINES[$SLURM_ARRAY_TASK_ID]}"
DATASET="$(printf '%s' "$LINE" | cut -f1)"
SCENE="$(printf '%s' "$LINE" | cut -f2)"

echo "Building Viser demo: dataset=$DATASET scene=$SCENE"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

# One demo package per scene under demo/<dataset>/<scene>/
DEMO_CASE="${DATASET}/${SCENE}.npz" \
OUTPUT_DIR="demo/${DATASET}/${SCENE}" \
NUM_FRAMES=64 \
bash run_build_worldtrack_demo.sh
