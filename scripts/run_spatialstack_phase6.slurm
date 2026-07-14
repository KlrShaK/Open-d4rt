#!/bin/bash
# Phase 6: native SpatialStack metric-motion baseline on basketball_6.
# Uses SpatialStack's defaults, including 32 uniformly sampled frames.

#SBATCH --job-name=d4rt-spatialstack-p6
#SBATCH --gpus=nvidia_a100_80gb_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --time=01:00:00
#SBATCH --output=/cluster/work/igp_psr/spanwar/Open-d4rt/logs/d4rt_agent/spatialstack_phase6_%j.out

set -euo pipefail

OPEN_D4RT=/cluster/work/igp_psr/spanwar/Open-d4rt
SPATIALSTACK=/cluster/work/igp_psr/spanwar/SpatialStack
MODEL=/cluster/work/igp_psr/spanwar/datasets/models/SpatialStack-Qwen3.5-4B
VIDEO="$OPEN_D4RT/demo/pstudio_mini/basketball_6/assets/input_video.mp4"
PROMPTS="$OPEN_D4RT/d4rt_agent/spatialstack_phase6_prompts.json"
RESULTS="$OPEN_D4RT/d4rt_agent/results/basketball_6"
RAW_RESULT="$RESULTS/phase6_spatialstack_raw.json"
SCORED_RESULT="$RESULTS/phase6_spatialstack.json"
SCENES_DIR="${TMPDIR:-/tmp/$USER}/spatialstack-phase6-$SLURM_JOB_ID/scenes"

source /cluster/work/igp_psr/spanwar/envs/spatialstack-qwen35/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export DATASETS_OFFLINE=1
export NCCL_NVLS_ENABLE=0
export TRITON_CACHE_DIR="${TMPDIR:-/tmp/$USER}/spatialstack-triton/$SLURM_JOB_ID/0"
mkdir -p "$OPEN_D4RT/logs/d4rt_agent" "$RESULTS" "$TRITON_CACHE_DIR" "$SCENES_DIR"

echo "Phase 6 prompts:"
python -m json.tool "$PROMPTS"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

python "$OPEN_D4RT/d4rt_agent/spatialstack_phase6.py" prepare \
  --video "$VIDEO" \
  --scenes-dir "$SCENES_DIR"

cd "$SPATIALSTACK"
python -u scripts/inference/infer_batch.py \
  --model-path "$MODEL" \
  --prompts-file "$PROMPTS" \
  --scenes-dir "$SCENES_DIR" \
  --output-file "$RAW_RESULT" \
  --add-frame-index \
  --disable-thinking

cd "$OPEN_D4RT"
python d4rt_agent/spatialstack_phase6.py score \
  --raw "$RAW_RESULT" \
  --output "$SCORED_RESULT"
