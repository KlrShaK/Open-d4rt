#!/usr/bin/env bash
#SBATCH --job-name=d4rt-qwen-v1
#SBATCH --account=es_schin
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --tmp=2G
#SBATCH --time=02:00:00
#SBATCH --exclude=eu-lo-g2-028
#SBATCH --output=logs/d4rt_agent/qwen_simple_%j.out

set -euo pipefail

REPO_ROOT=/cluster/work/igp_psr/spanwar/Open-d4rt
D4RT_PYTHON=/cluster/work/igp_psr/spanwar/envs/d4rt/bin/python
DEPS_DIR="$REPO_ROOT/.cache/qwen3vl_python"
LOCAL_DEPS="$TMPDIR/qwen3vl_python"
HF_HOME=/cluster/work/igp_psr/spanwar/hf_cache
MODEL_CACHE="$HF_HOME/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots"
PHASE="${PHASE:-qwen}"

cd "$REPO_ROOT"
mkdir -p logs/d4rt_agent "$DEPS_DIR"
mkdir -p "$LOCAL_DEPS"
cp -a "$DEPS_DIR/." "$LOCAL_DEPS/"
export HF_HOME
export PYTHONPATH="$LOCAL_DEPS${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

MODEL_DIR="$(find "$MODEL_CACHE" -mindepth 1 -maxdepth 1 -type d -print -quit)"
if [[ -z "$MODEL_DIR" ]]; then
  echo "Qwen snapshot is missing. Run scripts/prepare_qwen3vl_login.sh on a login node." >&2
  exit 2
fi
export MODEL_DIR

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

"$D4RT_PYTHON" - <<'PY'
import os
from transformers import AutoConfig
import accelerate, transformers

assert tuple(map(int, transformers.__version__.split(".")[:2])) >= (4, 57)
assert os.environ["HF_HUB_OFFLINE"] == "1"
AutoConfig.from_pretrained(os.environ["MODEL_DIR"], local_files_only=True)
print(f"Offline preflight passed: transformers={transformers.__version__}, accelerate={accelerate.__version__}")
PY

if [[ "$PHASE" == "direct" ]]; then
  "$D4RT_PYTHON" -m d4rt_agent.simple_v1 \
    --phase direct \
    --model "$MODEL_DIR" \
    --output d4rt_agent/results/basketball_6/phase5_direct_vlm.json
else
  "$D4RT_PYTHON" -m d4rt_agent.simple_v1 \
    --phase qwen \
    --model "$MODEL_DIR" \
    --output d4rt_agent/results/basketball_6/phase2_qwen.json
  "$D4RT_PYTHON" -m d4rt_agent.simple_v1 --phase aggregate
fi
