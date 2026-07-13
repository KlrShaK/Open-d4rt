#!/usr/bin/env bash
#SBATCH --job-name=d4rt-qwen-v1
#SBATCH --partition=gpupr.4h
#SBATCH --account=es_schin
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --exclude=eu-lo-g2-028
#SBATCH --output=logs/d4rt_agent/qwen_simple_%j.out

set -euo pipefail

REPO_ROOT=/cluster/work/igp_psr/spanwar/Open-d4rt
D4RT_PYTHON=/cluster/work/igp_psr/spanwar/envs/d4rt/bin/python
DEPS_DIR="$REPO_ROOT/.cache/qwen3vl_python"
HF_HOME=/cluster/work/igp_psr/spanwar/hf_cache

cd "$REPO_ROOT"
mkdir -p logs/d4rt_agent "$DEPS_DIR"
export HF_HOME
export PYTHONPATH="$DEPS_DIR${PYTHONPATH:+:$PYTHONPATH}"

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

if ! "$D4RT_PYTHON" -c 'import transformers, accelerate; assert tuple(map(int, transformers.__version__.split(".")[:2])) >= (4, 57)' 2>/dev/null; then
  "$D4RT_PYTHON" -m pip install \
    --target "$DEPS_DIR" \
    --upgrade \
    'transformers>=4.57,<5' \
    'accelerate>=1.1' \
    sentencepiece
fi

"$D4RT_PYTHON" -m d4rt_agent.simple_v1 \
  --phase qwen \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --output d4rt_agent/results/basketball_6/phase2_qwen.json
