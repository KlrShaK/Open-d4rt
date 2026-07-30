#!/usr/bin/env bash
# Full DSI-Bench census (all 1769 `std` questions) for the simple_v2 agent, on one
# Blackwell (RTX PRO 6000) GPU. Derived from run_dsi_bench_blackwell.slurm.
#
# Build the manifest once, on the login node (CPU only, a couple of seconds):
#
#   "$D4RT_PYTHON" -m d4rt_agent.dsi_bench_run --all-questions --dry-run \
#       --results-dir d4rt_agent/results/dsi_bench_full
#
# Then submit, and resubmit the identical command to resume:
#
#   sbatch --export=ALL,POINT_MODE=ensemble5 scripts/run_dsi_bench_full_blackwell.slurm
#   sbatch --export=ALL,POINT_MODE=ensemble5,MODE=baseline scripts/run_dsi_bench_full_blackwell.slurm
#
# One GPU, one question at a time, ~45-50 GPU-hours for the agent pass. That does
# not fit one 24 h job, and it is not meant to: every question is written as soon
# as it finishes and finished questions are skipped, so this is submitted as many
# times as it takes. Nothing is lost when the wall clock runs out mid-question --
# only that question is repeated.
#
# To run it to completion in a single submission instead:
#   sbatch --partition=cuda13pr.120h --time=120:00:00 \
#          --export=ALL,POINT_MODE=ensemble5 scripts/run_dsi_bench_full_blackwell.slurm

#SBATCH --job-name=d4rt-dsi-full
#SBATCH --partition=cuda13pr.24h
#SBATCH --gpus=nvidia_rtx_pro_6000:1
# The site submit filter flattens the typed GPU request and also adds gpupr.4h,
# whose A100 nodes share the cuda13/gpupr features -- so pin to the Blackwell
# nodes explicitly. EPYC_9654 is unique to eu-g7-[001-010] among reachable GPU
# nodes. This also keeps every resumed job on the same GPU model, which the
# comparison tooling requires of a single results directory.
#SBATCH --constraint=EPYC_9654
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --tmp=8G
#SBATCH --time=24:00:00
#SBATCH --output=/cluster/work/igp_psr/spanwar/Open-d4rt/logs/d4rt_agent/dsi_bench_full_%j.out

set -euo pipefail

REPO_ROOT=/cluster/work/igp_psr/spanwar/Open-d4rt
D4RT_PYTHON=/cluster/work/igp_psr/spanwar/envs/d4rt_bw/bin/python
DEPS_DIR="$REPO_ROOT/.cache/qwen3vl_python"
LOCAL_DEPS="${TMPDIR:-/tmp/$USER}/qwen3vl_python"
HF_HOME=/cluster/work/igp_psr/spanwar/hf_cache
MODEL_CACHE="$HF_HOME/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots"
# Deliberately on /cluster/work, never /cluster/scratch: this directory is the
# only copy of ~45 GPU-hours of answers and scratch is not a safe home for it.
RESULTS_DIR="${RESULTS_DIR:-d4rt_agent/results/dsi_bench_full}"
MANIFEST_PATH="${MANIFEST_PATH:-$REPO_ROOT/$RESULTS_DIR/manifest.json}"
FORCE="${FORCE:-0}"
MODE="${MODE:-agent}"
POINT_MODE="${POINT_MODE:?submit with POINT_MODE=ensemble5}"

if [[ "$POINT_MODE" != "centroid" && "$POINT_MODE" != "ensemble5" ]]; then
  echo "Unsupported POINT_MODE=$POINT_MODE" >&2
  exit 2
fi
if [[ "$MODE" != "agent" && "$MODE" != "baseline" ]]; then
  echo "Unsupported MODE=$MODE (expected agent or baseline)" >&2
  exit 2
fi
if [[ "$MODE" == "agent" && "$POINT_MODE" != "ensemble5" ]]; then
  echo "Agent evaluation requires host-fixed POINT_MODE=ensemble5" >&2
  exit 2
fi
if [[ "$FORCE" != "0" && "$FORCE" != "1" ]]; then
  echo "Unsupported FORCE=$FORCE (expected 0 or 1)" >&2
  exit 2
fi

cd "$REPO_ROOT"
# Built once, up front, so a job never spends its allocation deciding which
# questions to ask -- and so a mistyped results dir fails here instead of
# quietly starting a different run.
if [[ ! -f "$MANIFEST_PATH" ]]; then
  echo "Census manifest does not exist: $MANIFEST_PATH" >&2
  echo "Build it first with:" >&2
  echo "  $D4RT_PYTHON -m d4rt_agent.dsi_bench_run --all-questions --dry-run \\" >&2
  echo "      --results-dir $RESULTS_DIR" >&2
  exit 2
fi

DESIGN="$("$D4RT_PYTHON" -c "import json,sys;print(json.load(open(sys.argv[1]))['sampling']['design'])" "$MANIFEST_PATH")"
if [[ "$DESIGN" != "census" ]]; then
  echo "Refusing to run: $MANIFEST_PATH has design=$DESIGN, expected census." >&2
  echo "That is the 25-question sample, not the full split." >&2
  exit 2
fi

TOTAL="$("$D4RT_PYTHON" -c "import json,sys;print(len(json.load(open(sys.argv[1]))['questions']))" "$MANIFEST_PATH")"
DONE_ALREADY="$(find "$RESULTS_DIR/answers" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)"

echo "RESULTS_DIR=$RESULTS_DIR"
echo "MANIFEST_PATH=$MANIFEST_PATH (design=$DESIGN, $TOTAL questions)"
echo "ANSWERS ALREADY ON DISK=$DONE_ALREADY"
echo "FORCE=$FORCE"
echo "MODE=$MODE  immutable POINT_MODE=$POINT_MODE  LIMIT=${LIMIT:-all}"
echo "ATTEMPTS=1 (ungated; the strict/relaxed retry pair is not used)"
echo "GROUNDING_TOOL=isolated_qwen bbox_tokens=96 points_tokens=192 seed=42"
echo "CODE_SHA=$(git rev-parse HEAD)"

mkdir -p logs/d4rt_agent "$RESULTS_DIR" "$LOCAL_DEPS"
cp -a "$DEPS_DIR/." "$LOCAL_DEPS/"

export HF_HOME
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export DATASETS_OFFLINE=1
export NCCL_NVLS_ENABLE=0
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$LOCAL_DEPS${PYTHONPATH:+:$PYTHONPATH}"

MODEL_DIR="$(find "$MODEL_CACHE" -mindepth 1 -maxdepth 1 -type d -print -quit)"
if [[ -z "$MODEL_DIR" ]]; then
  echo "Offline Qwen snapshot missing. Run scripts/prepare_qwen3vl_login.sh first." >&2
  exit 2
fi
export MODEL_DIR

EXTRA=()
if [[ "$MODE" == "baseline" ]]; then
  EXTRA+=(--baseline)
fi
if [[ -n "${LIMIT:-}" ]]; then
  EXTRA+=(--limit "$LIMIT")
fi
if [[ "$FORCE" == "1" ]]; then
  EXTRA+=(--force)
fi

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

"$D4RT_PYTHON" -m d4rt_agent.dsi_bench_run \
  --all-questions \
  --results-dir "$RESULTS_DIR" \
  --manifest "$MANIFEST_PATH" \
  --point-mode "$POINT_MODE" \
  --qwen-model "$MODEL_DIR" \
  --d4rt-config checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml \
  --d4rt-checkpoint checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt \
  --d4rt-dtype bfloat16 \
  --device cuda \
  --seed 42 \
  --max-new-tokens 768 \
  --max-steps 14 \
  "${EXTRA[@]}"
