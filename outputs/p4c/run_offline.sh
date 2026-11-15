#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=src HF_HUB_OFFLINE=1 HF_HOME=/tmp/cr3-p4c-hf MPLCONFIGDIR=/tmp/cr3-mpl
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl MESA_SHADER_CACHE_DISABLE=true
out=${1:-outputs/p4c}
lerobot_py=${LEROBOT_PYTHON:-"$HOME/miniconda3/envs/lerobot_sim/bin/python"}
"$lerobot_py" outputs/p4c/export_chunks.py "$out/raw" "$out/dataset" >"$out/export.log" 2>&1
"$lerobot_py" outputs/p4c/verify_chunks.py "$out/raw" "$out/dataset" >"$out/chunk-check.log" 2>&1
"$lerobot_py" outputs/p4c/cache_loader.py "$out/dataset" >"$out/loader.log" 2>&1
.venv/bin/python outputs/p4c/bc.py "$out/dataset" "$out/train" >"$out/train.log" 2>&1
.venv/bin/python outputs/p4c/evaluate.py "$out/raw" "$out/train/model.pt" "$out/eval" >"$out/eval.log" 2>&1
