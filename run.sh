#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run the full binary-segmentation pipeline:
  1) Loss compare (BCE vs Lovasz-hinge) on unet_resnet50
  2) Pick best loss by val IoU
  3) Model compare (4 U-Net variants) with best loss
  4) Ablation (loss x attention on/off)
  5) Generate paper-style CSV tables

Outputs are written to:
  - run/train/exp*/ (weights, curves, metrics, vis)
  - run/tables/     (table_3_1_loss_compare.csv, table_3_2_model_compare.csv, table_4_2_ablation.csv)

Examples:
  bash run.sh --device cuda --epochs 50 --batch-size 16 --input-size 512 --data-config no-ai
  bash run.sh --data-config full

Options:
  --data-config   no-ai|full (default: no-ai)
  --device        cuda|cpu   (default: cuda)
  --epochs        int        (default: 50)
  --batch-size    int        (default: 8)
  --input-size    int        (default: 512)
  --workers       int        (default: 4)
  --seed          int        (default: 11)
  --weights       path       (default: weights/unet_resnet_voc.pth)
  --python        path       (default: .venv/bin/python)
  --cache-dir     path       (default: .hf-cache/datasets)
  -h, --help
EOF
}

DATA_CONFIG="no-ai"
DEVICE="cuda"
EPOCHS="50"
BATCH_SIZE="8"
INPUT_SIZE="512"
WORKERS="4"
SEED="11"
WEIGHTS="weights/unet_resnet_voc.pth"
PYTHON=".venv/bin/python"
CACHE_DIR=".hf-cache/datasets"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-config) DATA_CONFIG="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --input-size) INPUT_SIZE="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --weights) WEIGHTS="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --cache-dir) CACHE_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

if [[ "$DATA_CONFIG" != "no-ai" && "$DATA_CONFIG" != "full" ]]; then
  echo "Invalid --data-config: $DATA_CONFIG (expected: no-ai|full)"
  exit 1
fi

if [[ -x "$PYTHON" ]]; then
  :
elif command -v "$PYTHON" >/dev/null 2>&1; then
  PYTHON="$(command -v "$PYTHON")"
else
  echo "Python not found: $PYTHON"
  echo "Tip: create venv and install deps, or pass --python /path/to/python"
  exit 1
fi

mkdir -p "$CACHE_DIR" ".hf-cache" ".mpl-cache" "run/train" "run/tables"
export HF_HOME=".hf-cache"
export HF_DATASETS_CACHE="$CACHE_DIR"
export MPLCONFIGDIR=".mpl-cache"

latest_exp_dir() {
  ls -dt run/train/exp* 2>/dev/null | head -n 1
}

get_test_iou() {
  local exp_dir="$1"
  "$PYTHON" - <<PY
import json, sys
from pathlib import Path
p = Path("$exp_dir") / "test_metrics.json"
data = json.loads(p.read_text(encoding="utf-8"))
print(float(data["IoU"]))
PY
}

get_best_val_score() {
  local exp_dir="$1"
  "$PYTHON" - <<PY
import json
from pathlib import Path
p = Path("$exp_dir") / "summary.json"
data = json.loads(p.read_text(encoding="utf-8"))
print(float(data.get("best_score", -1.0)))
PY
}

run_train() {
  local model="$1"
  local loss="$2"
  echo ""
  echo "=============================="
  echo "Train: model=$model loss=$loss data=$DATA_CONFIG device=$DEVICE"
  echo "=============================="
  "$PYTHON" train.py \
    --task binary \
    --data-config "$DATA_CONFIG" \
    --device "$DEVICE" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --workers "$WORKERS" \
    --input-size "$INPUT_SIZE" \
    --seed "$SEED" \
    --model "$model" \
    --loss "$loss" \
    --weights "$WEIGHTS" \
    --cache-dir "$CACHE_DIR"

  LAST_EXP_DIR="$(latest_exp_dir)"
  if [[ -z "${LAST_EXP_DIR:-}" ]]; then
    echo "Failed to locate latest exp directory under run/train/"
    exit 1
  fi
  echo "Saved to: $LAST_EXP_DIR"
}

LOSS_A="bce"
LOSS_B="lovasz_hinge"
MODEL_LOSS_COMPARE="unet_resnet50"
MODELS=("unet_plain" "unet_resnet50" "attention_unet" "dualdense_unet")
ABLATION_MODELS=("unet_plain" "attention_unet")

echo "Python: $PYTHON"
echo "Data config: $DATA_CONFIG"
echo "Device: $DEVICE"
echo "Epochs: $EPOCHS  Batch: $BATCH_SIZE  Input: $INPUT_SIZE  Workers: $WORKERS  Seed: $SEED"
echo ""

# 1) loss compare on unet_resnet50
run_train "$MODEL_LOSS_COMPARE" "$LOSS_A"
EXP_A="$LAST_EXP_DIR"
VAL_A="$(get_best_val_score "$EXP_A")"
IOU_A="$(get_test_iou "$EXP_A")"
echo "Val IoU ($MODEL_LOSS_COMPARE, $LOSS_A):  $VAL_A"
echo "Test IoU ($MODEL_LOSS_COMPARE, $LOSS_A): $IOU_A"

run_train "$MODEL_LOSS_COMPARE" "$LOSS_B"
EXP_B="$LAST_EXP_DIR"
VAL_B="$(get_best_val_score "$EXP_B")"
IOU_B="$(get_test_iou "$EXP_B")"
echo "Val IoU ($MODEL_LOSS_COMPARE, $LOSS_B):  $VAL_B"
echo "Test IoU ($MODEL_LOSS_COMPARE, $LOSS_B): $IOU_B"

BEST_LOSS="$("$PYTHON" - <<PY
a=float("$VAL_A")
b=float("$VAL_B")
print("$LOSS_B" if b >= a else "$LOSS_A")
PY
)"
echo ""
echo ">>> Best loss by val IoU: $BEST_LOSS"

# 2) model compare with best loss
for model in "${MODELS[@]}"; do
  run_train "$model" "$BEST_LOSS"
done

# 3) ablation: (loss x attention on/off)
for loss in "$LOSS_A" "$LOSS_B"; do
  for model in "${ABLATION_MODELS[@]}"; do
    run_train "$model" "$loss"
  done
done

# 4) generate tables
echo ""
echo "=============================="
echo "Generate tables"
echo "=============================="
"$PYTHON" scripts/make_tables.py --data-config "$DATA_CONFIG" --task binary

echo ""
echo "Done."
echo "  - Experiments: run/train/exp*/"
echo "  - Tables:      run/tables/"
