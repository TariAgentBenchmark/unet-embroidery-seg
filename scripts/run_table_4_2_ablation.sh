#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run Table 4-2 module ablation on U-Net:
  - baseline
  - ASPP
  - ECA
  - SA
  - ASPP+ECA
  - ASPP+SA
  - ECA+SA
  - ASPP+ECA+SA

Outputs:
  - run/train/exp*/               training runs
  - run/tables/table_4_2_ablation.csv

Example:
  bash scripts/run_table_4_2_ablation.sh --data-config no-ai --device cuda --epochs 50 --batch-size 8

Options:
  --data-config   no-ai|full|sam3|sam3-label (default: no-ai)
  --task          binary (default: binary)
  --device        cuda|cpu (default: cuda)
  --epochs        int (default: 50)
  --batch-size    int (default: 8)
  --input-size    int (default: 512)
  --workers       int (default: 4)
  --seed          int (default: 11)
  --loss          bce|lovasz_hinge|ce|focal (default: lovasz_hinge)
  --weights       path (default: empty)
  --python        path (default: .venv/bin/python)
  --cache-dir     path (default: .hf-cache/datasets)
  --data-path     path (default: hf_datasets/merged_dataset_v2)
  --hf-repo       repo_id (default: tari-tech/13803867589-unet-image-seg)
  --hf-revision   revision (default: empty)
  --hf-local-dir  path (default: hf_datasets/merged_dataset_v2)
  --max-train-batches int (default: 0)
  --max-val-batches   int (default: 0)
  --max-test-batches  int (default: 0)
  -h, --help
EOF
}

DATA_CONFIG="no-ai"
TASK="binary"
DEVICE="cuda"
EPOCHS="50"
BATCH_SIZE="8"
INPUT_SIZE="512"
WORKERS="4"
SEED="11"
LOSS="lovasz_hinge"
WEIGHTS=""
PYTHON=".venv/bin/python"
CACHE_DIR=".hf-cache/datasets"
DATA_PATH="hf_datasets/merged_dataset_v2"
HF_REPO="tari-tech/13803867589-unet-image-seg"
HF_REVISION=""
HF_LOCAL_DIR="hf_datasets/merged_dataset_v2"
MAX_TRAIN_BATCHES="0"
MAX_VAL_BATCHES="0"
MAX_TEST_BATCHES="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-config) DATA_CONFIG="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --input-size) INPUT_SIZE="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --loss) LOSS="$2"; shift 2 ;;
    --weights) WEIGHTS="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --cache-dir) CACHE_DIR="$2"; shift 2 ;;
    --data-path) DATA_PATH="$2"; shift 2 ;;
    --hf-repo) HF_REPO="$2"; shift 2 ;;
    --hf-revision) HF_REVISION="$2"; shift 2 ;;
    --hf-local-dir) HF_LOCAL_DIR="$2"; shift 2 ;;
    --max-train-batches) MAX_TRAIN_BATCHES="$2"; shift 2 ;;
    --max-val-batches) MAX_VAL_BATCHES="$2"; shift 2 ;;
    --max-test-batches) MAX_TEST_BATCHES="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

if [[ -x "$PYTHON" ]]; then
  :
elif command -v "$PYTHON" >/dev/null 2>&1; then
  PYTHON="$(command -v "$PYTHON")"
else
  echo "Python not found: $PYTHON"
  exit 1
fi

mkdir -p "$CACHE_DIR" ".hf-cache" ".mpl-cache" "run/train" "run/tables"
export HF_HOME=".hf-cache"
export HF_DATASETS_CACHE="$CACHE_DIR"
export MPLCONFIGDIR=".mpl-cache"

ensure_dataset() {
  local cfg="$1"
  local expected_dir="$HF_LOCAL_DIR/$cfg"
  if [[ -d "$expected_dir" ]]; then
    return 0
  fi

  echo "Dataset not found: $expected_dir"
  echo "Downloading from Hugging Face: $HF_REPO"
  mkdir -p "$HF_LOCAL_DIR"

  if command -v huggingface-cli >/dev/null 2>&1; then
    args=(download "$HF_REPO" --repo-type dataset --local-dir "$HF_LOCAL_DIR")
    if [[ -n "$HF_REVISION" ]]; then
      args+=(--revision "$HF_REVISION")
    fi
    set +e
    huggingface-cli "${args[@]}" --resume-download
    rc=$?
    set -e
    if [[ $rc -ne 0 ]]; then
      "$PYTHON" - <<PY
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="$HF_REPO",
    repo_type="dataset",
    local_dir="$HF_LOCAL_DIR",
    local_dir_use_symlinks=False,
    revision="$HF_REVISION".strip() or None,
    resume_download=True,
)
PY
    fi
  else
    "$PYTHON" - <<PY
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="$HF_REPO",
    repo_type="dataset",
    local_dir="$HF_LOCAL_DIR",
    local_dir_use_symlinks=False,
    revision="$HF_REVISION".strip() or None,
    resume_download=True,
)
PY
  fi

  if [[ ! -d "$expected_dir" ]]; then
    echo "Download finished, but still missing: $expected_dir"
    exit 1
  fi
}

run_one() {
  local label="$1"
  local use_aspp="$2"
  local use_eca="$3"
  local use_sa="$4"

  echo ""
  echo "=============================="
  echo "Table 4-2 run: $label"
  echo "task=$TASK model=unet_plain loss=$LOSS data=$DATA_CONFIG"
  echo "ASPP=$use_aspp ECA=$use_eca SA=$use_sa"
  echo "=============================="

  cmd=(
    "$PYTHON" train.py
    --task "$TASK"
    --data-path "$DATA_PATH"
    --data-config "$DATA_CONFIG"
    --device "$DEVICE"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --workers "$WORKERS"
    --input-size "$INPUT_SIZE"
    --seed "$SEED"
    --model "unet_plain"
    --loss "$LOSS"
    --cache-dir "$CACHE_DIR"
    --max-train-batches "$MAX_TRAIN_BATCHES"
    --max-val-batches "$MAX_VAL_BATCHES"
    --max-test-batches "$MAX_TEST_BATCHES"
  )
  if [[ -n "$WEIGHTS" ]]; then
    cmd+=(--weights "$WEIGHTS")
  fi
  if [[ "$use_aspp" == "true" ]]; then
    cmd+=(--use-aspp)
  fi
  if [[ "$use_eca" == "true" ]]; then
    cmd+=(--use-eca)
  fi
  if [[ "$use_sa" == "true" ]]; then
    cmd+=(--use-sa)
  fi

  "${cmd[@]}"
}

echo "Python: $PYTHON"
echo "Data config: $DATA_CONFIG"
echo "Task: $TASK"
echo "Loss: $LOSS"
echo "Device: $DEVICE"
echo "Epochs: $EPOCHS  Batch: $BATCH_SIZE  Input: $INPUT_SIZE  Workers: $WORKERS  Seed: $SEED"

ensure_dataset "$DATA_CONFIG"

if [[ "$TASK" != "binary" ]]; then
  echo "This script only supports --task binary."
  exit 1
fi

run_one "baseline" false false false
run_one "ASPP" true false false
run_one "ECA" false true false
run_one "SA" false false true
run_one "ASPP+ECA" true true false
run_one "ASPP+SA" true false true
run_one "ECA+SA" false true true
run_one "ASPP+ECA+SA" true true true

"$PYTHON" scripts/make_ablation_table_4_2.py \
  --data-config "$DATA_CONFIG" \
  --task "$TASK" \
  --model "unet_plain" \
  --loss "$LOSS"

echo ""
echo "Done."
echo "  - Experiments: run/train/exp*/"
echo "  - Table:       run/tables/table_4_2_ablation.csv"
