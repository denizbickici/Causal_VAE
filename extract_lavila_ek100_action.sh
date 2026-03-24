#!/bin/bash
set -euo pipefail

# LaViLa EK100 action feature extraction.
# Usage:
#   ./extract_lavila_ek100_action.sh /path/to/checkpoint.pt [output_dir] [extra args...]
#
# Example:
#   ./extract_lavila_ek100_action.sh \
#     /scratch/users/bickici/data/EK100/model_checkpoints/lavila_ek100_action/checkpoint_best.pt \
#     /scratch/users/bickici/data/lavila_ek100_action_features

if [ $# -lt 1 ]; then
    echo "Usage: $0 PRETRAIN_MODEL [OUTPUT_DIR] [extra args...]"
    exit 1
fi

PRETRAIN_MODEL="$1"
shift

OUTPUT_DIR="${1:-/scratch/users/bickici/data/lavila_ek100_action_features}"
if [ $# -gt 0 ]; then
    shift
fi

ROOT_DIR="${ROOT_DIR:-/scratch/users/bickici/data/EK100/EK100_256p}"
TRAIN_META="${TRAIN_META:-/scratch/users/bickici/data/EK100/meta_data/EPIC_100_train.csv}"
VAL_META="${VAL_META:-/scratch/users/bickici/data/EK100/meta_data/EPIC_100_validation.csv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-24}"
SEED="${SEED:-42}"
CLIP_LENGTH="${CLIP_LENGTH:-16}"
CLIP_STRIDE="${CLIP_STRIDE:-2}"
NUM_CLIPS="${NUM_CLIPS:-1}"
NUM_CROPS="${NUM_CROPS:-1}"
USE_VN_CLASSIFIER="${USE_VN_CLASSIFIER:-1}"

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Extracting LaViLa EK100 action features"
echo "Checkpoint: $PRETRAIN_MODEL"
echo "Output dir: $OUTPUT_DIR"
echo "Root dir:   $ROOT_DIR"
echo "Train csv:  $TRAIN_META"
echo "Val csv:    $VAL_META"
echo "=========================================="

CMD=(
    "$PYTHON_BIN" main_extract_feature.py
    --dataset ek100_cls
    --metadata-train "$TRAIN_META"
    --metadata-val "$VAL_META"
    --root "$ROOT_DIR"
    --pretrain-model "$PRETRAIN_MODEL"
    --batch-size "$BATCH_SIZE"
    --use-sgd
    --wd 4e-5
    --output-dir "$OUTPUT_DIR"
    --egtea_finetune_type action
    --model_type lavila
    --num-clips "$NUM_CLIPS"
    --num-crops "$NUM_CROPS"
    --clip-length "$CLIP_LENGTH"
    --clip-stride "$CLIP_STRIDE"
    --workers "$NUM_WORKERS"
    --seed "$SEED"
    --use-checkpoint
)

if [ "$USE_VN_CLASSIFIER" = "1" ]; then
    CMD+=(--use-vn-classifier --num-classes 97 300 3806)
else
    CMD+=(--num-classes 3806)
fi

if [ $# -gt 0 ]; then
    CMD+=("$@")
fi

echo "Executing:"
printf ' %q' "${CMD[@]}"
echo
echo "=========================================="
"${CMD[@]}"
echo "=========================================="
echo "Done. Results in: $OUTPUT_DIR"
echo "Expected files:"
echo "  $OUTPUT_DIR/ek100_cls_train_feat.pt"
echo "  $OUTPUT_DIR/ek100_cls_test_feat.pt"
echo "=========================================="
