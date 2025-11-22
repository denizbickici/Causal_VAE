#!/bin/bash

# Launcher for training V-JEPA2 (or MViT) backbones with LaViLa's clean trainer.
# Usage:
#   ./train_vjepa2.sh [MODEL_TYPE] [DATASET] [TASK_TYPE] [extra args...]
# Examples:
#   ./train_vjepa2.sh vjepa2_large ek100_cls verb
#   ./train_vjepa2.sh vjepa2_giant_384 ek100_cls action --batch-size 4
#   ./train_vjepa2.sh mvit_spatial egtea action
#
# MODEL_TYPE options (from main_train_vjepa.py):
#   vjepa2_large | vjepa2_huge | vjepa2_giant | vjepa2_giant_384 | mvit_spatial | mvit_temporal
# DATASET: egtea | ek100_cls
# TASK_TYPE: action | verb | noun

MODEL_TYPE="${1:-vjepa2_huge}"
DATASET="${2:-ek100_cls}"
TASK_TYPE="${3:-verb}"

# Dataset-specific paths (update to your setup)
if [ "$DATASET" = "ek100_cls" ]; then
    ROOT_DIR="/mnt/k/EK100_256p"
    TRAIN_META="/home/dz/Projects/multi-modal_AR/data/EK/data/EPIC_100_train.csv"
    VAL_META="/home/dz/Projects/multi-modal_AR/data/EK/data/EPIC_100_validation.csv"
elif [ "$DATASET" = "egtea" ]; then
    ROOT_DIR="/mnt/j/video_clips/cropped_clips/"
    TRAIN_META="../data/EGTEA/raw/annotation/split/train_split1.txt"
    VAL_META="../data/EGTEA/raw/annotation/split/test_split1.txt"
else
    echo "Unknown dataset: $DATASET"
    exit 1
fi

# Output/checkpoints
OUTPUT_DIR="/mnt/k/checkpoints/${MODEL_TYPE}_${DATASET}_${TASK_TYPE}"
mkdir -p "$OUTPUT_DIR"

# Optional: set to an existing checkpoint to fine-tune or resume
PRETRAIN_MODEL=""

# Core hyperparameters
BATCH_SIZE=8
LR=3e-4
WD=0.01
EPOCHS=50
WARMUP_EPOCHS=5
EVAL_FREQ=1
SAVE_FREQ=1

# Data sampling
NUM_CLIPS=1
NUM_CROPS=1
CLIP_LENGTH=16
CLIP_STRIDE=2

# System
NUM_WORKERS=4
SEED=42

echo "=========================================="
echo "Training model: $MODEL_TYPE"
echo "Dataset: $DATASET | Task: $TASK_TYPE"
echo "Output dir: $OUTPUT_DIR"
echo "=========================================="

CMD="python main_train_vjepa.py \
    --dataset $DATASET \
    --root $ROOT_DIR \
    --metadata-train $TRAIN_META \
    --metadata-val $VAL_META \
    --output-dir $OUTPUT_DIR \
    --model-type $MODEL_TYPE \
    --task-type $TASK_TYPE \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --wd $WD \
    --epochs $EPOCHS \
    --warmup-epochs $WARMUP_EPOCHS \
    --eval-freq $EVAL_FREQ \
    --save-freq $SAVE_FREQ \
    --num-clips $NUM_CLIPS \
    --num-crops $NUM_CROPS \
    --clip-length $CLIP_LENGTH \
    --clip-stride $CLIP_STRIDE \
    --workers $NUM_WORKERS \
    --seed $SEED \
    --print-freq 100 \
    --use-timestamps"

if [ -n "$PRETRAIN_MODEL" ]; then
    CMD="$CMD --pretrain-model $PRETRAIN_MODEL"
fi

# Append any extra CLI args passed to this script
if [ $# -gt 3 ]; then
    shift 3
    CMD="$CMD $@"
fi

echo "Executing:"
echo "$CMD"
echo "=========================================="
$CMD
echo "=========================================="
echo "Done. Results in: $OUTPUT_DIR"
echo "=========================================="
