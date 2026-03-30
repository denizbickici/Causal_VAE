#!/bin/bash

# Clean MViT Training Script for LaViLa Infrastructure
# Simple and focused training launcher
#
# Usage:
#   ./train_mvit_clean.sh [MODEL_TYPE] [DATASET] [TASK_TYPE] [additional_args]
#
# Examples:
#   # EGTEA dataset
#   ./train_mvit_clean.sh mvit_temporal egtea verb
#   ./train_mvit_clean.sh mvit_spatial egtea action
#   
#   # Epic Kitchen 100 dataset
#   ./train_mvit_clean.sh mvit_temporal ek100_cls verb
#   ./train_mvit_clean.sh mvit_temporal ek100_cls action
#   ./train_mvit_clean.sh mvit_spatial ek100_cls noun
#
# Supported values:
#   MODEL_TYPE: mvit_spatial, mvit_temporal
#   DATASET: egtea, ek100_cls
#   TASK_TYPE: action, verb, noun

# Configuration
MODEL_TYPE="${1:-mvit_temporal}"  # Default to temporal, can be overridden
DATASET="${2:-ek100_cls}"              # Default to EGTEA
TASK_TYPE="${3:-verb}"             # Default to verb classification

# Paths - Update these for your environment
# Set paths based on dataset
if [ "$DATASET" = "ek100_cls" ]; then
    # Epic Kitchen 100 paths
    ROOT_DIR="/mnt/k/EK100_256p"  # Update with your EK100 path
    TRAIN_META="/home/dz/Projects/multi-modal_AR/data/EK/data/EPIC_100_train.csv"
    VAL_META="/home/dz/Projects/multi-modal_AR/data/EK/data/EPIC_100_validation.csv"
elif [ "$DATASET" = "egtea" ]; then
    # EGTEA paths
    ROOT_DIR="/mnt/j/video_clips/cropped_clips/"
    TRAIN_META="../data/EGTEA/raw/annotation/split/train_split1.txt"
    VAL_META="../data/EGTEA/raw/annotation/split/test_split1.txt"
else
    echo "Error: Unknown dataset $DATASET"
    echo "Supported datasets: egtea, ek100_cls"
    exit 1
fi

OUTPUT_DIR="/mnt/k/checkpoints/mvit_${MODEL_TYPE}_${DATASET}_${TASK_TYPE}_clean"

# Model checkpoint (optional - for fine-tuning)
PRETRAIN_MODEL="/mnt/k/checkpoints/mvit_${MODEL_TYPE}_${DATASET}_${TASK_TYPE}_clean/checkpoint_best.pt"
# Uncomment and set path to use pretrained weights:
# PRETRAIN_MODEL="/path/to/checkpoint.pt"

# Training hyperparameters
BATCH_SIZE=16
LR=1e-3
WD=0.01
EPOCHS=50
WARMUP_EPOCHS=5
EVAL_FREQ=1
SAVE_FREQ=1

# Data parameters
NUM_CLIPS=1      # For training
NUM_CROPS=1      # For validation
CLIP_LENGTH=16
CLIP_STRIDE=2

# System settings
NUM_WORKERS=4
SEED=42

# Create output directory
mkdir -p $OUTPUT_DIR

echo "=========================================="
echo "Training MViT Model"
echo "Model Type: $MODEL_TYPE"
echo "Dataset: $DATASET"
echo "Task: $TASK_TYPE"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# Build the command
CMD="python main_train_mvit_clean.py \
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

# Add pretrain model if specified
if [ ! -z "$PRETRAIN_MODEL" ]; then
    CMD="$CMD --pretrain-model $PRETRAIN_MODEL"
fi

# Optional flags you can add:
# --wandb                    # Enable WandB logging
# --use-sgd                  # Use SGD instead of AdamW
# --disable-amp              # Disable mixed precision (use fp32)
# --resume checkpoint.pt     # Resume from checkpoint
# --find-unused-parameters   # For DDP debugging

# Add any additional arguments passed to the script
if [ $# -gt 3 ]; then
    shift 3
    CMD="$CMD $@"
fi

# Execute the training
echo "Executing command:"
echo "$CMD"
echo "=========================================="

$CMD

echo "=========================================="
echo "Training completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="