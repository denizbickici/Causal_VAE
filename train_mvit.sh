#!/bin/bash

# MViT Training Script for LaViLa Infrastructure
# This script trains MViT models (spatial or temporal) using the LaViLa training framework

# Configuration
MODEL_TYPE="mvit_temporal"  # Options: mvit_spatial, mvit_temporal
DATASET="egtea" #"egtea"
FINETUNE_TYPE="verb"  # Options: action, verb, noun

# Paths
ROOT_DIR="/mnt/j/video_clips/cropped_clips/"
TRAIN_META="../data/EGTEA/raw/annotation/split/train_split1.txt"
VAL_META="../data/EGTEA/raw/annotation/split/test_split1.txt"
OUTPUT_DIR="output_ckpt/mvit_${MODEL_TYPE}_${DATASET}_${FINETUNE_TYPE}"

# Model paths (update these with your pretrained models)
if [ "$MODEL_TYPE" = "mvit_spatial" ]; then
    if [ "$FINETUNE_TYPE" = "noun" ]; then
        PRETRAIN_MODEL="" #"/home/dz/Projects/multi-modal_AR/mvit_spat_EGTEA_noun_stage2_fluent-cosmos-238/model.safetensors"
    elif [ "$FINETUNE_TYPE" = "verb" ]; then
        PRETRAIN_MODEL="/home/dz/Projects/multi-modal_AR/mvit_spat_EGTEA_verb_stage2_trim-aardvark-235/model.safetensors"
    else  # action
        PRETRAIN_MODEL=""  # Empty string for training from scratch
        # PRETRAIN_MODEL="/home/dz/Projects/multi-modal_AR/mvit_spat_EGTEA_action_stage3_exalted-shape-195/model.safetensors"
    fi
elif [ "$MODEL_TYPE" = "mvit_temporal" ]; then
    if [ "$FINETUNE_TYPE" = "verb" ]; then
        PRETRAIN_MODEL="/home/dz/Projects/multi-modal_AR/LaViLa/output_ckpt/mvit_mvit_temporal_egtea_verb/checkpoint_best.pt"
        #"/home/dz/Projects/multi-modal_AR/mvit_temp_EK_verb_fanciful-wood-99/model.safetensors" #"../mvit_temp_EGTEA_verb_ethereal-valley-107/model.safetensors"
    else
        # Add paths for other temporal models as needed
        PRETRAIN_MODEL=""
    fi
fi

# Training hyperparameters
BATCH_SIZE=16
LR=8e-4
WD=4e-5
EPOCHS=50
WARMUP_EPOCHS=5
EVAL_FREQ=3
SAVE_FREQ=10

# Number of clips and frames
NUM_CLIPS=1
NUM_CROPS=1
CLIP_LENGTH=16
CLIP_STRIDE=2

# System settings
NUM_WORKERS=12
SEED=42

# Create output directory
mkdir -p $OUTPUT_DIR

# Build command with optional pretrained model
# Use accelerate for better mixed precision handling
CMD="accelerate launch --mixed_precision fp16 main_train_mvit.py \
    --dataset $DATASET \
    --root $ROOT_DIR \
    --metadata-train $TRAIN_META \
    --metadata-val $VAL_META \
    --output-dir $OUTPUT_DIR \
    --model-type $MODEL_TYPE \
    --egtea-finetune-type $FINETUNE_TYPE \
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
    --use-checkpoint \
    --wandb"

# Add pretrain-model argument only if it's not empty
if [ ! -z "$PRETRAIN_MODEL" ]; then
    CMD="$CMD --pretrain-model $PRETRAIN_MODEL"
fi

# Execute the command with any additional arguments
$CMD ${@}

# Additional options you can add:
# --use-sgd                  # Use SGD instead of AdamW
# --disable-amp              # Disable mixed precision training (runs in fp32)
# Note: If you get OOM errors, try reducing batch size or add --disable-amp
# --fix-lr                   # Disable cosine learning rate decay
# --sparse-sample            # Use sparse sampling
# --use-zero                 # Use ZeroRedundancyOptimizer
# --resume checkpoint.pt     # Resume from checkpoint
# --lr-multiplier-on-backbone 0.1  # Different learning rate for backbone

echo "Training completed. Results saved to: $OUTPUT_DIR"