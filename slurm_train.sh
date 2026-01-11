#!/bin/bash
#SBATCH --nodelist=csardas
#SBATCH --job-name="train_vjepa_probe"
#SBATCH --output=/home/bickicdz/projects/Causal_VAE/slurm/logs/output_%A_%a.txt
#SBATCH --error=/home/bickicdz/projects/Causal_VAE/slurm/logs/error_%A_%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32        # match to your --num_workers
#SBATCH --gres=gpu:1
#SBATCH --mem=240G
#SBATCH --time=96:00:00
#SBATCH --chdir=/scratch/users/bickici/projects/Causal_VAE   # start here (no permission errors)


# Print node and GPU info
echo "Running on node: ${SLURMD_NODENAME}"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Job started at: $(date)"

# Activate your virtual environment
source /scratch/users/bickici/environments/lavila/bin/activate
echo "ENV loaded"

export MASTER_ADDR=localhost
export MASTER_PORT=12355   # any free port
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

# Verify Python and check GPU
echo "Python version: $(python --version)"
echo "Python path: $(which python)"
python -c "import torch; print(f'PyTorch version: {torch.version}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Launcher for training V-JEPA2 (or MViT) backbones with LaViLa's clean trainer.
# Usage:
#   ./train_vjepa2.sh [MODEL_TYPE] [DATASET] [TASK_TYPE] [extra args...]
# Examples:
#   ./train_vjepa2.sh vjepa2_large ek100_cls verb
#   ./train_vjepa2.sh vjepa2_giant_384 ek100_cls action --batch-size 4
#   ./train_vjepa2.sh mvit_spatial egtea action
#
# MODEL_TYPE options (from main_train_vjepa_probe.py):
#   vjepa2_large | vjepa2_huge | vjepa2_giant | vjepa2_giant_384 | mvit_spatial | mvit_temporal
# DATASET: egtea | ek100_cls
# TASK_TYPE: action | verb | noun

MODEL_TYPE="${1:-vjepa2_huge}"
DATASET="${2:-ek100_cls}"
TASK_TYPE="${3:-action}"

# Dataset-specific paths (update to your setup)
if [ "$DATASET" = "ek100_cls" ]; then
    ROOT_DIR="/scratch/users/bickici/data/EK100/EK100_256p"
    TRAIN_META="/scratch/users/bickici/data/EK100/meta_data/EPIC_100_train.csv"
    VAL_META="/scratch/users/bickici/data/EK100/meta_data/EPIC_100_validation.csv"
elif [ "$DATASET" = "egtea" ]; then
    ROOT_DIR="/mnt/j/video_clips/cropped_clips/"
    TRAIN_META="../data/EGTEA/raw/annotation/split/train_split1.txt"
    VAL_META="../data/EGTEA/raw/annotation/split/test_split1.txt"
else
    echo "Unknown dataset: $DATASET"
    exit 1
fi

# Optional: set to an existing checkpoint to fine-tune or resume
PRETRAIN_MODEL=""

# Core hyperparameters
BATCH_SIZE=32
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
NUM_WORKERS=24
SEED=42

MULTI_TASK_FLAGS=""
if [ "$DATASET" = "ek100_cls" ] && [[ "$MODEL_TYPE" == vjepa2_* ]]; then
    TASK_TYPE="action"
    OUTPUT_DIR="/scratch/users/bickici/data/EK100/model_checkpoints/${MODEL_TYPE}_${DATASET}_multitask"
    MULTI_TASK_FLAGS="--multi-task --probe-num-blocks 4 --probe-num-heads 16"
else
    OUTPUT_DIR="/scratch/users/bickici/data/EK100/model_checkpoints/${MODEL_TYPE}_${DATASET}_${TASK_TYPE}"
fi

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Training model: $MODEL_TYPE"
echo "Dataset: $DATASET | Task: $TASK_TYPE"
echo "Output dir: $OUTPUT_DIR"
echo "=========================================="

CMD="python main_train_vjepa_probe.py \
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
    --use-timestamps \
    $MULTI_TASK_FLAGS"

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

 
echo "Job finished at: $(date)"
