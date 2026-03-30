#!/bin/bash
# Epic Kitchen 100 Feature Extraction Script
# Uses MViT Temporal model for optical flow features

python3 main_extract_feature_flow2.py \
--dataset ek100_cls \
--metadata-train /home/dz/Projects/multi-modal_AR/data/EK/data/EPIC_100_train.csv \
--metadata-val /home/dz/Projects/multi-modal_AR/data/EK/data/EPIC_100_validation.csv \
--root /mnt/k/EK100_256p/ \
--pretrain-model /mnt/k/checkpoints_mvit/mvit_mvit_temporal_ek100_cls_verb_clean/checkpoint_best.pt \
--num-classes 97 \
--batch-size 4 \
--use-sgd --wd 4e-5 \
--output-dir /mnt/k/checkpoints_mvit/features/temp/large \
--egtea_finetune_type 'verb' \
--num-clips 10 \
--num-crops 3 \
--use-checkpoint \
--use-timestamps