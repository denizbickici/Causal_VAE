python3 main_extract_feature_vjepa.py \
--dataset ek100_cls \
--task-type noun \
--metadata-train /scratch/users/bickici/data/EK100/meta_data/EPIC_100_train.csv \
--metadata-val /scratch/users/bickici/data/EK100/meta_data/EPIC_100_validation.csv \
--root /scratch/users/bickici/data/EK100/EK100_256p \
--pretrain-model /path/to/vjepa_checkpoint.pt \
--model-type vjepa2_huge \
--num-classes 97 \
--batch-size 4 \
--num-clips 10 \
--num-crops 3 \
--clip-length 16 \
--clip-stride 2 \
--output-dir /scratch/users/bickici/data/EK100/model_checkpoints/${MODEL_TYPE}_${DATASET}_${TASK_TYPE}

# verb=97, noun=300, action=3806