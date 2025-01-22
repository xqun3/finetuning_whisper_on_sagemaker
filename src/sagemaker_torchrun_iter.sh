#!/bin/bash

# The following commented code needs to be uncommented when running locally
# SM_NUM_GPUS=8
# NODE_NUMBER=1
# NODE_INDEX=0
# SM_MASTER_ADDR=127.0.0.1

DISTRIBUTED_ARGS="--nproc_per_node $SM_NUM_GPUS --nnodes $NODE_NUMBER --node_rank $NODE_INDEX --master_addr $SM_MASTER_ADDR --master_port 12345"

torchrun $DISTRIBUTED_ARGS \
    whisper_finetuning_iter.py \
    --deepspeed ds_z0_config.json \
    --model_name_or_path="openai/whisper-large-v3" \
    --dataset_dir="/opt/ml/input/data/train" \
    --dataset_config_name="zh-CN" \
    --json_file="gt_transcript.json" \
    --language="zh" \
    --task="transcribe" \
    --train_split_name="train+validation" \
    --max_steps="1024" \
    --dataloader_drop_last True \
    --output_dir="/tmp/finetuned_model" \
    --per_device_train_batch_size="16" \
    --per_device_eval_batch_size="16" \
    --logging_steps="25" \
    --max_eval_samples "128" \
    --learning_rate="1e-5" \
    --warmup_steps="500" \
    --eval_split_name="eval" \
    --eval_strategy="steps" \
    --eval_steps="50" \
    --save_strategy="steps" \
    --save_steps="200" \
    --generation_max_length="225" \
    --preprocessing_num_workers="16" \
    --text_column_name="sentence" \
    --freeze_feature_encoder="False" \
    --gradient_checkpointing \
    --fp16 \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --predict_with_generate \
    --trust_remote_code True

