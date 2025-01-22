#!/bin/bash

# The following commented code needs to be uncommented when running locally
# SM_NUM_GPUS=8
# NODE_NUMBER=1
# NODE_INDEX=0
# SM_MASTER_ADDR=127.0.0.1

DISTRIBUTED_ARGS="--nproc_per_node $SM_NUM_GPUS --nnodes $NODE_NUMBER --node_rank $NODE_INDEX --master_addr $SM_MASTER_ADDR --master_port 12345"

torchrun ${DISTRIBUTED_ARGS} \
    whisper_finetuning.py \
    --model_name_or_path="openai/whisper-large-v3-turbo" \
    --dataset_dir="/opt/ml/input/data/train" \
    --json_file="gt_transcript.json" \
    --language="zh" \
    --task="transcribe" \
    --train_split_name="train+validation" \
    --max_steps="50" \
    --output_dir="/tmp/finetuned_model" \
    --per_device_train_batch_size="2" \
    --per_device_eval_batch_size="2" \
    --logging_steps="25" \
    --learning_rate="1e-5" \
    --warmup_steps="20" \
    --eval_split_name="test" \
    --eval_strategy="steps" \
    --eval_steps="50" \
    --save_strategy="steps" \
    --save_steps="20" \
    --generation_max_length="225" \
    --preprocessing_num_workers="16" \
    --max_duration_in_seconds="30" \
    --text_column_name="sentence" \
    --freeze_feature_encoder="False" \
    --gradient_checkpointing \
    --fp16 \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --predict_with_generate \
    --trust_remote_code True