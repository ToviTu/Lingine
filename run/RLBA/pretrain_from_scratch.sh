#!/bin/bash

export WANDB_PROJECT=RLBA

MODEL_NAME=llava-llama-3_2-1b-rlba
accelerate launch \
    --config_file=configs/accelerate/4gpus/deepspeed_zero2.yaml \
    --main_process_port 29501 \
    scripts/rlba/rlba_pretrain.py \
    --config_file configs/llava/llama-3_2-1b-instruct-224.json \
    --dataset_name "$MY_HOME/datasets/ShareGPT4V/sharegpt4v/share-captioner_coco_lcs_sam_1246k_1107.json" \
    --image_path "$MY_HOME/datasets/ShareGPT4V" \
    --reward_model "zer0int/LongCLIP-GmP-ViT-L-14" \
    --attn_implementation flash_attention_2 \
    --freeze_vision_model True \
    --freeze_mm_projector False \
    --freeze_language_model True \
    --per_device_train_batch_size 1 \
    --num_generations 32 \
    --temperature 1.3 \
    --gradient_accumulation_steps 1 \
    --max_prompt_length 1024 \
    --max_completion_length 128 \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_steps 1000 \
    --lr_scheduler_type cosine \
    --learning_rate 1e-3 \
    --logging_step 1 \
    --save_strategy steps \
    --save_steps 10000 \
    --save_total_limit 1 \
    --output_dir "$MY_HOME/models/$MODEL_NAME" \
    --bf16 \
    --tf32 true \
    --gradient_checkpointing \
    --dataloader_num_workers 8 \
    --run_name $MODEL_NAME \
    --do_train \
    --verbose True \
    --report_to wandb
