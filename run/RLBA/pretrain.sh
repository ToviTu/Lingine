#!/bin/bash

export WANDB_PROJECT=RLBA

accelerate launch \
    --config_file=configs/accelerate/4gpus/deepspeed_zero2.yaml \
    --main_process_port 29501 \
    scripts/rlba/rlba_pretrain.py \
    --model_name_or_path "$MY_HOME/models/llava-llama-3_2-1b-instruct" \
    --dataset_name "$MY_HOME/datasets/ShareGPT4V/sharegpt4v/share-captioner_coco_lcs_sam_1246k_1107.json" \
    --image_path "$MY_HOME/datasets/ShareGPT4V" \
    --reward_model "zer0int/LongCLIP-GmP-ViT-L-14" \
    --supervised False \
    --supervised_frequency 3 \
    --attn_implementation flash_attention_2 \
    --freeze_vision_model True \
    --freeze_mm_projector False \
    --freeze_language_model True \
    --per_device_train_batch_size 1 \
    --num_generations 32 \
    --temperature 0.8 \
    --gradient_accumulation_steps 1 \
    --max_prompt_length 1024 \
    --max_completion_length 128 \
    --beta 0.04 \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_steps 1000 \
    --lr_scheduler_type cosine \
    --learning_rate 1e-4 \
    --logging_step 1 \
    --save_strategy steps \
    --save_steps 10000 \
    --save_total_limit 1 \
    --output_dir "$MY_HOME/models/llava-llama-3_2-1b-instruct-rlba" \
    --bf16 True \
    --tf32 True  \
    --gradient_checkpointing \
    --dataloader_num_workers 8 \
    --run_name llava-llama-3_2-1b-instruct-rlba \
    --do_train \
    --verbose True \
    --report_to wandb
