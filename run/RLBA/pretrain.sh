#!/bin/bash

export WANDB_PROJECT=RLBA

MODEL_NAME=llava-llama-3_2-3b-instruct-rlba-beta0
accelerate launch \
    --config_file=configs/accelerate/8gpus/deepspeed_zero2.yaml \
    --main_process_port 29500 \
    scripts/rlba/rlba_pretrain.py \
    --model_name_or_path "$MY_HOME/models/llava-llama-3_2-3b-instruct" \
    --dataset_name "$MY_HOME/datasets/ShareGPT4V/llava/llava_pretrain/blip_laion_cc_sbu_558k.json" \
    --image_path "$MY_HOME/datasets/ShareGPT4V/llava/llava_pretrain/images" \
    --reward_model "zer0int/LongCLIP-GmP-ViT-L-14" \
    --attn_implementation flash_attention_2 \
    --freeze_vision_model True \
    --freeze_mm_projector False \
    --freeze_language_model True \
    --per_device_train_batch_size 32 \
    --num_generations 8 \
    --temperature 0.8 \
    --top_p 1.0 \
    --gradient_accumulation_steps 1 \
    --max_prompt_length 1024 \
    --max_completion_length 128 \
    --beta 0.00 \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_steps 4000 \
    --lr_scheduler_type cosine \
    --learning_rate 1e-4 \
    --logging_step 1 \
    --save_strategy steps \
    --save_steps 10000 \
    --save_total_limit 1 \
    --output_dir "$MY_HOME/models/$MODEL_NAME" \
    --bf16 True \
    --tf32 True  \
    --gradient_checkpointing \
    --dataloader_num_workers 8 \
    --run_name $MODEL_NAME \
    --do_train \
    --log_completions False \
    --report_to wandb
