#!/bin/bash

export WANDB_PROJECT=RLBA
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

accelerate launch \
    --config_file=configs/accelerate/4gpus/deepspeed_zero2.yaml \
    --main_process_port 29501 \
    scripts/mlan/llava_pretrain.py \
    --config_file configs/llava/llama-3_2-3b-instruct-336.json \
    --mean_resizing True \
    --dataset_name "$MY_HOME/datasets/ShareGPT4V/llava/llava_pretrain/blip_laion_cc_sbu_558k.json" \
    --use_prompt_template False \
    --image_path "$MY_HOME/datasets/ShareGPT4V/llava/llava_pretrain/images" \
    --attn_implementation flash_attention_2 \
    --freeze_vision_model True \
    --freeze_mm_projector False \
    --freeze_language_model True \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --num_train_epochs 1 \
    --lr_scheduler_type cosine \
    --learning_rate 1e-3 \
    --logging_step 1 \
    --save_strategy steps \
    --save_steps 10000 \
    --save_total_limit 1 \
    --output_dir "$MY_HOME/models/llava-llama-3_2-3b-instruct" \
    --bf16 True \
    --tf32 True \
    --gradient_checkpointing \
    --dataloader_num_workers 8 \
    --run_name llava-llama-3_2-3b-instruct \
    --do_train \
    --report_to wandb