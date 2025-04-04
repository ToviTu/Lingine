#!/bin/bash

export MY_HOME=/ib-scratch/chenguang03/scratch1/jianhong.t
export WANDB_PROJECT=MLAN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

#ratios="06l 125l 25l 375l 5l 625l 75l 875l"
ratios="875l"

# for ratio in $ratios;
# do
#     accelerate launch \
#         --config_file=configs/accelerate/8gpus/deepspeed_zero3.yaml \
#         scripts/mlan/llava_pretrain.py \
#         --model_name_or_path $MY_HOME/models/llava-llama-3_2-3b-instruct-spt \
#         --dataset_name $MY_HOME/datasets/VisionFlan/mix_0.$ratio.json \
#         --use_prompt_template True \
#         --group_by_modality True \
#         --image_path $MY_HOME/datasets/VisionFlan/images_191task_1k \
#         --attn_implementation flash_attention_2 \
#         --freeze_vision_model True \
#         --freeze_mm_projector False \
#         --freeze_language_model False \
#         --per_device_train_batch_size 8 \
#         --gradient_accumulation_steps 2 \
#         --weight_decay 0 \
#         --warmup_ratio 0.03 \
#         --num_train_epochs 1 \
#         --lr_scheduler_type cosine \
#         --learning_rate 2e-5 \
#         --logging_step 1 \
#         --save_strategy steps \
#         --save_steps 10000 \
#         --save_total_limit 1 \
#         --output_dir $MY_HOME/models/llava-llama-3_2-3b-instruct-$ratio \
#         --bf16 True \
#         --tf32 True \
#         --gradient_checkpointing True \
#         --dataloader_num_workers 8 \
#         --run_name llava-llama-3_2-3b-instruct-$ratio \
#         --do_train \
#         --report_to wandb 
# done

accelerate launch \
        --config_file=configs/accelerate/4gpus/deepspeed_zero3.yaml \
        scripts/mlan/llava_pretrain.py \
        --model_name_or_path $MY_HOME/models/llava-llama-3_2-3b-spt \
        --dataset_name $MY_HOME/datasets/VisionFlan/mix_0.75l_low_diversity2.json \
        --use_prompt_template True \
        --group_by_modality True \
        --image_path $MY_HOME/datasets/VisionFlan/images_191task_1k \
        --attn_implementation flash_attention_2 \
        --freeze_vision_model True \
        --freeze_mm_projector False \
        --freeze_language_model False \
        --per_device_train_batch_size 16 \
        --gradient_accumulation_steps 2 \
        --weight_decay 0 \
        --warmup_ratio 0.03 \
        --num_train_epochs 1 \
        --lr_scheduler_type cosine \
        --learning_rate 2e-5 \
        --logging_step 1 \
        --save_strategy steps \
        --save_steps 10000 \
        --save_total_limit 1 \
        --output_dir $MY_HOME/models/llava-llama-3_2-3b-75l_low_diversity2 \
        --bf16 True \
        --tf32 True \
        --gradient_checkpointing True \
        --dataloader_num_workers 8 \
        --run_name llava-llama-3_2-3b-75l_low_diversity2 \
        --do_train \
        --report_to wandb 

# accelerate launch \
#         --config_file=configs/accelerate/8gpus/deepspeed_zero3.yaml \
#         scripts/mlan/llava_pretrain.py \
#         --model_name_or_path $MY_HOME/models/llava-llama-3_2-3b-spt \
#         --dataset_name $MY_HOME/datasets/VisionFlan/mix_0.75l_low_diversity.json \
#         --use_prompt_template True \
#         --group_by_modality True \
#         --image_path $MY_HOME/datasets/VisionFlan/images_191task_1k \
#         --attn_implementation flash_attention_2 \
#         --freeze_vision_model True \
#         --freeze_mm_projector False \
#         --freeze_language_model False \
#         --per_device_train_batch_size 8 \
#         --gradient_accumulation_steps 2 \
#         --weight_decay 0 \
#         --warmup_ratio 0.03 \
#         --num_train_epochs 1 \
#         --lr_scheduler_type cosine \
#         --learning_rate 2e-5 \
#         --logging_step 1 \
#         --save_strategy steps \
#         --save_steps 10000 \
#         --save_total_limit 1 \
#         --output_dir $MY_HOME/models/llava-llama-3_2-3b-75l_low_diversity \
#         --bf16 True \
#         --tf32 True \
#         --gradient_checkpointing True \
#         --dataloader_num_workers 8 \
#         --run_name llava-llama-3_2-3b-75l_low_diversity \
#         --do_train \
#         --report_to wandb 