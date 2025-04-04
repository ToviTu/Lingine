MY_HOME=/ib-scratch/chenguang03/scratch1/jianhong.t

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 \
accelerate launch \
    --config_file=configs/accelerate/default_config.yaml \
    scripts/mlan/llava_pretrain.py \
    --config_file configs/llava/llama-3_2-1b-instruct.json \
    --dataset_name "$MY_HOME/datasets/ShareGPT4V/sharegpt4v/share-captioner_coco_lcs_sam_1246k_1107.json" \
    --image_path "$MY_HOME/datasets/ShareGPT4V" \
    --attn_implementation flash_attention_2 \
    --freeze_vision_model True \
    --freeze_mm_projector False \
    --freeze_language_model True \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_steps 500 \
    --num_train_epochs 1 \
    --lr_scheduler_type cosine \
    --learning_rate 1e-3 \
    --logging_step 1 \
    --save_strategy no \
    --save_steps 10000 \
    --save_total_limit 1 \
    --output_dir "$MY_HOME/models/llava-llama-3_2-1b-spt" \
    --bf16 \
    --gradient_checkpointing \
    --dataloader_num_workers 8 \
    --run_name llava-llama-3_2-1b-spt \
    --do_train \
    --report_to wandb