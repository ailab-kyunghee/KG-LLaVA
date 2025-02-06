#!/bin/bash
#SBATCH --job-name=llava-llama-lora-7b
#SBATCH --gres=gpu:4
#SBATCH --time=160:00:00 # 160 hour
#SBATCH --output=llava-llama-lora-7b.out
#SBATCH --error=llava-llama-lora-7b.err
#SBATCH --nodelist=augi4
#SBATCH --partition=batch
#SBATCH --cpus-per-gpu=15
#SBATCH --mem-per-gpu=50G

eval "$(conda shell.bash hook)"
conda activate /data/user/anaconda3/envs/llava
cd /data/user/LLaVA

srun deepspeed llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path liuhaotian/llava-v1.5-7b \
    --version v1 \
    --data_path ./mimic-nle-train-kg.json \
    --image_folder ./mimic-cxr-jpg/mimic-cxr-jpg/2.0.0/files/ \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-7b-task-lora-mimic-nle \
    --num_train_epochs 5 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
