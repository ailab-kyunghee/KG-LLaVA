#!/bin/bash

#SBATCH --job-name=infer-llava
#SBATCH --gres=gpu:4
#SBATCH --time=160:00:00 # 160 hours
#SBATCH --output=infer-llava.out
#SBATCH --error=infer-llava.err
#SBATCH --nodelist=augi4
#SBATCH --partition=batch
#SBATCH --cpus-per-gpu=15
#SBATCH --mem-per-gpu=50G

eval "$(conda shell.bash hook)"
conda activate /data/user/anaconda3/envs/llava
cd /data/user/LLaVA

# Set CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1,2,3

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}


# Create the necessary directory for the output file
mkdir -p /data/user/LLaVA/results/

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path /data/user/LLaVA-old/mimic-nle-xinstruct-models/llava-lora-llama-kg-model-5thEpoch \
        --question-file /data/user/test_dataset.jsonl \
        --image-folder /mimic-cxr-jpg-nle/2.0.0/files/ \
        --answers-file /data/user/LLaVA/results/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=/data/user/LLaVA/results/test-output.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat /data/user/LLaVA/results/test-output${CHUNKS}_${IDX}.jsonl >> "$output_file"
done
