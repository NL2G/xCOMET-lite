#!/bin/bash
#SBATCH --job-name=dsbs-3way-debug
#SBATCH --output=./logs/dsbs-3way-debug-%A-[%a].txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_4_h100,gpu_4_a100

nvitop -1

WANDB_NAME="dsbs-3way-debug" \
WANDB_PROJECT="dsbs" \
WANDB_RUN_GROUP="dsbs-3way-debug" \
srun accelerate launch --config_file="./config/1xgpu.yaml" train.py \
    -c ./config/full3way.yaml \
    -o "dsbs-3way-debug" \
    -s 0