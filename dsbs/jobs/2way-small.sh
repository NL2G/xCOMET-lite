#!/bin/bash
#SBATCH --job-name=dsbs-2way-small
#SBATCH --output=./logs/dsbs-2way-small-%A-[%a].txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu_4_h100,gpu_4_a100
#SBATCH --array=0-2

nvitop -1

WANDB_NAME="dsbs-2way-small-${SLURM_ARRAY_TASK_ID}" \
WANDB_PROJECT="dsbs" \
WANDB_RUN_GROUP="dsbs-2way-small" \
srun accelerate launch train.py \
    -c ./config/full2way.yaml \
    -o "dsbs-2way-small-${SLURM_ARRAY_TASK_ID}" \
    -s ${SLURM_ARRAY_TASK_ID}