#!/usr/bin/env bash

#SBATCH --job-name=no-labse-train-regressor
#SBATCH --gres=gpu:4
#SBATCH --partition=intel-gpu
#SBATCH --time=72:00:00
#SBATCH --mem=256GB
#SBATCH --output=./outputs-%j.txt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1

echo "Starting at `date` on `hostname` at `pwd`"
echo "Job name: $SLURM_JOB_NAME Job ID: $SLURM_JOB_ID"
echo "==============================="
nvidia-smi
echo "==============================="
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo "==============================="
accelerate launch --config_file="4xgpu.yaml" train.py --no-use-labse --save-file-name="mt0-large-no-labse.pth" --batch-size=8
