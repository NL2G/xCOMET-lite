#!/usr/bin/env bash

#SBATCH --job-name=cl-no-labse-train-regressor
#SBATCH --gres=gpu:8
#SBATCH --partition=intel-gpu
#SBATCH --time=72:00:00
#SBATCH --mem=256GB
#SBATCH --output=./outputs-%j.txt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1

echo "Starting at `date` on `hostname` at `pwd`"
echo "Job name: $SLURM_JOB_NAME Job ID: $SLURM_JOB_ID"
echo "==============================="
nvidia-smi
echo "==============================="
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo "==============================="
accelerate launch --config_file="8xgpu.yaml" train.py --checkpoint="wmtcl.ckpt" --no-use-labse --save-file-name="mt0-large-wmtcl-no-labse.pth"
