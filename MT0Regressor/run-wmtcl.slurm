#!/usr/bin/env bash

#SBATCH --job-name=cl-labse-train-regressor
#SBATCH --gres=gpu:2
#SBATCH --partition=intel-a100-pci3
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
accelerate launch --config_file="2xgpu.yaml" train.py --checkpoint="./wmtcl.ckpt" --save-file-name="mt0-large-wmtcl-labse.pth"
