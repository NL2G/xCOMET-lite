#!/usr/bin/env bash 
#SBATCH --job-name="prepare-data-mt0-labse" 
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00 
#SBATCH --mem=256GB 
#SBATCH --cpus-per-task=20
#SBATCH --exclude=worker-1,worker-2,worker-3,worker-4

echo "Starting at `date` on `hostname` at `pwd`"
echo "Job name: $SLURM_JOB_NAME Job ID: $SLURM_JOB_ID"
echo "==============================="
nvitop -1
echo "==============================="
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo "==============================="

python create_dataset.py