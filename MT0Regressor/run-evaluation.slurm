#!/usr/bin/env bash

#SBATCH --job-name="Evaluate labse"
#SBATCH --gres=gpu:1
#SBATCH --mem=256GB
#SBATCH --cpus-per-task=20
#SBATCH --time=2:00:00
#SBATCH --begin=now+4hours

echo "Starting at `date` on `hostname` at `pwd`"
echo "Job name: $SLURM_JOB_NAME Job ID: $SLURM_JOB_ID"
echo "==============================="
nvitop -1
echo "==============================="
echo "Script-side visible GPU ids: [$CUDA_VISIBLE_DEVICES]"
echo "==============================="

python evaluate.py --model-file="mt0-large-wmtcl-labse.pth" --use-labse
