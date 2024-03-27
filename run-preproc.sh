#!/bin/bash
#SBATCH --job-name=preproc
#SBATCH --output=./logs/preproc-%A-[%a].txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --partition=single

python check.py