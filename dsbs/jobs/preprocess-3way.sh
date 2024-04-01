#!/bin/bash
#SBATCH --job-name=dsbs-pretokenize
#SBATCH --output=./logs/dsbs-pretokenize-%A.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --partition=single

srun python cache-data.py --config ./config/full3way.yaml
