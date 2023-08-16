#!/usr/bin/env bash

#SBATCH --job-name=EvalBloomz560m
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=128GB
#SBATCH --output=./experiments/prompt-regressor/logs/eval_%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --exclude=worker-8


echo "============="
echo "Job start: "
echo $(date) $(hostname) $(pwd)
echo "============="

python prompt-regressor/evaluate.py --path experiments/prompt-regressor/model --n-bits=16 --activation=sigmoid --batch-size=64