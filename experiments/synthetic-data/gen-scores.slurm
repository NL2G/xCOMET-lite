#!/usr/bin/env bash

#SBATCH --job-name=gen_scores
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --mem=64GB
#SBATCH --output=./experiments/synthetic-data/logs/gen_scores_%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1

echo "============="
echo "Job start: "
echo $(date) $(hostname) $(pwd)
echo "============="
python ./synthetic-data/gen_scores.py \
    --data="synthetic-data/data/he-en-nllb600m.csv" \
    --metrics=bleurt --metrics=comet

echo "============="
echo "Job end: "
echo $(date)
echo "============="

echo "============="
echo "Job start: "
echo $(date) $(hostname) $(pwd)
echo "============="
python ./synthetic-data/gen_scores.py \
    --data="synthetic-data/data/he-en-nllb1.3B.csv" \
    --metrics=bleurt --metrics=comet

echo "============="
echo "Job end: "
echo $(date)
echo "============="

echo "============="
echo "Job start: "
echo $(date) $(hostname) $(pwd)
echo "============="
python ./synthetic-data/gen_scores.py \
    --data="synthetic-data/data/he-en-nllb3.3B.csv" \
    --metrics=bleurt --metrics=comet

echo "============="
echo "Job end: "
echo $(date)
echo "============="