#!/usr/bin/env bash

#SBATCH --job-name=translate-NLLB-3.3B
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --mem=128GB
#SBATCH --output=./experiments/synthetic-data/logs/translate-NLLB-3.3B-%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1

echo "============="
echo "Job start: "
echo $(date) $(hostname) $(pwd)
echo "============="
python ./synthetic-data/gen_translations.py \
    --model="facebook/nllb-200-3.3b" \
    --nbits=4 \
    --seed=84 \
    --sample=15000 \
    --save_path="./synthetic-data/data/he-en-nllb3.3B.csv" \
    --batch_size=8

echo "============="
echo "Job end: "
echo $(date)
echo "============="