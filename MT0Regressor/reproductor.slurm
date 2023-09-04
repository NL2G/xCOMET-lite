#!/usr/bin/env bash 
#SBATCH --job-name="reproduce-the-error" 
#SBATCH --time=3:00:00 
#SBATCH --mem=256GB 
#SBATCH --cpus-per-task=20
#SBATCH --nodelist=worker-4

echo "Starting at `date` on `hostname` at `pwd`"
echo "Job name: $SLURM_JOB_NAME Job ID: $SLURM_JOB_ID"
echo "==============================="
nvitop -1
echo "==============================="
echo "Script-side visible GPU ids: [$CUDA_VISIBLE_DEVICES]"
echo "==============================="
echo 'Python-side visible gpus count: '
python -c "import torch, os, rich;\
    print('Device count ', torch.cuda.device_count());\
    print('Is cuda available? ',torch.cuda.is_available());\
    rich.print(vars(os.environ))"
echo "==============================="