#!/bin/bash
#SBATCH --job-name=get-gold-from-pl
#SBATCH --output=/export/fs06/shuan148/asr-research/cpd_pl/errors_output/get-gold-from-pl-%j.out
#SBATCH --error=/export/fs06/shuan148/asr-research/cpd_pl/errors_output/get-gold-from-pl-%j.err
#SBATCH --time=00:05:00
#SBATCH --partition=gpu-a100            # Partition (A100 GPU)
#SBATCH --account=a100acct
#SBATCH --gres=gpu:1                    # Request one A100 GPU
#SBATCH --mem=32G

source /home/shuan148/miniconda3/etc/profile.d/conda.sh
conda activate whisper_env

cd /export/fs06/shuan148/asr-research/cpd_pl/dataset_gen/

python get_gold_from_pl.py \
    --input_csv /export/fs06/shuan148/asr-research/cpd_pl/qwen_csv/train_pl_24hr.csv \
    --output_csv /export/fs06/shuan148/asr-research/cpd_pl/qwen_csv/train_gold_24hr.csv

echo "Job finished with exit code: $?"
