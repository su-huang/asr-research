#!/bin/bash
#SBATCH --job-name=get_csv_duration
#SBATCH --output=/export/fs06/shuan148/asr-research/cpd_pl/errors_output/duration-star-qwen-%j.out
#SBATCH --error=/export/fs06/shuan148/asr-research/cpd_pl/errors_output/duration-star-qwen-%j.err
#SBATCH --partition=gpu-a100            # Partition (A100 GPU)
#SBATCH --account=a100acct
#SBATCH --gres=gpu:1                    # Request one A100 GPU
#SBATCH --mem=32G

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/kchapar1/.local/share/mamba/envs/qwen3-asr

python /export/fs06/shuan148/asr-research/cpd_pl/misc/analysis/get_csv_duration.py /export/fs06/shuan148/asr-research/cpd_pl/models/qwen/star-1/training_data_audit.csv