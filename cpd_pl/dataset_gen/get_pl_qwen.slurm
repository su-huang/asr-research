#!/bin/bash
#SBATCH --job-name=qwen_pseudolabel
#SBATCH --output=/export/fs06/shuan148/asr-research/cpd_pl/errors_output/qwen_pseudolabel_%j.out
#SBATCH --error=/export/fs06/shuan148/asr-research/cpd_pl/errors_output/qwen_pseudolabel_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu-a100            # Partition (A100 GPU)
#SBATCH --account=a100acct
#SBATCH --gres=gpu:1                    # Request one A100 GPU
#SBATCH --mem=32G

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/kchapar1/.local/share/mamba/envs/qwen3-asr

python /export/fs06/shuan148/asr-research/cpd_pl/dataset_gen/get_pl_qwen3.py \
    --input_csv /export/fs06/shuan148/asr-research/cpd_pl/misc_data/1.25hr_val.csv \
    --pl_csv_save_path /export/fs06/shuan148/asr-research/cpd_pl/whisper_csv/val/1.25hr.csv
