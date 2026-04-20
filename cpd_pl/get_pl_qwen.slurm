#!/bin/bash
#SBATCH --job-name=qwen_pseudolabel
#SBATCH --output=/export/fs06/shuan148/asr-research/cpd_pl/errors_output/qwen_pseudolabel_%j.out
#SBATCH --error=/export/fs06/shuan148/asr-research/cpd_pl/errors_output/qwen_pseudolabel_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/kchapar1/.local/share/mamba/envs/qwen3-asr

python /export/fs06/kchapar1/bpd_asr/pyfiles/get_pseudolabels_qwen3.py \
    --input_csv /export/fs06/shuan148/asr-research/cpd_pl/24hr.csv \
    --pl_csv_save_path /export/fs06/shuan148/asr-research/cpd_pl/24hr_pl.csv
