#!/bin/bash
#SBATCH --job-name=get_jsonl_duration
#SBATCH --output=/export/fs06/shuan148/asr-research/cpd_pl/errors_output/get_jsonl_duration_%j.out
#SBATCH --error=/export/fs06/shuan148/asr-research/cpd_pl/errors_output/get_jsonl_duration_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=32G

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/kchapar1/.local/share/mamba/envs/qwen3-asr

python /export/fs06/shuan148/asr-research/cpd_pl/get_jsonl_duration.py /export/fs06/kchapar1/bpd_asr/jsonls/fnlo_train.jsonl
