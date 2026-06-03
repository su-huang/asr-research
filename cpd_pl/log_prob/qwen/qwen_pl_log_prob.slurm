#!/bin/bash
#SBATCH --job-name=log_prob_qwen_pl
#SBATCH --nodes=1
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --exclude=e02,e04
#SBATCH --account=a100acct
#SBATCH --output=/export/fs06/shuan148/asr-research/cpd_pl/errors_output/log-prob-qwen-pl-%j.out
#SBATCH --error=/export/fs06/shuan148/asr-research/cpd_pl/errors_output/log-prob-qwen-pl-%j.err
#SBATCH --mail-user=shuan148@jh.edu
#SBATCH --mail-type=BEGIN,END,FAIL

cat $0

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/kchapar1/.local/share/mamba/envs/qwen3-asr

cd /export/fs06/shuan148/asr-research/cpd_pl/log_prob/qwen/

python qwen_pl_log_prob.py \
    --model_path       Qwen/Qwen3-ASR-1.7B \
    --input_csv        /export/fs06/shuan148/asr-research/cpd_pl/qwen_csv/train_gold_24hr.csv \
    --audio_column     audio \
    --pl_csv_save_path /export/fs06/shuan148/asr-research/cpd_pl/log_prob/qwen/train_pl_24hr.csv \
    --batch_size       8 \
    --include_gt \
    --gt_column        text