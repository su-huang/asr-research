#!/bin/bash
#SBATCH --job-name=plot-lp-wer
#SBATCH --nodes=1
#SBATCH --partition=cpu
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --account=a100acct
#SBATCH --output=/export/fs06/shuan148/asr-research/cpd_pl/errors_output/plot-lp-wer-%j.out
#SBATCH --error=/export/fs06/shuan148/asr-research/cpd_pl/errors_output/plot-lp-wer-%j.err
#SBATCH --mail-user=shuan148@jh.edu
#SBATCH --mail-type=BEGIN,END,FAIL

source ~/miniconda3/etc/profile.d/conda.sh
conda activate whisper_env

cd /export/fs06/kchapar1/bpd_asr/pyfiles/

python plot_correlation.py \
    --csv        /export/fs06/shuan148/asr-research/cpd_pl/log_prob/qwen/qwen_train_pl_24hr_log_prob_filtered.csv \
    --text_col   text \
    --gt_col     ground_truth \
    --logprob_col avg_logprob

echo "Exit code: $?"