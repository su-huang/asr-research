#!/bin/bash
#SBATCH --job-name=plot-wer-by-duration
#SBATCH --output=/export/fs06/shuan148/asr-research/cpd_pl/errors_output/plot-wer-by-duration-%j.out
#SBATCH --error=/export/fs06/shuan148/asr-research/cpd_pl/errors_output/plot-wer-by-duration-%j.err
#SBATCH --partition=gpu-a100            # Partition (A100 GPU)
#SBATCH --account=a100acct
#SBATCH --gres=gpu:1                    # Request one A100 GPU
#SBATCH --mem=32G

source ~/miniconda3/etc/profile.d/conda.sh
conda activate whisper_env

python /export/fs06/shuan148/asr-research/cpd_pl/dataset_gen/plot_wer_by_duration.py \
    --csvs "qwen pl:/export/fs06/shuan148/asr-research/cpd_pl/misc/unfiltered_qwen_pl.csv" \
    --output /export/fs06/shuan148/asr-research/cpd_pl/misc/qwen_pl_wer_by_duration.png