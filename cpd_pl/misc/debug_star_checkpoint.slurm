#!/bin/bash
#SBATCH --job-name=debug-star
#SBATCH --nodes=1
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --exclude=e02,e04
#SBATCH --account=a100acct
#SBATCH --time=00:30:00
#SBATCH --output=/export/fs06/shuan148/asr-research/cpd_pl/errors_output/debug-star-%j.out
#SBATCH --error=/export/fs06/shuan148/asr-research/cpd_pl/errors_output/debug-star-%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate whisper_env

PYTHONUNBUFFERED=1 python /export/fs06/shuan148/asr-research/cpd_pl/misc/debug_star_checkpoint.py \
    --checkpoint_path /export/fs06/shuan148/asr-research/cpd_pl/models/whisper/star-small-1_large-v3_1672703/last_checkpoint.pth \
    --test_audio      /secure/fs00/afield6/police/chicago/wavs/wavs/sidX-utt000060.wav