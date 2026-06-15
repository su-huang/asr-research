#!/bin/bash
#SBATCH --job-name=star-whisper-inference
#SBATCH --nodes=1
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --exclude=e02,e04
#SBATCH --account=a100acct
#SBATCH --output=/export/fs06/shuan148/asr-research/cpd_pl/errors_output/star-whisper-inference-%j.out
#SBATCH --error=/export/fs06/shuan148/asr-research/cpd_pl/errors_output/star-whisper-inference-%j.err
#SBATCH --mail-user=shuan148@jh.edu
#SBATCH --mail-type=BEGIN,END,FAIL

source ~/miniconda3/etc/profile.d/conda.sh
conda activate whisper_env

cd /export/fs06/shuan148/asr-research/cpd_pl/star/whisper/

python whisper_star_inference_from_checkpoint.py \
    --model_path /export/fs06/shuan148/asr-research/cpd_pl/models/whisper/star-1/best_checkpoint.pth \
    --test_csv   /export/fs06/shuan148/asr-research/cpd_pl/whisper_csv/test/test_gold_2.25hr.csv \
    --output_csv /export/fs06/shuan148/asr-research/cpd_pl/models/whisper/star-1/test_set_transcript.csv 