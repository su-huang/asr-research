#!/bin/bash
#SBATCH --job-name=star-whisper
#SBATCH --nodes=1
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --exclude=e02,e04
#SBATCH --account=a100acct
#SBATCH --output=/export/fs06/shuan148/asr-research/cpd_pl/errors_output/star-whisper-%j.out
#SBATCH --error=/export/fs06/shuan148/asr-research/cpd_pl/errors_output/star-whisper-%j.err
#SBATCH --mail-user=shuan148@jh.edu
#SBATCH --mail-type=BEGIN,END,FAIL

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/kchapar1/.local/share/mamba/envs/star-active-learning

cd /export/fs06/shuan148/asr-research/cpd_pl/star/whisper/

PYTHONUNBUFFERED=1 python whisper_star_from_filtered.py \
    --MODEL          openai/whisper-large-v3 \
    --DATASET        star-2 \
    --TRAIN_CSV      /export/fs06/shuan148/asr-research/cpd_pl/star/whisper/whisper_star_filtered.csv \
    --DEV_DATA       /export/fs06/shuan148/asr-research/cpd_pl/star/whisper/whisper_kaldi/val/ \
    --SAVE_EVERY     10 \
    --GRADIENT_ACCUMULATION_STEPS 4 \
    --LEARNING_RATE  1e-5 \
    --EPOCHS         100 \
    --PATIENCE       5 \
    --SAVE_DIR       /export/fs06/shuan148/asr-research/cpd_pl/models/whisper \
    --TEST_CSV       /export/fs06/shuan148/asr-research/cpd_pl/whisper_csv/test/test_gold_2.25hr.csv \
    --RUN_ID         $SLURM_JOB_ID

echo "Exit code: $?"
