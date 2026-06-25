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

PYTHONUNBUFFERED=1 python whisper_star.py \
    --MODEL             openai/whisper-large-v3 \
    --DATASET           star-additional \
    --TRAIN_DATA        /export/fs06/shuan148/asr-research/cpd_pl/star/whisper/whisper_kaldi/train/ \
    --DEV_DATA          /export/fs06/shuan148/asr-research/cpd_pl/star/whisper/whisper_kaldi/val/ \
    --EPOCHS            100 \
    --GRADIENT_ACCUMULATION_STEPS 16 \
    --LEARNING_RATE     1e-5 \
    --THRESOLD          2.0 \
    --TOP_PERCENT       1.0 \
    --TAU               10 \
    --SAVE_EVERY        1000 \
    --SAVE_DIR          /export/fs06/shuan148/asr-research/cpd_pl/models/whisper \
    --TEST_CSV          /export/fs06/shuan148/asr-research/cpd_pl/whisper_csv/test/test_gold_2.25hr.csv \
    --RUN_ID            $SLURM_JOB_ID

echo "Exit code: $?"
