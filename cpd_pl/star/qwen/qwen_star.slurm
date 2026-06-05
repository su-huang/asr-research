#!/bin/bash
#SBATCH --job-name=star-qwen
#SBATCH --nodes=1
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --exclude=e02,e04
#SBATCH --account=a100acct
#SBATCH --output=/export/fs06/shuan148/asr-research/cpd_pl/errors_output/star-qwen-%j.out
#SBATCH --error=/export/fs06/shuan148/asr-research/cpd_pl/errors_output/star-qwen-%j.err
#SBATCH --mail-user=shuan148@jh.edu
#SBATCH --mail-type=BEGIN,END,FAIL

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/kchapar1/.local/share/mamba/envs/qwen3-asr

cd /export/fs06/kchapar1/bpd_asr/STAR-Active-Learning/

python finetune_qwen3_star.py \
    --MODEL         Qwen/Qwen3-ASR-1.7B \
    --DATASET       star-small \
    --TRAIN_CSV     /export/fs06/shuan148/asr-research/cpd_pl/star/qwen/small_train_gold.csv \
    --DEV_CSV       /export/fs06/shuan148/asr-research/cpd_pl/star/qwen/small_val_gold.csv \
    --TEST_CSV      /export/fs06/shuan148/asr-research/cpd_pl/star/qwen/small_test_gold.csv \
    --LAYER_ID      26 \
    --HEAD_ID       10 \
    --EPOCHS        100 \
    --GRAD_ACC      8 \
    --LR            2e-5 \
    --THRESOLD      2.0 \
    --TOP_PERCENT   0.8 \
    --TAU           10 \
    --SAVE_EVERY    884 \
    --PATIENCE      5 \
    --SAVE_DIR      /export/fs06/shuan148/asr-research/cpd_pl/models/qwen \
    --RUN_ID        $SLURM_JOB_ID

echo "Exit code: $?"