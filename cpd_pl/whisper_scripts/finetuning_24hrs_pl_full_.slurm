#!/bin/bash
#SBATCH --job-name=ft-24hrs-pl-full
#SBATCH --nodes=1
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --exclude=e02,e04
#SBATCH --account=a100acct
#SBATCH --output=/export/fs06/shuan148/asr-research/cpd_pl/errors_outputft-24hrs-pl-full-%j.out
#SBATCH --error=/export/fs06/shuan148/asr-research/cpd_pl/errors_outputft-24hrs-pl-full-%j.err
#SBATCH --mail-user=shuan148@jh.edu
#SBATCH --mail-type=BEGIN,END,FAIL

cat $0

source /etc/profile
source ~/.bashrc

module load ffmpeg
mamba activate /home/kchapar1/miniconda3/envs/whisper-finetune

cd /export/fs06/kchapar1/bpd_asr/pyfiles/
which python

python finetune_whisper.py \
    large-v3 \
    100 \
    8 \
    /export/fs06/kchapar1/bpd_asr/datasets/datasets_with_paths/datasets_with_paths_nonzero_durations/non_null_transcripts/pseudo_labeled_data/whisper-lgv3_PLs_from_base_PLs_iter_1_csv.csv \
    /export/fs06/kchapar1/bpd_asr/datasets/datasets_with_paths/datasets_with_paths_nonzero_durations/non_null_transcripts/fnlo_splits_preprocessed_w_largev3 \
    /export/fs06/kchapar1/bpd_asr/finetuned_models/finetuned_on_pseudolabeled_data/lv3-24hr-PLs-full \
    --eval_steps 884

echo "Exit code: $?"
