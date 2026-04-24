#!/bin/bash
#SBATCH --job-name=ft-gold-full
#SBATCH --nodes=1
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --exclude=e02,e04
#SBATCH --account=a100acct
#SBATCH --output=/export/fs06/shuan148/asr-research/cpd_pl/errors_output/ft-gold-full-%j.out
#SBATCH --error=/export/fs06/shuan148/asr-research/cpd_pl/errors_output/ft-gold-full-%j.err
#SBATCH --mail-user=shuan148@jh.edu
#SBATCH --mail-type=BEGIN,END,FAIL

cat $0

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/kchapar1/.local/share/mamba/envs/qwen3-asr

cd /export/fs06/kchapar1/bpd_asr/pyfiles/
which python

# model_id_or_path, num_epochs, batch_size, training_data_path, eval_dataset_path, save_path 
python finetune_whisper.py \
    large-v3 \
    100 \
    8 \
    /export/fs06/shuan148/asr-research/cpd_pl/whisper_datasets/train/train_gold_24hr \
    /export/fs06/shuan148/asr-research/cpd_pl/whisper_datasets/hf_dict/hf_dict_val_gold_1.25hr_test_gold_2.25hr \
    /export/fs06/shuan148/asr-research/cpd_pl/models/whisper/gold-full \
    --eval_steps 884

echo "Exit code: $?"
