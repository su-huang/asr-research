#!/bin/bash
#SBATCH --job-name=whisper-pseudolabel
#SBATCH --nodes=1
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --exclude=e02,e04
#SBATCH --account=a100acct
#SBATCH --output=/export/fs06/shuan148/asr-research/cpd_pl/errors_output/whisper-pseudolabel-%j.out
#SBATCH --error=/export/fs06/shuan148/asr-research/cpd_pl/errors_output/whisper-pseudolabel-%j.err
#SBATCH --mail-user=shuan148@jh.edu
#SBATCH --mail-type=BEGIN,END,FAIL

source /home/shuan148/miniconda3/etc/profile.d/conda.sh
conda activate whisper_env

cd /export/fs06/shuan148/asr-research/cpd_pl/dataset_gen/

python get_whisper_pseudolabels.py \
    --model_path openai/whisper-large-v3 \
    --input_csv /export/fs06/shuan148/asr-research/cpd_pl/whisper_csv/train/train_gold_24hr.csv \
    --pl_csv_save_path /export/fs06/shuan148/asr-research/cpd_pl/whisper_csv/train/train_pl_24hr_3.csv \
    --batch_size 32

echo "Job finished with exit code: $?"
