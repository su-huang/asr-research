#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --output=/export/fs06/shuan148/asr-research/cpd_pl/errors_output/preprocess-%j.out
#SBATCH --error=/export/fs06/shuan148/asr-research/cpd_pl/errors_output/preprocess-%j.err
#SBATCH --nodes=1
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --exclude=e02,e04
#SBATCH --account=a100acct
#SBATCH --mail-user=shuan148@jh.edu
#SBATCH --mail-type=BEGIN,END,FAIL

source /home/shuan148/miniconda3/etc/profile.d/conda.sh
conda activate whisper_env

cd /export/fs06/shuan148/asr-research/cpd_pl/dataset_gen/

python preprocess_dataset_from_csv.py \
    --input_csv /export/fs06/shuan148/asr-research/cpd_pl/whisper_csv/train/train_pl_24hr.csv \
    --output_path /export/fs06/shuan148/asr-research/cpd_pl/whisper_datasets/train/train_pl_24hr_preprocessed \
    --text "text"