#!/bin/bash
#SBATCH --job-name=prepare_dataset
#SBATCH --output=/export/fs06/shuan148/asr-research/cpd_pl/errors_output/prepare_dataset_%j.out
#SBATCH --error=/export/fs06/shuan148/asr-research/cpd_pl/errors_output/prepare_dataset_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=gpu-a100
#SBATCH --account=a100acct
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --mail-user=shuan148@jh.edu
#SBATCH --mail-type=BEGIN,END,FAIL

source /home/shuan148/miniconda3/etc/profile.d/conda.sh
conda activate whisper_env

cd /export/fs06/shuan148/asr-research/cpd_pl/dataset_gen/

export PYTHONPATH=/export/fs06/kchapar1/bpd_asr/pyfiles:$PYTHONPATH
python /export/fs06/kchapar1/bpd_asr/pyfiles/prepare_pseudolabel_dataset_from_csv.py \
    --csv_path /export/fs06/shuan148/asr-research/cpd_pl/whisper_csv/val/val_gold_1.25hr.csv \
    --save_path /export/fs06/shuan148/asr-research/cpd_pl/whisper_datasets/val_gold_1.25hr \
    --processor_id openai/whisper-large-v3