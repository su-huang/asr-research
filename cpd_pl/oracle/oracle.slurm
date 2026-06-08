#!/bin/bash
#SBATCH --job-name=oracle-filter-whisper
#SBATCH --nodes=1
#SBATCH --partition=cpu
#SBATCH --mem=32G
#SBATCH --output=/export/fs06/shuan148/asr-research/cpd_pl/errors_output/oracle-filter-whisper-%j.out
#SBATCH --error=/export/fs06/shuan148/asr-research/cpd_pl/errors_output/oracle-filter-whisper-%j.err
#SBATCH --mail-user=shuan148@jh.edu
#SBATCH --mail-type=BEGIN,END,FAIL

source /home/shuan148/miniconda3/etc/profile.d/conda.sh
conda activate whisper_env

cd /export/fs06/shuan148/asr-research/cpd_pl/oracle

python oracle.py \
    --pl_csv        /export/fs06/shuan148/asr-research/cpd_pl/llm_judge/whisper/whisper_train_24hr.csv \
    --threshold     0.2 \
    --out_csv       /export/fs06/shuan148/asr-research/cpd_pl/oracle/whisper/whisper_train_24hr_oracle.csv \

echo "Exit code: $?"
