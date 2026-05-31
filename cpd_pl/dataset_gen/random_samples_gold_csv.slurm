#!/bin/bash
#SBATCH --job-name=random-samples
#SBATCH --output=/export/fs06/shuan148/asr-research/cpd_pl/errors_output/random-samples-test-24hr%j.out
#SBATCH --error=/export/fs06/shuan148/asr-research/cpd_pl/errors_output/random-samples-test-24hr%j.err
#SBATCH --nodes=1
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --exclude=e02,e04
#SBATCH --account=a100acct
#SBATCH --mail-user=shuan148@jh.edu
#SBATCH --mail-type=BEGIN,END,FAIL

source /home/shuan148/miniconda3/etc/profile.d/conda.sh
conda activate whisper_env

python /export/fs06/shuan148/asr-research/cpd_pl/dataset_gen/random_samples_gold_csv.py \
    "/secure/fs00/afield6/police/shuan148/test_wav.scp" \
    "/secure/fs00/afield6/police/chicago/data/data/test/text" \
    --output_csv "/export/fs06/shuan148/asr-research/cpd_pl/whisper_csv/test/test_gold_24hr_2.csv" \
    --target_hours 24.0

echo "Exit code: $?"