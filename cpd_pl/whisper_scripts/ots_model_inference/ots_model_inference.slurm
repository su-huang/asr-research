#!/bin/bash
#SBATCH --job-name=whisper_ots
#SBATCH --output=/export/fs06/shuan148/asr-research/cpd_pl/errors_output/whisper_ots_%j.out
#SBATCH --error=/export/fs06/shuan148/asr-research/cpd_pl/errors_output/whisper_ots_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu-a100
#SBATCH --account=a100acct
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shuan148@jh.edu

# --- Load modules ---
module purge
/bin/hostname

# --- Activate your conda environment ---
source /home/shuan148/miniconda3/etc/profile.d/conda.sh
conda activate whisper_env

# --- Move to your working directory ---
cd /export/fs06/shuan148/asr-research/cpd_pl/whisper_scripts/ots_model_inference/

# --- Run inference ---
python ots_model_inference.py \
    /export/fs06/shuan148/asr-research/cpd_pl/whisper_csv/test/test_gold_2.25hr.csv \
    --output_csv /export/fs06/shuan148/asr-research/cpd_pl/whisper_results/whisper_ots_${SLURM_JOB_ID}.csv
