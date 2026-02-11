#!/bin/bash
#SBATCH --job-name=whisper_zeroshot_and_finetune         # Job name
#SBATCH --output=/export/fs06/shuan148/asr-research/cpd_audio/zeroshot_and_finetuned_on_testset/errors_output/whisper_%j.out         # Standard output log
#SBATCH --error=/export/fs06/shuan148/asr-research/cpd_audio/zeroshot_and_finetuned_on_testset/errors_output/whisper_%j.err          # Standard error log
#SBATCH --time=36:00:00                 # Max run time (HH:MM:SS)
#SBATCH --partition=gpu-a100                 # GPU partition
#SBATCH --account=a100acct
#SBATCH --gres=gpu:1                    # Number of GPUs
#SBATCH --cpus-per-task=8               # Number of CPU cores
#SBATCH --mem=32G                       # Memory
#SBATCH --mail-type=ALL            # Notify when done/fails
#SBATCH --mail-user=shuan148@jh.edu    # Your email

# --- Load your conda ---
source /home/shuan148/miniconda3/etc/profile.d/conda.sh
conda activate cpd_zeroshot_finetuned

# --- Move to your working directory ---
cd /export/fs06/shuan148/asr-research/cpd_audio/zeroshot_and_finetuned_on_testset

# --- Run your script ---
python zeroshot_and_finetuned_on_testset.py

