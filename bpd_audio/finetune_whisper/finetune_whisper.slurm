#!/bin/bash
#SBATCH --job-name=whisper_finetune         # Job name
#SBATCH --output=errors_output/whisper_%j.out         # Standard output log
#SBATCH --error=errors_output/whisper_%j.err          # Standard error log
#SBATCH --time=36:00:00                 # Max run time (HH:MM:SS)
#SBATCH --partition=gpu-a100                 # GPU partition
#SBATCH --account=a100acct
#SBATCH --gres=gpu:1                    # Number of GPUs
#SBATCH --cpus-per-task=8               # Number of CPU cores
#SBATCH --mem=32G                       # Memory
#SBATCH --mail-type=ALL            # Notify when done/fails
#SBATCH --mail-user=shuan148@jh.edu    # Your email

# --- Load your conda ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bpd_finetune

# --- Move to your working directory ---
cd /export/fs06/shuan148/bpd_audio/finetune_whisper/ 

# --- Run your script ---
# python finetune_whisper.py small 5 8
python finetune_whisper.py large-v3 5 8
