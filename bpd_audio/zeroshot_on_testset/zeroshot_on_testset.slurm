#!/bin/bash
#SBATCH --job-name=zeroshot_whisper     # Job name
#SBATCH --output=errors_output/zeroshot_%j.out        # Standard output log
#SBATCH --error=errors_output/zeroshot_%j.err         # Standard error log
#SBATCH --time=12:00:00                 # Max run time (HH:MM:SS)
#SBATCH --partition=gpu-a100            # Partition (A100 GPU)
#SBATCH --account=a100acct
#SBATCH --gres=gpu:1                    # Request one A100 GPU
#SBATCH --cpus-per-task=8               # Number of CPU cores
#SBATCH --mem=64G                       # Memory
#SBATCH --mail-type=ALL                 # Notify when done/fails
#SBATCH --mail-user=shuan148@jh.edu     # Your email

# --- Load modules ---
module purge
/bin/hostname

# --- Activate your conda environment ---
source /home/shuan148/miniconda3/etc/profile.d/conda.sh
conda activate bpd_zeroshot

# --- Move to your working directory ---
cd /export/fs06/shuan148/asr-research/bpd_audio/zeroshot_on_testset/ 

# --- Run your zeroshot script ---
python zeroshot_on_testset.py \
    --outfile "/export/fs06/shuan148/asr-research/bpd_audio/zeroshot_on_testset/zeroshot_whisper_on_new_testset.csv" \
    --do_sample False \
    --temp 1.0 \
    --top_p 1.0