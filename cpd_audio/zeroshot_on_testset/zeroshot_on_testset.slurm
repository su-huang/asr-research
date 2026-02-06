#!/bin/bash
#SBATCH --job-name=zeroshot_cpd     # Job name
#SBATCH --output=errors_output/zeroshot_%j.out        # Standard output log
#SBATCH --error=errors_output/zeroshot_%j.err         # Standard error log
#SBATCH --time=36:00:00                 # Max run time (HH:MM:SS)
#SBATCH --partition=gpu-a100            # Partition (A100 GPU)
#SBATCH --account=a100acct
#SBATCH --gres=gpu:1                    # Request one A100 GPU
#SBATCH --cpus-per-task=8               # Number of CPU cores
#SBATCH --mem=64G                       # Memory
#SBATCH --mail-type=ALL                 # Notify when done/fails
#SBATCH --mail-user=shuan148@jh.edu     # Your emails

# --- Load modules ---
module purge
/bin/hostname

# --- Activate your conda environment ---
source /home/shuan148/miniconda3/etc/profile.d/conda.sh
conda activate cpd_zeroshot

# --- Move to your working directory ---
cd /export/fs06/shuan148/cpd_audio/zeroshot_on_testset/

# --- Run your zeroshot script ---
python zeroshot_on_testset.py \
    --scp_file /export/fs06/shuan148/cpd_audio/data/test/test_wav.scp \
    --text_file /secure/fs00/afield6/police/chicago/data/data/test/text \
    --outfile results/whisper_lgv3_test.csv \
    --do_sample False \
    --temp 1.0 \
    --top_p 1.0