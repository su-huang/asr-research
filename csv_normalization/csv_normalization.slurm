#!/bin/bash
#SBATCH --job-name=csv_norm     # Job name
#SBATCH --output=/export/fs06/shuan148/asr-research/csv_normalization/errors_output/csv_norm_%j.out        # Standard output log
#SBATCH --error=/export/fs06/shuan148/asr-research/csv_normalization/errors_output/csv_norm_%j.err         # Standard error log
#SBATCH --time=00:30:00                 # Max run time (HH:MM:SS)
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --mail-type=ALL                 # Notify when done/fails
#SBATCH --mail-user=shuan148@jh.edu     # Your emails

module purge
/bin/hostname

source /home/shuan148/miniconda3/etc/profile.d/conda.sh
conda activate csv_normalization

cd /export/fs06/shuan148/asr-research/csv_normalization/
python main.py 