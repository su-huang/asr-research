#!/bin/bash
#SBATCH --job-name=cross_test    # Job name
#SBATCH --output=/export/fs06/shuan148/asr-research/bpd_cpd_cross_test/errors_output/cross_test_%j.out        # Standard output log
#SBATCH --error=/export/fs06/shuan148/asr-research/bpd_cpd_cross_test/errors_output/cross_test_%j.err         # Standard error log
#SBATCH --time=36:00:00                 # Max run time (HH:MM:SS)
#SBATCH --partition=gpu-a100            # Partition (A100 GPU)
#SBATCH --account=a100acct
#SBATCH --gres=gpu:1                    # Request one A100 GPU
#SBATCH --cpus-per-task=8               # Number of CPU cores
#SBATCH --mem=64G                       # Memory
#SBATCH --mail-type=ALL                 # Notify when done/fails
#SBATCH --mail-user=shuan148@jh.edu     # Your email

module purge
/bin/hostname

source ~/anaconda3/etc/profile.d/conda.sh
conda activate cross_test

cd /export/fs06/shuan148/asr-research/bpd_cpd_cross_test/  

# Format for status args: test train val (as booleans)
python /export/fs06/shuan148/asr-research/bpd_cpd_cross_test/main_cross_test.py \
    --outfile "/export/fs06/shuan148/asr-research/bpd_cpd_cross_test/results/cpd_model_evaluation_results_$(date +%Y%m%d).csv" \
    --finetuned_model_path "/export/fs06/shuan148/asr-research/cpd_audio/finetune_whisper/finetuned_models/whisper-large-v3-finetuned" \
    --cpd_file_status True False False \
    --bpd_file_status True True True

