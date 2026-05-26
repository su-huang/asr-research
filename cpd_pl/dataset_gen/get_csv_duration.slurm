#!/bin/bash
#SBATCH --job-name=get_csv_duration
#SBATCH --output=/export/fs06/shuan148/asr-research/cpd_pl/errors_output/get_csv_duration_whisper_pl_%j.out
#SBATCH --error=/export/fs06/shuan148/asr-research/cpd_pl/errors_output/get_csv_duration_whisper_pl_%j.err
#SBATCH --time=2:00:00
#SBATCH --partition=gpu-a100            # Partition (A100 GPU)
#SBATCH --account=a100acct
#SBATCH --gres=gpu:1                    # Request one A100 GPU
#SBATCH --mem=32G

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/kchapar1/.local/share/mamba/envs/qwen3-asr

python /export/fs06/shuan148/asr-research/cpd_pl/dataset_gen/get_csv_duration.py /export/fs06/shuan148/asr-research/cpd_pl/llm_judge/whisper/whisper_train_24hr_llm_results.csv
