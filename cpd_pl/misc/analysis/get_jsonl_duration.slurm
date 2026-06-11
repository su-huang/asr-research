#!/bin/bash
#SBATCH --job-name=get_jsonl_duration
#SBATCH --output=/export/fs06/shuan148/asr-research/cpd_pl/errors_output/get_jsonl_duration_qwen_judged_70b_%j.out
#SBATCH --error=/export/fs06/shuan148/asr-research/cpd_pl/errors_output/get_jsonl_duration_qwen_judged_70b_%j.err
#SBATCH --time=2:00:00
#SBATCH --partition=gpu-a100            # Partition (A100 GPU)
#SBATCH --account=a100acct
#SBATCH --gres=gpu:1                    # Request one A100 GPU
#SBATCH --mem=32G

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/kchapar1/.local/share/mamba/envs/qwen3-asr

python /export/fs06/shuan148/asr-research/cpd_pl/misc/analysis/get_jsonl_duration.py /export/fs06/shuan148/asr-research/cpd_pl/qwen_jsonl/train/qwen_train_pl_24hr_llm_judged_70b.jsonl
