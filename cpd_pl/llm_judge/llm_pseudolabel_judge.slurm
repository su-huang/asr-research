#!/bin/bash
#SBATCH --job-name=llm-pl-judge
#SBATCH --nodes=1
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --exclude=e02,e04
#SBATCH --account=a100acct
#SBATCH --output=/export/fs06/shuan148/asr-research/cpd_pl/errors_output/llm-pl-judge-%j.out
#SBATCH --error=/export/fs06/shuan148/asr-research/cpd_pl/errors_output/llm-pl-judge-%j.err
#SBATCH --mail-user=shuan148@jh.edu
#SBATCH --mail-type=BEGIN,END,FAIL

cat $0

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/kchapar1/.local/share/mamba/envs/qwen3-asr

cd /export/fs06/shuan148/asr-research/cpd_pl/llm_judge/
which python

PYTHONUNBUFFERED=1 python llm_pseudolabel_judge.py \
    --input_csv  /export/fs06/shuan148/asr-research/cpd_pl/llm_judge/qwen/qwen_train_24hr.csv \
    --output_csv /export/fs06/shuan148/asr-research/cpd_pl/llm_judge/qwen/qwen_train_pl_24hr_llm_results.csv \
    --model      meta-llama/Meta-Llama-3-8B-Instruct \
    --save_every 50

echo "Exit code: $?"
