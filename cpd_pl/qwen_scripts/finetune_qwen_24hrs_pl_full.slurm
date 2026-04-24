#!/bin/bash
#SBATCH --job-name=qwen-ft-24hrs_pl_full
#SBATCH --nodes=1
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --exclude=e02,e04
#SBATCH --account=a100acct
#SBATCH --output=/export/fs06/shuan148/asr-research/cpd_pl/errors_output/qwen-ft-24hrs_pl_full-%j.out
#SBATCH --error=/export/fs06/shuan148/asr-research/cpd_pl/errors_output/qwen-ft-24hrs_pl_full-%j.err
#SBATCH --mail-user=shuan148@jh.edu
#SBATCH --mail-type=BEGIN,END,FAIL

cat $0

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/kchapar1/.local/share/mamba/envs/qwen3-asr

cd /export/fs06/kchapar1/bpd_asr/pyfiles/
which python

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python qwen3_asr_sft.py \
    --model_path /export/fs06/shuan148/asr-research/cpd_pl/models/qwen3-asr-1.7b \
    --train_file /export/fs06/shuan148/asr-research/cpd_pl/qwen_jsonl/train/train_pl_24hr.jsonl \
    --eval_file /export/fs06/shuan148/asr-research/cpd_pl/qwen_jsonl/val/val_gold_1.25hr.jsonl \
    --output_dir /export/fs06/shuan148/asr-research/cpd_pl/qwen_results/qwen_finetune_24hrs_pl_full \
    --batch_size 4 \
    --grad_acc 8 \
    --lr 2e-5 \
    --epochs 10 \
    --save_steps 713 \
    --save_total_limit 5 \
    --test_file /export/fs06/shuan148/asr-research/cpd_pl/qwen_jsonl/test/test_gold_2.25hr.jsonl

echo "Exit code: $?"
