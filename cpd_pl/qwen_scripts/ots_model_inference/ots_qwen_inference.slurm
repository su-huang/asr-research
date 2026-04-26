#!/bin/bash
#SBATCH --job-name=qwen-ots
#SBATCH --nodes=1
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --exclude=e02,e04
#SBATCH --account=a100acct
#SBATCH --output=/export/fs06/shuan148/asr-research/cpd_pl/errors_output/qwen-ots-%j.out
#SBATCH --error=/export/fs06/shuan148/asr-research/cpd_pl/errors_output/qwen-ots-%j.err
#SBATCH --mail-user=shuan148@jh.edu
#SBATCH --mail-type=BEGIN,END,FAIL

INPUT_JSONL="/export/fs06/shuan148/asr-research/cpd_pl/qwen_jsonl/test/test_gold_2.25hr.jsonl"
MODEL_ID="/export/fs06/shuan148/asr-research/cpd_pl/models/qwen3-asr-1.7b"
OUTPUT_CSV="/export/fs06/shuan148/asr-research/cpd_pl/qwen_results/qwen_ots_%j.csv"

cat $0

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/kchapar1/.local/share/mamba/envs/qwen3-asr

cd /export/fs06/shuan148/asr-research/cpd_pl/qwen_scripts/ots_model_inference
which python

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python ots_qwen_inference.py \
    --input_jsonl  "$INPUT_JSONL" \
    --model_id     "$MODEL_ID"    \
    --output_csv   "$OUTPUT_CSV"

echo "Exit code: $?"
