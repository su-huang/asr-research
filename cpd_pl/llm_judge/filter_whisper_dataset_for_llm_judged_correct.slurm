#!/bin/bash
#SBATCH --job-name=whisper-filter-pl
#SBATCH --nodes=1
#SBATCH --partition=cpu
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=/export/fs06/shuan148/asr-research/cpd_pl/errors_output/whisper-filter-pl-%j.out
#SBATCH --error=/export/fs06/shuan148/asr-research/cpd_pl/errors_output/whisper-filter-pl-%j.err
#SBATCH --mail-user=shuan148@jh.edu
#SBATCH --mail-type=BEGIN,END,FAIL

cat $0

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/kchapar1/.local/share/mamba/envs/qwen3-asr

cd /export/fs06/shuan148/asr-research/cpd_pl/llm_judge/
which python 

python filter_whisper_dataset_for_llm_judged_correct.py \
    --judge_csv /export/fs06/shuan148/asr-research/cpd_pl/llm_judge/qwen/qwen_train_pl_24hr_llm_results_3.csv \
    --output_dataset /export/fs06/shuan148/asr-research/cpd_pl/whisper_datasets/train/train_pl_24hrs_preprocessed_qwen_judged_3

echo "Exit code: $?"
