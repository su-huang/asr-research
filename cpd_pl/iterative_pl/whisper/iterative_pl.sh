#!/bin/bash
#SBATCH --job-name=ipl-whisper-llm
#SBATCH --nodes=1
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --exclude=e02,e04
#SBATCH --account=a100acct
#SBATCH --output=/export/fs06/shuan148/asr-research/cpd_pl/errors_output/ipl-whisper-%j.out
#SBATCH --error=/export/fs06/shuan148/asr-research/cpd_pl/errors_output/ipl-whisper-%j.err
#SBATCH --mail-user=shuan148@jh.edu
#SBATCH --mail-type=BEGIN,END,FAIL

source ~/miniconda3/etc/profile.d/conda.sh
source ~/.bashrc

cd /export/fs06/shuan148/asr-research/cpd_pl/

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

MODEL_PATH="/export/fs06/shuan148/asr-research/cpd_pl/models/whisper-large-v3"

EPOCHS=100
BATCH_SIZE=8

EVAL_TEST_DATASET="/export/fs06/shuan148/asr-research/cpd_pl/whisper_datasets/hf_dict/hf_dict_val_gold_1.25hr_test_gold_2.25hr" 

FNLO_TRAIN_GT_CSV="/export/fs06/shuan148/asr-research/cpd_pl/whisper_csv/train/train_gold_24hr.csv"

# Starting PL CSV (pre-generated before loop begins)
INITIAL_PL_CSV="/export/fs06/shuan148/asr-research/cpd_pl/llm_judge/whisper/whisper_train_24hr.csv"

BASE_SAVE_DIR="/export/fs06/shuan148/asr-research/cpd_pl/models/whisper/iterative-pl"
mkdir -p "$BASE_SAVE_DIR"

export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN}"
export HF_TOKEN="${HF_TOKEN}"

# ─────────────────────────────────────────────
# ITERATIVE PSEUDO-LABELING LOOP (3 iterations)
# Each iteration:
#   1. LLM judge PL CSV → judged CSV
#   2. Filter to is_correct=1 → JSONL
#   3. Finetune from current model checkpoint
#   4. Generate new PLs with finetuned model (skipped after iteration 3)
# ─────────────────────────────────────────────
CURRENT_MODEL_PATH="$MODEL_PATH"
CURRENT_PL_CSV="$INITIAL_PL_CSV"

for i in {1..3}
do
    echo "======================================="
    echo "  Iteration $i"
    echo "======================================="

    JUDGED_CSV="$BASE_SAVE_DIR/pl_iter_${i}_llm_judged.csv"
    PREPROCESSED_DATASET="$BASE_SAVE_DIR/pl_iter_${i}_filtered"
    MODEL_SAVE_PATH="$BASE_SAVE_DIR/model_iter_${i}"

    # Step 1: LLM judge
    echo "--- LLM judging PLs: $CURRENT_PL_CSV ---"
    conda activate /home/kchapar1/.local/share/mamba/envs/qwen3-asr
    python llm_judge/llm_pseudolabel_judge.py \
        --input_csv  "$CURRENT_PL_CSV" \
        --output_csv "$JUDGED_CSV" \
        --model      "meta-llama/Meta-Llama-3-8B-Instruct"

    # Step 2: Filter to is_correct=1, preprocess dataset
    echo "--- Filtering to is_correct=1, preprocess dataset ---"
    export LD_LIBRARY_PATH=/home/kchapar1/.local/share/mamba/envs/qwen3-asr/lib:$LD_LIBRARY_PATH
    python llm_judge/filter_whisper_dataset_for_llm_judged_correct.py \
        --judge_csv     "$JUDGED_CSV" \
        --output_dataset "$PREPROCESSED_DATASET" 

    # Step 3: Finetune
    echo "--- Finetuning from $CURRENT_MODEL_PATH, saving to $MODEL_SAVE_PATH ---"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python whisper_scripts/finetune_whisper_full/finetune_whisper_full.py \
        "$CURRENT_MODEL_PATH" \
        "$EPOCHS" \
        "$BATCH_SIZE" \
        "$PREPROCESSED_DATASET" \
        "$EVAL_TEST_DATASET" \
        "$MODEL_SAVE_PATH" \
        --eval_steps 884

    CURRENT_MODEL_PATH="$MODEL_SAVE_PATH"
    echo "Iteration $i complete. Model saved to $MODEL_SAVE_PATH"

    # Step 4: Generate new PLs for next iteration (skip after final iteration)
    if [ $i -lt 3 ]; then
        NEXT_PL_CSV="$BASE_SAVE_DIR/pl_iter_$((i+1)).csv"
        echo "--- Generating pseudo-labels with $CURRENT_MODEL_PATH ---"
        conda activate whisper_env
        python dataset_gen/get_whisper_pseudolabels.py \
            --model_path "$CURRENT_MODEL_PATH" \
            --input_csv "$FNLO_TRAIN_GT_CSV" \
            --pl_csv_save_path "$NEXT_PL_CSV" \
            --batch_size 32

        FINAL_PL_CSV="$BASE_SAVE_DIR/pl_iter_$((i+1))_formatted.csv"
        echo "--- Formatting for llm judging ---"
        conda activate /home/kchapar1/.local/share/mamba/envs/qwen3-asr
        python dataset_gen/llm_judge_csv.py \
            --csv_gold "$FNLO_TRAIN_GT_CSV" \
            --csv_pl "$NEXT_PL_CSV" \
            --output "$FINAL_PL_CSV"

        python dataset_gen/format_csv.py \
            --input_csv "$FINAL_PL_CSV" \
            --pair 

        CURRENT_PL_CSV="$FINAL_PL_CSV"
    fi
done

echo "======================================="
echo "Iterative training complete."
echo "======================================="