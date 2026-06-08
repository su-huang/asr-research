#!/bin/bash
#SBATCH --job-name=ipl-qwen3-llm
#SBATCH --nodes=1
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --exclude=e02,e04
#SBATCH --account=a100acct
#SBATCH --output=/export/fs06/shuan148/asr-research/cpd_pl/errors_output/ipl-qwen-%j.out
#SBATCH --error=/export/fs06/shuan148/asr-research/cpd_pl/errors_output/ipl-qwen-%j.err
#SBATCH --mail-user=shuan148@jh.edu
#SBATCH --mail-type=BEGIN,END,FAIL

source ~/miniconda3/etc/profile.d/conda.sh
source ~/.bashrc

cd /export/fs06/shuan148/asr-research/cpd_pl/

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

MODEL="Qwen/Qwen3-ASR-1.7B"

EPOCHS=10
BATCH_SIZE=4
GRAD_ACC=8

EVAL_FILE="/export/fs06/shuan148/asr-research/cpd_pl/qwen_jsonl/val/val_gold_1.25hr.jsonl"
TEST_FILE="/export/fs06/shuan148/asr-research/cpd_pl/qwen_jsonl/test/test_gold_2.25hr.jsonl"

FNLO_TRAIN_GT_CSV="/export/fs06/shuan148/asr-research/cpd_pl/qwen_csv/train_gold_24hr.csv"

# Starting PL CSV (pre-generated before loop begins)
INITIAL_PL_CSV="/export/fs06/shuan148/asr-research/cpd_pl/llm_judge/qwen/qwen_train_24hr.csv"

BASE_SAVE_DIR="/export/fs06/shuan148/asr-research/cpd_pl/models/qwen/iterative-pl"
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
CURRENT_MODEL="$MODEL"
CURRENT_PL_CSV="$INITIAL_PL_CSV"

for i in {1..3}
do
    echo "======================================="
    echo "  Iteration $i"
    echo "======================================="

    JUDGED_CSV="$BASE_SAVE_DIR/pl_iter_${i}_llm_judged.csv"
    FILTERED_JSONL="$BASE_SAVE_DIR/pl_iter_${i}_filtered.jsonl"
    MODEL_SAVE_PATH="$BASE_SAVE_DIR/model_iter_${i}"

    # Step 1: LLM judge
    echo "--- LLM judging PLs: $CURRENT_PL_CSV ---"
    conda activate /home/kchapar1/.local/share/mamba/envs/qwen3-asr
    python llm_judge/llm_pseudolabel_judge.py \
        --input_csv  "$CURRENT_PL_CSV" \
        --output_csv "$JUDGED_CSV" \
        --model      "meta-llama/Meta-Llama-3-8B-Instruct"

    # Step 2: Filter to is_correct=1, write JSONL
    echo "--- Filtering to is_correct=1, writing JSONL ---"
    export LD_LIBRARY_PATH=/home/kchapar1/.local/share/mamba/envs/qwen3-asr/lib:$LD_LIBRARY_PATH
    python llm_judge/filter_qwen_jsonl_for_llm_judged_correct.py \
        --csv     "$JUDGED_CSV" \
        --output "$FILTERED_JSONL" 

    # Step 3: Finetune
    echo "--- Finetuning from $CURRENT_MODEL, saving to $MODEL_SAVE_PATH ---"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python /export/fs06/kchapar1/bpd_asr/pyfiles/qwen3_asr_sft.py \
        --model_path "$CURRENT_MODEL" \
        --train_file "$FILTERED_JSONL" \
        --eval_file  "$EVAL_FILE" \
        --output_dir "$MODEL_SAVE_PATH" \
        --batch_size $BATCH_SIZE \
        --grad_acc   $GRAD_ACC \
        --lr         2e-5 \
        --epochs     $EPOCHS \
        --save_steps 884 \
        --save_total_limit 5 \
        --test_file  "$TEST_FILE"

    CURRENT_MODEL="$MODEL_SAVE_PATH"
    echo "Iteration $i complete. Model saved to $MODEL_SAVE_PATH"

    # Step 4: Generate new PLs for next iteration (skip after final iteration)
    if [ $i -lt 3 ]; then
        NEXT_PL_CSV="$BASE_SAVE_DIR/pl_iter_$((i+1)).csv"
        echo "--- Generating pseudo-labels with $CURRENT_MODEL ---"
        python dataset_gen/get_pl_qwen3.py \
            --model_path       "$CURRENT_MODEL" \
            --input_csv        "$FNLO_TRAIN_GT_CSV" \
            --pl_csv_save_path "$NEXT_PL_CSV"

        FINAL_PL_CSV="$BASE_SAVE_DIR/pl_iter_$((i+1))_formatted.csv"
        echo "--- Formatting for llm judging ---"
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