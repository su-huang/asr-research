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

cd /export/fs06/kchapar1/bpd_asr/pyfiles/

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
FNLO_DATASET_PATH=""

# Starting PL CSV (pre-generated before loop begins)
INITIAL_PL_CSV="/export/fs06/shuan148/asr-research/cpd_pl/qwen_csv/train_pl_24hr.csv"

BASE_SAVE_DIR="/export/fs06/shuan148/asr-research/cpd_pl/models/qwen/iterative-pl"
mkdir -p "$BASE_SAVE_DIR"

export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN}"
export HF_TOKEN="${HF_TOKEN}"

# ─────────────────────────────────────────────
# STEP 0: Export FNLO train ground-truth CSV
# ─────────────────────────────────────────────
if [ ! -f "$FNLO_TRAIN_GT_CSV" ]; then
    echo "Exporting FNLO train ground-truth CSV to $FNLO_TRAIN_GT_CSV ..."
    conda activate /home/kchapar1/.local/share/mamba/envs/qwen3-asr
    python - <<EOF
from datasets import load_from_disk
ds = load_from_disk("$FNLO_DATASET_PATH")["train"]
df = ds.select_columns(["absolute_path", "text"]).to_pandas()
df = df.rename(columns={"absolute_path": "audio_filepath", "text": "ground_truth"})
df.to_csv("$FNLO_TRAIN_GT_CSV", index=False)
print(f"Exported {len(df)} rows to $FNLO_TRAIN_GT_CSV")
EOF
else
    echo "FNLO train GT CSV already exists: $FNLO_TRAIN_GT_CSV"
fi

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
    PYTHONUNBUFFERED=1 python llm_pseudolabel_judge.py \
        --input_csv  "$CURRENT_PL_CSV" \
        --output_csv "$JUDGED_CSV" \
        --model      meta-llama/Meta-Llama-3-8B-Instruct \
        --audio_col  audio \
        --pseudo_col text

    # Step 2: Filter to is_correct=1, write JSONL
    echo "--- Filtering to is_correct=1, writing JSONL ---"
    export LD_LIBRARY_PATH=/home/kchapar1/.local/share/mamba/envs/qwen3-asr/lib:$LD_LIBRARY_PATH
    python filter_qwen_jsonl_for_llm_judged.py \
        --csv_path     "$JUDGED_CSV" \
        --output_jsonl "$FILTERED_JSONL" \
        --audio_col    audio \
        --pseudo_col   text \
        --gt_csv       "$FNLO_TRAIN_GT_CSV" \
        --gt_audio_col audio \
        --gt_text_col  ground_truth

    # Step 3: Finetune
    echo "--- Finetuning from $CURRENT_MODEL, saving to $MODEL_SAVE_PATH ---"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python qwen3_asr_sft.py \
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
        python get_pseudolabels_qwen3_with_logprobs.py \
            --model_path       "$CURRENT_MODEL" \
            --input_csv        "$FNLO_TRAIN_GT_CSV" \
            --audio_column     audio_filepath \
            --gt_column        ground_truth \
            --include_gt \
            --pl_csv_save_path "$NEXT_PL_CSV"
        CURRENT_PL_CSV="$NEXT_PL_CSV"
    fi
done

echo "======================================="
echo "Iterative training complete."
echo "======================================="