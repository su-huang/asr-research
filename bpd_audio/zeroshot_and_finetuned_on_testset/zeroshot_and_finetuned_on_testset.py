import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_from_disk
from tqdm import tqdm
from jiwer import wer
import pandas as pd

# --- Config ---
base_model_name = "openai/whisper-large-v3"
# fine_tuned_model_path = "/export/fs06/shuan148/asr-research/bpd_audio/finetune_whisper/finetuned_models/whisper-small-finetuned2" 
fine_tuned_model_path = "/export/fs06/shuan148/asr-research/bpd_audio/finetune_whisper/finetuned_models/whisper-large-v3-finetuned2" 
test_dataset_path = "/home/kchapar1/bpd_asr/datasets/datasets_with_paths/test_dataset"
output_csv = "whisper_transcription_comparison.csv"

# --- Load processor (shared tokenizer) ---
processor = WhisperProcessor.from_pretrained(base_model_name)
tokenizer = processor.tokenizer

# --- Load models ---
base_model = WhisperForConditionalGeneration.from_pretrained(base_model_name).eval().cuda()
ft_model = WhisperForConditionalGeneration.from_pretrained(fine_tuned_model_path).eval().cuda()

# --- Load test dataset ---
test_dataset = load_from_disk(test_dataset_path)

# --- Transcription function ---
def transcribe(example, model):
    inputs = processor(
        example["audio"]["array"],
        sampling_rate=example["audio"]["sampling_rate"],
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        predicted_ids = model.generate(inputs.input_features)
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

# --- Transcribe each example ---
rows = []

for example in tqdm(test_dataset):
    gt = example["text"]

    # Transcribe with base and fine-tuned models
    base_pred = transcribe(example, base_model)
    ft_pred = transcribe(example, ft_model)

    # Normalize all
    gt_norm = tokenizer._normalize(gt)
    base_pred_norm = tokenizer._normalize(base_pred)
    ft_pred_norm = tokenizer._normalize(ft_pred)

    # --- Per-sample WER with empty-string protection ---
    def safe_wer(ref, hyp):
        if not ref and not hyp:
            return 0.0
        elif ref and not hyp:
            return 1.0
        elif not ref and hyp:
            return 1.0
        else:
            return wer(ref, hyp)

    wer_base_sample = safe_wer(gt_norm, base_pred_norm)
    wer_ft_sample = safe_wer(gt_norm, ft_pred_norm)

    # Store row
    rows.append({
        "ground_truth": gt,
        "base_pred": base_pred,
        "ft_pred": ft_pred,
        "ground_truth_norm": gt_norm,
        "base_pred_norm": base_pred_norm,
        "ft_pred_norm": ft_pred_norm,
        "WER_base_sample": wer_base_sample,
        "WER_ft_sample": wer_ft_sample,
        "WER_improvement": wer_base_sample - wer_ft_sample
    })

# --- Compute global WERs, skipping empty references ---
from jiwer import wer

def safe_aggregate_wer(refs, hyps):
    total_wer = 0.0
    count = 0

    for ref, hyp in zip(refs, hyps):
        ref = ref.strip()
        hyp = hyp.strip()

        if not ref and not hyp:
            sample_wer = 0.0
        elif ref and not hyp:
            sample_wer = 1.0
        elif not ref and hyp:
            sample_wer = 1.0
        else:
            sample_wer = wer(ref, hyp)

        total_wer += sample_wer
        count += 1

    return total_wer / count if count > 0 else 0.0



gt_list = [row["ground_truth_norm"] for row in rows]
base_list = [row["base_pred_norm"] for row in rows]
ft_list = [row["ft_pred_norm"] for row in rows]

base_wer = safe_aggregate_wer(gt_list, base_list)
ft_wer = safe_aggregate_wer(gt_list, ft_list)

print(f"\nGlobal Base WER: {base_wer:.3f}")
print(f"Global Fine-tuned WER: {ft_wer:.3f}")
print(f"Relative Improvement: {((base_wer - ft_wer) / base_wer if base_wer > 0 else 0):.2%}")

# --- Write to CSV ---
df = pd.DataFrame(rows)
df.to_csv(output_csv, index=False)
