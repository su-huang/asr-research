import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_from_disk
from tqdm import tqdm
from jiwer import wer
import pandas as pd
import librosa 
from pathlib import Path
import soundfile as sf
import numpy as np

# --- Config ---
base_model_name = "openai/whisper-small"
fine_tuned_model_path = "/export/fs06/shuan148/asr-research/cpd_audio/finetune_whisper/finetuned_models/whisper-small-finetuned" 
scp_path = "/secure/fs00/afield6/police/shuan148/test_wav.scp"
text_path = "/secure/fs00/afield6/police/chicago/data/data/test/text"
output_csv = "/export/fs06/shuan148/asr-research/cpd_audio/zeroshot_and_finetuned_on_testset/results/cpd_whisper_small_transcription_comparison.csv"

# --- Load processor (shared tokenizer) ---
processor = WhisperProcessor.from_pretrained(base_model_name)
tokenizer = processor.tokenizer

# --- Load models ---
base_model = WhisperForConditionalGeneration.from_pretrained(base_model_name).eval().cuda()
ft_model = WhisperForConditionalGeneration.from_pretrained(fine_tuned_model_path).eval().cuda()

# --- Load test dataset ---
def load_scp_text(scp_file, text_file):
    scp_dict = {}
    with open(scp_file, "r") as f_scp:
        for line in f_scp:
            utt_id, path = line.strip().split(maxsplit=1)
            scp_dict[utt_id] = path

    text_dict = {}
    with open(text_file, "r") as f_text:
        for line in f_text:
            utt_id, transcription = line.strip().split(maxsplit=1)
            text_dict[utt_id] = transcription
    
    data = []
    for utt_id in scp_dict:
        if utt_id in text_dict:
            data.append({
                "utt_id": utt_id,
                "audio_path": scp_dict[utt_id],
                "text": text_dict[utt_id]
            })
    return data

test_dataset = load_scp_text(scp_path, text_path)

# --- Transcription function ---
def transcribe(example, model):
    inputs = processor(
        example["audio"]["array"],
        sampling_rate=example["audio"]["sampling_rate"],
        return_tensors="pt"
    ).to(model.device)

    forced_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")

    with torch.no_grad():
        predicted_ids = model.generate(
            inputs.input_features, 
            forced_decoder_ids=forced_ids,
            max_new_tokens=444
        )
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

# -------------------- Read audio as numpy array --------------------
def load_audio(item, target_sr=16000):
    # Read the audio
    audio, sr = sf.read(item["audio_path"])
    
    # Convert to mono if stereo
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    
    # Resample if needed
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    
    return audio

# --- Transcribe each example ---
rows = []

for example in tqdm(test_dataset):
    gt = example["text"]

    audio_array = load_audio(example)
    audio_example = {
        "audio": {
            "array": audio_array,
            "sampling_rate": 16000
        }
    }


    # Transcribe with base and fine-tuned models
    base_pred = transcribe(audio_example, base_model)
    ft_pred = transcribe(audio_example, ft_model)

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
