import argparse
import json
import torch
import pandas as pd
from tqdm import tqdm
from qwen_asr import Qwen3ASRModel
from transformers import WhisperProcessor
from torch.utils.data import Dataset, DataLoader
from jiwer import wer
 
parser = argparse.ArgumentParser()
parser.add_argument("--input_jsonl", type=str, required=True,
                    help="Path to input JSONL file with 'audio' and 'text' fields.")
parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-ASR-0.6B")
parser.add_argument("--output_csv", type=str, default=None)
args = parser.parse_args()
 
# --- 1. Load JSONL ---
print(f"Loading JSONL from: {args.input_jsonl}")
records = []
with open(args.input_jsonl, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))
 
# Extract audio paths and ground truth text
# Strip "language English<asr_text>" prefix from text field
def parse_text(raw: str) -> str:
    tag = "<asr_text>"
    if tag in raw:
        return raw.split(tag, 1)[1].strip()
    return raw.strip()
 
test_paths = [r["audio"] for r in records]
test_gts   = [parse_text(r["text"]) for r in records]
 
# --- 2. Dataset & DataLoader ---
class AudioPathDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        return self.paths[idx]
 
processor  = WhisperProcessor.from_pretrained('openai/whisper-large-v3', low_cpu_mem_usage=True)
normalizer = processor.tokenizer._normalize
 
model = Qwen3ASRModel.from_pretrained(
    args.model_id,
    dtype=torch.bfloat16,
    device_map="cuda:0",
    max_inference_batch_size=32,
)
 
loader = DataLoader(AudioPathDataset(test_paths), batch_size=32, num_workers=4)
 
# --- 3. Inference Loop ---
all_predictions = []
for batch_paths in tqdm(loader, desc="Transcribing"):
    results = model.transcribe(audio=list(batch_paths), language=None)
    all_predictions.extend([r.text for r in results])
 
# --- 4. Normalization & WER Calculation ---
df = pd.DataFrame({
    "path":         test_paths,
    "ground_truth": test_gts,
    "prediction":   all_predictions
})
 
print("Normalizing and calculating WER...")
df["norm_ground_truth"] = df["ground_truth"].apply(lambda x: normalizer(x) if pd.notnull(x) else "")
df["norm_prediction"]   = df["prediction"].apply(lambda x: normalizer(x) if pd.notnull(x) else "")
 
def safe_wer(row):
    ref = row["norm_ground_truth"].strip()
    hyp = row["norm_prediction"].strip()
    return wer(ref, hyp)
 
df["WER"] = df.apply(safe_wer, axis=1)
 
print(f"Mean Per-Row WER: {df['WER'].mean():.4f}")
 
model_name = args.model_id.replace("/", "_")
out_csv = args.output_csv 
df.to_csv(out_csv, index=False)
print(f"Saved to: {out_csv}")
