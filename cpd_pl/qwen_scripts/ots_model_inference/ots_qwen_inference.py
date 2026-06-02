import argparse
import functools
import json
import types

import pandas as pd
import torch
from jiwer import wer
from qwen_asr import Qwen3ASRModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import WhisperProcessor

parser = argparse.ArgumentParser()
parser.add_argument("--input_jsonl", type=str, required=True)
parser.add_argument("--model_id",    type=str, default="Qwen/Qwen3-ASR-0.6B")
parser.add_argument("--output_csv",  type=str, required=True)
parser.add_argument("--temperature", type=float, default=0.2)
args = parser.parse_args()


# --- Monkey-patch ---
def _infer_asr_transformers_sampled(self, contexts, wavs, languages, temperature):
    outs = []
    texts = [self._build_text_prompt(context=c, force_language=fl) for c, fl in zip(contexts, languages)]
    batch_size = self.max_inference_batch_size
    if batch_size is None or batch_size < 0:
        batch_size = len(texts)
    for i in range(0, len(texts), batch_size):
        sub_text = texts[i : i + batch_size]
        sub_wavs = wavs[i : i + batch_size]
        inputs = self.processor(text=sub_text, audio=sub_wavs, return_tensors="pt", padding=True)
        inputs = inputs.to(self.model.device).to(self.model.dtype)
        text_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=temperature,
        )
        decoded = self.processor.batch_decode(
            text_ids.sequences[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        outs.extend(list(decoded))
    return outs


# --- Load JSONL ---
print(f"Loading JSONL from: {args.input_jsonl}")
records = []
with open(args.input_jsonl, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))


def parse_text(raw: str) -> str:
    tag = "<asr_text>"
    if tag in raw:
        return raw.split(tag, 1)[1].strip()
    return raw.strip()


test_paths = [r["audio"] for r in records]
test_gts   = [parse_text(r["text"]) for r in records]


# --- Dataset ---
class AudioPathDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        return self.paths[idx]


# --- Load model and apply patch ---
processor  = WhisperProcessor.from_pretrained("openai/whisper-large-v3", low_cpu_mem_usage=True)
normalizer = processor.tokenizer._normalize

model = Qwen3ASRModel.from_pretrained(
    args.model_id,
    dtype=torch.bfloat16,
    device_map="cuda:0",
    max_inference_batch_size=32,
)

model._infer_asr_transformers = types.MethodType(
    functools.partial(_infer_asr_transformers_sampled, temperature=args.temperature),
    model
)
print(f"Sampling patch applied: temperature={args.temperature}")

# --- Inference ---
loader = DataLoader(AudioPathDataset(test_paths), batch_size=32, num_workers=4)
all_predictions = []
for batch_paths in tqdm(loader, desc="Transcribing"):
    results = model.transcribe(audio=list(batch_paths), language=None)
    all_predictions.extend([r.text for r in results])

# --- WER ---
df = pd.DataFrame({
    "path":         test_paths,
    "ground_truth": test_gts,
    "prediction":   all_predictions,
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

df.to_csv(args.output_csv, index=False)
print(f"Saved to: {args.output_csv}")