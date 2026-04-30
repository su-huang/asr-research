import argparse
import pandas as pd
import torch
import soundfile as sf
import librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from jiwer import wer as jwer
from whisper.normalizers import EnglishTextNormalizer

parser = argparse.ArgumentParser()
parser.add_argument("input_csv", type=str)
parser.add_argument("--output_csv", type=str, default="whisper_results.csv")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--model_path", type=str, default="/export/fs06/shuan148/asr-research/cpd_pl/models/whisper-large-v3")
args = parser.parse_args()

normalizer = EnglishTextNormalizer()

# Load model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print("Loading Whisper large-v3...")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    args.model_path,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
)
model.to(device)
model.generation_config.forced_decoder_ids = None
model.generation_config.language = "en"
model.generation_config.task = "transcribe"

processor = AutoProcessor.from_pretrained(args.model_path)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
    generate_kwargs={"language": "en", "task": "transcribe"},  # set defaults here
)

# Load CSV
df = pd.read_csv(args.input_csv)
df = df.dropna(subset=["text", "audio"])
df = df[df["text"].str.strip() != ""].reset_index(drop=True)
print(f"Running inference on {len(df)} files...")

# Load all audio into a list
audio_inputs = []
valid_indices = []
for i, path in enumerate(df["audio"].tolist()):
    try:
        audio, sr = sf.read(path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        audio_inputs.append({"raw": audio, "sampling_rate": 16000})
        valid_indices.append(i)
    except Exception as e:
        print(f"Error loading {path}: {e}")

# Run batched inference
print(f"Running batched inference (batch_size={args.batch_size})...")
results = pipe(
    audio_inputs,
    batch_size=args.batch_size,
    generate_kwargs={"language": "en", "task": "transcribe"},
)

# Map results back to dataframe rows
hypotheses = [None] * len(df)  # None marks failed audio loads
for result, idx in zip(results, valid_indices):
    hypotheses[idx] = result["text"].strip()

df["hypothesis"] = hypotheses
df["load_failed"] = df["hypothesis"].isna()  # flag failed rows for later inspection
df["hypothesis"] = df["hypothesis"].fillna("")  # fill so normalizer doesn't crash

# Normalize both reference and hypothesis before WER
df["ref_norm"] = df["text"].apply(normalizer)
df["hyp_norm"] = df["hypothesis"].apply(normalizer)

# Per-sample WER
def safe_wer(ref, hyp):
    ref = ref.strip()
    hyp = hyp.strip()
    if not ref:
        return None
    if not hyp:
        return 1.0  # complete miss
    return round(jwer(ref, hyp), 4)

df["wer"] = df.apply(lambda row: safe_wer(row["ref_norm"], row["hyp_norm"]), axis=1)

# Global WER using jiwer on lists (proper corpus-level WER)
valid = df[df["wer"].notna() & ~df["load_failed"]]  # exclude failed audio loads
global_wer = jwer(valid["ref_norm"].tolist(), valid["hyp_norm"].tolist())
avg_wer = valid["wer"].mean()

print(f"\nCorpus-level WER:        {global_wer:.4f} ({global_wer*100:.2f}%)")
print(f"Per-sample average WER:  {avg_wer:.4f} ({avg_wer*100:.2f}%)")
print(f"Samples evaluated:       {len(valid)} / {len(df)}")

df.to_csv(args.output_csv, index=False)
print(f"Results saved to {args.output_csv}")