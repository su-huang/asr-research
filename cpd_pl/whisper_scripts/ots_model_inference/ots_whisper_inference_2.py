import argparse
import pandas as pd
import torch
import soundfile as sf
import librosa
import jiwer
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

parser = argparse.ArgumentParser()
parser.add_argument("input_csv", type=str)
parser.add_argument("--output_csv", type=str, default="whisper_results.csv")
parser.add_argument("--batch_size", type=int, default=16)
args = parser.parse_args()

# Load model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print("Loading Whisper large-v3...")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "/export/fs06/shuan148/asr-research/cpd_pl/models/whisper-large-v3",
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
)
model.to(device)
processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

# Load CSV
df = pd.read_csv(args.input_csv)
df = df.dropna(subset=["text", "audio"])
df = df[df["text"].str.strip() != ""]
print(f"Running inference on {len(df)} files...")

hypotheses = []
for path in df["audio"].tolist():
    try:
        audio, sr = sf.read(path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        result = pipe({"raw": audio, "sampling_rate": 16000}, generate_kwargs={"language": "english"})
        hyp = result["text"].strip().lower()
    except Exception as e:
        print(f"Error on {path}: {e}")
        hyp = ""
    hypotheses.append(hyp)

df["hypothesis"] = hypotheses

# Per-utterance WER
def safe_wer(ref, hyp):
    try:
        if not ref.strip():
            return None
        return round(jiwer.wer(ref.strip(), hyp.strip() if hyp.strip() else " "), 4)
    except Exception:
        return None

df["wer"] = df.apply(lambda row: safe_wer(row["text"], row["hypothesis"]), axis=1)

# Global WER using list-based jiwer (corpus-level, no boundary contamination)
all_refs = df["text"].tolist()
all_hyps = df["hypothesis"].tolist()

out = jiwer.process_words(all_refs, all_hyps)

print(f"\nGlobal WER:  {out.wer:.4f} ({out.wer*100:.2f}%)")
print(f"Substitutions: {out.substitutions}")
print(f"Deletions:     {out.deletions}")
print(f"Insertions:    {out.insertions}")
print(f"Hits:          {out.hits}")

df.to_csv(args.output_csv, index=False)
print(f"\nResults saved to {args.output_csv}")
