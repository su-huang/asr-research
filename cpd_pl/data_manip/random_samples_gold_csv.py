import os
import random
import pandas as pd
import soundfile as sf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input_scp", type=str, help="Path to .scp file with utt_id and audio paths")
parser.add_argument("input_text", type=str, help="Path to text file with utt_id and transcriptions")
parser.add_argument("--output_csv", type=str, default="sampled.csv")
parser.add_argument("--target_hours", type=float, default=24.0)
args = parser.parse_args()

MAX_DURATION_S = 30.0

# Load transcripts into a dictionary
transcripts = {}
with open(args.input_text, "r") as f:
    for line in f:
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            utt_id, text = parts
            text = text.lower()
            if "<unintelligible>" not in text:
                transcripts[utt_id] = text

# Load scp file
rows = []
with open(args.input_scp, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            utt_id, path = parts
            if utt_id in transcripts:
                rows.append({"utt_id": utt_id, "audio": path, "text": transcripts[utt_id]})

# Shuffle and select up to target hours
random.shuffle(rows)
target_seconds = args.target_hours * 3600
selected = []
total = 0

for row in rows:
    try:
        info = sf.info(row["audio"])
        duration = info.duration
        was_truncated = duration > MAX_DURATION_S
        if total + duration <= target_seconds:
            selected.append({
                "audio": row["audio"],
                "text": row["text"],
                "truncated": was_truncated,
                "duration_s": round(min(duration, MAX_DURATION_S), 2),
            })
            total += duration
    except Exception as e:
        print(f"Skipping {row['audio']}: {e}")

print(f"Selected {len(selected)} files, total duration: {total/3600:.2f} hours")

df = pd.DataFrame(selected)
df.to_csv(args.output_csv, index=False)
print(f"Saved to {args.output_csv}")
