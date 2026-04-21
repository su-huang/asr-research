import os
import random
import pandas as pd
import soundfile as sf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input_file", type=str)
parser.add_argument("--output_csv", type=str, default="24hr_sampled.csv")
parser.add_argument("--target_hours", type=float, default=24.0)
args = parser.parse_args()

rows = []
with open(args.input_file, "r") as f:
    for line in f:
        parts = line.strip().split()
        utt_id, path = parts[0], parts[1]
        rows.append({"utt_id": utt_id, "audio_filepath": path})

random.shuffle(rows)

target_seconds = args.target_hours * 3600
selected = []
total = 0

for row in rows:
    try:
        info = sf.info(row["audio_filepath"])
        duration = info.duration
        if total + duration <= target_seconds:
            row["duration_s"] = round(duration, 2)
            selected.append(row)
            total += duration
    except Exception as e:
        print(f"Skipping {row['audio_filepath']}: {e}")

df = pd.DataFrame(selected)
df.to_csv(args.output_csv, index=False)
