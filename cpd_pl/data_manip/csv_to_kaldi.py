import argparse
import csv
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--input_csv", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--audio_col", default="audio")
parser.add_argument("--text_col", default="text")
args = parser.parse_args()

input_path = Path(args.input_csv)
out_dir = Path(args.output_dir)
out_dir.mkdir(parents=True, exist_ok=True)

rows = []
with open(input_path, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

pad_width = len(str(len(rows)))

with open(out_dir / "wav.scp", "w") as wav_f, open(out_dir / "text", "w") as text_f:
    for i, row in enumerate(rows):
        utt_id = f"utt_{str(i).zfill(pad_width)}"
        wav_f.write(f"{utt_id} {row[args.audio_col]}\n")
        text_f.write(f"{utt_id} {row[args.text_col]}\n")

print(f"Wrote {len(rows)} utterances to {out_dir}")
