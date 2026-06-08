import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--csv1", required=True, help="first CSV (source of audio paths to keep)")
parser.add_argument("--csv2", required=True, help="second CSV to filter")
parser.add_argument("--output", required=True, help="path to save filtered CSV")
parser.add_argument("--audio_col", default="audio", help="name of the audio column in both CSVs")
args = parser.parse_args()

df1 = pd.read_csv(args.csv1)
df2 = pd.read_csv(args.csv2)

valid_paths = set(df1[args.audio_col].tolist())
filtered = df2[df2[args.audio_col].isin(valid_paths)]

print(f"CSV1 paths:     {len(df1)}")
print(f"CSV2 rows:      {len(df2)}")
print(f"Filtered rows:  {len(filtered)}")

filtered.to_csv(args.output, index=False)
print(f"Saved to: {args.output}")
