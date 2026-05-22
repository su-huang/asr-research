import pandas as pd
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input_csv", type=str)
parser.add_argument("--output_csv", type=str, default=None)
args = parser.parse_args()

df = pd.read_csv(args.input_csv)

def normalize_text(text):
    if not isinstance(text, str):
        return text
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s'\-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["text_gold"] = df["text_gold"].apply(normalize_text)
df["text_pl"] = df["text_pl"].apply(normalize_text)

n_before = len(df)
df = df.dropna(subset=["text_gold", "text_pl"])
df = df[(df["text_gold"].str.strip() != "") & (df["text_pl"].str.strip() != "")]
n_dropped = n_before - len(df)
if n_dropped > 0:
    print(f"Dropped {n_dropped} rows with null or empty text")

output_path = args.output_csv if args.output_csv else args.input_csv
df.to_csv(output_path, index=False)
print(f"Saved {len(df)} rows to {output_path}")
