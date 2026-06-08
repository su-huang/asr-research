import os
import argparse
import pandas as pd
import soundfile as sf
import random

TARGET_GOLD_HOURS = 5.0

def get_duration(path):
    try:
        info = sf.info(path)
        return info.duration
    except Exception:
        return 0.0

def mix_csvs(gold_csv_path, pseudo_csv_path, output_path):
    print("Loading CSVs...")
    gold_df = pd.read_csv(gold_csv_path)
    pseudo_df = pd.read_csv(pseudo_csv_path)

    # Validate required columns
    for df, name in [(gold_df, "gold"), (pseudo_df, "pseudo")]:
        for col in ["audio", "text"]:
            if col not in df.columns:
                raise ValueError(f"Missing column '{col}' in {name} CSV")

    # Drop nulls and empty transcripts
    gold_df = gold_df.dropna(subset=["audio", "text"])
    gold_df = gold_df[gold_df["text"].str.strip() != ""].reset_index(drop=True)
    pseudo_df = pseudo_df.dropna(subset=["audio", "text"])
    pseudo_df = pseudo_df[pseudo_df["text"].str.strip() != ""].reset_index(drop=True)

    # Shuffle gold data and select up to TARGET_GOLD_HOURS
    gold_df = gold_df.sample(frac=1, random_state=2).reset_index(drop=True)
    target_seconds = TARGET_GOLD_HOURS * 3600
    accumulated = 0.0
    cutoff_idx = 0

    for i, row in gold_df.iterrows():
        duration = get_duration(row["audio"])
        accumulated += duration
        cutoff_idx = i + 1
        if accumulated >= target_seconds:
            break

    actual_hours = accumulated / 3600
    gold_subset = gold_df.iloc[:cutoff_idx].copy()
    gold_subset["source"] = "gold"
    pseudo_df["source"] = "pseudo"

    print(f"Target gold: {TARGET_GOLD_HOURS}h | Selected {cutoff_idx} samples (~{actual_hours:.2f}h)")
    print(f"Pseudo samples: {len(pseudo_df)}")

    # Find common columns and keep them
    common_cols = list(set(gold_subset.columns).intersection(set(pseudo_df.columns)))
    # Always ensure audio, text, source are included
    for col in ["audio", "text", "source"]:
        if col not in common_cols:
            common_cols.append(col)

    combined = pd.concat(
        [gold_subset[common_cols], pseudo_df[common_cols]],
        ignore_index=True
    )
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    combined.to_csv(output_path, index=False)
    print(f"Combined dataset: {len(combined)} samples")
    print(f"Saved to {output_path}")

    # Print summary
    print(f"\n--- Mix Summary ---")
    print(f"Gold hours contributed:   {actual_hours:.2f}h ({cutoff_idx} samples)")
    print(f"Pseudo samples included:  {len(pseudo_df)}")
    print(f"Total samples:            {len(combined)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_csv", type=str, required=True, help="Path to gold/manual CSV")
    parser.add_argument("--pseudo_csv", type=str, required=True, help="Path to pseudo-labelled CSV (taken in full)")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save the mixed output CSV")
    args = parser.parse_args()

    mix_csvs(args.gold_csv, args.pseudo_csv, args.output_csv)
