import argparse
import os
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser(description="Merge Gold and Pseudo-Label CSVs on the 'audio' column.")
    parser.add_argument("--csv_gold", type=str, required=True, help="Path to the gold standard CSV file.")
    parser.add_argument("--csv_pl", type=str, required=True, help="Path to the pseudo-labeled CSV file.")
    parser.add_argument("--output", type=str, required=True, help="Path where the merged CSV will be saved.")
    parser.add_argument("--suffix_gold", type=str, default="_gold", help="Suffix to append to gold CSV text columns (default: _gold).")
    parser.add_argument("--suffix_pl", type=str, default="_pl", help="Suffix to append to pseudo-label CSV text columns (default: _pl).",)
    return parser.parse_args()


def merge_csvs():
    args = get_args()

    print(f"Reading Gold CSV: {args.csv_gold}")
    if not os.path.exists(args.csv_gold):
        raise FileNotFoundError(f"Could not find file at {args.csv_gold}")
    df1 = pd.read_csv(args.csv_gold)

    print(f"Reading PL CSV: {args.csv_pl}")
    if not os.path.exists(args.csv_pl):
        raise FileNotFoundError(f"Could not find file at {args.csv_pl}")
    df2 = pd.read_csv(args.csv_pl)

    # Lowercase column names for consistency
    df1.columns = [col.lower() for col in df1.columns]
    df2.columns = [col.lower() for col in df2.columns]

    # Validate presence of required columns
    if "audio" not in df1.columns or "text" not in df1.columns:
        raise ValueError("Gold CSV must contain both 'audio' and 'text' columns.")
    if "audio" not in df2.columns or "text" not in df2.columns:
        raise ValueError("PL CSV must contain both 'audio' and 'text' columns.")

    # Keep only target columns
    df1 = df1[["audio", "text"]]
    df2 = df2[["audio", "text"]]

    print("Merging dataframes on the 'audio' column...")
    merged_df = pd.merge(
        df1, df2, on="audio", suffixes=(args.suffix_gold, args.suffix_pl)
    )

    # Ensure the parent directory exists before saving
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print(f"Saving merged results ({len(merged_df)} rows) to: {args.output}")
    merged_df.to_csv(args.output, index=False)
    print("Merge complete!")


if __name__ == "__main__":
    merge_csvs()