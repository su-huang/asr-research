import argparse
import pandas as pd


def main(args):
    df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df)} samples from {args.input_csv}")

    # Filter: keep TOP_PERCENT with lowest avg_wer * diversity
    df["filter_score"] = df[args.avg_wer_col] * df[args.diversity_col]
    n_keep = int(len(df) * args.top_percent)
    filtered_df = df.nsmallest(n_keep, "filter_score").drop(columns=["filter_score"])

    print(f"Keeping {len(filtered_df)}/{len(df)} samples (top {args.top_percent*100:.0f}% lowest avg_wer * diversity)")

    filtered_df.to_csv(args.output_csv, index=False)
    print(f"Saved to: {args.output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter STAR audit CSV by avg_wer * diversity")
    parser.add_argument("--input_csv",     type=str, required=True)
    parser.add_argument("--output_csv",    type=str, required=True)
    parser.add_argument("--top_percent",   type=float, default=1.0, help="Fraction of samples to keep (default: 0.8)")
    parser.add_argument("--avg_wer_col",   type=str, default="avg_wer")
    parser.add_argument("--diversity_col", type=str, default="diversity")
    args = parser.parse_args()
    main(args)