"""
Filter a pseudo-label CSV by per-sample WER against ground truth.

Inputs:
  --pl_csv        CSV with pseudo-labels (must have audio + text_pl columns)
  --gt_csv        Optional separate ground-truth CSV. If omitted, gt_text_col
                  is read from pl_csv directly (single-CSV mode).
  --threshold     Keep samples with WER <= this value
  --out_csv       Where to save the filtered PL CSV
"""
import argparse
import os

import jiwer
import pandas as pd
from whisper.normalizers import EnglishTextNormalizer

normalizer = EnglishTextNormalizer()


def safe_wer(ref, hyp):
    if not ref.strip():
        return 1.0
    return jiwer.wer(ref, hyp)


def main(args):
    pl_df = pd.read_csv(args.pl_csv)
    pl_df = pl_df.rename(columns={
        args.pl_audio_col: "audio",
        args.pl_text_col:  "text",
    })
    pl_df = pl_df.drop_duplicates(subset=["audio"])

    if args.gt_csv:
        gt_df = pd.read_csv(args.gt_csv)
        gt_df = gt_df.rename(columns={
            args.gt_audio_col: "audio",
            args.gt_text_col:  "ref_text",
        })
        gt_df = gt_df.drop_duplicates(subset=["audio"])
        merged = pl_df.merge(gt_df[["audio", "ref_text"]], on="audio", how="inner")
        print(f"PL: {len(pl_df)}  GT: {len(gt_df)}  Merged: {len(merged)}")
    else:
        if args.gt_text_col not in pl_df.columns:
            raise ValueError(
                f"Column '{args.gt_text_col}' not found in pl_csv. "
                f"Available columns: {list(pl_df.columns)}"
            )
        merged = pl_df.rename(columns={args.gt_text_col: "ref_text"}).dropna(subset=["ref_text"]).copy()
        print(f"PL: {len(pl_df)}  Using column '{args.gt_text_col}' as ground truth  Merged: {len(merged)}")

    # Compute per-sample WER
    merged["wer"] = merged.apply(
        lambda row: safe_wer(normalizer(str(row["ref_text"])), normalizer(str(row["text"]))),
        axis=1
    )

    # Filter
    kept = merged[merged["wer"] <= args.threshold].copy()
    print(
        f"Kept {len(kept)}/{len(merged)} samples with WER <= {args.threshold} "
        f"(mean WER: {kept['wer'].mean():.4f})"
    )

    # Save filtered CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    save_cols = [c for c in ["audio", "text", "truncated", "duration_s", "wer", "ref_text"] if c in kept.columns]
    kept[save_cols].to_csv(args.out_csv, index=False)
    print(f"Filtered CSV saved: {args.out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pl_csv",        type=str, required=True)
    parser.add_argument("--gt_csv",        type=str, default="", help="Separate GT CSV. If omitted, gt_text_col is read from pl_csv.")
    parser.add_argument("--pl_audio_col",  type=str, default="audio")
    parser.add_argument("--pl_text_col",   type=str, default="text_pl")
    parser.add_argument("--gt_audio_col",  type=str, default="audio", help="Only used when --gt_csv is provided.")
    parser.add_argument("--gt_text_col",   type=str, default="text_gold", help="GT text column — in gt_csv if provided, else in pl_csv.")
    parser.add_argument("--threshold",     type=float, required=True)
    parser.add_argument("--out_csv",       type=str, required=True)
    args = parser.parse_args()
    main(args)