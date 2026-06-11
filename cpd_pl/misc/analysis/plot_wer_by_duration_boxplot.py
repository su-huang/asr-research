import argparse
import os
import re
import soundfile as sf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from jiwer import wer as jiwer_wer
from whisper.normalizers import EnglishTextNormalizer
from word2number import w2n

whisper_normalizer = EnglishTextNormalizer()

# Using the bins and labels from the boxplot reference script
DUR_BINS = [0, 1, 2, 3, 4, 5, 10, 30]
DUR_LABELS = ['0-1s', '1-2s', '2-3s', '3-4s', '4-5s', '5-10s', '10-30s']

# ─── Normalization ──────────────────────────────────────────────────────────

tens_map = {'twenty':20,'thirty':30,'forty':40,'fifty':50,
            'sixty':60,'seventy':70,'eighty':80,'ninety':90}
ones_map = {'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,
            'six':6,'seven':7,'eight':8,'nine':9,'ten':10,
            'eleven':11,'twelve':12,'thirteen':13,'fourteen':14,
            'fifteen':15,'sixteen':16,'seventeen':17,'eighteen':18,'nineteen':19}
all_map = {**ones_map, **tens_map}

ones = r'(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen)'
tens = r'(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)'
single_num = r'(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|and)'

def aggressive_normalize(text):
    text = text.replace("<UNINTELLIGIBLE>", "")
    text = text.lower()
    text = re.sub(r'[\u2018\u2019\u201c\u201d`"]', '', text)
    text = text.replace('-', ' ')

    real_structural = rf'\b{single_num}(?:\s+{single_num})*\b'

    def replace_structural(match):
        text_chunk = match.group(0).strip()
        if not re.search(r'\b(hundred|thousand|and)\b', text_chunk, re.IGNORECASE):
            return text_chunk
        try:
            num = w2n.word_to_num(text_chunk)
            return ' '.join(list(str(num)))
        except:
            return text_chunk

    text = re.sub(real_structural, replace_structural, text, flags=re.IGNORECASE)

    spoken_pair = rf'\b({ones}|{tens})\s+({tens})\b'

    def resolve_spoken_pair(match):
        a, b = match.group(1).lower(), match.group(2).lower()
        if a in all_map and b in tens_map:
            combined = str(all_map[a]) + str(tens_map[b])
            return ' '.join(list(combined))
        return match.group(0)

    text = re.sub(spoken_pair, resolve_spoken_pair, text, flags=re.IGNORECASE)

    tens_ones_pair = rf'\b({tens})\s+({ones})\b'

    def resolve_tens_ones(match):
        a, b = match.group(1).lower(), match.group(2).lower()
        if a in tens_map and b in ones_map:
            combined = str(tens_map[a] + ones_map[b])
            return ' '.join(list(combined))
        return match.group(0)

    text = re.sub(tens_ones_pair, resolve_tens_ones, text, flags=re.IGNORECASE)

    single_pattern = rf'\b{single_num}\b'

    def resolve_single(match):
        word = match.group(0).lower()
        if word in ('and', 'hundred', 'thousand'):
            return match.group(0)
        if word in all_map:
            return ' '.join(list(str(all_map[word])))
        return match.group(0)

    text = re.sub(single_pattern, resolve_single, text, flags=re.IGNORECASE)
    text = re.sub(r"[^a-z0-9'\s]", '', text)
    text = re.sub(r'(\d)', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize(text):
    return aggressive_normalize(whisper_normalizer(str(text)))


def sample_wer(ref, hyp):
    r = normalize(ref)
    h = normalize(hyp)
    if not r and not h:
        return 0.0
    if not r or not h:
        return 1.0
    return jiwer_wer(r, h)


def get_duration(path):
    try:
        return sf.info(path.strip()).duration
    except Exception:
        return None


def load_csv_source(path, audio_col, gt_col, pl_col):
    df = pd.read_csv(path, na_filter=False)
    records = []
    for i, row in df.iterrows():
        audio_path = str(row[audio_col]).strip()
        ref = str(row[gt_col])
        hyp = str(row[pl_col])
        dur = get_duration(audio_path)
        if dur is not None:
            wer_val = sample_wer(ref, hyp)
            records.append({"duration": dur, "wer": wer_val})
        
        if (i + 1) % 5000 == 0:
            print(f"  Processed {i + 1}/{len(df)} files", flush=True)
            
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csvs",      nargs="+", required=True,
                        help="label:path pairs for CSV files")
    parser.add_argument("--audio_col", type=str, default="audio",
                        help="Column name for audio paths")
    parser.add_argument("--gt_col",    type=str, default="text_gold",
                        help="Column name for ground truth text")
    parser.add_argument("--pl_col",    type=str, default="text_pl",
                        help="Column name for pseudo-label text")
    parser.add_argument("--output_plot", type=str, default="wer_by_duration_boxplot.png")
    parser.add_argument("--title",     type=str, default="Pseudolabel WER by Audio Duration")
    parser.add_argument("--ymax",      type=float, default=2.0)
    args = parser.parse_args()

    sources = {}
    for spec in args.csvs:
        label, path = spec.split(":", 1)
        print(f"Loading CSV: {label} ...", flush=True)
        sources[label] = load_csv_source(path, args.audio_col, args.gt_col, args.pl_col)

    n_sources = len(sources)
    
    # Setup plotting canvas side-by-side if multiple sources are passed
    fig, axes = plt.subplots(1, n_sources, figsize=(10 * n_sources, 6), squeeze=False)
    axes = axes.flatten()

    for idx, (label, df_records) in enumerate(sources.items()):
        ax = axes[idx]
        
        if df_records.empty:
            ax.set_title(f"{label} (No Data)")
            continue
            
        # Bin data using the reference script parameters
        df_records['dur_bin'] = pd.cut(df_records['duration'], bins=DUR_BINS, labels=DUR_LABELS, right=False)
        groups = [df_records[df_records['dur_bin'] == lbl]['wer'].values for lbl in DUR_LABELS]
        
        # Plot using precise formatting styles from your reference script
        bp = ax.boxplot(groups, labels=DUR_LABELS, patch_artist=True,
                        medianprops=dict(color='red', linewidth=2),
                        flierprops=dict(marker='.', markersize=2, alpha=0.3))
        
        for patch in bp['boxes']:
            patch.set_facecolor('steelblue')
            patch.set_alpha(0.7)

        ax.set_xlabel('Duration')
        ax.set_ylabel('WER')
        ax.set_title(f"{args.title} ({label})")
        ax.set_ylim(0, args.ymax)

        # Print metrics to console and plot the sample size (n=) annotations
        print(f"\n{label} ({len(df_records)} samples):")
        print(df_records['wer'].describe())
        
        for i, (lbl, grp) in enumerate(zip(DUR_LABELS, groups)):
            n_count = len(grp)
            ax.text(i + 1, args.ymax * 0.97, f'n={n_count:,}',
                    ha='center', va='top', fontsize=8, color='gray')

    plt.tight_layout()
    plt.savefig(args.output_plot, dpi=150)
    print(f"\nPlot saved to {args.output_plot}")


if __name__ == "__main__":
    main()