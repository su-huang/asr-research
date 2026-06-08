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

BINS = [(0, 3), (3, 6), (6, 9), (9, 12), (12, 15), (15, float("inf"))]
BIN_LABELS = ["0–3s", "3–6s", "6–9s", "9–12s", "12–15s", "15+s"]

# ─── Normalization (from csv_aggressive_normalization.py) ───────────────────

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
    for _, row in df.iterrows():
        audio_path = str(row[audio_col]).strip()
        ref = str(row[gt_col])
        hyp = str(row[pl_col])
        dur = get_duration(audio_path)
        if dur is not None:
            records.append({"duration": dur, "wer": sample_wer(ref, hyp)})
    return records


def bin_wers(records):
    bins = [[] for _ in BINS]
    for r in records:
        for i, (lo, hi) in enumerate(BINS):
            if lo <= r["duration"] < hi:
                bins[i].append(r["wer"])
                break
    return [np.mean(b) if b else np.nan for b in bins], [len(b) for b in bins]


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
    parser.add_argument("--output",    type=str, default="wer_by_duration.png")
    args = parser.parse_args()

    sources = {}
    for spec in args.csvs:
        label, path = spec.split(":", 1)
        print(f"Loading CSV: {label} ...", flush=True)
        sources[label] = load_csv_source(path, args.audio_col, args.gt_col, args.pl_col)

    results = {}
    for label, records in sources.items():
        avg_wers, counts = bin_wers(records)
        results[label] = (avg_wers, counts)
        print(f"\n{label} ({len(records)} samples):")
        for bl, aw, c in zip(BIN_LABELS, avg_wers, counts):
            if not np.isnan(aw):
                print(f"  {bl:6s}  n={c:5d}  avg WER={aw:.4f}")
            else:
                print(f"  {bl:6s}  n={c:5d}  avg WER=N/A")

    n_sources = len(results)
    n_bins    = len(BINS)
    x         = np.arange(n_bins)
    width     = 0.8 / n_sources

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (label, (avg_wers, _)) in enumerate(results.items()):
        offset = (i - n_sources / 2 + 0.5) * width
        ax.bar(x + offset, avg_wers, width, label=label)

    ax.set_xlabel("Audio Duration")
    ax.set_ylabel("Average Per-Sample WER (normalized)")
    ax.set_title("Pseudolabel WER by Audio Duration")
    ax.set_xticks(x)
    ax.set_xticklabels(BIN_LABELS)
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"\nPlot saved to {args.output}")


if __name__ == "__main__":
    main()