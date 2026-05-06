import os
os.environ["HF_USE_TORCH_CODEC"] = "0"  # Must come before datasets/audio imports
 
import torch
import pandas as pd
import argparse
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from whisper.normalizers import EnglishTextNormalizer
import jiwer
from jiwer import wer
from pathlib import Path
import soundfile as sf
import numpy as np
import librosa
import re
from word2number import w2n
from collections import Counter
  
# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
# Model & processor
model_name = "openai/whisper-large-v3"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float32).to(device).eval()

# -------------------- Command-line arguments --------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Re-transcribe bad Whisper rows")
    parser.add_argument("input_csv", type=str,
                        help="Path to input CSV with 'audio' and 'text' columns")
    parser.add_argument("--output_csv", type=str, default="whisper_results.csv",
                        help="Base path for output CSVs (suffixes will be added)")
    parser.add_argument("--model_path", type=str,
                        default="openai/whisper-large-v3",
                        help="HF model ID or local path to Whisper model")
    parser.add_argument("--do_sample", type=lambda x: x.lower() in ["true", "1"],
                        default=False, help="Whether to use sampling (True/False)")
    parser.add_argument("--temp", type=float, default=1.0,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Top-p sampling probability")
    return parser.parse_args()
 
 
# -------------------- Transcription function --------------------
def transcribe(audio_array, do_sample, temp, top_p):
    inputs = processor(
        audio_array,
        sampling_rate=16000,
        return_tensors="pt"
    )
    input_features = inputs.input_features.to(device)
    input_features = input_features.to(model.dtype)  # match processor output to model dtype
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            do_sample=do_sample,
            temperature=temp if do_sample else None,
            top_p=top_p if do_sample else None,
            forced_decoder_ids=forced_decoder_ids
        )
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription
 
 
# -------------------- Normalization --------------------
normalizer = EnglishTextNormalizer()

def get_bad_word_fixes():
    bad_words_fixed = {}
    path = '/secure/fs00/afield6/police/shuan148/bad_words_fixed.csv'
    
    with open(path, 'r', encoding='latin1') as f:
        next(f)  # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Split on first comma only, in case fixed value contains commas
            parts = line.split(',', 1)
            
            if len(parts) == 2:
                orig = parts[0].strip()
                fixed = parts[1].strip()
                
                if orig and fixed:  # only add if both orig and fixed are non-empty
                    bad_words_fixed[orig] = fixed

    # Add known confusions (just to be sure)
    bad_words_fixed['FOURTY'] = 'FORTY'
    bad_words_fixed['OK'] = 'OKAY'
    bad_words_fixed['O'] = 'OH'
   
    # Two and three letter abbreviations were common, let's make absolute certain to fix a few really common ones
    bad_words_fixed['DOB'] = 'D O B'
    bad_words_fixed['EMS'] = 'E M S'
   
    # While we're at it, there are some common contractions without apostrophes
    bad_words_fixed['DONT'] = "DON'T"
   
    # Also, phoenetically identical but different spellings so just pick one
    bad_words_fixed['EDDY'] = 'EDDIE'
   
    # Also, 'ALRIGHT' -> 'ALL RIGHT'
    #bad_words_fixed['ALRIGHT'] = 'ALL RIGHT'
   
    # Also, 'GONNA' often confused with 'TO' but...
    # Investigation shows this is actually 'GONNA' <==> 'GOING TO' (so no change)

    return bad_words_fixed

def fix_bad_words(text, bad_word_fixes):
    # Fix "bad" words using manually generated dictionary
    text = text.strip()
    words = text.split()
    words_fixed = []
    for w in words:
        if str(w.upper()) in bad_word_fixes:
            fixed_word = bad_word_fixes[str(w.upper())]
            if fixed_word!=' ':
                words_fixed.append(fixed_word)
        else:
            words_fixed.append(w)
    return ' '.join(words_fixed).replace('  ', ' ').strip()
 
def extensive_normalization(text, debug=False):
    text = text.replace("<UNINTELLIGIBLE>", "")
    text = text.lower()
    text = re.sub(r'[\u2018\u2019\u201c\u201d`"]', '', text)
    text = text.replace('-', ' ')
 
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
 
 
# -------------------- Load audio --------------------
def load_audio(path, target_sr=16000):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio
 
 
# -------------------- Metrics --------------------
def get_detailed_metrics(predictions, references):
    out = jiwer.process_words(references, predictions)
    S = out.substitutions
    D = out.deletions
    I = out.insertions
    H = out.hits
    N = S + D + H
    return {
        "WER": (S + D + I) / N * 100,
        "S_rate": (S / N) * 100,
        "D_rate": (D / N) * 100,
        "I_rate": (I / N) * 100,
    }
 
 
def print_common_errors(predictions, references, top_n=20):
    substitutions, deletions, insertions = [], [], []
    for ref, hyp in zip(references, predictions):
        out = jiwer.process_words(ref, hyp)
        for op in out.alignments[0]:
            r_words = ref.split()[op.ref_start_idx:op.ref_end_idx]
            h_words = hyp.split()[op.hyp_start_idx:op.hyp_end_idx]
            if op.type == 'substitute':
                substitutions.append(f"{' '.join(r_words)} -> {' '.join(h_words)}")
            elif op.type == 'delete':
                deletions.append(' '.join(r_words))
            elif op.type == 'insert':
                insertions.append(' '.join(h_words))
 
    print("Most common substitutions:")
    for error, count in Counter(substitutions).most_common(top_n):
        print(f"  {count:4d}x | {error}")
    print("Most common deletions:")
    for error, count in Counter(deletions).most_common(top_n):
        print(f"  {count:4d}x | {error}")
    print("Most common insertions:")
    for error, count in Counter(insertions).most_common(top_n):
        print(f"  {count:4d}x | {error}")
 
 
def print_wer_per_duration(df, bins=[0, 2, 5, 10, 20, 30]):
    labels = [f"{bins[i]}-{bins[i+1]}s" for i in range(len(bins)-1)]
    df_analysis = df.copy()
    df_analysis['duration_bin'] = pd.cut(df_analysis['duration'], bins=bins, labels=labels)
    stats = df_analysis.groupby('duration_bin', observed=True).agg(
        avg_wer=('wer', 'mean'),
        sample_count=('path', 'count')
    )
    stats['avg_wer'] = stats['avg_wer'].round(4).astype(str)
    print("WER per duration bucket:")
    print(stats)
 
 
def compute_wer_for_pair(hyp, ref):
    if ref == "":
        return 0.0 if hyp == "" else 1.0
    if hyp == "":
        return 1.0
    return wer(ref, hyp)
 
 
def build_and_save_csv(paths, durations, groundtruths, hypotheses, col_name, outfile):
    wers = [compute_wer_for_pair(h, r) for h, r in zip(hypotheses, groundtruths)]
    global_wer = sum(wers) / len(wers) if wers else 0.0
 
    df = pd.DataFrame({
        "path": paths,
        "duration": durations,
        "groundtruth": groundtruths,
        col_name: hypotheses,
        "wer": wers,
    })
 
    summary_row = pd.DataFrame({
        "path": ["GLOBAL_WER"],
        "duration": [None],
        "groundtruth": [""],
        col_name: [""],
        "wer": [global_wer],
    })
 
    df = pd.concat([df, summary_row], ignore_index=True)
    df.to_csv(outfile, index=False)
    return global_wer, wers
 
 
# -------------------- Main --------------------
def main() -> None:
    args = parse_args()
 
    # reload model from --model_path if specified
    global processor, model
    if args.model_path != "openai/whisper-large-v3":
        print(f"Loading model from {args.model_path}...")
        processor = WhisperProcessor.from_pretrained(args.model_path)
        model = WhisperForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.float32).to(device).eval()
 
    # derive output paths
    base = Path(args.output_csv)
    outfile_raw      = base.with_name(base.stem + "_raw"      + base.suffix)
    outfile_bad_word = base.with_name(base.stem + "_bad_word" + base.suffix)
    outfile_norm     = base.with_name(base.stem + "_norm"     + base.suffix)
 
    # load CSV — expects 'audio' (path) and 'text' columns
    df_in = pd.read_csv(args.input_csv)
    df_in = df_in.dropna(subset=["audio", "text"])
    df_in = df_in[df_in["text"].str.strip() != ""].reset_index(drop=True)
    print(f"Loaded {len(df_in)} rows from {args.input_csv}")
 
    bad_word_fixes = get_bad_word_fixes()
 
    paths, durations = [], []
    raw_hyps,     raw_refs     = [], []
    badword_hyps, badword_refs = [], []
    norm_hyps,    norm_refs    = [], []
 
    for _, row in tqdm(df_in.iterrows(), total=len(df_in), desc="Transcribing"):
        path = row["audio"]
        raw_gt = row["text"]
 
        try:
            audio_array = load_audio(path)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue
 
        duration = len(audio_array) / 16000
        if duration < 0.5 or duration > 30:
            continue
 
        raw_whisper = transcribe(audio_array, args.do_sample, args.temp, args.top_p)
 
        whisper_fix_bad = fix_bad_words(raw_whisper, bad_word_fixes)
        gt_fix_bad      = fix_bad_words(raw_gt, bad_word_fixes)
 
        whisper_norm = extensive_normalization(whisper_fix_bad)
        gt_norm      = extensive_normalization(gt_fix_bad)
 
        if gt_norm.strip() == "":
            continue
 
        paths.append(path)
        durations.append(duration)
 
        raw_hyps.append(raw_whisper);         raw_refs.append(raw_gt)
        badword_hyps.append(whisper_fix_bad); badword_refs.append(gt_fix_bad)
        norm_hyps.append(whisper_norm);       norm_refs.append(gt_norm)
 
    # save CSVs
    global_wer_raw, _ = build_and_save_csv(
        paths, durations, raw_refs, raw_hyps,
        col_name="whisper_raw", outfile=str(outfile_raw)
    )
    global_wer_bw, _ = build_and_save_csv(
        paths, durations, badword_refs, badword_hyps,
        col_name="whisper_bad_word_fixed", outfile=str(outfile_bad_word)
    )
    global_wer_norm, _ = build_and_save_csv(
        paths, durations, norm_refs, norm_hyps,
        col_name="normalized_whisper_lgv3", outfile=str(outfile_norm)
    )
 
    results = get_detailed_metrics(norm_hyps, norm_refs)
    print(f"\nGlobal WER — raw:          {global_wer_raw:.4f}")
    print(f"Global WER — bad-word fix: {global_wer_bw:.4f}")
    print(f"Global WER — full norm:    {global_wer_norm:.4f}")
    print(f"Substitutions (S): {results['S_rate']:.1f}%")
    print(f"Deletions (D):     {results['D_rate']:.1f}%")
    print(f"Insertions (I):    {results['I_rate']:.1f}%")
 
    df_norm = pd.read_csv(outfile_norm).dropna(subset=["duration"])
    print_wer_per_duration(df_norm)
    print_common_errors(norm_hyps, norm_refs)
 
 
if __name__ == "__main__":
    main()