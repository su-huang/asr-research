import os
os.environ["HF_USE_TORCH_CODEC"] = "0"  # Must come before datasets/audio imports

import torch
import pandas as pd
import argparse
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_from_disk, Audio  # Import after disabling TorchCodec
from whisper.normalizers import EnglishTextNormalizer
import jiwer
from jiwer import wer
from pathlib import Path
import soundfile as sf
import numpy as np
import librosa
import re
from num2words import num2words
from word2number import w2n
from collections import Counter

from normalization import replace_question_marks, get_bad_word_fixes, fix_bad_words

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model & processor
model_name = "openai/whisper-large-v3"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device).eval()
tokenizer = processor.tokenizer  # for normalization

# -------------------- Command-line arguments --------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Re-transcribe bad Whisper rows")
    parser.add_argument("--scp_file", type=str, required=True,
                        help="Path to test_wav.scp")
    parser.add_argument("--text_file", type=str, required=True,
                        help="Path to transcription text file")
    parser.add_argument("--outfile", required=True,
                        help="CSV output file for results")
    parser.add_argument("--do_sample", type=lambda x: x.lower() in ["true", "1"], required=True,
                        help="Whether to use sampling (True/False)")
    parser.add_argument("--temp", type=float, default=1.0,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Top-p sampling probability")
    return parser.parse_args()

# -------------------- Transcription function --------------------
def transcribe(example, do_sample, temp, top_p):
    inputs = processor(
        example["audio"]["array"],
        sampling_rate=example["audio"]["sampling_rate"],
        return_tensors="pt"
    )
    input_features = inputs.input_features.to(device)
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

# -------------------- Normalization function --------------------
normalizer = EnglishTextNormalizer()
bad_word_fixes = get_bad_word_fixes()

def normalize_example(predicted, text):
    return normalizer(predicted), normalizer(text)

def extensive_normalization(text): 
    text = text.lower()
    text = re.sub(r'["`‘’]', "", text)
    text = text.replace('-', ' ') 

    # isolate digits in compound numbers 
    num_word_list = r'(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand)'
    compound_pattern = rf'\b{num_word_list}(?:\s+{num_word_list})*\b'

    def replace_with_num(match):
        text_chunk = match.group(0).strip()
        words = text_chunk.split()
        if len(words) == 2 and words[0] != "and" and words[1] != "and":
            try:
                return str(w2n.word_to_num(words[0])) + str(w2n.word_to_num(words[1]))
            except:
                pass
        
        try:
            return str(w2n.word_to_num(text_chunk))
        except:
            return text_chunk

    text = re.sub(compound_pattern, replace_with_num, text, flags=re.IGNORECASE)
    
    # strip everything except letters, numbers, and apostrophes
    text = re.sub(r"[^a-z0-9'\s]", "", text)

    # isolate every digit
    text = re.sub(r'(\d)', r' \1 ', text)

    # clean extra whitespace 
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# -------------------- Load SCP + text --------------------
def load_scp_text(scp_file, text_file):
    scp_dict = {}
    with open(scp_file, "r") as f_scp:
        for line in f_scp:
            utt_id, path = line.strip().split(maxsplit=1)
            scp_dict[utt_id] = path

    text_dict = {}
    with open(text_file, "r") as f_text:
        for line in f_text:
            utt_id, transcription = line.strip().split(maxsplit=1)
            text_dict[utt_id] = transcription
    
    data = []
    for utt_id in scp_dict:
        if utt_id in text_dict:
            data.append({
                "utt_id": utt_id,
                "audio_path": scp_dict[utt_id],
                "text": text_dict[utt_id]
            })
    return data

# -------------------- Read audio as numpy array --------------------
def load_audio(item, target_sr=16000):
    # Read the audio
    audio, sr = sf.read(item["audio_path"])
    
    # Convert to mono if stereo
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    
    # Resample if needed
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    
    return audio

def get_detailed_metrics(predictions, references):
    out = jiwer.process_words(references, predictions)

    S = out.substitutions
    D = out.deletions
    I = out.insertions
    H = out.hits  # Correct words
    N = S + D + H  # Total words in reference
    
    # calculate rates 
    metrics = {
        "WER": (S + D + I) / N * 100,
        "S_rate": (S / N) * 100,
        "D_rate": (D / N) * 100,
        "I_rate": (I / N) * 100,
    }
    
    return metrics

def print_common_errors(predictions, references, top_n=20):
    substitutions = []
    deletions = []
    insertions = []

    for ref, hyp in zip(references, predictions):
        # Get word-level alignment
        out = jiwer.process_words(ref, hyp)
        for op in out.alignments[0]:
            # op is an object containing type, ref_start/end, hyp_start/end
            r_words = ref.split()[op.ref_start_idx:op.ref_end_idx]
            h_words = hyp.split()[op.hyp_start_idx:op.hyp_end_idx]

            if op.type == 'substitute':
                substitutions.append(f"{' '.join(r_words)} -> {' '.join(h_words)}")
            elif op.type == 'delete':
                deletions.append(' '.join(r_words))
            elif op.type == 'insert':
                insertions.append(' '.join(h_words))

    print(f"most common substitutions")
    for error, count in Counter(substitutions).most_common(top_n):
        print(f"{count:4d}x | {error}")

    print(f"most common deletions")
    for error, count in Counter(deletions).most_common(top_n):
        print(f"{count:4d}x | {error}")

    print(f"most common insertions")
    for error, count in Counter(insertions).most_common(top_n):
        print(f"{count:4d}x | {error}")

def print_wer_per_duration(df, bins=[0, 2, 5, 10, 20, 30]): 
    labels = [f"{bins[i]}-{bins[i+1]}s" for i in range(len(bins)-1)]
    df_analysis = df.copy()
    df_analysis['duration_bin'] = pd.cut(df_analysis['duration'], bins=bins, labels=labels)
    stats = df_analysis.groupby('duration_bin', observed=True).agg(
        avg_wer=('wer', 'mean'),
        sample_count=('path', 'count')
    )
    
    stats['avg_wer'] = (stats['avg_wer']).round(4).astype(str)
    
    print("wer per duration")
    print(stats)
    print("\n")

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

# -------------------- Main script --------------------
def main() -> None:
    args = parse_args()

    # derive output paths
    base = Path(args.outfile)
    outfile_raw      = base.with_name(base.stem + "_raw"      + base.suffix)
    outfile_bad_word = base.with_name(base.stem + "_bad_word" + base.suffix)
    outfile_norm     = base.with_name(base.stem + "_norm"     + base.suffix)

    # load SCP + text
    test_dataset = load_scp_text(args.scp_file, args.text_file)

    # lists to store results
    paths, durations = [], []
    raw_hyps,      raw_refs      = [], []
    badword_hyps,  badword_refs  = [], []
    norm_hyps,     norm_refs     = [], []

    # iterate over dataset
    for item in tqdm(test_dataset, desc="Transcribing Test Set"):
        audio_array = load_audio(item)

        # skip audio segments that are too long/short
        duration = len(audio_array) / 16000
        if duration < 0.5 or duration > 30:
            continue

        audio_example = {
            "audio": {
                "array": audio_array,
                "sampling_rate": 16000
            }
        }

        # raw whisper output
        raw_whisper = transcribe(audio_example, args.do_sample, args.temp, args.top_p)
        raw_gt = item["text"]

        # bad-word fixes
        whisper_fix_bad = fix_bad_words(raw_whisper, bad_word_fixes)
        gt_fix_bad      = fix_bad_words(raw_gt,      bad_word_fixes)

        # full normalization
        whisper_norm = extensive_normalization(whisper_fix_bad)
        gt_norm      = extensive_normalization(gt_fix_bad)

        # skip empty transcriptions
        if gt_norm.strip() == "":
            continue

        paths.append(item["audio_path"])
        durations.append(duration)

        raw_hyps.append(raw_whisper);         raw_refs.append(raw_gt)
        badword_hyps.append(whisper_fix_bad); badword_refs.append(gt_fix_bad)
        norm_hyps.append(whisper_norm);       norm_refs.append(gt_norm)

    # save results
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

    # global WER
    results = get_detailed_metrics(norm_hyps, norm_refs)
    print(f"\nGlobal WER — raw:          {global_wer_raw:.4f}")
    print(f"Global WER — bad-word fix: {global_wer_bw:.4f}")
    print(f"Global WER — full norm:    {global_wer_norm:.4f}  (expect ~0.508 test / 51.4 dev)")
    print(f"Substitutions (S): {results['S_rate']:.1f}%  (expect ~26.2% dev)")
    print(f"Deletions (D):     {results['D_rate']:.1f}%  (expect ~11.2% dev)")
    print(f"Insertions (I):    {results['I_rate']:.1f}%  (expect ~14.0% dev)")

    df_norm = pd.read_csv(outfile_norm).dropna(subset=["duration"])
    print_wer_per_duration(df_norm)
    print_common_errors(norm_hyps, norm_refs)

# -------------------- Entry point --------------------
if __name__ == "__main__":
    main()
