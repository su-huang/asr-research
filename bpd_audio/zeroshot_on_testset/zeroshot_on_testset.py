import os
os.environ["HF_USE_TORCH_CODEC"] = "0"  # Must come before datasets/audio imports

import torch
import pandas as pd
import argparse
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_from_disk, Audio  # Import after disabling TorchCodec
from whisper.normalizers import EnglishTextNormalizer
import librosa 
import re
from num2words import num2words
from word2number import w2n


import jiwer
from jiwer import wer
from pathlib import Path
import soundfile as sf
import numpy as np
from collections import Counter

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
def transcribe_v3_from_disk(example, do_sample, temp, top_p):
    path = example["absolute_path"]
    
    # sr=16000 is required by Whisper
    audio, _ = librosa.load(path, sr=16000)

    # Extract 128-bin features specifically for v3
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            do_sample = do_sample,
            temperature = temp if do_sample else None,
            top_p = top_p if do_sample else None,
            forced_decoder_ids = forced_decoder_ids,
            max_new_tokens = 444
        )
    
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

# -------------------- Normalization function --------------------
def normalize_example(predicted, text):
    normalizer = EnglishTextNormalizer()
    return normalizer(predicted), normalizer(text)

def extensive_normalization(text): 
    text = text.lower()

    # handle specific tags and quotes
    text = text.replace("<unintelligible>", "").replace("<x>", "")
    text = text.replace('"', "").replace('`', "'")
    text = text.replace('‘', "'").replace('’', "'")
    text = text.replace('-', ' ')

    number_pattern = r'\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion)\b'
    
    # replace matched words with digits
    def replace_with_num(match):
        try:
            return str(w2n.word_to_num(match.group(0)))
        except:
            return match.group(0)

    # replace spelled out numbers with digit strings
    text = re.sub(number_pattern, replace_with_num, text, flags=re.IGNORECASE)
    
    # keeps lowercase letters, numbers, apostrophes, and whitespace
    text = re.sub(r"[^a-z0-9'\s]", "", text)

    # isolate each digit
    text = re.sub(r'(\d)', r'\1 ', text)

    # clean extra whitespace 
    text = re.sub(r'\s+', ' ', text).strip()

    return text

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

# -------------------- Main script --------------------
def main() -> None:
    args = parse_args()
    test_dataset = load_from_disk("/export/fs06/kchapar1/bpd_asr/datasets/datasets_with_paths/test_dataset/")
    
    # We reset the format to make sure we can read the strings (paths)
    test_dataset.set_format(None) 

    paths, groundtruth, whisper_transcript_list, wers = [], [], [], []

    for i in tqdm(range(len(test_dataset)), desc="Transcribing (v3 Re-processing)"):
        item = test_dataset[i]
        
        # Get Path
        audio_path = item['absolute_path']
        paths.append(audio_path)

        # Transcribe (re-processing on the fly)
        whisper_transcript = transcribe_v3_from_disk(item, args.do_sample, args.temp, args.top_p)

        # Decode the Ground Truth (labels column is still useful)
        # Note: 'item' is now a dict, so labels is likely a list of IDs
        label_ids = [l if l != -100 else tokenizer.pad_token_id for l in item["labels"]]
        gt_text = tokenizer.decode(label_ids, skip_special_tokens=True)

        # Normalize and Score
        whisper_norm = extensive_normalization(whisper_transcript)
        gt_norm = extensive_normalization(gt_text)
        whisper_transcript_list.append(whisper_norm)
        groundtruth.append(gt_norm)

        # Compute WER
        if gt_norm == "":
            whisper_wer = 0 if whisper_norm == "" else 1
        elif whisper_norm == "":
            whisper_wer = 1
        else:
            whisper_wer = wer(gt_norm, whisper_norm)
        wers.append(whisper_wer)

    # Global WER 
    global_wer = sum(wers) / len(wers) if wers else 0.0

    # Save results
    df = pd.DataFrame({
        "path": paths,
        "duration": durations,
        "normalized_groundtruth": groundtruth,
        "normalized_whisper_lgv3": whisper_transcript_list,
        "wer": wers
    })

    results = get_detailed_metrics(whisper_transcript_list, groundtruth)
    print(f"Global Whisper WER: {global_wer:.4f} test - expect: 0.508 test, 51.4 dev") 
    print(f"Substitutions (S): {results['S_rate']:.1f}% test - expect: 26.2% dev")
    print(f"Deletions (D):     {results['D_rate']:.1f}% test - expect: 11.2% dev")
    print(f"Insertions (I):    {results['I_rate']:.1f}% test - expect: 14.0% dev")

    print_wer_per_duration(df) 
    print_common_errors(whisper_transcript_list, groundtruth) 

    # Create summary row at end of CSV 
    summary_row = pd.DataFrame({
        "path": ["GLOBAL_WER"],
        "duration": [None], 
        "normalized_groundtruth": [""],
        "normalized_whisper_lgv3": [""],
        "wer": [global_wer]
    })

    df = pd.concat([df, summary_row], ignore_index=True)
    df.to_csv(args.outfile, index=False)

# -------------------- Entry point --------------------
if __name__ == "__main__":
    main()
