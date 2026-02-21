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
def normalize_example(predicted, text):
    return normalizer(predicted), normalizer(text)

def extensive_normalization(text): 
    text = text.upper()

    text = text.replace("<UNINTELLIGIBLE>", "")
    text = text.replace("<X>", "")
    text = text.replace('"', "")
    text = text.replace('-', ' ')
    text = text.replace('`', "'")
    text = text.replace('‘', "'").replace('’', "'")

    # remove punctation except apostrophe 
    text = re.sub(r"[^A-Z0-9'\s]", "", text)

    # verbalize numbers 
    def replace_numbers(match):
        number_str = match.group(0)
        if len(number_str) in [3, 4]:
            try:
                if len(number_str) == 3:
                    parts = [number_str[0], number_str[1:]]
                else:
                    parts = [number_str[:2], number_str[2:]]
                return " ".join([num2words(int(p)) for p in parts]).upper()
            except:
                return num2words(int(number_str)).upper()
        
        return num2words(int(number_str)).upper()

    text = re.sub(r'\d+', replace_numbers, text)

    # remove extra whitespace and strip
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

    print(f"Top {top_n} Most Common Substitutions")
    for error, count in Counter(substitutions).most_common(top_n):
        print(f"{count:4d}x | {error}")

    print(f"Top {top_n} Most Common Deletions (Words the model misses)")
    for error, count in Counter(deletions).most_common(top_n):
        print(f"{count:4d}x | {error}")

    print(f"Top {top_n} Most Common Insertions (Hallucinations)")
    for error, count in Counter(insertions).most_common(top_n):
        print(f"{count:4d}x | {error}")

# -------------------- Main script --------------------
def main() -> None:
    args = parse_args()

    # Load SCP + text
    test_dataset = load_scp_text(args.scp_file, args.text_file)

    # Lists to store results
    paths, groundtruth, whisper_transcript_list, wers = [], [], [], []

    # Iterate over dataset
    for item in tqdm(test_dataset, desc="Transcribing Test Set"):
        paths.append(item["audio_path"])

        # Load audio
        audio_array = load_audio(item)

        # Wrap as Whisper-style input
        audio_example = {
            "audio": {
                "array": audio_array,
                "sampling_rate": 16000
            }
        }

        # Transcribe
        whisper_transcript = transcribe(audio_example, args.do_sample, args.temp, args.top_p)

        # Normalize
        # whisper_norm, gt_norm = normalize_example(whisper_transcript, item['text'])
        whisper_norm = extensive_normalization(whisper_transcript)
        gt_norm = extensive_normalization(item['text'])

        # skip empty transcriptions
        if gt_norm.strip() == "":
            continue

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
        "normalized_groundtruth": groundtruth,
        "normalized_whisper_lgv3": whisper_transcript_list,
        "wer": wers
    })

    # Create summary row at end of CSV 
    summary_row = pd.DataFrame({
        "path": ["GLOBAL_WER"],
        "normalized_groundtruth": [""],
        "normalized_whisper_lgv3": [""],
        "wer": [global_wer]
    })

    df = pd.concat([df, summary_row], ignore_index=True)
    df.to_csv(args.outfile, index=False)
    
    results = get_detailed_metrics(whisper_transcript_list, groundtruth)
    print(f"Global Whisper WER: {global_wer:.4f} test - expect: 0.508 test, 51.4 dev") 
    print(f"Substitutions (S): {results['S_rate']:.1f}% test - expect: 26.2% dev")
    print(f"Deletions (D):     {results['D_rate']:.1f}% test - expect: 11.2% dev")
    print(f"Insertions (I):    {results['I_rate']:.1f}% test - expect: 14.0% dev")

    print_common_errors(whisper_transcript_list, groundtruth) 

# -------------------- Entry point --------------------
if __name__ == "__main__":
    main()
