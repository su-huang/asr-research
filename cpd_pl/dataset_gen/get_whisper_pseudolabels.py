import argparse
import os
from collections import Counter

import librosa
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

MAX_AUDIO_SAMPLES = 30 * 16000  # truncate to 30s at 16kHz


def has_excessive_ngrams(text, max_repeats=8):
    if not isinstance(text, str) or text.strip() == "":
        return False
    words = text.split()
    for n in range(1, 6):
        ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        counts = Counter(ngrams)
        if any(count > max_repeats for count in counts.values()):
            return True
    return False


def load_audio(path):
    """Returns (audio_array, duration_s, was_truncated)."""
    audio_array, sr = sf.read(path)
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)
    if sr != 16000:
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
    duration_s = len(audio_array) / 16000
    was_truncated = len(audio_array) > MAX_AUDIO_SAMPLES
    if was_truncated:
        audio_array = audio_array[:MAX_AUDIO_SAMPLES]
    return audio_array, duration_s, was_truncated


def main(args):
    print(f"Loading Whisper model: {args.model_path}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
        torch_dtype = torch.bfloat16

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True
    ).to(device)

    processor = AutoProcessor.from_pretrained(args.model_path)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        torch_dtype=torch_dtype,
        device=device,
    )

    df = pd.read_csv(args.input_csv)
    audio_paths = df[args.audio_column].tolist()
    if args.max_samples is not None:
        audio_paths = audio_paths[:args.max_samples]

    print(f"Starting transcription of {len(audio_paths)} files in batches of {args.batch_size}...")
    print(f"Sampling enabled: temperature={args.temperature}")

    records = []
    total_truncated = 0

    for batch_start in tqdm(range(0, len(audio_paths), args.batch_size)):
        batch_paths = audio_paths[batch_start:batch_start + args.batch_size]
        valid_paths, audio_arrays, durations, truncated_flags = [], [], [], []

        for path in batch_paths:
            if not os.path.exists(path):
                print(f"Skipping missing file: {path}")
                continue
            try:
                audio_array, duration_s, was_truncated = load_audio(path)
                valid_paths.append(path)
                audio_arrays.append(audio_array)
                durations.append(duration_s)
                truncated_flags.append(was_truncated)
            except Exception as e:
                print(f"Error loading {path}: {e}")

        if not valid_paths:
            continue

        try:
            results = pipe(
                audio_arrays,
                batch_size=len(audio_arrays),
                generate_kwargs={
                    "language": "english",
                    "task": "transcribe",
                    "do_sample": True,
                    "temperature": args.temperature,
                }
            )

            for path, res, duration, truncated in zip(valid_paths, results, durations, truncated_flags):
                if truncated:
                    total_truncated += 1
                records.append({
                    "audio": path,
                    "text": res["text"].strip(),
                    "truncated": truncated,
                    "duration_s": round(float(duration), 2),
                })
        except Exception as e:
            print(f"Error processing batch at {batch_start}: {e}")

    print(f"Total transcribed: {len(records)}")
    print(f"Files truncated to 30s: {total_truncated} / {len(records)}")

    out_df = pd.DataFrame(records)
    out_df = out_df[~out_df["text"].apply(has_excessive_ngrams)]
    print(f"After n-gram filtering: {len(out_df)} records")
    total_duration_s = out_df["duration_s"].sum()
    print(f"Total duration: {total_duration_s / 3600:.2f} hours ({total_duration_s:.1f} seconds)")

    os.makedirs(os.path.dirname(os.path.abspath(args.pl_csv_save_path)), exist_ok=True)
    out_df.to_csv(args.pl_csv_save_path, index=False)
    print(f"CSV saved to: {args.pl_csv_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pseudo-labels using Whisper")
    parser.add_argument("--model_path", type=str, default="openai/whisper-large-v3")
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--audio_column", type=str, default="audio")
    parser.add_argument("--pl_csv_save_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()
    main(args)