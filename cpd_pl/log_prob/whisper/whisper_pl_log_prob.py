import pandas as pd
import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from collections import Counter
import soundfile as sf
import librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

MAX_AUDIO_SAMPLES = 30 * 16000
BATCH_SIZE = 8


def load_audio(path):
    audio_array, sr = sf.read(path)
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)
    if sr != 16000:
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
    was_truncated = len(audio_array) > MAX_AUDIO_SAMPLES
    if was_truncated:
        audio_array = audio_array[:MAX_AUDIO_SAMPLES]
    return audio_array, was_truncated


def compute_avg_logprob_batch(output, special_ids, batch_size):
    num_prompt_tokens = output.sequences.shape[1] - len(output.scores)
    avg_logprobs = []
    for i in range(batch_size):
        token_log_probs = []
        for step, step_scores in enumerate(output.scores):
            token_id_int = output.sequences[i, num_prompt_tokens + step].item()
            if token_id_int in special_ids:
                continue
            log_probs = torch.nn.functional.log_softmax(step_scores[i], dim=-1)
            token_log_probs.append(log_probs[token_id_int].item())
        avg_logprobs.append(np.mean(token_log_probs) if token_log_probs else -99.0)
    return avg_logprobs


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


def main(args):
    model_id = args.whisper_size
    if not os.path.isdir(model_id) and not model_id.startswith("openai/whisper-"):
        model_id = f"openai/whisper-{model_id}"

    print(f"Loading model: {model_id}")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    ).to(device)
    model.generation_config.forced_decoder_ids = None
    processor = AutoProcessor.from_pretrained(model_id)
    special_ids = set(processor.tokenizer.all_special_ids) if args.record_logprobs else None

    df = pd.read_csv(args.input_csv)
    audio_paths = df[args.audio_column].tolist()
    gt_lookup = dict(zip(df[args.audio_column], df[args.gt_column])) if args.gt_column and args.gt_column in df.columns else {}
    if args.max_samples is not None:
        audio_paths = audio_paths[:args.max_samples]

    metadata_records = []
    total_truncated = 0
    print(f"Starting transcription of {len(audio_paths)} files in batches of {BATCH_SIZE}...")

    for batch_start in tqdm(range(0, len(audio_paths), BATCH_SIZE)):
        batch_paths = audio_paths[batch_start:batch_start + BATCH_SIZE]
        valid_paths, audio_arrays, truncated_flags, durations = [], [], [], []

        for path in batch_paths:
            if not os.path.exists(path):
                print(f"Skipping missing file: {path}")
                continue
            try:
                audio_array, was_truncated = load_audio(path)
                audio_arrays.append(audio_array)
                valid_paths.append(path)
                truncated_flags.append(was_truncated)
                durations.append(sf.info(path).duration)
            except Exception as e:
                print(f"Error loading {path}: {e}")

        if not valid_paths:
            continue

        try:
            input_features = processor.feature_extractor(
                audio_arrays, sampling_rate=16000, return_tensors="pt"
            ).input_features.to(device=device, dtype=torch_dtype)

            with torch.no_grad():
                output = model.generate(
                    input_features,
                    language="en",
                    task="transcribe",
                    do_sample=False,
                    return_dict_in_generate=args.record_logprobs,
                    output_scores=args.record_logprobs,
                )

            if args.record_logprobs:
                texts = processor.tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
                avg_logprobs = compute_avg_logprob_batch(output, special_ids, len(valid_paths))
            else:
                texts = processor.tokenizer.batch_decode(output, skip_special_tokens=True)
                avg_logprobs = [None] * len(valid_paths)

            for path, text, was_truncated, duration, avg_logprob in zip(
                valid_paths, texts, truncated_flags, durations, avg_logprobs
            ):
                if was_truncated:
                    total_truncated += 1
                record = {
                    "audio": path,
                    "text": text.strip(),
                    "truncated": was_truncated,
                    "duration_s": round(float(duration), 2),
                }
                if gt_lookup:
                    record["ground_truth"] = gt_lookup.get(path, "")
                if args.record_logprobs:
                    record["avg_logprob"] = round(float(avg_logprob), 4)
                metadata_records.append(record)

        except Exception as e:
            print(f"Error processing batch at {batch_start}: {e}")

    print(f"Total transcribed: {len(metadata_records)}")
    print(f"Files truncated to 30s: {total_truncated} / {len(metadata_records)}")

    filtered_df = pd.DataFrame(metadata_records)
    total_duration_s = filtered_df["duration_s"].sum()
    print(f"Total duration: {total_duration_s / 3600:.2f} hours ({total_duration_s:.1f} seconds)")

    os.makedirs(os.path.dirname(os.path.abspath(args.pl_csv_save_path)), exist_ok=True)
    filtered_df.to_csv(args.pl_csv_save_path, index=False)
    print(f"CSV saved to: {args.pl_csv_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Whisper Pseudolabels")
    parser.add_argument("--whisper_size",      type=str, required=True)
    parser.add_argument("--input_csv",         type=str, required=True)
    parser.add_argument("--audio_column",      type=str, default="audio_filepath")
    parser.add_argument("--pl_csv_save_path",  type=str, required=True)
    parser.add_argument("--max_samples",       type=int, default=None)
    parser.add_argument("--gt_column",         type=str, default=None)
    parser.add_argument("--record_logprobs",   action="store_true")
    args = parser.parse_args()
    main(args)