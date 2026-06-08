import argparse
import os
import functools
import numpy as np
import soundfile as sf
import librosa
import pandas as pd
from datasets import Dataset
from transformers import WhisperProcessor

MAX_INPUT_LENGTH = 25 * 16000
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")


def prepare_dataset(batch, audio_col, text_col):
    audio_path = batch[audio_col]
    audio_array, sr = sf.read(audio_path)
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)
    if sr != 16000:
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000, res_type="soxr_hq")
    if len(audio_array) > MAX_INPUT_LENGTH:
        audio_array = audio_array[:MAX_INPUT_LENGTH]
    input_features = processor.feature_extractor(audio_array, sampling_rate=16000).input_features[0]
    text = batch[text_col]
    input_ids = processor.tokenizer(text, truncation=True).input_ids
    return {
        "input_features": np.array(input_features, dtype=np.float32),
        "labels": input_ids,
    }


def main(args):
    print(f"Reading CSV: {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df)} rows")

    dataset = Dataset.from_pandas(df, preserve_index=False)
    cols_to_remove = [col for col in dataset.column_names if col != args.audio_path]
    print(f"Preprocessing {len(dataset)} samples with {args.num_proc} workers...")

    prepare_fn = functools.partial(prepare_dataset, audio_col=args.audio_path, text_col=args.text)
    processed = dataset.map(
        prepare_fn,
        remove_columns=cols_to_remove,
        num_proc=args.num_proc,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    processed.save_to_disk(args.output_path)
    print(f"Saved preprocessed dataset to: {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess a CSV into a Whisper-ready HuggingFace dataset")
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, default="audio")
    parser.add_argument("--text", type=str, default="text_pl")
    parser.add_argument("--num_proc", type=int, default=4)
    args = parser.parse_args()
    main(args)