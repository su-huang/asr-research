import argparse
import os

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperProcessor
from whisper.normalizers import EnglishTextNormalizer

normalizer = EnglishTextNormalizer()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def safe_wer(ref, hyp):
    from jiwer import wer as calculate_wer
    if not ref.strip() and not hyp.strip():
        return 0.0
    if not ref.strip() or not hyp.strip():
        return 1.0
    return calculate_wer(ref, hyp)


def main(args):
    print(f"Loading base model: {args.base_model}")
    model = WhisperForConditionalGeneration.from_pretrained(args.base_model).to(device)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.base_model)
    processor = WhisperProcessor.from_pretrained(args.base_model)

    print(f"Loading weights from: {args.model_path}")
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    model_dtype = next(model.parameters()).dtype
    print("Model loaded.")

    test_df = pd.read_csv(args.test_csv)
    audio_paths = test_df[args.audio_col].tolist()
    refs = [str(t).strip() for t in test_df[args.text_col].tolist()]

    all_audio, all_gt_raw, all_pred_raw, all_gt_norm, all_pred_norm, all_wers = [], [], [], [], [], []

    for i in tqdm(range(0, len(audio_paths), args.batch_size), desc="Test inference"):
        batch_paths = audio_paths[i: i + args.batch_size]
        batch_refs  = refs[i: i + args.batch_size]

        audio_arrays, valid_paths, valid_refs = [], [], []
        for path, ref in zip(batch_paths, batch_refs):
            try:
                audio_arr, sr = sf.read(path)
                if audio_arr.ndim > 1:
                    audio_arr = audio_arr.mean(axis=1)
                if sr != 16000:
                    audio_arr = librosa.resample(audio_arr, orig_sr=sr, target_sr=16000)
                if len(audio_arr) > 30 * 16000:
                    audio_arr = audio_arr[:30 * 16000]
                audio_arrays.append(audio_arr)
                valid_paths.append(path)
                valid_refs.append(ref)
            except Exception as e:
                print(f"  Error loading {path}: {e}")

        if not audio_arrays:
            continue

        input_features = feature_extractor(
            audio_arrays, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device=device, dtype=model_dtype)

        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                language="en",
                task="transcribe",
                do_sample=False,
            )

        predictions = processor.batch_decode(predicted_ids, skip_special_tokens=True)

        for path, ref, pred in zip(valid_paths, valid_refs, predictions):
            gt_norm   = normalizer(ref) or "<UNK>"
            pred_norm = normalizer(pred.strip()) or "<UNK>"
            w = safe_wer(gt_norm, pred_norm)
            all_audio.append(path)
            all_gt_raw.append(ref)
            all_pred_raw.append(pred.strip())
            all_gt_norm.append(gt_norm)
            all_pred_norm.append(pred_norm)
            all_wers.append(w)

    avg_wer = sum(all_wers) / len(all_wers) if all_wers else 0.0
    print(f"Test Average WER (per-sample mean, n={len(all_wers)}): {avg_wer:.4f}")

    out_df = pd.DataFrame({
        "audio":        all_audio,
        "ground_truth": all_gt_raw,
        "prediction":   all_pred_raw,
        "gt_norm":      all_gt_norm,
        "pred_norm":    all_pred_norm,
        "wer":          all_wers,
    })
    out_df.to_csv(args.output_csv, index=False)
    print(f"Results saved to: {args.output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Whisper test set inference from .pth checkpoint")
    parser.add_argument("--model_path",  type=str, required=True, help="Path to .pth checkpoint file")
    parser.add_argument("--base_model",  type=str, default="openai/whisper-large-v3", help="Base HuggingFace model ID")
    parser.add_argument("--test_csv",    type=str, required=True)
    parser.add_argument("--output_csv",  type=str, required=True)
    parser.add_argument("--audio_col",   type=str, default="audio")
    parser.add_argument("--text_col",    type=str, default="text")
    parser.add_argument("--batch_size",  type=int, default=8)
    args = parser.parse_args()
    main(args)