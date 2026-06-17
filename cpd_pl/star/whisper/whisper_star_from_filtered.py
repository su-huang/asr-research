"""
STAR finetuning for Whisper, starting from a precomputed filtered training CSV
(with STAR scores already calculated) while dev data is still loaded from the
original Kaldi-style directory (wav.scp + text).

Expects TRAIN_CSV to contain columns:
  audio, ground_truth, pseudo_label, sample_star_score, avg_wer, diversity,
  star_scores, confidence_scores, attention_scores, conflict_scores, no_conflict_scores

star_scores (and friends) are expected to be JSON-encoded lists, as written by
the original audit CSV.
"""

import json
import os
import random

import fire
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from jiwer import wer as calculate_wer
from tqdm import tqdm
from transformers import (
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)
from whisper.normalizers import EnglishTextNormalizer

normalizer = EnglishTextNormalizer()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def safe_wer(ref, hyp):
    if not ref.strip() and not hyp.strip():
        return 0.0
    if not ref.strip() or not hyp.strip():
        return 1.0
    return calculate_wer(ref, hyp)


def build_item_from_row(row, feature_extractor, tokenizer, prompt_ids):
    """Reconstructs the in-memory training item from a filtered CSV row.
    Loads the audio fresh (mel features) and re-tokenizes the pseudo_label
    text into ids, since the CSV stores text/scores but not raw tensors.
    """
    audio_path = row["audio"]
    audio, sr = torchaudio.load(audio_path)
    if sr != 16000:
        audio = torchaudio.functional.resample(audio, sr, 16000)

    mel = feature_extractor(
        audio.squeeze(0).numpy(), sampling_rate=16_000, return_tensors="pt"
    )["input_features"]

    pseudo_label_text = str(row["pseudo_label"])
    label_ids = tokenizer(pseudo_label_text, add_special_tokens=False).input_ids
    pseudo_label_ids = torch.tensor([prompt_ids + label_ids]).long()

    star_scores = json.loads(row["star_scores"])

    n_label_toks = len(label_ids)
    if len(star_scores) < n_label_toks:
        star_scores = star_scores + [1.0] * (n_label_toks - len(star_scores))
    elif len(star_scores) > n_label_toks:
        star_scores = star_scores[:n_label_toks]

    item = {
        "audio_path": audio_path,
        "text": str(row["ground_truth"]),
        "mel": mel,
        "pseudo_label_ids": pseudo_label_ids,
        "pseudo_text": pseudo_label_text,
        "probs": torch.tensor(star_scores).unsqueeze(0),
        "sample_star_score": float(row.get("sample_star_score", 1.0)),
        "avg_wer": float(row.get("avg_wer", 0.0)) if pd.notna(row.get("avg_wer", None)) else 0.0,
        "diversity": int(row.get("diversity", 1)) if pd.notna(row.get("diversity", None)) else 1,
    }
    return item


def load_train_dataset_from_csv(csv_path, feature_extractor, tokenizer, prompt_ids):
    df = pd.read_csv(csv_path)
    dataset = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Loading {os.path.basename(csv_path)}"):
        try:
            item = build_item_from_row(row, feature_extractor, tokenizer, prompt_ids)
            dataset.append(item)
        except Exception as e:
            print(f"  Skipping row (audio={row.get('audio', '?')}): {e}")
    return dataset


def load_dev_dataset_kaldi(data_path, feature_extractor):
    """Loads dev data from a Kaldi-style directory (wav.scp + text).
    Only mel features + ground-truth text are needed for dev/eval — no STAR
    scoring required since dev is never used for loss weighting.
    """
    with open(data_path + "wav.scp", "r") as f1:
        wave_data = f1.readlines()
    with open(data_path + "text", "r") as f2:
        trans_data = f2.readlines()

    dataset = []
    for audio_line, text_line in tqdm(zip(wave_data, trans_data), total=len(wave_data), desc="Loading dev (Kaldi)"):
        audio_path = audio_line.strip().split(None, 1)[1].strip()
        text = " ".join(text_line.split()[1:]).lower().strip()

        audio, sr = torchaudio.load(audio_path)
        if sr != 16000:
            audio = torchaudio.functional.resample(audio, sr, 16000)

        mel = feature_extractor(
            audio.squeeze(0).numpy(), sampling_rate=16_000, return_tensors="pt"
        )["input_features"]

        dataset.append({
            "audio_path": audio_path,
            "text": text,
            "mel": mel,
        })

    return dataset


def evaluate(model, dataset, processor, forced_decoder_ids):
    with torch.no_grad():
        all_pred, all_gt = [], []
        for item in dataset:
            mel = item["mel"]
            generated_ids = model.generate(
                input_features=mel.to(device=device, dtype=next(model.parameters()).dtype),
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=150,
            )
            generated_text = processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

            pred = normalizer(generated_text)
            pred = pred if len(pred) > 0 else "<UNK>"

            gt = normalizer(item["text"])
            gt = gt if len(gt) > 0 else "<UNK>"

            all_pred.append(pred)
            all_gt.append(gt)

    return calculate_wer(all_gt, all_pred)


def train(
    MODEL="openai/whisper-large-v3",
    DATASET="chime4",
    TRAIN_CSV="",       # CHANGED: filtered audit CSV with STAR scores
    DEV_DATA="",         # UNCHANGED: Kaldi-style dir with wav.scp + text
    SAVE_EVERY=10,
    GRADIENT_ACCUMULATION_STEPS=4,
    LEARNING_RATE=1e-5,
    EPOCHS=100,
    SAVE_DIR="runs",
    TEST_CSV="",
    RUN_ID="",
    PATIENCE=5,
    MAX_RATIO=10.0,
    MIN_RATIO=0.01,
):
    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL)
    processor = WhisperProcessor.from_pretrained(
        MODEL, language="en", task="transcribe"
    )
    tokenizer = WhisperTokenizer.from_pretrained(
        MODEL, language="en", task="transcribe"
    )

    # fp32 throughout for training stability
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL, attn_implementation="eager", torch_dtype=torch.float32
    ).to(device)

    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="en", task="transcribe"
    )

    prompt_and_eos = tokenizer("")["input_ids"]
    prompt_ids, eos_id = prompt_and_eos[:-1], prompt_and_eos[-1]
    n_prompt_toks = 4

    model_size = MODEL.replace("openai/whisper-", "")
    run_suffix = f"_{RUN_ID}" if RUN_ID else ""
    exp_dir = os.path.join(SAVE_DIR, f"{DATASET}_{model_size}{run_suffix}")
    os.makedirs(exp_dir, exist_ok=True)

    # ── Load precomputed STAR-filtered training data from CSV ───────────────
    print(f"Loading filtered training data from: {TRAIN_CSV}")
    filtered_train_dataset = load_train_dataset_from_csv(TRAIN_CSV, feature_extractor, tokenizer, prompt_ids)
    print(f"Loaded {len(filtered_train_dataset)} training samples.")

    # ── Load dev data from Kaldi-style directory (unchanged format) ─────────
    print(f"Loading dev data from Kaldi dir: {DEV_DATA}")
    dev_dataset = load_dev_dataset_kaldi(DEV_DATA, feature_extractor)
    print(f"Loaded {len(dev_dataset)} dev samples.")

    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")

    steps, loss = 0, 0
    best_loss, best_wer = 10000, 10000
    patience_counter = 0
    stop_training = False
    skipped_nan_batches = 0

    print("Starting training...")
    for Epoch in range(EPOCHS):
        if stop_training:
            break
        print("Epoch: ", Epoch + 1)

        random.shuffle(filtered_train_dataset)
        print("Training...")
        optimizer.zero_grad()

        for item in filtered_train_dataset:
            mel = item["mel"].to(device=device, dtype=next(model.parameters()).dtype)
            labels = item["pseudo_label_ids"].to(device)
            ratios = item["probs"].to(device)

            y_in = labels[:, :-1]
            y_out = labels[:, 1:]

            logits = model(input_features=mel, decoder_input_ids=y_in).logits
            loss_items = loss_fn(logits.permute(0, 2, 1), y_out)

            ratio_mean = torch.mean(ratios)
            ratio_mean = ratio_mean if torch.isfinite(ratio_mean) and ratio_mean.abs() > 1e-8 else torch.tensor(1.0, device=device)
            ratios = ratios / ratio_mean
            ratios = torch.clamp(ratios, min=MIN_RATIO, max=MAX_RATIO)

            n_pl_positions = loss_items.shape[1] - (n_prompt_toks - 1)
            if ratios.shape[-1] != n_pl_positions:
                n = min(ratios.shape[-1], n_pl_positions)
                ratios = ratios[:, :n]
                pl_loss = loss_items[:, n_prompt_toks - 1: n_prompt_toks - 1 + n]
            else:
                pl_loss = loss_items[:, n_prompt_toks - 1:]

            loss = (
                torch.sum(loss_items[:, : n_prompt_toks - 1])
                + torch.sum(pl_loss * ratios)
            ) / (n_prompt_toks - 1 + ratios.shape[-1])

            if not torch.isfinite(loss):
                skipped_nan_batches += 1
                print(f"  WARNING: non-finite loss detected, skipping batch "
                      f"(total skipped: {skipped_nan_batches})", flush=True)
                optimizer.zero_grad()
                continue

            (loss / GRADIENT_ACCUMULATION_STEPS).backward()

            grad_is_finite = all(
                torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
            )
            if not grad_is_finite:
                skipped_nan_batches += 1
                print(f"  WARNING: non-finite gradient detected, skipping optimizer step "
                      f"(total skipped: {skipped_nan_batches})", flush=True)
                optimizer.zero_grad()
                continue

            steps += 1

            if steps % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if steps % SAVE_EVERY == 0:  # Evaluate
                torch.save(model.state_dict(), f"{exp_dir}/Iter_{steps}.pth")

                model.eval()
                dev_wer = evaluate(model, dev_dataset, processor, forced_decoder_ids)
                model.train()

                if dev_wer < best_wer or (
                    dev_wer == best_wer and loss < best_loss
                ):
                    torch.save(model.state_dict(), f"{exp_dir}/best_checkpoint.pth")
                    best_loss, best_wer = loss, dev_wer
                    patience_counter = 0
                    print(f"  -> New best (dev WER: {best_wer:.4f})", flush=True)
                else:
                    patience_counter += 1
                    print(f"  -> No improvement. Patience: {patience_counter}/{PATIENCE}", flush=True)
                    if patience_counter >= PATIENCE:
                        print("Early stopping triggered.")
                        stop_training = True
                        break

    print(f"Total batches skipped due to NaN/Inf: {skipped_nan_batches}")

    torch.save(model.state_dict(), f"{exp_dir}/last_checkpoint.pth")

    best_ckpt_path = os.path.join(exp_dir, "best_checkpoint.pth")
    if not os.path.exists(best_ckpt_path):
        print("Warning: 'best_checkpoint.pth' not found (training exited before evaluation checkpoint). Falling back to 'last_checkpoint.pth'.")
        best_ckpt_path = os.path.join(exp_dir, "last_checkpoint.pth")

    print(f"Loading weights from {best_ckpt_path} for HuggingFace save...")

    best_model = WhisperForConditionalGeneration.from_pretrained(
        MODEL, attn_implementation="eager", torch_dtype=torch.float32
    ).to(device)

    best_weights = torch.load(best_ckpt_path, map_location=device)

    has_nan = any(torch.isnan(v.float()).any() for v in best_weights.values())
    if has_nan:
        print("ERROR: best checkpoint contains NaN weights. Aborting HuggingFace save and test inference.")
        return

    best_model.load_state_dict(best_weights)

    best_model.save_pretrained(exp_dir)
    print(f"Model saved to: {exp_dir}")
    print(f"Processor saved to: {os.path.join(exp_dir, 'processor')}")

    # ── Test set inference ────────────────────────────────────────────────
    if TEST_CSV:
        print("Running test set inference...")
        test_df = pd.read_csv(TEST_CSV)
        audio_paths = test_df["audio"].tolist()
        refs = [str(t).strip() for t in test_df["text"].tolist()]

        best_model.eval()
        (
            all_audio,
            all_gt_raw,
            all_pred_raw,
            all_gt_norm,
            all_pred_norm,
            all_wers,
        ) = ([], [], [], [], [], [])

        BATCH_SIZE_INFER = 8
        for i in tqdm(
            range(0, len(audio_paths), BATCH_SIZE_INFER), desc="Test inference"
        ):
            batch_paths = audio_paths[i : i + BATCH_SIZE_INFER]
            batch_refs = refs[i : i + BATCH_SIZE_INFER]

            audio_arrays, valid_paths, valid_refs = [], [], []
            for path, ref in zip(batch_paths, batch_refs):
                try:
                    audio_arr, sr = sf.read(path)
                    if audio_arr.ndim > 1:
                        audio_arr = audio_arr.mean(axis=1)
                    if sr != 16000:
                        audio_arr = librosa.resample(
                            audio_arr, orig_sr=sr, target_sr=16000
                        )
                    if len(audio_arr) > 30 * 16000:
                        audio_arr = audio_arr[: 30 * 16000]
                    audio_arrays.append(audio_arr)
                    valid_paths.append(path)
                    valid_refs.append(ref)
                except Exception as e:
                    print(f"  Error loading {path}: {e}")

            if not audio_arrays:
                continue

            model_dtype = next(best_model.parameters()).dtype
            input_features = (
                feature_extractor(
                    audio_arrays, sampling_rate=16000, return_tensors="pt"
                )
                .input_features.to(device=device, dtype=model_dtype)
            )

            with torch.no_grad():
                predicted_ids = best_model.generate(
                    input_features,
                    language="en",
                    task="transcribe",
                    do_sample=False,
                )

            predictions = processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )

            for path, ref, pred in zip(valid_paths, valid_refs, predictions):
                gt_norm_str = normalizer(ref)
                pred_norm_str = normalizer(pred.strip())
                gt_norm_str = gt_norm_str if gt_norm_str else "<UNK>"
                pred_norm_str = pred_norm_str if pred_norm_str else "<UNK>"
                w = safe_wer(gt_norm_str, pred_norm_str)
                all_audio.append(path)
                all_gt_raw.append(ref)
                all_pred_raw.append(pred.strip())
                all_gt_norm.append(gt_norm_str)
                all_pred_norm.append(pred_norm_str)
                all_wers.append(w)

        avg_wer = sum(all_wers) / len(all_wers) if all_wers else 0.0
        print(
            f"Test Average WER (per-sample mean, n={len(all_wers)}): {avg_wer:.4f}"
        )
        csv_path = os.path.join(exp_dir, "test_set_transcriptions.csv")
        pd.DataFrame(
            {
                "audio": all_audio,
                "ground_truth": all_gt_raw,
                "prediction": all_pred_raw,
                "gt_norm": all_gt_norm,
                "pred_norm": all_pred_norm,
                "wer": all_wers,
            }
        ).to_csv(csv_path, index=False)
        print(f"Test transcriptions saved to: {csv_path}")


if __name__ == "__main__":
    fire.Fire(train)