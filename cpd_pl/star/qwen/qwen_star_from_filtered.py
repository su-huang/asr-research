"""
STAR finetuning for Qwen3-ASR-1.7B (Bypassing precomputation using Audit CSV).

This script skips eager attention precomputation and directly ingests a 
pre-computed training data audit CSV containing STAR scores.
"""

import copy
import heapq
import json
import os
import random

import librosa
import numpy as np
import pandas as pd
import torch
from jiwer import wer as calculate_wer
from qwen_asr import Qwen3ASRModel
from qwen_asr.inference.utils import parse_asr_output
from whisper.normalizers import EnglishTextNormalizer

normalizer = EnglishTextNormalizer()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def safe_wer(ref, hyp):
    if not ref.strip() and not hyp.strip():
        return 0.0
    if not ref.strip() or not hyp.strip():
        return 1.0
    return calculate_wer(ref, hyp)


def load_audio(path, sr=16000):
    return librosa.load(path, sr=sr, mono=True)[0]


def build_prefix(processor, audio_array):
    msgs = [{"role": "user", "content": [{"type": "audio", "audio": audio_array}]}]
    return processor.apply_chat_template(
        [msgs], add_generation_prompt=True, tokenize=False
    )[0]


def cast_inputs(inputs, dtype, dev):
    out = {}
    for k, v in inputs.items():
        if torch.is_tensor(v):
            out[k] = v.to(dtype=dtype, device=dev) if v.is_floating_point() else v.to(dev)
        else:
            out[k] = v
    return out


def train(
    MODEL="Qwen/Qwen3-ASR-1.7B",
    DATASET="fnlo",
    TRAIN_CSV="",  
    DEV_CSV="",
    TEST_CSV="",
    AUDIO_COL="audio",
    TEXT_COL="ground_truth",      
    PSEUDO_COL="pseudo_label",    
    STAR_SCORES_COL="star_scores",
    DEV_AUDIO_COL="audio",
    DEV_TEXT_COL="text",
    LAYER_ID=26,
    HEAD_ID=10,
    EPOCHS=100,
    BATCH_SIZE=4,
    GRAD_ACC=8,
    LR=2e-5,
    SAVE_EVERY=884,
    PATIENCE=5,
    SAVE_DIR="runs",
    RUN_ID="",
):
    # ── Load model ──────────────────────────────────────────────────────────
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    model_dtype = torch.bfloat16 if use_bf16 else torch.float16

    asr_wrapper = Qwen3ASRModel.from_pretrained(
        MODEL, dtype=model_dtype, device_map="cuda:0"
    )
    thinker = asr_wrapper.model.thinker
    processor = asr_wrapper.processor
    tokenizer = processor.tokenizer
    eos_token = tokenizer.eos_token or ""

    exp_dir = os.path.join(SAVE_DIR, DATASET)
    os.makedirs(exp_dir, exist_ok=True)

    # ── Evaluation ───────────────────────────────────────────────────────────
    def evaluate(csv_path):
        df = pd.read_csv(csv_path)
        all_pred, all_gt = [], []
        audio_paths = df[DEV_AUDIO_COL].tolist()
        refs = [str(t).strip() for t in df[DEV_TEXT_COL].tolist()]
        batch_size = 8
        for i in range(0, len(audio_paths), batch_size):
            results = asr_wrapper.transcribe(audio_paths[i: i + batch_size], language="English")
            for r, ref in zip(results, refs[i: i + batch_size]):
                hyp = r.text.strip() if hasattr(r, "text") else str(r)
                all_pred.append(normalizer(hyp) or "<UNK>")
                all_gt.append(normalizer(ref) or "<UNK>")
        return calculate_wer(all_gt, all_pred) if all_gt else 0.0

    # ── Load and Ingest Pre-audited Dataset ──────────────────────────────────
    print(f"Ingesting precomputed audit dataset from {TRAIN_CSV}...")
    df = pd.read_csv(TRAIN_CSV)
    filtered = []
    all_pred, all_gt = [], []

    for idx, row in enumerate(df.itertuples()):
        audio_path = str(getattr(row, AUDIO_COL))
        gt_text = str(getattr(row, TEXT_COL)).strip()
        pseudo_text_clean = str(getattr(row, PSEUDO_COL)).strip()
        
        # Safely extract and unpack JSON star scores list
        try:
            star_scores = json.loads(getattr(row, STAR_SCORES_COL))
        except Exception as e:
            print(f"  [{idx}] Error parsing STAR scores JSON string: {e}")
            continue

        try:
            audio_array = load_audio(audio_path)
        except Exception as e:
            print(f"  [{idx}] Error loading audio file {audio_path}: {e}")
            continue

        prefix_text = build_prefix(processor, audio_array)

        # Reconstruct full Qwen pseudo text target format (<asr_text>...)
        pseudo_text_raw = f"English<asr_text>{pseudo_text_clean}"

        # Get exact prefix lengths for target masking alignment
        with torch.no_grad():
            prefix_inputs = processor(
                text=[prefix_text], audio=[audio_array],
                return_tensors="pt", padding=True
            )
            prefix_len = prefix_inputs["input_ids"].shape[1]

        item = {
            "audio_array": audio_array,
            "text": gt_text,
            "pseudo_text": pseudo_text_raw,
            "prefix_text": prefix_text,
            "prefix_len": prefix_len,
            "star_weights": torch.tensor(star_scores),
        }
        filtered.append(item)
        all_pred.append(normalizer(pseudo_text_clean) or "<UNK>")
        all_gt.append(normalizer(gt_text) or "<UNK>")

        if (idx + 1) % 100 == 0:
            print(f"  Ingested {idx + 1}/{len(df)} samples", flush=True)

    train_wer = calculate_wer(all_gt, all_pred) if all_gt else 0.0
    print(f"Loaded {len(filtered)} items for training. Ingested Audit Data WER: {train_wer:.4f}")

    # Ensure SDPA attention is forced for optimized training footprint
    thinker.set_attn_implementation("sdpa")

    # ── Training ─────────────────────────────────────────────────────────────
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    optimizer = torch.optim.AdamW(thinker.parameters(), lr=LR)

    steps = 0
    best_wer = float("inf")
    best_loss = float("inf")
    patience_counter = 0
    stop_training = False

    print("Starting training...")
    for epoch in range(EPOCHS):
        if stop_training:
            break
        print(f"Epoch {epoch + 1}", flush=True)

        random.shuffle(filtered)
        optimizer.zero_grad()

        for idx, item in enumerate(filtered):
            audio_array = item["audio_array"]
            prefix_text = item["prefix_text"]
            pseudo_text = item["pseudo_text"]
            prefix_len  = item["prefix_len"]
            star_weights = item["star_weights"].to(device=device, dtype=model_dtype)

            full_text = prefix_text + pseudo_text + eos_token
            full_inputs = processor(
                text=[full_text], audio=[audio_array],
                return_tensors="pt", padding=True
            )
            full_inputs = cast_inputs(full_inputs, model_dtype, device)

            labels = full_inputs["input_ids"].clone()
            labels[0, :prefix_len] = -100

            out = thinker(
                input_ids=full_inputs["input_ids"],
                input_features=full_inputs.get("input_features"),
                feature_attention_mask=full_inputs.get("feature_attention_mask"),
                attention_mask=full_inputs.get("attention_mask"),
            )
            logits = out.logits  # (1, seq_len, vocab)

            # CE loss, per-token, with -100 masking
            loss_items = loss_fn(
                logits[:, :-1].permute(0, 2, 1).float(),
                labels[:, 1:].long()
            )  # (1, seq_len-1)

            # Apply parsed STAR weights to pseudo-label token positions
            target_start = prefix_len - 1
            n_available = loss_items.shape[1] - target_start
            n_tokens = min(len(star_weights), n_available)
            ratios = star_weights[:n_tokens] / star_weights[:n_tokens].mean().clamp(min=1e-8)
            loss_items[0, target_start: target_start + n_tokens] *= ratios.float()

            n_valid = (labels[0] != -100).sum().clamp(min=1)
            loss = loss_items.sum() / n_valid
            (loss / GRAD_ACC).backward()
            steps += 1

            if steps % GRAD_ACC == 0:
                torch.nn.utils.clip_grad_norm_(thinker.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if steps % SAVE_EVERY == 0:
                thinker.eval()
                dev_wer = evaluate(DEV_CSV) if DEV_CSV else float("inf")
                thinker.train()
                print(f"  Step {steps} | loss: {loss.item():.4f} | dev WER: {dev_wer:.4f}", flush=True)

                if dev_wer < best_wer or (dev_wer == best_wer and loss.item() < best_loss):
                    ckpt_path = os.path.join(exp_dir, "best_checkpoint")
                    asr_wrapper.model.thinker.save_pretrained(ckpt_path)
                    processor.save_pretrained(ckpt_path)
                    best_wer, best_loss = dev_wer, loss.item()
                    patience_counter = 0
                    print(f"  -> New best (dev WER: {best_wer:.4f})", flush=True)
                else:
                    patience_counter += 1
                    if patience_counter >= PATIENCE:
                        print("Early stopping.")
                        stop_training = True
                        break

    print("Training complete.")

    # ── Test set inference ────────────────────────────────────────────────────
    if TEST_CSV:
        print("Running test set inference...")
        test_df = pd.read_csv(TEST_CSV)
        audio_paths = test_df[DEV_AUDIO_COL].tolist()
        refs = [str(t).strip() for t in test_df[DEV_TEXT_COL].tolist()]
        preds, gt_raw, pred_norm, gt_norm, wers = [], [], [], [], []

        batch_size = 32
        for i in range(0, len(audio_paths), batch_size):
            results = asr_wrapper.transcribe(audio_paths[i: i + batch_size], language="English")
            for r, ref in zip(results, refs[i: i + batch_size]):
                hyp = r.text.strip() if hasattr(r, "text") else str(r)
                preds.append(hyp)
                gt_raw.append(ref)
                pn = normalizer(hyp) or "<UNK>"
                gn = normalizer(ref) or "<UNK>"
                pred_norm.append(pn)
                gt_norm.append(gn)
                wers.append(safe_wer(gn, pn))

        avg_wer = sum(wers) / len(wers)
        print(f"Test Average WER (per-sample mean, n={len(wers)}): {avg_wer:.4f}")
        pd.DataFrame({
            "audio": audio_paths, "ground_truth": gt_raw, "prediction": preds,
            "gt_norm": gt_norm, "pred_norm": pred_norm, "wer": wers,
        }).to_csv(os.path.join(exp_dir, "test_set_transcriptions.csv"), index=False)
        print(f"Test transcriptions saved to: {os.path.join(exp_dir, 'test_set_transcriptions.csv')}")

    # Save final model
    asr_wrapper.model.thinker.save_pretrained(exp_dir)
    processor.save_pretrained(exp_dir)
    print(f"Model saved to: {exp_dir}")


if __name__ == "__main__":
    import fire
    fire.Fire(train)