"""
STAR finetuning for Qwen3-ASR-1.7B.

Attention score uses self-attention at layer LAYER_ID, head HEAD_ID,
measuring how much each generated token attends to the audio token positions.
This is the decoder-only analog of Whisper's cross-attention STAR score.

Precomputation (eager attention):
  1. Generate pseudo-label with asr_wrapper.transcribe()
  2. Run teacher-forcing forward pass to get logits + self-attention weights
  3. Compute per-token confidence (logit prob) and attention (to audio tokens)
  4. Combine into STAR scores

Training (SDPA attention, faster):
  - Custom loss loop with per-token STAR weights applied to pseudo-label positions
"""

import copy
import heapq
import json
import os

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


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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
    TEXT_COL="text",
    DEV_AUDIO_COL="audio",
    DEV_TEXT_COL="text",
    LAYER_ID=26,
    HEAD_ID=10,
    THRESOLD=2.0,
    TAU=10,
    TOP_PERCENT=0.8,
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
    audio_tok_id = thinker.config.audio_token_id

    model_size = MODEL.split("/")[-1]
    run_suffix = f"_{RUN_ID}" if RUN_ID else ""
    exp_dir = os.path.join(SAVE_DIR, DATASET)
    os.makedirs(exp_dir, exist_ok=True)

    # ── Attention hook ───────────────────────────────────────────────────────
    # Fires once per forward pass. During KV-cached generation each step has
    # q_len == 1 (one new token); the prefill step has q_len > 1. We only
    # collect generation steps so attn_steps[k] aligns with generated token k.
    attn_steps = []

    def attn_hook(module, inp, output):
        # output: (attn_output, attn_weights)
        # attn_weights: (batch, heads, q_len, k_len) — None with SDPA
        if output[1] is not None:
            attn_w = output[1][0, HEAD_ID]  # (q_len, k_len)
            if attn_w.shape[0] == 1:        # generation step, not prefill
                attn_steps.append(attn_w[0].detach().cpu())  # (k_len,)

    hook_handle = thinker.model.layers[LAYER_ID].self_attn.register_forward_hook(attn_hook)

    # ── Helpers ──────────────────────────────────────────────────────────────
    def compute_star_scores(norm_probs, norm_weights):
        stars, conflicts, no_conflicts = [], [], []
        for ci, ai in zip(norm_probs, norm_weights):
            c_over_a = ci * ci / ai if ai != 0 else float("inf")
            a_over_c = ai * ai / ci if ci != 0 else float("inf")
            conflict = (
                sigmoid((c_over_a - THRESOLD) * TAU) +
                sigmoid((a_over_c - THRESOLD) * TAU)
            ) * ai
            no_conflict = (
                sigmoid((THRESOLD - c_over_a) * TAU) *
                sigmoid((THRESOLD - a_over_c) * TAU)
            ) * ai * np.exp((ci - ai) / TAU)
            conflicts.append(round(float(conflict), 5))
            no_conflicts.append(round(float(no_conflict), 5))
            stars.append(round(float(conflict + no_conflict), 5))
        return stars, conflicts, no_conflicts

    # ── Data preparation (eager attention, autoregressive STAR scores) ───────
    def data_preparation(csv_path):
        df = pd.read_csv(csv_path)
        dataset = []
        all_pred, all_gt = [], []

        thinker.set_attn_implementation("eager")
        original_state_dict = copy.deepcopy(thinker.state_dict())

        for idx, row in enumerate(df.itertuples()):
            audio_path = str(getattr(row, AUDIO_COL))
            gt_text = str(getattr(row, TEXT_COL)).strip()

            try:
                audio_array = load_audio(audio_path)
            except Exception as e:
                print(f"  [{idx}] Error loading {audio_path}: {e}")
                continue

            prefix_text = build_prefix(processor, audio_array)

            # Process prompt (prefix only) to get prefix length + audio positions
            prefix_inputs = processor(
                text=[prefix_text], audio=[audio_array],
                return_tensors="pt", padding=True
            )
            prefix_inputs = cast_inputs(prefix_inputs, model_dtype, device)
            prefix_len = prefix_inputs["input_ids"].shape[1]

            # Identify audio token positions in the prefix
            prefix_ids = prefix_inputs["input_ids"][0]
            audio_pos = (prefix_ids == audio_tok_id).nonzero(as_tuple=True)[0].cpu()

            # Run autoregressive generation — hook fires once per generated token
            attn_steps.clear()
            with torch.no_grad():
                gen_out = thinker.generate(
                    **prefix_inputs,
                    max_new_tokens=150,
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            # gen_out.sequences: (1, prefix_len + n_generated)
            # gen_out.scores: tuple of n_generated tensors each (1, vocab_size)
            generated_ids = gen_out.sequences[0, prefix_len:].cpu()  # (n_tokens,)

            # Truncate at the first newline — keeps only the first transcription
            # (e.g. "language English<asr_text>One one one.") and discards
            # any hallucinated repetitions that follow.
            newline_ids = tokenizer.encode("\n", add_special_tokens=False)
            if newline_ids:
                nl_id = newline_ids[0]
                nl_pos = (generated_ids == nl_id).nonzero(as_tuple=True)[0]
                if len(nl_pos) > 0:
                    generated_ids = generated_ids[:nl_pos[0].item()]

            n_tokens = len(generated_ids)
            if n_tokens == 0:
                continue

            # pseudo_text_raw: full Qwen3-ASR format used as the training target
            # pseudo_text_clean: plain transcription text used for WER
            pseudo_text_raw = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            _, pseudo_text_clean = parse_asr_output(pseudo_text_raw)
            if not pseudo_text_clean:
                continue

            # Confidence: softmax prob of each chosen token from generation scores
            token_probs = []
            for k, score in enumerate(gen_out.scores[:n_tokens]):
                probs = torch.softmax(score[0].float() / 1.2, dim=-1)
                token_probs.append(float(probs[generated_ids[k]]))
            mean_p = sum(token_probs) / len(token_probs) if token_probs else 1.0
            norm_probs = [round(p / mean_p, 5) for p in token_probs]

            # Attentive score for token k (mirrors Whisper STAR formula):
            #   term1: sum of attention FROM token k TO content positions only
            #          — audio token positions + generated token positions (0..k)
            #          — excludes chat-template/system prefix tokens, analogous to
            #            Whisper's attn[:, :n_prompt_toks-1] = 0 zeroing
            #   term2: sum(attn_steps[j][prefix_len + k] for j > k)
            #          — how much subsequent tokens attend back to token k (backward)
            #   weight[k] = term1 + term2
            # No diagonal subtraction needed since term2 starts at j > k (no overlap).
            raw_attn = []
            for k in range(n_tokens):
                if k < len(attn_steps):
                    step_attn = attn_steps[k]  # (k_len,) where k_len = prefix_len + k + 1
                    # Content positions: audio tokens + all generated tokens so far (incl. k)
                    gen_pos = torch.arange(prefix_len, prefix_len + k + 1)
                    keep_pos = torch.cat([audio_pos, gen_pos])
                    keep_pos = keep_pos[keep_pos < step_attn.shape[0]]
                    term1 = float(step_attn[keep_pos].sum())
                else:
                    term1 = 0.0
                term2 = 0.0
                for j in range(k + 1, n_tokens):
                    if j < len(attn_steps):
                        pos = prefix_len + k
                        if pos < attn_steps[j].shape[0]:
                            term2 += float(attn_steps[j][pos])
                raw_attn.append(term1 + term2)
            mean_a = sum(raw_attn) / len(raw_attn) if raw_attn else 1.0
            norm_weights = [round(a / mean_a, 5) if mean_a != 0 else 1.0 for a in raw_attn]

            star_scores, conflict_scores, no_conflict_scores = compute_star_scores(
                norm_probs, norm_weights
            )

            # Utterance-level uncertainty: run 5 noisy forward passes and measure
            # how much the transcription changes (avg_wer and diversity).
            # Samples with low avg_wer * diversity are most reliable (kept for training).
            # avg_wer_val = 0.0
            # generated_texts = []
            # for _ in range(5):
            #     noisy_state = copy.deepcopy(original_state_dict)
            #     for key in noisy_state:
            #         std = torch.std(noisy_state[key].float())
            #         noise = torch.randn_like(noisy_state[key].float()) * std * 0.1
            #         noisy_state[key] = (noisy_state[key].float() + noise).to(noisy_state[key].dtype)
            #     thinker.load_state_dict(noisy_state)
            #     results = asr_wrapper.transcribe([audio_path], language="English")
            #     noisy_text = results[0].text.strip() if results else ""
            #     generated_texts.append(noisy_text)
            #     avg_wer_val += calculate_wer([pseudo_text_clean], [noisy_text]) / 5
            # diversity = len(set(generated_texts))

            item = {
                "audio_path": audio_path,
                "audio_array": audio_array,
                "text": gt_text,
                "pseudo_text": pseudo_text_raw,        # full Qwen3-ASR format for training
                "pseudo_text_clean": pseudo_text_clean, # plain text for WER / display
                "prefix_text": prefix_text,
                "prefix_len": prefix_len,
                "star_weights": torch.tensor(star_scores),
                "sample_star_score": float(np.mean(star_scores)),
                "norm_probs": norm_probs,
                "norm_weights": norm_weights,
                "star_scores": star_scores,
                "conflict_scores": conflict_scores,
                "no_conflict_scores": no_conflict_scores,
                # "avg_wer": avg_wer_val,
                # "diversity": diversity,
            }
            dataset.append(item)
            all_pred.append(normalizer(pseudo_text_clean) or "<UNK>")
            all_gt.append(normalizer(gt_text) or "<UNK>")

            if (idx + 1) % 100 == 0:
                print(f"  Prepared {idx + 1}/{len(df)} samples", flush=True)

        thinker.load_state_dict(original_state_dict)
        corpus_wer = calculate_wer(all_gt, all_pred) if all_gt else 0.0
        return dataset, corpus_wer

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

    # ── Precompute ───────────────────────────────────────────────────────────
    data_dir = os.path.join(SAVE_DIR, DATASET)
    os.makedirs(data_dir, exist_ok=True)
    saved_data_path = os.path.join(data_dir, f"train_{DATASET}_{RUN_ID}.pt" if RUN_ID else f"train_{DATASET}.pt")

    if os.path.exists(saved_data_path):
        print(f"Loading precomputed training data from {saved_data_path}...")
        train_dataset = torch.load(saved_data_path)
        all_pred = [normalizer(item["pseudo_text_clean"]) or "<UNK>" for item in train_dataset]
        all_gt   = [normalizer(item["text"]) or "<UNK>" for item in train_dataset]
        train_wer = calculate_wer(all_gt, all_pred) if all_gt else 0.0
        print(f"Loaded {len(train_dataset)} samples. Corpus WER: {train_wer:.4f}")
    else:
        print("Preparing training data (STAR inference)...")
        train_dataset, train_wer = data_preparation(TRAIN_CSV)
        print(f"Train pseudo-label corpus WER: {train_wer:.4f}  ({len(train_dataset)} samples)")
        torch.save(train_dataset, saved_data_path)
        print(f"Saved precomputed data to {saved_data_path}")

        # Save audit CSV
        audit_rows = [{
            "audio":               item["audio_path"],
            "ground_truth":        item["text"],
            "pseudo_label":        item["pseudo_text_clean"],
            "sample_star_score":   item["sample_star_score"],
            "star_scores":         json.dumps(item["star_scores"]),
            "confidence_scores":   json.dumps(item["norm_probs"]),
            "attention_scores":    json.dumps(item["norm_weights"]),
            "conflict_scores":     json.dumps(item["conflict_scores"]),
            "no_conflict_scores":  json.dumps(item["no_conflict_scores"]),
        } for item in train_dataset]
        audit_path = os.path.join(exp_dir, "training_data_audit.csv")
        pd.DataFrame(audit_rows).to_csv(audit_path, index=False)
        print(f"Audit CSV saved to: {audit_path}")

    # Sample-level filtering: keep TOP_PERCENT with lowest avg_wer * diversity
    # (low values = model consistently produces the pseudo-label under perturbation = high reliability)
    # n_keep = int(len(train_dataset) * TOP_PERCENT)
    # filtered = heapq.nsmallest(n_keep, train_dataset, key=lambda x: x["avg_wer"] * x["diversity"])
    # print(f"Keeping {n_keep}/{len(train_dataset)} samples (lowest avg_wer * diversity)")
    # Keep 100% of the successfully prepared data
    filtered = train_dataset
    print(f"Keeping all {len(filtered)} successfully prepared samples for training.")

    # ── Switch to SDPA for training ──────────────────────────────────────────
    hook_handle.remove()
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

        import random
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

            # Apply STAR weights to pseudo-label token positions.
            # Clamp to the actual generated length — re-encoding pseudo_text via
            # skip_special_tokens=True may produce fewer tokens than were generated.
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
