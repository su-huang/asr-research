from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor
from transformers import AutoFeatureExtractor, WhisperModel
from transformers import LlamaTokenizer
from datasets import load_dataset
import torch, torchaudio
from torch import nn
import numpy as np
from jiwer import wer as calculate_wer
import pickle
import fire
from datasets import Dataset, Audio, Value
import os, random, json
import pandas as pd
import soundfile as sf
import librosa
from tqdm import tqdm
from typing import Optional
from whisper.normalizers import EnglishTextNormalizer
import math
from sentencepiece import SentencePieceProcessor, SentencePieceTrainer
from pathlib import Path
import whisper
import copy, heapq
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

def train(
    MODEL = "openai/whisper-large-v3",
    DATASET = "chime4",
    TRAIN_DATA = "",
    DEV_DATA = "",
    SAVE_EVERY = 10,
    BATCH_SIZE = 32,
    GRADIENT_ACCUMULATION_STEPS = 4,
    LEARNING_RATE = 1e-3,
    EPOCHS = 100,
    THRESOLD=2.0,
    TOP_PERCENT=0.8,
    TAU=10,
    SAVE_DIR="runs",
    TEST_CSV="",
    RUN_ID="",
    ):
    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL)
    processor = WhisperProcessor.from_pretrained(MODEL, language="en", task="transcribe")
    tokenizer = WhisperTokenizer.from_pretrained(MODEL, language="en", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(MODEL).to(device)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
    state_dict = copy.deepcopy(model.state_dict())

    prompt_and_eos = tokenizer('')['input_ids']
    prompt_ids, eos_id = prompt_and_eos[:-1], prompt_and_eos[-1]
    n_prompt_toks = 4

    def data_preparation(data_path, feature_extractor, tokenizer):
        with open(data_path + "wav.scp", 'r') as f1:
            wave_data = f1.readlines()
        with open(data_path + "text", 'r') as f2:
            trans_data = f2.readlines()

        audio_data, txt_data = [], []
        for i in range(len(wave_data)):
            audio_data.append(wave_data[i])
            txt_data.append(trans_data[i])

        audio_dataset = []
        all_pred, all_gt = [], []
        for audio_line, text_line in zip(audio_data, txt_data):
            audio_path = audio_line.strip().split(None, 1)[1].strip()
            text = ' '.join(text_line.split()[1:]).lower().strip()
            audio, sr = torchaudio.load(audio_path)
            if sr != 16000:
                audio = torchaudio.functional.resample(audio, sr, 16000)
            item = {'audio': audio, 'audio_path': audio_path, 'text': text}

            item['mel'] = feature_extractor(audio.squeeze(0).numpy(), sampling_rate=16_000, return_tensors="pt")['input_features']
            item['decoder_input_ids'] = tokenizer(text, max_length=1024, truncation=True).input_ids

            model.load_state_dict(state_dict)
            hidden_feature = model.model.encoder(input_features=item['mel'].to(device=device, dtype=next(model.parameters()).dtype)).last_hidden_state

            # prompt: '<|startoftranscript|><|en|><|transcribe|><|notimestamps|>'
            pseudo_label_ids = torch.tensor([prompt_ids]).long().to(device)

            ### probs: confidence score
            probs, decoder_outputs = [], None
            for _ in range(150):
                decoder_outputs = model(encoder_outputs=(hidden_feature), decoder_input_ids=pseudo_label_ids, output_attentions=True)
                logits = torch.softmax(decoder_outputs.logits / 1.2, dim=-1)
                next_token = logits[0, -1, :].topk(1)[1]
                probs.append(float(logits[0, -1, next_token]))
                pseudo_label_ids = torch.cat((pseudo_label_ids, next_token.unsqueeze(0)), dim=-1)
                if next_token == eos_id:     # EOS
                    break
            
            # normlization
            mean_probs = sum(probs) / len(probs)
            for k in range(len(probs)):
                probs[k] = round(probs[k] / mean_probs, 3)

            ### weights: attentive score
            # Run one final forward pass with the complete sequence (including EOS)
            # so the attention matrix covers all n_generated tokens and aligns with probs.
            # Without this, decoder_outputs is from the step before EOS was appended,
            # causing a 1-position offset when zipping probs and weights.
            n_prompt_toks = 4
            layer_id, head_id = 30, 13  # suggest: layer_id \in [30,31], head_id \in [0,1,...,19]
            with torch.no_grad():
                decoder_outputs = model(
                    encoder_outputs=(hidden_feature),
                    decoder_input_ids=pseudo_label_ids,
                    output_attentions=True,
                )
            attn = decoder_outputs.decoder_attentions[layer_id][0, head_id, :, :]
            attn[:, :n_prompt_toks] = 0   # remove all prompt tokens
            weights = []
            for i in range(n_prompt_toks, attn.shape[-1]):
                weight = torch.sum(attn[i, :]) + torch.sum(attn[:, i]) - attn[i, i]
                weights.append(float(weight))
            
            # normalization
            mean_weights = sum(weights) / len(weights)
            for j in range(len(weights)):
                weights[j] = round(weights[j] / mean_weights, 3)

            ### final_weights: star score
            final_weights = []
            conflict_scores, no_conflict_scores, star_scores = [], [], []
            for ci, ai in zip(probs, weights):
                c_over_a = ci * ci / ai if ai != 0 else float('inf')
                a_over_c = ai * ai / ci if ci != 0 else float('inf')
                conflict = (sigmoid((c_over_a - THRESOLD) * TAU) + sigmoid((a_over_c - THRESOLD) * TAU)) * ai
                no_conflict = (sigmoid((THRESOLD - c_over_a) * TAU) * sigmoid((THRESOLD - a_over_c) * TAU)) * ai * np.exp((ci - ai) / TAU)
                final_weights.append(conflict + no_conflict)
                conflict_scores.append(round(float(conflict), 5))
                no_conflict_scores.append(round(float(no_conflict), 5))
                star_scores.append(round(float(conflict + no_conflict), 5))

            item['pseudo_label_ids'] = pseudo_label_ids
            item['probs'] = torch.tensor(final_weights).unsqueeze(0)
            item['norm_probs'] = probs
            item['norm_weights'] = weights
            item['star_scores'] = star_scores
            item['conflict_scores'] = conflict_scores
            item['no_conflict_scores'] = no_conflict_scores
            item['sample_star_score'] = float(np.mean(star_scores))
            pseudo_text = processor.batch_decode(pseudo_label_ids, skip_special_tokens=True)[0]
            item['pseudo_text'] = pseudo_text

            ### utt-level uncertainty
            if 'train' in data_path:
                avg_wer, generated_texts = 0, []
                for _ in range(5):
                    new_state_dict = copy.deepcopy(state_dict)
                    for k in new_state_dict.keys():
                        std = torch.std(new_state_dict[k])
                        noise = torch.randn_like(new_state_dict[k])
                        new_state_dict[k] = new_state_dict[k] + noise * std * 0.1

                    model.load_state_dict(new_state_dict)
                    generated_ids = model.generate(inputs=item['mel'].to(device), forced_decoder_ids=forced_decoder_ids, max_new_tokens=150)
                    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    generated_texts.append(generated_text)
                    avg_wer += calculate_wer([pseudo_text], [generated_text]) / 5

                item['avg_wer'] = avg_wer
                item['diversity'] = len(list(set(generated_texts)))

            ## text normalization
            pseudo_text = normalizer(pseudo_text)
            pseudo_text = pseudo_text if len(pseudo_text) > 0 else '<UNK>'

            gt = normalizer(text)
            gt = gt if len(gt) > 0 else '<UNK>'

            audio_dataset.append(item)
            all_pred.append(pseudo_text)
            all_gt.append(gt)

        model.load_state_dict(state_dict)
        return audio_dataset, calculate_wer(all_gt, all_pred)


    def evaluate(model, dataset):
        with torch.no_grad():
            all_pred, all_gt = [], []
            for item in dataset:
                mel = item['mel']
                generated_ids = model.generate(inputs=mel.to(device), forced_decoder_ids=forced_decoder_ids, max_new_tokens=150)
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                ## text normalization
                pred = normalizer(generated_text)
                pred = pred if len(pred) > 0 else '<UNK>'

                gt = normalizer(item['text'])
                gt = gt if len(gt) > 0 else '<UNK>'

                all_pred.append(pred)
                all_gt.append(gt)

        return calculate_wer(all_gt, all_pred)


    model_size = MODEL.replace('openai/whisper-', '')
    run_suffix = f'_{RUN_ID}' if RUN_ID else ''
    exp_dir = os.path.join(SAVE_DIR, f'{DATASET}_{model_size}{run_suffix}')
    os.makedirs(exp_dir, exist_ok=True)

    model.eval()
    train_dataset, train_wer = data_preparation(TRAIN_DATA, feature_extractor, tokenizer)
    dev_dataset, dev_wer = data_preparation(DEV_DATA, feature_extractor, tokenizer)
    os.system('mkdir -p data')
    torch.save(train_dataset, f'data/train_{DATASET}.pt')
    torch.save(dev_dataset, f'data/dev_{DATASET}.pt')

    # Save detailed audit CSV
    audit_rows = [{
        'audio': item['audio_path'],
        'ground_truth': item['text'],
        'pseudo_label': item['pseudo_text'],
        'sample_star_score': item['sample_star_score'],
        'avg_wer': item.get('avg_wer', ''),
        'diversity': item.get('diversity', ''),
        'star_scores': json.dumps(item['star_scores']),
        'confidence_scores': json.dumps(item['norm_probs']),
        'attention_scores': json.dumps(item['norm_weights']),
        'conflict_scores': json.dumps(item['conflict_scores']),
        'no_conflict_scores': json.dumps(item['no_conflict_scores']),
    } for item in train_dataset]
    audit_csv_path = os.path.join(exp_dir, 'whisper-STAR-training_data-w-STAR-scores.csv')
    pd.DataFrame(audit_rows).to_csv(audit_csv_path, index=False)
    print(f"Audit CSV saved to: {audit_csv_path}")

    model.train()

    ## load saved data
    # train_dataset = torch.load(f'data/train_{DATASET}.pt')
    # dev_dataset = torch.load(f'data/dev_{DATASET}.pt')

    ## utt-level filtering
    def product(item):
        return item['avg_wer'] * item['diversity']
    filtered_train_dataset = heapq.nsmallest(int(len(train_dataset) * TOP_PERCENT), train_dataset, key=product)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

    steps, loss = 0, 0
    best_loss, best_wer = 10000, 10000
    for Epoch in range(EPOCHS):
        print("Epoch: ", Epoch + 1)

        random.shuffle(filtered_train_dataset)
        print('Training...')
        optimizer.zero_grad()

        for item in filtered_train_dataset:
            mel    = item['mel'].to(device)
            labels = item['pseudo_label_ids'].to(device)
            ratios = item['probs'].to(device)

            y_in = labels[:, :-1]
            y_out = labels[:, 1:]

            logits = model(input_features=mel, decoder_input_ids=y_in).logits
            loss_items = loss_fn(logits.permute(0, 2, 1), y_out)

            # uncertainty calibration — use this item's own STAR weights
            ratios = ratios / torch.mean(ratios)
            loss = (torch.sum(loss_items[:, :n_prompt_toks-1]) + torch.sum(loss_items[:, n_prompt_toks-1:] * ratios)) / (n_prompt_toks-1 + ratios.shape[-1])

            (loss / GRADIENT_ACCUMULATION_STEPS).backward()
            steps += 1

            if steps % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if steps % SAVE_EVERY == 0:   # Evaluate
                torch.save(model, f"{exp_dir}/Iter_{steps}.pth")

                model.eval()
                dev_wer = evaluate(model, dev_dataset)
                model.train()

                if dev_wer < best_wer or (dev_wer == best_wer and loss < best_loss):
                    torch.save(model, f"{exp_dir}/best_checkpoint.pth")
                    best_loss, best_wer = loss, dev_wer

    torch.save(model, f"{exp_dir}/last_checkpoint.pth")

    # Load best checkpoint and save in HuggingFace format
    print("Loading best checkpoint for HuggingFace save...")
    best_model = torch.load(f"{exp_dir}/best_checkpoint.pth", map_location=device)
    best_model.save_pretrained(exp_dir)
    processor.save_pretrained(os.path.join(exp_dir, "processor"))
    print(f"Model saved to: {exp_dir}")
    print(f"Processor saved to: {os.path.join(exp_dir, 'processor')}")

    # Test set inference
    if TEST_CSV:
        print("Running test set inference...")
        test_df = pd.read_csv(TEST_CSV)
        audio_paths = test_df["audio"].tolist()
        refs = [str(t).strip() for t in test_df["text"].tolist()]

        best_model.eval()
        all_audio, all_gt_raw, all_pred_raw, all_gt_norm, all_pred_norm, all_wers = [], [], [], [], [], []

        BATCH_SIZE_INFER = 8
        for i in tqdm(range(0, len(audio_paths), BATCH_SIZE_INFER), desc="Test inference"):
            batch_paths = audio_paths[i: i + BATCH_SIZE_INFER]
            batch_refs  = refs[i: i + BATCH_SIZE_INFER]

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

            model_dtype = next(best_model.parameters()).dtype
            input_features = feature_extractor(
                audio_arrays, sampling_rate=16000, return_tensors="pt"
            ).input_features.to(device=device, dtype=model_dtype)

            with torch.no_grad():
                predicted_ids = best_model.generate(
                    input_features, language="en", task="transcribe", do_sample=False
                )

            predictions = processor.batch_decode(predicted_ids, skip_special_tokens=True)

            for path, ref, pred in zip(valid_paths, valid_refs, predictions):
                gt_norm_str   = normalizer(ref)
                pred_norm_str = normalizer(pred.strip())
                gt_norm_str   = gt_norm_str   if gt_norm_str   else "<UNK>"
                pred_norm_str = pred_norm_str if pred_norm_str else "<UNK>"
                w = safe_wer(gt_norm_str, pred_norm_str)
                all_audio.append(path)
                all_gt_raw.append(ref)
                all_pred_raw.append(pred.strip())
                all_gt_norm.append(gt_norm_str)
                all_pred_norm.append(pred_norm_str)
                all_wers.append(w)

        avg_wer = sum(all_wers) / len(all_wers) if all_wers else 0.0
        print(f"Test Average WER (per-sample mean, n={len(all_wers)}): {avg_wer:.4f}")
        csv_path = os.path.join(exp_dir, "test_set_transcriptions.csv")
        pd.DataFrame({
            "audio": all_audio, "ground_truth": all_gt_raw, "prediction": all_pred_raw,
            "gt_norm": all_gt_norm, "pred_norm": all_pred_norm, "wer": all_wers,
        }).to_csv(csv_path, index=False)
        print(f"Test transcriptions saved to: {csv_path}")


if __name__ == "__main__":
    fire.Fire(train)

