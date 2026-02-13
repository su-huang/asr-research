import os
os.environ["HF_USE_TORCH_CODEC"] = "0"  

import torch
import pandas as pd
import argparse
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_from_disk, Audio 
from whisper.normalizers import EnglishTextNormalizer
from jiwer import wer
import librosa 
from pathlib import Path
import soundfile as sf
import numpy as np

from cpd_loader import cpd_load_text, cpd_load_audio
from bpd_loader import bpd_load_test, bpd_load_audio

# test, train, val paths 
BPD_PATHS = [
    "/export/fs06/kchapar1/bpd_asr/datasets/datasets_with_paths/bpd_dataset_dict/test",
    "/export/fs06/kchapar1/bpd_asr/datasets/datasets_with_paths/bpd_dataset_dict/train",
    "/export/fs06/kchapar1/bpd_asr/datasets/datasets_with_paths/bpd_dataset_dict/val"
]

# test, train, val paths 
CPD_PATHS = [
    {
        "scp": "/secure/fs00/afield6/police/shuan148/test_wav.scp", 
        "text": "/secure/fs00/afield6/police/chicago/data/data/test/text"
    },
    {
        "scp": "/secure/fs00/afield6/police/shuan148/train_wav.scp", 
        "text": "/secure/fs00/afield6/police/chicago/data/data/train/text"
    },
    {
        "scp": "/secure/fs00/afield6/police/shuan148/dev_wav.scp", 
        "text": "/secure/fs00/afield6/police/chicago/data/data/dev/text"
    }
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Whisper on CPD and BPD data")

    # outfile 
    parser.add_argument("--outfile", required=True,
                        help="CSV output file for results")

    # finetuned model path 
    parser.add_argument("--finetuned_model_path", type=str, required=True,
                        help="Path to the finetuned Whisper model directory or Hugging Face ID")

    def str2bool(v):
        if isinstance(v, bool):
            return v
        return v.lower() in ("true", "1", "yes", "t")

    # cpd files: test, train, val 
    parser.add_argument("--cpd_file_status", 
                        type=str2bool, 
                        nargs=3, 
                        required=True,
                        help="Indicate files for CPD with booleans (test, train, val)")

    # bpd files: test, train, val 
    parser.add_argument("--bpd_file_status", 
                        type=str2bool, 
                        nargs=3, 
                        required=True,
                        help="Indicate files for BPD with booleans (test, train, val)")

    return parser.parse_args()

def get_active_paths(config_list, status_list):
    config_status_map = zip(config_list, status_list)    
    return [item for item, is_active in config_status_map if is_active]

def transcribe(example, processor, device, model):
    inputs = processor(
        example["audio"]["array"],
        sampling_rate=example["audio"]["sampling_rate"],
        return_tensors="pt"
    )
    input_features = inputs.input_features.to(device)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
    with torch.no_grad():
        predicted_ids = model.generate (
            input_features,
            do_sample = False,
            temperature=1.0, 
            top_p= 1.0, 
            forced_decoder_ids=forced_decoder_ids
        )
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

def compute_wer(ref, hyp):
        if not ref and not hyp:
            return 0.0
        elif ref and not hyp:
            return 1.0
        elif not ref and hyp:
            return 1.0
        else:
            return wer(ref, hyp)

def compute_global_wer(wers): 
    return sum(wers) / len(wers) if wers else 0.0

def main() -> None:
    args = parse_args()

    base_model_name = "openai/whisper-large-v3"
    processor = WhisperProcessor.from_pretrained(base_model_name)
    normalizer = EnglishTextNormalizer()
    base_model = WhisperForConditionalGeneration.from_pretrained(base_model_name).eval().cuda()
    ft_model = WhisperForConditionalGeneration.from_pretrained(args.finetuned_model_path).eval().cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    active_bpd_paths = get_active_paths(BPD_PATHS, args.bpd_file_status)
    bpd_data = bpd_load_test(active_bpd_paths)

    active_cpd_paths = get_active_paths(CPD_PATHS, args.cpd_file_status)
    cpd_data = cpd_load_text(active_cpd_paths)

    all_data = bpd_data + cpd_data 
    all_results = []

    for item in tqdm(all_data): 
        if item["source"] == "CPD":
            audio_array = cpd_load_audio(item)
        else: 
            audio_array = bpd_load_audio(item["audio_path"])

        audio_example = {"audio": {"array": audio_array, "sampling_rate": 16000}}

        base_transcript = transcribe(audio_example, processor, device, base_model)
        ft_transcript = transcribe(audio_example, processor, device, ft_model)

        base_norm = normalizer(base_transcript)
        ft_norm = normalizer(ft_transcript)
        gt_norm = normalizer(item['text']) 

        wer_base = compute_wer(gt_norm, base_norm)
        wer_ft = compute_wer(gt_norm, ft_norm)

        if item["source"] == "CPD":
            source_label = "CPD"
        else:
            source_label = "BPD"

        all_results.append({
            "source": source_label,
            "path": item["audio_path"],
            "groundtruth": gt_norm,
            "base_prediction": base_norm,
            "ft_prediction": ft_norm,
            "wer_base": wer_base,
            "wer_ft": wer_ft
        })
    
    df = pd.DataFrame(all_results)

    print("\nResults")

    cpd_wer = df[df["source"] == "CPD"][["wer_base", "wer_ft"]].mean()
    c_base, c_ft = cpd_wer['wer_base'], cpd_wer['wer_ft']
    print(f"CPD Base WER: {c_base:.4f}") 
    print(f"CPD Fine-Tuned WER: {c_ft:.4f}")
    print(f"CPD Relative Improvement: {((c_base - c_ft) / c_base if c_base > 0 else 0):.2%}")

    bpd_wer = df[df["source"] == "BPD"][["wer_base", "wer_ft"]].mean()
    b_base, b_ft = bpd_wer['wer_base'], bpd_wer['wer_ft']
    print(f"\nBPD Base WER: {b_base:.4f}")
    print(f"BPD Fine-Tuned WER: {b_ft:.4f}")
    print(f"BPD Relative Improvement: {((b_base - b_ft) / b_base if b_base > 0 else 0):.2%}")

    # Save the full detailed CSV
    df.to_csv(args.outfile, index=False)



