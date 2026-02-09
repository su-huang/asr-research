import os
os.environ["HF_USE_TORCH_CODEC"] = "0"  # Must come before datasets/audio imports

import torch
import pandas as pd
import argparse
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_from_disk, Audio  # Import after disabling TorchCodec
from whisper.normalizers import EnglishTextNormalizer
from jiwer import wer

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
def transcribe_preprocessed(example, do_sample, temp, top_p):
    # The data is already a Log-Mel Spectrogram (input_features)
    # We just need to convert it to a tensor and add a batch dimension
    input_features = torch.tensor(example["input_features"]).to(device)
    if input_features.ndim == 2:
        input_features = input_features.unsqueeze(0) # Add batch dim if missing

    forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
    
    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            do_sample=do_sample,
            temperature=temp if do_sample else None,
            top_p=top_p if do_sample else None,
            forced_decoder_ids=forced_decoder_ids,
            max_new_tokens=448 
        )
    
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

# def transcribe(example, do_sample, temp, top_p):
#     inputs = processor(
#         example["audio"]["array"],
#         sampling_rate=example["audio"]["sampling_rate"],
#         return_tensors="pt"
#     )
#     input_features = inputs.input_features.to(device)
#     forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
#     with torch.no_grad():
#         predicted_ids = model.generate(
#             input_features,
#             do_sample=do_sample,
#             temperature=temp if do_sample else None,
#             top_p=top_p if do_sample else None,
#             forced_decoder_ids=forced_decoder_ids
#         )
#     transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
#     return transcription

# -------------------- Normalization function --------------------
def normalize_example(predicted, text):
    normalizer = EnglishTextNormalizer()
    return normalizer(predicted), normalizer(text)

# -------------------- Main script --------------------
def main() -> None:
    args = parse_args()
    test_dataset = load_from_disk("/export/fs06/kchapar1/bpd_asr/datasets/datasets_with_paths/test_dataset/")

    # IMPORTANT: Force the dataset to show all columns or at least the ones we need
    test_dataset.set_format(type="torch", columns=["input_features", "labels"], output_all_columns=True)

    paths, groundtruth, whisper_transcript_list, wers = [], [], [], []

    for i in tqdm(range(len(test_dataset)), desc="Transcribing Test Set"):
        item = test_dataset[i]
        
        # 1. Get Path
        paths.append(item.get('absolute_path', 'unknown'))

        # 2. Transcribe using the precomputed features
        whisper_transcript = transcribe_preprocessed(item, args.do_sample, args.temp, args.top_p)

        # 3. Decode the 'labels' back into text for ground truth
        # Whisper labels often use -100 to ignore loss; we must replace them for the tokenizer
        label_ids = item["labels"]
        label_ids = [l if l != -100 else tokenizer.pad_token_id for l in label_ids]
        gt_text = tokenizer.decode(label_ids, skip_special_tokens=True)

        # 4. Normalize and Score
        whisper_norm, gt_norm = normalize_example(whisper_transcript, gt_text)

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
    print(f"Global Whisper WER: {global_wer:.4f}") 

# -------------------- Entry point --------------------
if __name__ == "__main__":
    main()
