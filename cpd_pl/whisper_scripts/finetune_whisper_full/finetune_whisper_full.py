# %%
import csv, json
import sys
import numpy as np 
import argparse
from dataclasses import dataclass
import transformers
from pathlib import Path
from pydub.utils import mediainfo
import transformers
transformers.logging.set_verbosity_debug()
transformers.logging.enable_default_handler()
transformers.logging.enable_explicit_format()
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainer, TrainingArguments, Seq2SeqTrainingArguments, AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import DatasetDict, load_from_disk, Dataset, Audio
import jiwer
import re
import os
from typing import Any, Dict, List, Union
from accelerate import Accelerator
import torch 
from transformers import EarlyStoppingCallback
from whisper.normalizers import EnglishTextNormalizer



# torch.autograd.set_detect_anomaly(True)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

os.environ["WANDB_DISABLED"] = "true"


# %%
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def main(args): 

    def prepare_dataset(batch): #extract input features and make sure that the data samples do not exceed max length 
            MAX_DURATION = 25.0  # seconds
            SAMPLE_RATE = 16000
            MAX_INPUT_LENGTH = int(SAMPLE_RATE * MAX_DURATION)
            audio_array = batch["audio"]["array"]
            sr = batch["audio"]["sampling_rate"]
            text = batch["text"]

            # Truncate audio
            if len(audio_array) > MAX_INPUT_LENGTH:
                audio_array = audio_array[:MAX_INPUT_LENGTH]

            # Use processor to extract both input_features and aligned labels
            processed = processor(
                audio_array,
                sampling_rate=sr,
                text=text,
                return_attention_mask=False,
                truncation=True,  # Ensures tokenized transcript doesn't exceed max length
            )

            # Extract audio input features
            input_features = processor.feature_extractor(
                audio_array, sampling_rate=sr
            ).input_features[0]

            #USE IF TRAINING WITH FP16
            #input_features = np.array(input_features, dtype=np.float16)
        
            input_features = np.array(input_features, dtype=np.float32) #do not manually force things into fp16, leads to numeric instability 
        
            # Tokenize the text (target labels)
            input_ids = processor.tokenizer(
                text, return_tensors="pt", truncation=True
            ).input_ids[0]
        
            return {
                "input_features": input_features,
                "labels": input_ids
            }

    def flatten_preds(pred_ids):
        # Recursively flatten if too nested
        flat_preds = []
        for ids in pred_ids:
            while isinstance(ids, (list, np.ndarray)) and len(ids) == 1:
                ids = ids[0]
            flat_preds.append(ids)
        return flat_preds

    normalizer = EnglishTextNormalizer()

    def compute_metrics(pred):
        pred_ids  = pred.predictions
        label_ids = pred.label_ids

        # If the predictions are nested, flatten them
        if isinstance(pred_ids[0], (list, np.ndarray)) and isinstance(pred_ids[0][0], (list, np.ndarray)):
            pred_ids = [ids[0] for ids in pred_ids]

        if isinstance(label_ids[0], (list, np.ndarray)) and isinstance(label_ids[0][0], (list, np.ndarray)):
            label_ids = [ids[0] for ids in label_ids]

        # replace the ignore-index (-100) with pad_token_id so decoding works
        pad_id = processor.tokenizer.pad_token_id
        if isinstance(label_ids, np.ndarray):
            label_ids = label_ids.copy()
            label_ids[label_ids == -100] = pad_id
        else:
            label_ids = [
                [t if t != -100 else pad_id for t in seq] for seq in label_ids
            ]

        # Decode
        pred_str  = processor.batch_decode(pred_ids,  skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        # Normalize using Whisper's EnglishTextNormalizer
        pred_str  = [normalizer(text) for text in pred_str]
        label_str = [normalizer(text) for text in label_str]

        # Filter out empty references
        new_preds, new_refs = [], []
        for ref, pred in zip(label_str, pred_str):
            if ref.strip():
                new_preds.append(pred)
                new_refs.append(ref)

        if not new_refs:
            return {"wer": 1.0}

        wer = jiwer.wer(new_refs, new_preds)
        return {"wer": wer}


    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available(): 
        print("GPU available")
    torch_dtype = torch.float32


    #model_id = "openai/whisper-" + args.whisper_model 
    
    model_input = args.model_id_or_path.strip() # Remove any accidental whitespace

    # If it's an absolute path, a relative path with slashes, or a verified directory
    if os.path.exists(model_input) or "/" in model_input:
        model_id = model_input
    else:
        # If it's just a keyword like 'tiny' or 'large-v3'
        model_id = f"openai/whisper-{model_input}"

    print(f"LOADING {model_id}")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True) #using eager instead of sdpa because 

    processor = WhisperProcessor.from_pretrained(model_id)
    expected_bins = processor.feature_extractor.feature_size

    model.generation_config.forced_decoder_ids = None
    model.generation_config.language = "en"
    model.generation_config.task = "transcribe"
    model.config.use_cache = False
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False
    model.gradient_checkpointing_enable()

    if args.freeze_decoder:
        for param in model.model.decoder.parameters():
            param.requires_grad = False
        print("Decoder frozen: only encoder weights will be updated during training.")

    print(f"loading dataset from disk")
    train_dataset = load_from_disk(args.training_data_path)

    eval_dict = load_from_disk(args.eval_dataset_path)
    validation_dataset = eval_dict["validation"]
    test_dataset = eval_dict["test"]
    print(f"eval/test datasets loaded from: {args.eval_dataset_path}")

    # Limit to first N samples per split when --max_samples is set (for debugging)
    if args.max_samples is not None:
        train_dataset = train_dataset.select(range(min(args.max_samples, len(train_dataset))))
        validation_dataset = validation_dataset.select(range(min(args.max_samples, len(validation_dataset))))
        test_dataset = test_dataset.select(range(min(args.max_samples, len(test_dataset))))
        print(f"max_samples={args.max_samples}: using {len(train_dataset)} train, {len(validation_dataset)} val, {len(test_dataset)} test")

    # #use tiny subsets for debugging:
    # tiny_train_dataset = train_dataset.select(range(15))
    # tiny_validation_dataset = validation_dataset.select(range(15))
    # tiny_test_dataset = test_dataset.select(range(15))
    # print(f"created tiny subsets")


    # train_dataset = train_dataset.map(prepare_dataset, remove_columns=train_dataset.column_names)
    # validation_dataset = validation_dataset.map(prepare_dataset, remove_columns=validation_dataset.column_names)
    # test_dataset = test_dataset.map(prepare_dataset, remove_columns=test_dataset.column_names)


    dataset = DatasetDict({
            "train": train_dataset,
            "validation": validation_dataset,
            "test": test_dataset
        })


    # dataset.save_to_disk("/export/fs06/kchapar1/bpd_asr/datasets/datasets_with_paths/datasets_with_paths_nonzero_durations/non_null_transcripts/mixed_24hrsilver_24hrgold_train_data/preprocessed-w-lgv3")
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, decoder_start_token_id=model.config.decoder_start_token_id)
    print("defined the data collator")
    save_path = args.save_path
    checkpoint_path = save_path + "/checkpoints/" 

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=10, # e.g., wait 3 epochs
        early_stopping_threshold=0.001
    )

    training_args = Seq2SeqTrainingArguments(
            generation_max_length=64,
            per_device_eval_batch_size=args.batch_size, 
            per_device_train_batch_size=args.batch_size,
            torch_compile=False,
            output_dir= checkpoint_path,
            num_train_epochs=args.num_epochs,
            eval_strategy="steps",
            eval_steps = args.eval_steps,
            save_strategy="steps",
            save_steps = args.eval_steps,
            gradient_accumulation_steps=4,
            gradient_checkpointing = True,
            logging_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            learning_rate=args.learning_rate,
            warmup_steps=500,
            optim="adamw_torch",
            bf16=True,
            max_grad_norm=1.0, #change from 0.0 to 1.0 to prevent exploding gradients 
            report_to="none",
            remove_unused_columns=False,
            predict_with_generate=True, 
        )

    # replace Seq2SeqTrainer(
    trainer = Seq2SeqTrainer( 
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            callbacks=[early_stopping_callback],
            data_collator=data_collator,
            tokenizer=processor,
            compute_metrics=compute_metrics
    )
    print("defined the trainer and training arguments")

    torch.cuda.empty_cache()

    try:
        trainer.train()
    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)  

    print("finished training, running test set prediction with best model")
    import pandas as pd

    # Run inference on the test set using trainer.predict(), which uses the best
    # checkpoint loaded at end of training (load_best_model_at_end=True)
    test_output = trainer.predict(test_dataset)
    pred_ids = test_output.predictions
    label_ids = test_output.label_ids

    # Flatten nested predictions if needed (can happen with generate-based decoding)
    if isinstance(pred_ids[0], (list, np.ndarray)) and isinstance(pred_ids[0][0], (list, np.ndarray)):
        pred_ids = [ids[0] for ids in pred_ids]

    # Replace -100 padding sentinels in labels with the tokenizer's pad token id
    # so that batch_decode doesn't choke on them
    pad_id = processor.tokenizer.pad_token_id
    if isinstance(label_ids, np.ndarray):
        label_ids = label_ids.copy()
        label_ids[label_ids == -100] = pad_id
    else:
        label_ids = [[t if t != -100 else pad_id for t in seq] for seq in label_ids]

    # Decode token ids back to text strings
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    # Normalize text using Whisper's EnglishTextNormalizer (lowercasing,
    # punctuation removal, number expansion, etc.) before computing WER
    norm_preds = [normalizer(p) for p in pred_str]
    norm_labels = [normalizer(l) for l in label_str]

    # Compute per-sample WER and average across all samples.
    # This is the average (macro) WER, i.e. sum(per_sample_wer) / n_samples,
    # as opposed to corpus-level WER which weights longer utterances more heavily.
    # Edge cases:
    #   - both empty -> WER 0 (correct silence)
    #   - ref empty, hyp non-empty -> WER 1 (false alarm)
    #   - ref non-empty, hyp empty -> WER 1 (complete miss)
    #   - both non-empty -> standard jiwer.wer()
    def safe_wer(ref, hyp):
        ref_empty = not ref.strip()
        hyp_empty = not hyp.strip()
        if ref_empty and hyp_empty:
            return 0.0
        if ref_empty or hyp_empty:
            return 1.0
        return jiwer.wer(ref, hyp)

    per_sample_wers = [safe_wer(ref, hyp) for ref, hyp in zip(norm_labels, norm_preds)]

    avg_wer = sum(per_sample_wers) / len(per_sample_wers)
    print(f"Test Average WER (per-sample mean, n={len(per_sample_wers)}): {avg_wer:.4f}")

    # Pull audio paths from the dataset if they were retained during preprocessing
    # (absolute_path column is preserved by prepare_pseudolabel_dataset_from_csv.py)
    audio_paths = test_dataset["absolute_path"] if "absolute_path" in test_dataset.column_names else [""] * len(pred_str)

    # Build a DataFrame with per-sample results and save to CSV
    df = pd.DataFrame({
        "audio": audio_paths,
        "ground_truth": label_str,
        "prediction": pred_str,
        "gt_norm": norm_labels,
        "pred_norm": norm_preds,
        "wer": per_sample_wers,
    })
    csv_path = os.path.join(save_path, "test_set_transcriptions.csv")
    df.to_csv(csv_path, index=False)
    print(f"Test transcriptions saved to: {csv_path}")


    try:
        trainer.save_model(save_path)
        print(f"saved model to {save_path}")
    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()

        
    save_processor_path = save_path + "/processor/"
    processor.save_pretrained(save_processor_path)
    print(f"saved processor to {save_processor_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Whisper model")

    parser.add_argument("model_id_or_path", type=str, help="HF model id (ie. tiny or large-v3) OR local path to finetuned model")
    parser.add_argument("num_epochs", type=int, help="number of training epochs")
    parser.add_argument("batch_size", type=int, help="training batch size")
    parser.add_argument("training_data_path", type=str, help="where to load the training data from")
    parser.add_argument("eval_dataset_path", type=str, help="path to DatasetDict containing 'validation' and 'test' splits")
    parser.add_argument("save_path", type=str, help="where to save the checkpoints, model, processor")
    parser.add_argument("--freeze_decoder", action="store_true", help="freeze decoder weights during finetuning")
    parser.add_argument("--eval_steps", type=int, default=500, help="evaluate and save every N steps (default: 500)")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="peak learning rate (default: 5e-5)")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit train/val/test to first N samples (for debugging)")

    args = parser.parse_args()
    print(f"Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    main(args)
