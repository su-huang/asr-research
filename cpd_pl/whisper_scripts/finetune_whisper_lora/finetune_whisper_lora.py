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
from peft import LoraConfig, get_peft_model

torch.autograd.set_detect_anomaly(True)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

os.environ["WANDB_DISABLED"] = "true"


def safe_wer(ref, hyp):
    """Per-sample WER with edge-case handling for empty strings.
    - both empty  -> 0.0 (correct silence)
    - one empty   -> 1.0 (false alarm or complete miss)
    - both non-empty -> standard jiwer.wer()
    """
    ref_empty = not ref.strip()
    hyp_empty = not hyp.strip()
    if ref_empty and hyp_empty:
        return 0.0
    if ref_empty or hyp_empty:
        return 1.0
    return jiwer.wer(ref, hyp)


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

    def prepare_dataset(batch):
        import soundfile as sf
        import librosa

        MAX_DURATION = 25.0
        SAMPLE_RATE = 16000
        MAX_INPUT_LENGTH = int(SAMPLE_RATE * MAX_DURATION)

        # Load audio from path
        audio_array, sr = sf.read(batch["audio"])
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)
        if sr != SAMPLE_RATE:
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=SAMPLE_RATE)

        # Truncate
        if len(audio_array) > MAX_INPUT_LENGTH:
            audio_array = audio_array[:MAX_INPUT_LENGTH]

        text = batch["text"]

        # Extract features
        input_features = processor.feature_extractor(
            audio_array, sampling_rate=SAMPLE_RATE
        ).input_features[0]

        input_features = np.array(input_features, dtype=np.float32)

        # Tokenize text
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

        # Flatten nested predictions if needed (can happen with generate-based decoding)
        if isinstance(pred_ids[0], (list, np.ndarray)) and isinstance(pred_ids[0][0], (list, np.ndarray)):
            pred_ids = [ids[0] for ids in pred_ids]

        if isinstance(label_ids[0], (list, np.ndarray)) and isinstance(label_ids[0][0], (list, np.ndarray)):
            label_ids = [ids[0] for ids in label_ids]

        # Replace -100 padding sentinels so batch_decode doesn't choke on them
        pad_id = processor.tokenizer.pad_token_id
        if isinstance(label_ids, np.ndarray):
            label_ids = label_ids.copy()
            label_ids[label_ids == -100] = pad_id
        else:
            label_ids = [
                [t if t != -100 else pad_id for t in seq] for seq in label_ids
            ]

        # Decode token ids back to text strings
        pred_str  = processor.batch_decode(pred_ids,  skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        # Normalize using Whisper's EnglishTextNormalizer
        pred_str  = [normalizer(text) for text in pred_str]
        label_str = [normalizer(text) for text in label_str]

        # Compute average (macro) WER: per-sample WER summed and divided by n_samples.
        # Uses safe_wer to handle empty ref/hyp edge cases without crashing.
        per_sample = [safe_wer(ref, hyp) for ref, hyp in zip(label_str, pred_str)]
        return {"wer": sum(per_sample) / len(per_sample)}


    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        print("GPU available")
    torch_dtype = torch.float32

    model_input = args.model_id_or_path.strip()

    if os.path.exists(model_input) or "/" in model_input:
        model_id = model_input
    else:
        model_id = f"openai/whisper-{model_input}"

    print(f"LOADING {model_id}")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)

    processor = WhisperProcessor.from_pretrained(model_id)

    model.generation_config.forced_decoder_ids = None
    model.generation_config.language = "en"
    model.generation_config.task = "transcribe"

    # Apply LoRA
    # Note: task_type is intentionally omitted. TaskType.SEQ_2_SEQ_LM wraps the
    # model in PeftModelForSeq2SeqLM (a generic T5/BART wrapper) which injects
    # input_ids=None and inputs_embeds=None into every forward call — kwargs
    # Whisper's forward doesn't accept. Without task_type, PEFT uses the base
    # PeftModel class, which passes args straight through to Whisper unchanged
    # and inherits generate() directly from WhisperForConditionalGeneration.
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Gradient checkpointing + LoRA on encoder-decoder models requires this hook.
    # Without it, the encoder output enters the decoder without requires_grad=True,
    # breaking the backward pass through the LoRA parameters.
    def make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)
    model.base_model.model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)

    print(f"loading dataset from disk")

    train_dataset = load_from_disk(args.training_data_path)
    train_dataset = train_dataset.map(prepare_dataset, remove_columns=train_dataset.column_names, num_proc=1)

    eval_dict = load_from_disk(args.eval_dataset_path)
    test_audio_paths = eval_dict["test"]["audio"]
    validation_dataset = eval_dict["validation"].map(prepare_dataset, remove_columns=eval_dict["validation"].column_names, num_proc=1)
    test_dataset = eval_dict["test"].map(prepare_dataset, remove_columns=eval_dict["test"].column_names, num_proc=1)
    
    print(f"eval/test datasets loaded from: {args.eval_dataset_path}")

    # Limit to first N samples per split when --max_samples is set (for debugging)
    if args.max_samples is not None:
        train_dataset = train_dataset.select(range(min(args.max_samples, len(train_dataset))))
        validation_dataset = validation_dataset.select(range(min(args.max_samples, len(validation_dataset))))
        test_dataset = test_dataset.select(range(min(args.max_samples, len(test_dataset))))
        print(f"max_samples={args.max_samples}: using {len(train_dataset)} train, {len(validation_dataset)} val, {len(test_dataset)} test")

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, decoder_start_token_id=model.config.decoder_start_token_id)
    print("defined the data collator")
    save_path = args.save_path
    checkpoint_path = save_path + "/checkpoints/"

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=10,
        early_stopping_threshold=0.001
    )

    training_args = Seq2SeqTrainingArguments(
            generation_max_length=64,
            per_device_eval_batch_size=args.batch_size,
            per_device_train_batch_size=args.batch_size,
            torch_compile=False,
            output_dir=checkpoint_path,
            num_train_epochs=args.num_epochs,
            max_steps=args.max_steps,
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            save_strategy="steps",
            save_steps=args.eval_steps,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            logging_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            learning_rate=args.learning_rate,
            warmup_steps=500,
            optim="adamw_torch_fused",
            bf16=True,       # bf16 instead of fp16: same memory savings but much more
            fp16=False,      # numerically stable — avoids NaN on large models like Whisper large-v3
            max_grad_norm=1.0,
            report_to="none",
            remove_unused_columns=False,
            predict_with_generate=True,
        )

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

    # Compute per-sample WER and average across all samples (macro WER).
    # safe_wer handles empty ref/hyp edge cases without crashing.
    per_sample_wers = [safe_wer(ref, hyp) for ref, hyp in zip(norm_labels, norm_preds)]
    avg_wer = sum(per_sample_wers) / len(per_sample_wers)
    print(f"Test Average WER (per-sample mean, n={len(per_sample_wers)}): {avg_wer:.4f}")

    # Pull audio paths from the dataset if they were retained during preprocessing
    # (absolute_path column is preserved by prepare_pseudolabel_dataset_from_csv.py)
    audio_paths = test_audio_paths

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

    # Merge LoRA weights into base model before saving so the model can be
    # loaded without peft for inference and pseudo-label generation
    print("Merging LoRA weights into base model...")
    merged_model = trainer.model.merge_and_unload()

    try:
        merged_model.save_pretrained(save_path)
        print(f"saved merged model to {save_path}")
    except Exception as e:
        print(f"An error occurred during saving: {e}")
        import traceback
        traceback.print_exc()

    save_processor_path = save_path + "/processor/"
    processor.save_pretrained(save_processor_path)
    print(f"saved processor to {save_processor_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Whisper model with LoRA")

    parser.add_argument("model_id_or_path", type=str, help="HF model id (ie. tiny or large-v3) OR local path to finetuned model")
    parser.add_argument("num_epochs", type=int, help="number of training epochs")
    parser.add_argument("batch_size", type=int, help="training batch size")
    parser.add_argument("training_data_path", type=str, help="where to load the training data from")
    parser.add_argument("eval_dataset_path", type=str, help="path to DatasetDict containing 'validation' and 'test' splits")
    parser.add_argument("save_path", type=str, help="where to save the checkpoints, model, processor")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank (r). lora_alpha is set to 2*r.")
    parser.add_argument("--max_steps", type=int, default=-1, help="If positive, overrides num_train_epochs and stops after this many steps.")
    parser.add_argument("--eval_steps", type=int, default=500, help="evaluate and save every N steps (default: 500)")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="peak learning rate (default: 5e-5)")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit train/val/test to first N samples (for debugging)")

    args = parser.parse_args()
    print(f"Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    main(args)
