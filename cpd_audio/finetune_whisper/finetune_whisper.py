# %%
import csv, json
import sys
import numpy as np 
import argparse
from dataclasses import dataclass
import transformers
from pathlib import Path
from pydub.utils import mediainfo
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainer, TrainingArguments, Seq2SeqTrainingArguments, AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import DatasetDict, load_from_disk, Dataset, Audio
import jiwer
import re
import os
from typing import Any, Dict, List, Union
from accelerate import Accelerator
from datasets import set_caching_enabled
set_caching_enabled(False)


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
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
            MAX_LABEL_LENGTH = 448

            audio_array = batch["audio"]["array"]
            sr = batch["audio"]["sampling_rate"]
            text = batch["text"]

            # Truncate audio
            if len(audio_array) > MAX_INPUT_LENGTH:
                audio_array = audio_array[:MAX_INPUT_LENGTH]

            # Use processor to extract both input_features and aligned labels
        #  processed = processor(
        #      audio_array,
        #      sampling_rate=sr,
        #      text=text,
        #      return_attention_mask=False,
        #      truncation=True,  # Ensures tokenized transcript doesn't exceed max length
        # )

            # Extract audio input features
            input_features = processor.feature_extractor(
                audio_array, sampling_rate=sr
            ).input_features[0]

            #USE IF TRAINING WITH FP16
            #input_features = np.array(input_features, dtype=np.float16)
        
            input_features = np.array(input_features, dtype=np.float32) #do not manually force things into fp16, leads to numeric instability 
        
            # Tokenize the text (target labels)
            input_ids = processor.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=MAX_LABEL_LENGTH
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

    def compute_metrics(pred):
        pred_ids  = pred.predictions
        label_ids = pred.label_ids

        # If the predictions are nested, flatten them
        if isinstance(pred_ids[0], (list, np.ndarray)) and isinstance(pred_ids[0][0], (list, np.ndarray)):
            pred_ids = [ids[0] for ids in pred_ids]

        if isinstance(label_ids[0], (list, np.ndarray)) and isinstance(label_ids[0][0], (list, np.ndarray)):
            label_ids = [ids[0] for ids in label_ids]

        # ------------------------------------------------------------------
        # NEW: replace the ignore-index (-100) with pad_token_id so decoding works
        # ------------------------------------------------------------------
        pad_id = processor.tokenizer.pad_token_id
        # label_ids may be a list of lists, a NumPy array, or a torch tensor.
        if isinstance(label_ids, np.ndarray):
            label_ids = label_ids.copy()
            label_ids[label_ids == -100] = pad_id
        else:
            # generic python list / torch tensor handling
            label_ids = [
                [t if t != -100 else pad_id for t in seq] for seq in label_ids
            ]
        # ------------------------------------------------------------------

        # Decode
        pred_str  = processor.batch_decode(pred_ids,  skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        # Normalize
        pred_str  = [re.sub(r"[^\w\s]", "", text.lower()).strip() for text in pred_str]
        label_str = [re.sub(r"[^\w\s]", "", text.lower()).strip() for text in label_str]

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

    def old_compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # If the predictions are nested, flatten them
        if isinstance(pred_ids[0], (list, np.ndarray)) and isinstance(pred_ids[0][0], (list, np.ndarray)):
            pred_ids = [ids[0] for ids in pred_ids]

        if isinstance(label_ids[0], (list, np.ndarray)) and isinstance(label_ids[0][0], (list, np.ndarray)):
            label_ids = [ids[0] for ids in label_ids]

        # Decode
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        # Normalize
        pred_str = [re.sub(r"[^\w\s]", "", text.lower()).strip() for text in pred_str]
        label_str = [re.sub(r"[^\w\s]", "", text.lower()).strip() for text in label_str]

        # Filter out empty references
        new_preds, new_refs = [], []
        for ref, pred in zip(label_str, pred_str):
            if ref.strip() != "":
                new_preds.append(pred)
                new_refs.append(ref)

        if len(new_refs) == 0:
            # If all refs are empty, return dummy metric to avoid crashing
            return {"wer": 1.0}

        wer = jiwer.wer(new_refs, new_preds)

        return {"wer": wer}

    def load_scp_text(scp_file, text_file):
        # Load raw data from SCP file and test file 
        scp_dict = {}
        with open(scp_file, "r") as f_scp:
            for line in f_scp:
                utt_id, path = line.strip().split(maxsplit=1)
                scp_dict[utt_id] = path

        text_dict = {}
        with open(text_file, "r") as f_text:
            for line in f_text:
                utt_id, transcription = line.strip().split(maxsplit=1)
                text_dict[utt_id] = transcription
        
        data = []
        for utt_id in scp_dict:
            if utt_id in text_dict:
                data.append({
                    "utt_id": utt_id,
                    "audio": scp_dict[utt_id],
                    "text": text_dict[utt_id]
                })
        return data

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available(): 
        print("GPU available")
    torch_dtype = torch.float32

    # Loading model + data processor 
    model_id = "openai/whisper-" + args.whisper_model 

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True) #using eager instead of sdpa because 

    processor = WhisperProcessor.from_pretrained(model_id)

    forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")

    model.generation_config.forced_decoder_ids = forced_decoder_ids
    model.generation_config.language = "en"
    model.generation_config.task = "transcribe"

    # Defining and initial processing of datasets 
    # train_dataset = load_from_disk("/home/kchapar1/bpd_asr/datasets/datasets_with_paths/train_dataset")
    # validation_dataset = load_from_disk("/home/kchapar1/bpd_asr/datasets/datasets_with_paths/val_dataset")
    # test_dataset = load_from_disk("/home/kchapar1/bpd_asr/datasets/datasets_with_paths/test_dataset")

    # Create list of dicts from SCP and transcription files 
    raw_train_data = load_scp_text("/secure/fs00/afield6/police/shuan148/train_wav.scp", "/secure/fs00/afield6/police/chicago/data/data/train/text")
    raw_validation_data = load_scp_text("/secure/fs00/afield6/police/shuan148/dev_wav.scp", "/secure/fs00/afield6/police/chicago/data/data/dev/text")
    raw_test_data = load_scp_text("/secure/fs00/afield6/police/shuan148/test_wav.scp", "/secure/fs00/afield6/police/chicago/data/data/test/text")

    # Convert to Hugging Face Datasets format and set audio column
    train_dataset = Dataset.from_list(raw_train_data)
    validation_dataset = Dataset.from_list(raw_validation_data)
    test_dataset = Dataset.from_list(raw_test_data)
    
    # Convert string paths to actual audio objects
    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
    validation_dataset = validation_dataset.cast_column("audio", Audio(sampling_rate=16000))
    test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16000))

    train_dataset = train_dataset.map(prepare_dataset, remove_columns=train_dataset.column_names)
    validation_dataset = validation_dataset.map(prepare_dataset, remove_columns=validation_dataset.column_names)
    test_dataset = test_dataset.map(prepare_dataset, remove_columns=test_dataset.column_names) 

    dataset = DatasetDict({ 
            "train": train_dataset,
            "validation": validation_dataset,
            "test": test_dataset
        })

    
    #dataset.save_to_disk("/home/kchapar1/bpd_asr/complete_fnlo_ready_for_finetuning") -> can be used to cache data for later usage 
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, decoder_start_token_id=model.config.decoder_start_token_id)

    # Update to save to my folder 
    checkpoint_path = "/export/fs06/shuan148/asr-research/cpd_audio/finetune_whisper/finetuned_models/checkpoints/whisper-finetuned" + args.whisper_model

    # Defines arguments to pass to trainer 
    training_args = Seq2SeqTrainingArguments(
            generation_max_length=448,
            per_device_eval_batch_size=args.batch_size, 
            per_device_train_batch_size=args.batch_size,
            torch_compile=False,
            output_dir= checkpoint_path,
            num_train_epochs=args.num_epochs,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            gradient_accumulation_steps=1,
            logging_steps=1,
            fp16=False,
            gradient_checkpointing=False, 
            max_grad_norm=0.0,
            report_to="none",
            remove_unused_columns=False,
            predict_with_generate=True
        )

    # Defining object to do training 
    # replace Seq2SeqTrainer(
    trainer = Seq2SeqTrainer( 
            model=model,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['validation'],
            data_collator=data_collator,
            tokenizer=processor,
            compute_metrics=compute_metrics
    )

    # Training and evaluating 
    trainer.train()
    test_metrics = trainer.evaluate(eval_dataset=dataset['test'])
    wer_value = test_metrics.get("eval_wer", None)
    if wer_value not in (None, ""):
        print(f"Test WER: {wer_value:.3f}")
    else:
        print("Test WER: NA (no valid references)")

    # Update directories - Completed
    save_model_path = "/export/fs06/shuan148/asr-research/cpd_audio/finetune_whisper/finetuned_models/whisper-" + args.whisper_model + "-finetuned"
    trainer.save_model(save_model_path)

    save_processor_path = "/export/fs06/shuan148/asr-research/cpd_audio/finetune_whisper/finetuned_models/whisper-" + args.whisper_model + "-finetuned-processor"
    processor.save_pretrained(save_processor_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Whisper model")

    parser.add_argument("whisper_model", type=str, help="specify the model size ie tiny or large-v2")
    parser.add_argument("num_epochs", type=int, help="number of training epochs")
    parser.add_argument("batch_size", type=int, help="training batch size")

    args = parser.parse_args()
    print(f"Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    main(args)

