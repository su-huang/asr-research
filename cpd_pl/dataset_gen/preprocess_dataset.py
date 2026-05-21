import os
import numpy as np
import soundfile as sf
import librosa
from datasets import load_from_disk, DatasetDict, Dataset
from transformers import WhisperProcessor

DATASETS = ["/export/fs06/shuan148/asr-research/cpd_pl/whisper_datasets/hf_dict/hf_dict_val_gold_1.25hr_test_gold_2.25hr"]

# Instantiate once globally so workers don't reload it for every single audio file
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")

def prepare_dataset(batch):
    MAX_INPUT_LENGTH = 25 * 16000
    
    # Read audio array
    audio_path = batch.get("absolute_path") or batch.get("audio")
    audio_array, sr = sf.read(audio_path)
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)
        
    # Resample if necessary
    if sr != 16000:
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000, res_type="soxr_hq")
        
    # Truncate
    if len(audio_array) > MAX_INPUT_LENGTH:
        audio_array = audio_array[:MAX_INPUT_LENGTH]
        
    # Extract features and tokens
    input_features = processor.feature_extractor(audio_array, sampling_rate=16000).input_features[0]
    text = batch.get("text") or batch.get("transcript")
    input_ids = processor.tokenizer(text, truncation=True).input_ids
    
    return {"input_features": np.array(input_features, dtype=np.float32), "labels": input_ids}


# Loop through each path in your list
for path in DATASETS:
    print(f"\n{'='*50}\nLoading dataset from: {path}")
    loaded_data = load_from_disk(path)
    
    # Generate the output directory name by appending '_preprocessed'
    output_path = f"{path.rstrip('/')}_preprocessed"
    
    # Check if the loaded object is a DatasetDict (contains 'train', 'validation', 'test', etc.)
    if isinstance(loaded_data, DatasetDict):
        print(f"Detected DatasetDict with splits: {list(loaded_data.keys())}")
        processed_dict = DatasetDict()
        
        for split_name, dataset in loaded_data.items():
            print(f"Processing split: {split_name} ({len(dataset)} samples)...")
            # If you need to keep 'absolute_path' for evaluation, use the block below instead:
            cols_to_remove = [col for col in dataset.column_names if col != "absolute_path"]
            # cols_to_remove = dataset.column_names
            
            processed_dict[split_name] = dataset.map(
                prepare_dataset,
                remove_columns=cols_to_remove,
                num_proc=4
            )
        
        print(f"Saving processed DatasetDict to: {output_path}")
        processed_dict.save_to_disk(output_path)
        
    # Otherwise, treat it as a standard flat Dataset
    elif isinstance(loaded_data, Dataset):
        print(f"Detected flat Dataset ({len(loaded_data)} samples)...")
        
        processed_ds = loaded_data.map(
            prepare_dataset, 
            remove_columns=loaded_data.column_names, 
            num_proc=4 
        )
        
        print(f"Saving processed Dataset to: {output_path}")
        processed_ds.save_to_disk(output_path)
        
    else:
        print(f"Unknown data type loaded from {path}. Skipping.")

print("\nAll datasets processed successfully!")