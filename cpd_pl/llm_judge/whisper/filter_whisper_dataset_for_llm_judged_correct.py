import argparse
import pandas as pd
from transformers import WhisperTokenizer
from datasets import load_from_disk

parser = argparse.ArgumentParser()
parser.add_argument("--judge_csv",     required=True)
parser.add_argument("--audio_col",     default="audio")
parser.add_argument("--pseudo_col",    default="pseudo_label")
parser.add_argument("--input_dataset", default="/export/fs06/shuan148/asr-research/cpd_pl/whisper_datasets/train/train_pl_24hr_preprocessed")
parser.add_argument("--output_dataset", required=True)
args = parser.parse_args()

# 1. Load the CSV containing the pseudolabels with llm judging
df_meta = pd.read_csv(args.judge_csv)
# Filter CSV to only 'correct' rows and create a path -> pseudo_label map
valid_data_map = df_meta[df_meta['is_correct'] == 1].set_index(args.audio_col)[args.pseudo_col].to_dict()
print(f"Loaded {len(valid_data_map)} is_correct=1 samples from {args.judge_csv}", flush=True)

# 2. Setup your tokenizer
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v3", language="english", task="transcribe")

# 3. Load Dataset
# Rather than reprocessing audio, filter the already-preprocessed fnlo dataset
# to the LLM-judged-correct samples and swap in the pseudolabel transcripts.
dataset = load_from_disk(args.input_dataset)
dataset = dataset['train']
print(f"Loaded {len(dataset)} samples from dataset", flush=True)

# 4. Filter first (to avoid tokenizing things we are going to throw away)
dataset = dataset.filter(lambda x: x['absolute_path'] in valid_data_map)
print(f"After filtering: {len(dataset)} samples", flush=True)

# 5. Define the update function
def update_transcripts(batch):
    new_text = valid_data_map[batch['absolute_path']]
    batch['text'] = new_text
    batch['labels'] = tokenizer(new_text).input_ids
    return batch

# 6. Apply the transformation
updated_dataset = dataset.map(update_transcripts, num_proc=4)

# 7. Save
updated_dataset.save_to_disk(args.output_dataset)
print(f"Saved {len(updated_dataset)} samples to {args.output_dataset}", flush=True)
