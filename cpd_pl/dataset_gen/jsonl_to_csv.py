import json
import pandas as pd
import re
import os
import argparse

def clean_qwen_text(text):
    if not isinstance(text, str):
        return text
    # Strips out "language <AnyLanguage><asr_text>" or "language <AnyLanguage> <asr_text>"
    cleaned = re.sub(r"language\s+[A-Za-z]+\s*<asr_text>", "", text)
    return cleaned.strip()

def convert(INPUT_JSONL, OUTPUT_CSV):
    print(f"Reading JSONL file: {INPUT_JSONL}")
    if not os.path.exists(INPUT_JSONL):
        raise FileNotFoundError(f"Could not find {INPUT_JSONL}")
        
    records = []
    with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    if ",language" in line:
                        parts = line.split(",language", 1)
                        records.append({
                            "audio": parts[0].replace("audio,text", "").strip(),
                            "text": "language" + parts[1]
                        })

    df = pd.DataFrame(records)
    df.columns = [col.lower() for col in df.columns]
    
    if 'text' in df.columns:
        print("Stripping Qwen ASR special tokens from text column...")
        df['text'] = df['text'].apply(clean_qwen_text)
        
    df = df.dropna(subset=['audio', 'text'])
    df = df[df['text'] != ""]

    print(f"Saving {len(df)} rows to clean CSV: {OUTPUT_CSV}")
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    print("Success!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a JSONL (JSON Lines) file into a CSV file.")
    parser.add_argument("input_jsonl", type=str, help="Path to the input .jsonl file")
    parser.add_argument("output_csv", type=str, help="Path to save the output .csv file")
    
    args = parser.parse_args()
    convert(args.input_jsonl, args.output_csv)