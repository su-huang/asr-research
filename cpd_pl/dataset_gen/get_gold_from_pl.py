import pandas as pd
import argparse
import os
import re

def create_lookup_dict(lookup_file_path):
    lookup = {}
    pattern = re.compile(r'^(utt\d+)\s+(.*)$')
    with open(lookup_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().replace('\r', '')
            if not line: continue
            match = pattern.match(line)
            if match:
                lookup[match.group(1)] = match.group(2).strip()
    print(f"Loaded {len(lookup)} gold standard utterances into memory.")
    return lookup

def create_path_to_utt_dict(map_utts_path):
    path_to_utt = {}
    print(f"Parsing path mapping file: {map_utts_path}")
    with open(map_utts_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().replace('\r', '')
            if not line: continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                utt_id, wav_path = parts[0].strip(), parts[1].strip()
                path_to_utt[wav_path] = utt_id
                
    print(f"Loaded {len(path_to_utt)} file-path to utt_id mappings into memory.")
    
    # --- DEBUG PRINT: Show a few sample keys loaded from scp ---
    print("\n--- DEBUG: First 3 keys loaded from .scp dictionary ---")
    sample_keys = list(path_to_utt.keys())[:3]
    for i, sk in enumerate(sample_keys):
        print(f"  SCP Key {i}: '{sk}' (Length: {len(sk)})")
    print("----------------------------------------------------\n")
    
    return path_to_utt

def remap_csv(input_csv, lookup_file, map_utts_file, output_csv):
    if not os.path.exists(lookup_file): raise FileNotFoundError(f"Missing {lookup_file}")
    if not os.path.exists(map_utts_file): raise FileNotFoundError(f"Missing {map_utts_file}")
        
    utt_lookup = create_lookup_dict(lookup_file)
    path_to_utt_lookup = create_path_to_utt_dict(map_utts_file)
    
    print(f"Reading source CSV: {input_csv}")
    df = pd.read_csv(input_csv)
    df.columns = [col.lower() for col in df.columns]
    
    # --- DEBUG PRINT: Show first 3 paths from the CSV ---
    print("\n--- DEBUG: First 3 paths extracted from your CSV ---")
    for idx, row in df.head(3).iterrows():
        csv_path = str(row['audio']).strip().replace('\r', '')
        print(f"  CSV Path {idx}: '{csv_path}' (Length: {len(csv_path)})")
        print(f"  Direct Match Check: {csv_path in path_to_utt_lookup}")
    print("----------------------------------------------------\n")
    
    updated_rows = []
    matched_count = 0
    missing_mappings = 0
    
    for idx, row in df.iterrows():
        audio_path = str(row['audio']).strip().replace('\r', '')
        if audio_path in path_to_utt_lookup:
            utt_id = path_to_utt_lookup[audio_path]
            if utt_id in utt_lookup:
                true_text = utt_lookup[utt_id]
                matched_count += 1
            else:
                true_text = str(row['text']).strip() if 'text' in df.columns else ""
                missing_mappings += 1
        else:
            true_text = str(row['text']).strip() if 'text' in df.columns else ""
            missing_mappings += 1
            
        updated_rows.append({"audio": audio_path, "text": true_text})
        
    final_df = pd.DataFrame(updated_rows)
    final_df = final_df.dropna(subset=['audio'])
    final_df = final_df[final_df['audio'].str.endswith('.wav')]
    
    print(f"--- Alignment Diagnostics ---")
    print(f"✅ Successfully matched via exact path map: {matched_count}")
    print(f"⚠️ Failed to match (retained original text): {missing_mappings}")
    print(f"-----------------------------")
    final_df.to_csv(output_csv, index=False, encoding='utf-8')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--lookup_file", type=str, default="/secure/fs00/afield6/police/chicago/data/data/train/text")
    parser.add_argument("--map_utts", type=str, default="/secure/fs00/afield6/police/shuan148/train_wav.scp")
    parser.add_argument("--output_csv", type=str, required=True)
    args = parser.parse_args()
    remap_csv(args.input_csv, args.lookup_file, args.map_utts, args.output_csv)