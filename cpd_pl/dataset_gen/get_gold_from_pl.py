import pandas as pd
import argparse
import os
import re

def create_lookup_dict(lookup_file_path):
    """Parses the raw text file to build a rapid-mapping dictionary of {utt_id: text}."""
    lookup = {}
    print(f"Parsing lookup file: {lookup_file_path}")
    
    # Matches the 'utt' followed by digits at the start of a line, then captures the rest of the line
    pattern = re.compile(r'^(utt\d+)\s+(.*)$')
    
    with open(lookup_file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            match = pattern.match(line)
            if match:
                utt_id = match.group(1)
                true_text = match.group(2).strip()
                lookup[utt_id] = true_text
                
    print(f"Loaded {len(lookup)} gold standard utterances into memory.")
    return lookup

def remap_csv(input_csv, lookup_file, output_csv):
    # 1. Build our dictionary mapping
    if not os.path.exists(lookup_file):
        raise FileNotFoundError(f"Lookup file missing at {lookup_file}")
    utt_lookup = create_lookup_dict(lookup_file)
    
    # 2. Read the source evaluation CSV
    print(f"Reading source CSV: {input_csv}")
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Source CSV missing at {input_csv}")
    df = pd.read_csv(input_csv)
    
    # Standardize column naming structure
    df.columns = [col.lower() for col in df.columns]
    
    # 3. Extract utterance IDs from paths and map back to true text values
    updated_rows = []
    missing_mappings = 0
    
    print("Aligning audio file paths to gold-standard labels...")
    for idx, row in df.iterrows():
        audio_path = str(row['audio']).strip()
        
        # Regex to isolate the 'uttXXXXXX' sequence out of the absolute file path
        id_match = re.search(r'(utt\d+)', audio_path)
        
        if id_match:
            utt_id = id_match.group(1)
            
            # Check if we have the true label inside our text file dictionary
            if utt_id in utt_lookup:
                true_text = utt_lookup[utt_id]
            else:
                # If missing from this specific text file segment, fall back to current text 
                # or log it as unmapped for debugging.
                true_text = str(row['text']).strip() if 'text' in df.columns else ""
                missing_mappings += 1
        else:
            true_text = str(row['text']).strip() if 'text' in df.columns else ""
            missing_mappings += 1
            
        updated_rows.append({
            "audio": audio_path,
            "text": true_text
        })
        
    # 4. Export clean dataset
    final_df = pd.DataFrame(updated_rows)
    
    # Final cleanup: drop truncation failures or corrupt rows
    final_df = final_df.dropna(subset=['audio'])
    final_df = final_df[final_df['audio'].str.endswith('.wav')]
    
    if missing_mappings > 0:
        print(f"⚠️ Note: {missing_mappings} rows couldn't be matched in the lookup file and retained their original text.")
        
    print(f"Saving newly aligned data ({len(final_df)} rows) to: {output_csv}")
    final_df.to_csv(output_csv, index=False, encoding='utf-8')
    print("Success!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Map exact gold transcripts to an evaluation audio path file.")
    parser.add_argument("input_csv", type=str, help="Path to your current audio/text CSV")
    parser.add_argument("lookup_file", type=str, help="Path to the text file containing true utt values")
    parser.add_argument("output_csv", type=str, help="Path where the final corrected CSV will be saved")
    
    args = parser.parse_args()
    remap_csv(args.input_csv, args.lookup_file, args.output_csv)
