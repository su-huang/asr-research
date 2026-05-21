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

def create_path_to_utt_dict(map_utts_path):
    """Parses the map_utts file to build a dictionary of {absolute_wav_path: utt_id}."""
    path_to_utt = {}
    print(f"Parsing path mapping file: {map_utts_path}")
    
    with open(map_utts_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Splits by whitespace: [utt_id] [absolute_wav_path]
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                utt_id, wav_path = parts[0].strip(), parts[1].strip()
                path_to_utt[wav_path] = utt_id
                
    print(f"Loaded {len(path_to_utt)} file-path to utt_id mappings into memory.")
    return path_to_utt

def remap_csv(input_csv, lookup_file, map_utts_file, output_csv):
    # 1. Build our dictionary mappings
    if not os.path.exists(lookup_file):
        raise FileNotFoundError(f"Lookup file missing at {lookup_file}")
    if not os.path.exists(map_utts_file):
        raise FileNotFoundError(f"Path-to-utt mapping file missing at {map_utts_file}")
        
    utt_lookup = create_lookup_dict(lookup_file)
    path_to_utt_lookup = create_path_to_utt_dict(map_utts_file)
    
    # 2. Read the source evaluation CSV
    print(f"Reading source CSV: {input_csv}")
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Source CSV missing at {input_csv}")
    df = pd.read_csv(input_csv)
    
    # Standardize column naming structure
    df.columns = [col.lower() for col in df.columns]
    
    # 3. Use the file map to find utt_ids and match back to true text values
    updated_rows = []
    missing_mappings = 0
    
    print("Aligning audio file paths to gold-standard labels using map_utts...")
    for idx, row in df.iterrows():
        audio_path = str(row['audio']).strip()
        
        # Check if the exact audio path exists in our path-to-utt mapping file
        if audio_path in path_to_utt_lookup:
            utt_id = path_to_utt_lookup[audio_path]
            
            # Check if we have the true label inside our text file dictionary
            if utt_id in utt_lookup:
                true_text = utt_lookup[utt_id]
            else:
                true_text = str(row['text']).strip() if 'text' in df.columns else ""
                missing_mappings += 1
        else:
            # Fall back to current text if path isn't found inside the map_utts file
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
        print(f"⚠️ Note: {missing_mappings} rows couldn't be matched via the map files and retained their original text.")
        
    print(f"Saving newly aligned data ({len(final_df)} rows) to: {output_csv}")
    final_df.to_csv(output_csv, index=False, encoding='utf-8')
    print("Success!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Map exact gold transcripts to an evaluation audio path file via a map utility file.")
    parser.add_argument("input_csv", type=str, help="Path to your current audio/text CSV")
    parser.add_argument("lookup_file", type=str, nargs='?', default="/secure/fs00/afield6/police/chicago/data/data/train/text", help="Path to the text file containing true utt values")
    parser.add_argument("map_utts", type=str, nargs='?', default="/secure/fs00/afield6/police/shuan148/train_wav.scp", help="Path to the file mapping utt_ids to absolute paths")
    parser.add_argument("output_csv", type=str, help="Path where the final corrected CSV will be saved")
    
    args = parser.parse_args()
    remap_csv(args.input_csv, args.lookup_file, args.map_utts, args.output_csv)
