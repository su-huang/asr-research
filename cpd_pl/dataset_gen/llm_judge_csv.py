import pandas as pd
import os

CSV1_PATH = "/export/fs06/shuan148/asr-research/cpd_pl/whisper_csv/train/train_gold_24hr.csv"
CSV2_PATH = "/export/fs06/shuan148/asr-research/cpd_pl/whisper_csv/train/train_pl_24hr.csv"
OUTPUT_PATH = "/export/fs06/shuan148/asr-research/cpd_pl/llm_judge/train_24hr.csv"

SUFFIX1 = "_gold"
SUFFIX2 = "_pl"

def merge_csvs():
    print(f"Reading CSV 1: {CSV1_PATH}")
    if not os.path.exists(CSV1_PATH):
        raise FileNotFoundError(f"Could not find file at {CSV1_PATH}")
    df1 = pd.read_csv(CSV1_PATH)
    
    print(f"Reading CSV 2: {CSV2_PATH}")
    if not os.path.exists(CSV2_PATH):
        raise FileNotFoundError(f"Could not find file at {CSV2_PATH}")
    df2 = pd.read_csv(CSV2_PATH)
    
    df1.columns = [col.lower() for col in df1.columns]
    df2.columns = [col.lower() for col in df2.columns]
    
    if 'audio' not in df1.columns or 'text' not in df1.columns:
        raise ValueError("CSV 1 must contain both 'audio' and 'text' columns.")
    if 'audio' not in df2.columns or 'text' not in df2.columns:
        raise ValueError("CSV 2 must contain both 'audio' and 'text' columns.")
        
    df1 = df1[['audio', 'text']]
    df2 = df2[['audio', 'text']]
    
    print("Merging dataframes on the 'audio' column...")
    merged_df = pd.merge(df1, df2, on='audio', suffixes=(SUFFIX1, SUFFIX2))
    
    # Ensure the parent directory exists before saving
    output_dir = os.path.dirname(OUTPUT_PATH)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    print(f"Saving merged results ({len(merged_df)} rows) to: {OUTPUT_PATH}")
    merged_df.to_csv(OUTPUT_PATH, index=False)
    print("Merge complete!")

if __name__ == "__main__":
    merge_csvs()