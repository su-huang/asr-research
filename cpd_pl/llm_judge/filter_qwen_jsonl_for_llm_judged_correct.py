import csv
import json
import argparse

def process_csv_to_jsonl(input_csv, output_jsonl):
    count = 0
    with open(input_csv, mode='r', encoding='utf-8') as csv_file, \
         open(output_jsonl, mode='w', encoding='utf-8') as jsonl_file:
        
        # Use csv.DictReader to automatically map headers to row dictionaries
        reader = csv.DictReader(csv_file)
        
        for row in reader:
            # Check if 'is_correct' equals '1' (stripping any accidental whitespace)
            if row.get('is_correct', '').strip() == '1':
                # Grab the gold text target or fallback to pseudo-label text if empty
                asr_text = row.get('text_gold', '').strip()
                
                # Format the text according to your required prompt structure
                formatted_text = f"language English<asr_text>{asr_text}"
                
                # Construct the JSON payload structure
                json_data = {
                    "audio": row.get('audio', '').strip(),
                    "text": formatted_text
                }
                
                # Write directly to JSONL format
                jsonl_file.write(json.dumps(json_data) + '\n')
                count += 1
                
    print(f"Successfully processed {count} rows matching 'is_correct == 1'. Output saved to {output_jsonl}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter ASR metrics CSV and convert to specific JSONL schema.")
    parser.add_argument("--csv", required=True, help="Path to input metrics CSV file.")
    parser.add_argument("--output", default="output.jsonl", help="Path to write final JSONL dataset.")
    
    args = parser.parse_args()
    process_csv_to_jsonl(args.csv, args.output)