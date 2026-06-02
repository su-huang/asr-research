import csv
import json
import argparse


def process_csv_to_jsonl(input_csv, output_jsonl):
    count = 0
    with open(input_csv, mode='r', encoding='utf-8') as csv_file, \
         open(output_jsonl, mode='w', encoding='utf-8') as jsonl_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            asr_text = row.get('text', '').strip()
            json_data = {
                "audio": row.get('audio', '').strip(),
                "text": f"language English<asr_text>{asr_text}"
            }
            jsonl_file.write(json.dumps(json_data) + '\n')
            count += 1
    print(f"Successfully processed {count} rows. Output saved to {output_jsonl}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert filtered PL CSV to JSONL.")
    parser.add_argument("--csv",    required=True, help="Path to input CSV file.")
    parser.add_argument("--output", required=True, help="Path to write output JSONL file.")
    args = parser.parse_args()
    process_csv_to_jsonl(args.csv, args.output)