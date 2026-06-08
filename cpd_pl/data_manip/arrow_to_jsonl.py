import argparse
import os
from datasets import load_from_disk

def convert_arrow_to_jsonl(dataset_dir, output_jsonl):
    if not os.path.isdir(dataset_dir):
        print(f"Error: Dataset directory not found at {dataset_dir}")
        return

    print(f"Loading Hugging Face Arrow dataset from: {dataset_dir}")
    dataset = load_from_disk(dataset_dir)

    print(f"Writing JSONL format to: {output_jsonl}")
    os.makedirs(os.path.dirname(os.path.abspath(output_jsonl)), exist_ok=True)

    dataset.to_json(output_jsonl, force_ascii=False)

    print(f"Success! Packaged dataset into JSONL format at {output_jsonl}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a Hugging Face Arrow dataset directory to JSONL format"
    )
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    args = parser.parse_args()

    convert_arrow_to_jsonl(args.dataset_dir, args.output_jsonl)
