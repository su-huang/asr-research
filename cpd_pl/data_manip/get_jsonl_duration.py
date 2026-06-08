import json
import sys
import soundfile as sf

def calculate_duration(file_path):
    total_seconds = 0.0
    count = 0

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Skipping invalid JSON line: {e}", file=sys.stderr)
                    continue

                audio_path = entry.get("audio") or entry.get("audio_filepath")
                if not audio_path:
                    continue

                try:
                    info = sf.info(audio_path.strip())
                    total_seconds += info.duration
                    count += 1
                except Exception as e:
                    print(f"Error processing {audio_path}: {e}", file=sys.stderr)

        print(f"Processed {count} files.")
        print(f"Total duration: {total_seconds:.2f} seconds")
        print(f"Total duration: {total_seconds/60:.2f} minutes")
        print(f"Total duration: {total_seconds/3600:.2f} hours")

    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 get_duration.py <path_to_file.jsonl>")
    else:
        calculate_duration(sys.argv[1])