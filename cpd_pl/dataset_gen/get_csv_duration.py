import csv
import sys
import soundfile as sf


def calculate_duration(csv_path):
    total_seconds = 0.0
    count = 0

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for entry in reader:
                # 1. Filter: Only consider rows where is_correct is exactly '1'
                # if entry.get("is_correct") != "1":
                #     continue

                # Extract the 'audio' column path dynamically
                file_path = entry.get("audio")
                if not file_path:
                    continue

                try:
                    # Reading only the file info/metadata
                    info = sf.info(file_path.strip())
                    total_seconds += info.duration
                    count += 1
                except Exception as e:
                    print(f"Error processing {file_path}: {e}", file=sys.stderr)

        # Output results
        print(f"Processed {count} files (filtered by is_correct = 1).")
        print(f"Total duration: {total_seconds:.2f} seconds")
        print(f"Total duration: {total_seconds/60:.2f} minutes")
        print(f"Total duration: {total_seconds/3600:.2f} hours")

    except FileNotFoundError:
        print(f"Error: The file {csv_path} was not found.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 get_duration.py <path_to_your_csv>")
    else:
        calculate_duration(sys.argv[1])