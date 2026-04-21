import json
import sys
import soundfile as sf

def calculate_duration(jsonl_path):
    total_seconds = 0.0
    count = 0
    
    try:
        with open(jsonl_path, 'r') as f:
            for line in f:
                if not line.strip(): continue
                entry = json.loads(line)
                file_path = entry.get('audio')
                
                try:
                    # Reading only the file info/metadata
                    info = sf.info(file_path)
                    total_seconds += info.duration
                    count += 1
                except Exception as e:
                    print(f"Error processing {file_path}: {e}", file=sys.stderr)
        
        # Output results
        print(f"Processed {count} files.")
        print(f"Total duration: {total_seconds:.2f} seconds")
        print(f"Total duration: {total_seconds/60:.2f} minutes")
        print(f"Total duration: {total_seconds/3600:.2f} hours")
        
    except FileNotFoundError:
        print(f"Error: The file {jsonl_path} was not found.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 get_duration.py <path_to_your_jsonl>")
    else:
        calculate_duration(sys.argv[1])
