import os
import librosa
from datasets import load_from_disk

os.environ["HF_USE_TORCH_CODEC"] = "0"

def bpd_load_text(active_paths):
    processed_items = []
    
    for path in active_paths:
        ds = load_from_disk(path)
        
        for item in ds:
            primary_text = item.get("text", "").strip()
            abs_path = item.get("absolute_path")

            if primary_text:
                processed_items.append({
                    "audio_path": abs_path,
                    "text": primary_text,
                    "source": "BPD"
                })

            for i in range(2, 5):
                extra_text = item.get(f"Text.{i}")
                if extra_text and extra_text.strip():
                    processed_items.append({
                        "audio_path": abs_path,
                        "text": extra_text.strip(),
                        "source": f"BPD_Speaker_{i}"
                    })
                    
    return processed_items

def bpd_load_audio(path):
    audio, _ = librosa.load(path, sr=16000)
    return audio