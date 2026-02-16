import numpy as np
import soundfile as sf
import librosa

def cpd_load_text(active_paths):
    all_data = []

    for path in active_paths:
        scp_file = path["scp"]
        text_file = path["text"]
        
        scp_dict = {}
        with open(scp_file, "r") as f_scp:
            for line in f_scp:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    utt_id, path = parts
                    scp_dict[utt_id] = path

        text_dict = {}
        with open(text_file, "r") as f_text:
            for line in f_text:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    utt_id, transcription = parts
                    text_dict[utt_id] = transcription
        
        for utt_id in scp_dict:
            if utt_id in text_dict:
                all_data.append({
                    "utt_id": utt_id,
                    "audio_path": scp_dict[utt_id],
                    "text": text_dict[utt_id],
                    "source": "CPD"
                })

    return all_data

def cpd_load_audio(item, target_sr=16000):
    # Read the audio
    audio, sr = sf.read(item["audio_path"])
    
    # Convert to mono if stereo
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    
    # Resample if needed
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    
    return audio