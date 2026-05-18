import pandas as pd
import os
import whisper
import argparse
import torch
import numpy as np
from tqdm import tqdm
import torch
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor, WhisperForConditionalGeneration

def main(args): 
# Determine the model ID
    # If it's a directory, use it directly. If it's a size (e.g., 'large-v3'), add prefix.
    model_id = args.whisper_size
    if not os.path.isdir(model_id) and not model_id.startswith("openai/whisper-"):
        model_id = f"openai/whisper-{model_id}"

    print(f"Loading model for inference from: {model_id}")

    # Load via Transformers Pipeline (Much safer than manual weight mapping)
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)
    # dont worry about getting the logprobs for now
    # if os.path.isdir(args.whisper_size):
    #     # 1. Load your fine-tuned model via Transformers
    #     print("Loading fine-tuned weights...")
    #     hf_model = WhisperForConditionalGeneration.from_pretrained("/export/fs06/kchapar1/bpd_asr/finetuned_models/whisper-large-v3-finetuned/")

    #     # 2. Load a base Whisper model of the same size
    #     # Since you used large-v3, we load large-v3
    #     print("Initializing base Whisper architecture...")
    #     model = whisper.load_model(args.whisper_size)

    #     # 3. Rename and transfer the weights
    #     # OpenAI and Hugging Face use different names (e.g., 'blocks' vs 'layers')
    #     hf_state_dict = hf_model.model.state_dict()
    #     whisper_state_dict = model.state_dict()

    #     # Common mapping for converting HF Whisper to OpenAI Whisper
    #     mapping = {
    #         "encoder.layers": "encoder.blocks",
    #         "decoder.layers": "decoder.blocks",
    #         "encoder.embed_positions.weight": "encoder.positional_embedding",
    #         "decoder.embed_positions.weight": "decoder.positional_embedding",
    #         "decoder.embed_tokens.weight": "decoder.token_embedding.weight",
    #         "self_attn.k_proj": "attn.key",
    #         "self_attn.q_proj": "attn.query",
    #         "self_attn.v_proj": "attn.value",
    #         "self_attn.out_proj": "attn.out",
    #         "encoder_attn.k_proj": "cross_attn.key",
    #         "encoder_attn.q_proj": "cross_attn.query",
    #         "encoder_attn.v_proj": "cross_attn.value",
    #         "encoder_attn.out_proj": "cross_attn.out",
    #         "self_attn_layer_norm": "attn_ln",
    #         "encoder_attn_layer_norm": "cross_attn_ln",
    #         "final_layer_norm": "mlp_ln",
    #         "fc1": "mlp.0",
    #         "fc2": "mlp.2",
    #     }

    #     new_state_dict = {}
    #     for hf_key, va in hf_state_dict.items():
    #         new_key = hf_key
    #         for k, v in mapping.items():
    #             new_key = new_key.replace(k, v)
            
    #         if new_key in whisper_state_dict:
    #             new_state_dict[new_key] = va

    #     # Load the transformed weights into the OpenAI model object
    #     model.load_state_dict(new_state_dict, strict=False)
    #     print("✅ Fine-tuned weights successfully loaded into OpenAI Whisper object.")   
    # else: 
    #     print(f"loading whisper {args.whisper_size} model")
    #     model = whisper.load_model(args.whisper_size)
    #     print(f"loaded whisper {args.whisper_size} model")

    # --- CONFIGURATION ---
    CSV_INPUT = args.path 
    OUTPUT_CSV = args.pl_save_path

    # 2. Prepare the Input List
    df = pd.read_csv(CSV_INPUT)
    # Adjust "Post_Filter_Audio_Path" if your CSV column name is different
    audio_paths = df["audio"].tolist()

    metadata_records = []

    print(f"Starting transcription of {len(audio_paths)} files...")

    # 3. Transcription & Metadata Extraction Loop
    for path in tqdm(audio_paths):
        if not os.path.exists(path):
            print(f"Skipping missing file: {path}")
            continue
        try:
            # Transcribe using base whisper library
            # verbose=False keeps the console clean; fp16=True for faster GPU inference
            result = model.transcribe(path, task="transcribe", language="en", fp16=True)
            
            full_text = result["text"].strip()
            print(f"pseudo transcript: {full_text}")
            # Calculate Average LogProb (Confidence)
            # We average the logprobs across all segments to get a file-level score
            avg_logprobs = [seg["avg_logprob"] for seg in result["segments"]]
            mean_logprob = np.mean(avg_logprobs) if avg_logprobs else -99.0
            
            # Duration is the 'end' timestamp of the very last segment
            total_duration = result["segments"][-1]["end"] if result["segments"] else 0.0
            
            metadata_records.append({
                "audio_path": path,
                "pseudolabel": full_text,
                "avg_logprob": round(float(mean_logprob), 4),
                "duration_seconds_(from_whisper_output)": round(float(total_duration), 2)
            })
            
        except Exception as e:
            print(f"Error processing {path}: {e}")

    # 4. Final Export
    audit_df = pd.DataFrame(metadata_records)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    
    audit_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n✅ Audit Complete!")
    print(f"Total files processed: {len(metadata_records)}")
    print(f"Metadata saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Pseudolabels")
    parser.add_argument("--path", type=str)
    parser.add_argument("--whisper_size", type=str, help="HF model id (ie. tiny or large-v3) OR local path to finetuned model")
    parser.add_argument("--pl_save_path", type=str, help="path to save the generated PLs to")
    args = parser.parse_args()

    main(args)
