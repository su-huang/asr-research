import argparse
import torch
import soundfile as sf
import librosa
from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperProcessor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_audio(path, max_sec=30):
    audio_arr, sr = sf.read(path)
    if audio_arr.ndim > 1:
        audio_arr = audio_arr.mean(axis=1)
    if sr != 16000:
        audio_arr = librosa.resample(audio_arr, orig_sr=sr, target_sr=16000)
    if len(audio_arr) > max_sec * 16000:
        audio_arr = audio_arr[: max_sec * 16000]
    return audio_arr


def main(args):
    print("=" * 60)
    print("STEP 1: Checking checkpoint validity")
    print("=" * 60)
    sd = torch.load(args.checkpoint_path, map_location="cpu")
    print(f"Checkpoint type: {type(sd)}")
    print(f"Number of keys: {len(sd)}")
    if len(sd) == 0:
        print("WARNING: Checkpoint is empty!")
        return

    sample_keys = list(sd.keys())[:5]
    print(f"Sample keys: {sample_keys}")
    for k in sample_keys:
        v = sd[k]
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}, "
              f"mean={v.float().mean().item():.6f}, std={v.float().std().item():.6f}, "
              f"has_nan={torch.isnan(v.float()).any().item()}, "
              f"has_inf={torch.isinf(v.float()).any().item()}")

    print("\n" + "=" * 60)
    print("STEP 2: Loading model with checkpoint")
    print("=" * 60)
    for attn_impl in ["eager", "sdpa"]:
        print(f"\n--- Testing attn_implementation='{attn_impl}' ---")
        model = WhisperForConditionalGeneration.from_pretrained(
            args.base_model, attn_implementation=attn_impl
        ).to(device)
        model.load_state_dict(sd)
        model.eval()

        model_dtype = next(model.parameters()).dtype
        print(f"Model dtype: {model_dtype}")

        feature_extractor = WhisperFeatureExtractor.from_pretrained(args.base_model)
        processor = WhisperProcessor.from_pretrained(args.base_model)

        print("\n" + "=" * 60)
        print(f"STEP 3: Running inference on test audio ({attn_impl})")
        print("=" * 60)
        audio_arr = load_audio(args.test_audio)
        input_features = feature_extractor(
            [audio_arr], sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device=device, dtype=model_dtype)

        print(f"Input features shape: {input_features.shape}, dtype: {input_features.dtype}")
        print(f"Input features stats: mean={input_features.mean().item():.4f}, "
              f"std={input_features.std().item():.4f}, "
              f"has_nan={torch.isnan(input_features).any().item()}")

        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                language="en",
                task="transcribe",
                do_sample=False,
                max_new_tokens=150,
            )

        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        print(f"Transcription ({attn_impl}): '{transcription}'")
        print(f"Token ids: {predicted_ids[0].tolist()[:20]}...")

        del model
        torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("STEP 4: Comparing against base (untrained) model")
    print("=" * 60)
    base_model = WhisperForConditionalGeneration.from_pretrained(
        args.base_model, attn_implementation="sdpa"
    ).to(device)
    base_model.eval()
    model_dtype = next(base_model.parameters()).dtype

    input_features = feature_extractor(
        [audio_arr], sampling_rate=16000, return_tensors="pt"
    ).input_features.to(device=device, dtype=model_dtype)

    with torch.no_grad():
        predicted_ids = base_model.generate(
            input_features,
            language="en",
            task="transcribe",
            do_sample=False,
            max_new_tokens=150,
        )
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    print(f"Base model transcription: '{transcription}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug Whisper STAR checkpoint")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--base_model", type=str, default="openai/whisper-large-v3")
    parser.add_argument("--test_audio", type=str, required=True, help="Path to a single test wav file")
    args = parser.parse_args()
    main(args)