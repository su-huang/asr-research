"""
LLM-based pseudolabel quality judge using Llama-3.3-70B-Instruct.

For each sample in the input audit CSV, makes two LLM calls:
  1. Uncertainty score (0-1 float): how uncertain is the LLM that the pseudolabel is correct?
  2. Binary correctness (0 or 1): does the LLM think the transcript is correct?

Also computes WER between pseudo_label and ground_truth.

Output CSV columns: audio, pseudo_label, ground_truth, wer, uncertainty_score, is_correct
"""

import os
import re
import argparse
import pandas as pd
import torch
import transformers
from jiwer import wer as calculate_wer
from whisper.normalizers import EnglishTextNormalizer

normalizer = EnglishTextNormalizer()

# ── AUDIO CONTEXT (Baltimore Domain) ──────────────────────────────────────────
AUDIO_CONTEXT = """
police radio communication from the city of Baltimore, Maryland. The audio contains communication of dispatch assignments in response to calls for service from the 911/311 system in Baltimore, but they also capture police activity that emerges in response to events on the ground that require some degree of notification of officers working in the same police district and/or coordination among officers assigned to the same sector or post. The audio is recorded over radio channels and may contain background noise, police-specific vocabulary such as 10- codes, Baltimore street names and locations, call signs and dispatcher speech. 
"""
# ──────────────────────────────────────────────────────────────────────────────

UNCERTAINTY_SYSTEM = f"""You are an expert evaluator of automatic speech recognition (ASR) transcripts.
The audio being transcribed is {AUDIO_CONTEXT.strip()}

You will be given an ASR transcript, and your task is to assess how certain or uncertain you are that the transcript is correct. Consider whether the transcript looks like plausible speech from this domain — coherent words,
expected terminology, etc.

Output ONLY a single decimal number between 0.0 and 1.0:
  1.0 = very certain the transcript is correct
  0.0 = very uncertain / likely incorrect or garbled
No explanation, no other text — just the number."""

CORRECTNESS_SYSTEM = f"""You are an exceptionally strict, zero-tolerance quality control evaluator for police radio Automatic Speech Recognition (ASR) transcripts. Your goal is to isolate only the most flawless, high-fidelity transcripts for model training.

The audio context is: {AUDIO_CONTEXT.strip()}

CRITICAL INSTRUCTION ON FORMATTING: Do NOT reject a transcript for minor numerical formatting variations (e.g., "10-4" vs "ten four", or "1014" vs "ten fourteen"). A downstream normalizer handles these, so they count as CORRECT.

Instead, you must ruthlessly judge the SEMANTIC and OPERATIONAL validity. You will also be given the duration of the audio file associated with the transcript. Reject the transcript immediately and output 0 if it contains ANY of the following flaws:

1. CONTEXTUAL NONSENSE: The words technically form a grammatical English phrase but make absolutely zero sense in a law enforcement or dispatch context (e.g., phrases like "life coach records", "hard escape", "harper rope", "Adam Tutu", "riding an increase", or "put that through my blog"). 
2. MISMATCH BETWEEN AUDIO DURATION AND TRANSCRIPT: The audio duration is very short (1-2 seconds) and yet the transcript is several words long, or vice versa.  
3. CASUAL OR COLLOQUIAL HALLUCINATIONS: The transcript includes non-professional, casual filler words that are rare in formal dispatch (e.g., "Y'all", "man", "with all God's permission").
4. PHONETIC CODE MANGLES: Standard 10-codes or alphanumeric unit callsigns are corrupted into phonetically similar but wrong words (e.g. "Charlie for once" instead of "Charlie 41").

ONLY output a 1 if the transcript is a high-fidelity, pristine, and clean representation of standard professional police radio chatter. If you have even any doubt whatsoever that a phrase sounds corrupted or contextually bizarre, mark it 0.

Output ONLY the digit 1 or 0:
 1 = The transcript is operationally flawless and contextually correct.
 0 = The transcript has a contextual error, truncation, hallucination, or flaw.

No explanation, no other text — just 1 or 0."""


def build_pipeline(model_name):
    print(f"Loading model: {model_name}", flush=True)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    pipe = transformers.pipeline(
        "text-generation",
        model=model_name,
        model_kwargs={"dtype": torch.bfloat16},
        device_map="auto",
        token=token,
    )
    print("Model loaded.", flush=True)
    return pipe


def call_llm(pipe, system_prompt, user_text, max_new_tokens=16):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_text},
    ]
    output = pipe(messages, max_new_tokens=max_new_tokens, do_sample=False)
    return output[0]["generated_text"][-1]["content"].strip()


def parse_float(text, default=0.5):
    match = re.search(r"\d+(?:\.\d+)?", text)
    if match:
        val = float(match.group())
        return max(0.0, min(1.0, val))
    return default


def parse_binary(text, default=0):
    match = re.search(r"[01]", text)
    return int(match.group()) if match else default


def safe_wer(ref, hyp):
    ref_n = normalizer(ref) or "<UNK>"
    hyp_n = normalizer(hyp) or "<UNK>"
    if not ref_n.strip() and not hyp_n.strip():
        return 0.0
    if not ref_n.strip() or not hyp_n.strip():
        return 1.0
    return calculate_wer(ref_n, hyp_n)


def main(
    INPUT_CSV,
    OUTPUT_CSV,
    MODEL,
    DURATION_CSV,
    SAVE_EVERY=50,
    AUDIO_COL="audio",
    PSEUDO_COL="text_pl",
    GT_COL="text_gold",
    MAX_ROWS=None,
):
    df = pd.read_csv(INPUT_CSV)
    if MAX_ROWS is not None:
        df = df.head(int(MAX_ROWS))
    print(f"Loaded {len(df)} rows from {INPUT_CSV}", flush=True)

    dur_lookup = pd.read_csv(DURATION_CSV).set_index('audio')['duration_s'].to_dict()

    # Resume support evaluation mapping setup
    if os.path.exists(OUTPUT_CSV):
        try:
            done = pd.read_csv(OUTPUT_CSV)
            done_paths = set(done[AUDIO_COL].astype(str))
            print(f"Resuming — {len(done)} rows already done, skipping.", flush=True)
        except Exception as e:
            print(f"Could not parse existing output CSV, starting fresh. Error: {e}", flush=True)
            done = pd.DataFrame()
            done_paths = set()
    else:
        done = pd.DataFrame()
        done_paths = set()

    pipe = build_pipeline(MODEL)

    results = []
    for i, row in df.iterrows():
        audio_path   = str(row[AUDIO_COL])
        pseudo_label = str(row[PSEUDO_COL]).strip() if pd.notnull(row[PSEUDO_COL]) else ""
        ground_truth = str(row[GT_COL]).strip() if pd.notnull(row[GT_COL]) else ""

        if audio_path in done_paths:
            continue

        duration = dur_lookup.get(audio_path)
        duration_str = f"{duration:.1f} seconds" if duration is not None else "unknown"
        user_text = f"Audio duration: {duration_str}\nTranscript: {pseudo_label}"

        # Call 1: Uncertainty Score
        raw_uncertainty = call_llm(pipe, UNCERTAINTY_SYSTEM, user_text)
        uncertainty     = parse_float(raw_uncertainty)

        # Call 2: Strict Binary Correctness 
        raw_correct = call_llm(pipe, CORRECTNESS_SYSTEM, user_text)
        is_correct  = parse_binary(raw_correct)

        wer = safe_wer(ground_truth, pseudo_label)

        results.append({
            AUDIO_COL:           audio_path,
            PSEUDO_COL:          pseudo_label,
            GT_COL:              ground_truth,
            "wer":               round(wer, 4),
            "uncertainty_score":  uncertainty,
            "is_correct":        is_correct,
        })

        if (len(results) % 10) == 0:
            print(f"  [{i+1}/{len(df)}] wer={wer:.3f} uncertainty={uncertainty:.2f} correct={is_correct}", flush=True)

        if len(results) % SAVE_EVERY == 0:
            done = _save(done, results, OUTPUT_CSV)
            results = []  # Clear memory cache batch

    if results:
        _save(done, results, OUTPUT_CSV)
    print(f"Done. Results saved to {OUTPUT_CSV}", flush=True)


def _save(done, new_rows, path):
    batch = pd.DataFrame(new_rows)
    combined = pd.concat([done, batch], ignore_index=True) if not done.empty else batch
    combined.to_csv(path, index=False)
    print(f"  Saved {len(combined)} rows to {path}", flush=True)
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv",    required=True)
    parser.add_argument("--output_csv",   required=True)
    parser.add_argument("--duration_csv", required=True, help="Path to duration lookup CSV file")
    parser.add_argument("--model",        default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--audio_col",    default="audio")
    parser.add_argument("--pseudo_col",   default="text_pl")
    parser.add_argument("--gt_col",       default="text_gold")
    parser.add_argument("--max_rows",     default=None, type=int)
    parser.add_argument("--save_every",   default=50, type=int)
    args = parser.parse_args()
    
    main(
        INPUT_CSV=args.input_csv, 
        OUTPUT_CSV=args.output_csv, 
        MODEL=args.model,
        DURATION_CSV=args.duration_csv,
        AUDIO_COL=args.audio_col, 
        PSEUDO_COL=args.pseudo_col, 
        GT_COL=args.gt_col,
        MAX_ROWS=args.max_rows,
        SAVE_EVERY=args.save_every
    )