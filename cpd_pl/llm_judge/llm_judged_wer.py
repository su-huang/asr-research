import pandas as pd
import jiwer
import argparse
import re
from word2number import w2n

tens_map = {'twenty':20,'thirty':30,'forty':40,'fifty':50,
            'sixty':60,'seventy':70,'eighty':80,'ninety':90}
ones_map = {'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,
            'six':6,'seven':7,'eight':8,'nine':9,'ten':10,
            'eleven':11,'twelve':12,'thirteen':13,'fourteen':14,
            'fifteen':15,'sixteen':16,'seventeen':17,'eighteen':18,'nineteen':19}
all_map = {**ones_map, **tens_map}
 
ones = r'(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen)'
tens = r'(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)'
single_num = r'(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|and)'
 
def normalize(text, debug=False):
    if not isinstance(text, str):
        text = str(text) if pd.notna(text) else ""
        
    text = text.replace("<UNINTELLIGIBLE>", "")
    text = text.lower()
    text = re.sub(r'[\u2018\u2019\u201c\u201d`"]', '', text)
    text = text.replace('-', ' ')
 
    # structural compounds (hundred/thousand/and)
    real_structural = rf'\b{single_num}(?:\s+{single_num})*\b'
 
    def replace_structural(match):
        text_chunk = match.group(0).strip()
        if not re.search(r'\b(hundred|thousand|and)\b', text_chunk, re.IGNORECASE):
            return text_chunk
        if debug: print(f"  structural match: '{text_chunk}'")
        try:
            num = w2n.word_to_num(text_chunk)
            return ' '.join(list(str(num)))
        except:
            return text_chunk
 
    text = re.sub(real_structural, replace_structural, text, flags=re.IGNORECASE)
    if debug: print(f"After Step 1 (structural): '{text}'")
 
    # spoken pairs — ones/tens + tens (e.g. "two twenty" = 220)
    spoken_pair = rf'\b({ones}|{tens})\s+({tens})\b'
 
    def resolve_spoken_pair(match):
        a, b = match.group(1).lower(), match.group(2).lower()
        if debug: print(f"  spoken pair: '{a}' + '{b}'")
        if a in all_map and b in tens_map:
            combined = str(all_map[a]) + str(tens_map[b])
            return ' '.join(list(combined))
        return match.group(0)
 
    text = re.sub(spoken_pair, resolve_spoken_pair, text, flags=re.IGNORECASE)
    if debug: print(f"After Step 2 (spoken pairs): '{text}'")
 
    # tens + ones (e.g. "twenty one" = 21)
    tens_ones_pair = rf'\b({tens})\s+({ones})\b'
 
    def resolve_tens_ones(match):
        a, b = match.group(1).lower(), match.group(2).lower()
        if debug: print(f"  tens+ones pair: '{a}' + '{b}'")
        if a in tens_map and b in ones_map:
            combined = str(tens_map[a] + ones_map[b])
            return ' '.join(list(combined))
        return match.group(0)
 
    text = re.sub(tens_ones_pair, resolve_tens_ones, text, flags=re.IGNORECASE)
    if debug: print(f"After Step 2b (tens+ones): '{text}'")
 
    # single number words
    single_pattern = rf'\b{single_num}\b'
 
    def resolve_single(match):
        word = match.group(0).lower()
        if word in ('and', 'hundred', 'thousand'):
            return match.group(0)
        if word in all_map:
            return ' '.join(list(str(all_map[word])))
        return match.group(0)
 
    text = re.sub(single_pattern, resolve_single, text, flags=re.IGNORECASE)
    if debug: print(f"After Step 3 (singles): '{text}'")
 
    # strip punctuation
    text = re.sub(r"[^a-z0-9'\s]", '', text)
 
    # isolate literal digits
    text = re.sub(r'(\d)', r' \1 ', text)
 
    # clean whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    if debug: print(f"Final: '{text}'")
    return text

def calculate_corpus_wer(df, gold_col, pl_col, use_normalization=False):
    if df.empty:
        return 0.0
        
    if use_normalization:
        references = df[gold_col].apply(normalize).tolist()
        hypotheses = df[pl_col].apply(normalize).tolist()
    else:
        references = df[gold_col].fillna("").astype(str).tolist()
        hypotheses = df[pl_col].fillna("").astype(str).tolist()
    
    filtered_pairs = [(r, h) for r, h in zip(references, hypotheses) if r.strip()]
    
    if not filtered_pairs:
        return 0.0
        
    final_refs, final_hyps = zip(*filtered_pairs)
    return jiwer.wer(list(final_refs), list(final_hyps))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--pl_col", default="text_pl")
    parser.add_argument("--gold_col", default="text_gold")
    parser.add_argument("--correct_col", default="is_correct")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    # split by is_correct subset condition
    correct_df = df[df[args.correct_col] == 1]
    incorrect_df = df[df[args.correct_col] == 0]
        
    # calculate original wer
    orig_overall_wer = calculate_corpus_wer(df, args.gold_col, args.pl_col, use_normalization=False)
    orig_correct_wer = calculate_corpus_wer(correct_df, args.gold_col, args.pl_col, use_normalization=False)
    orig_incorrect_wer = calculate_corpus_wer(incorrect_df, args.gold_col, args.pl_col, use_normalization=False)

    # calculate normalized wer
    norm_overall_wer = calculate_corpus_wer(df, args.gold_col, args.pl_col, use_normalization=True)
    norm_correct_wer = calculate_corpus_wer(correct_df, args.gold_col, args.pl_col, use_normalization=True)
    norm_incorrect_wer = calculate_corpus_wer(incorrect_df, args.gold_col, args.pl_col, use_normalization=True)

    print(f"\ntotal samples:               {len(df)}")
    print(f"is_correct=1 samples:        {len(correct_df)}")
    print(f"is_correct=0 samples:        {len(incorrect_df)}")
    print("\noriginal wer:")
    print(f"overall WER:                 {orig_overall_wer:.4f}")
    print(f"wer (is_correct=1):          {orig_correct_wer:.4f}")
    print(f"wer (is_correct=0):          {orig_incorrect_wer:.4f}")
    print("\nnormalized wer:")
    print(f"overall wer:                 {norm_overall_wer:.4f}")
    print(f"wer (is_correct=1):          {norm_correct_wer:.4f}")
    print(f"wer (is_correct=0):          {norm_incorrect_wer:.4f}")