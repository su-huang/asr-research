import re
import csv
from jiwer import wer
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
  
def compute_wer(reference: str, hypothesis: str) -> float:
    if not reference and not hypothesis:
        return 0.0
    if not reference or not hypothesis:
        return 1.0
    return wer(reference, hypothesis)
 
def reprocess(input_path: str, output_path: str) -> None:
    with open(input_path, newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        rows = list(reader)
 
    gt_col, pred_col, wer_col = "", "", ""

    if "text_gold" in fieldnames: 
        gt_col = "text_gold"
    elif "ref_text" in fieldnames:
        gt_col = "ref_text"
    elif "norm_ground_truth" in fieldnames: 
        gt_col = "norm_ground_truth"
    elif "gt_norm" in fieldnames: 
        gt_col = "gt_norm"
    elif "ground_truth" in fieldnames:
        gt_col = "ground_truth"
    elif "text" in fieldnames: 
        gt_col = "text"
    elif "groundtruth" in fieldnames: 
        gt_col = "groundtruth"

    if "text_pl" in fieldnames:
        pred_col = "text_pl"
    elif "norm_prediction" in fieldnames: 
        pred_col = "norm_prediction"
    elif "pred_norm" in fieldnames: 
        pred_col = "pred_norm"
    elif "pseudo_label" in fieldnames:
        pred_col = "pseudo_label"
    elif "hypothesis" in fieldnames: 
        pred_col = "hypothesis"
    elif "whisper_raw" in fieldnames: 
        pred_col = "whisper_raw"
    elif "text" in fieldnames:
        pred_col = "text"

    if "norm_wer" in fieldnames: 
        wer_col = "norm_wer"
    elif "wer" in fieldnames: 
        wer_col = "wer"
    elif "WER" in fieldnames: 
        wer_col = "WER"
    else: 
        wer_col = "wer"
        fieldnames.append(wer_col)
    
    if not gt_col or not pred_col or not wer_col:
        raise ValueError(f"could not detect expected column names in {input_path}. found: {fieldnames}")

    all_pre_refs, all_pre_hyps = [], [] 
    all_pre_wers, num_pre_wers = 0, 0

    all_refs, all_hyps = [], [] 
    all_wers, num_wers = 0, 0

    for row in rows:
        if row[gt_col].strip():
            all_pre_refs.append(row[gt_col])
            all_pre_hyps.append(row[pred_col])
            all_pre_wers += compute_wer(row[gt_col], row[pred_col])
            num_pre_wers += 1

        norm_gt = normalize(row[gt_col])
        norm_pred = normalize(row[pred_col])
 
        row[gt_col] = norm_gt
        row[pred_col] = norm_pred
        row[wer_col] = compute_wer(norm_gt, norm_pred)

        if norm_gt: 
            all_refs.append(norm_gt)
            all_hyps.append(norm_pred)

            all_wers += row[wer_col]
            num_wers += 1 
    
    with open(output_path, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
 
    global_wer = wer(" ".join(all_refs), " ".join(all_hyps))
    avg_sample_wer = all_wers / num_wers 

    pre_global_wer = wer(" ".join(all_pre_refs), " ".join(all_pre_hyps))
    pre_avg_wer = all_pre_wers / num_pre_wers

    return global_wer, avg_sample_wer, pre_global_wer, pre_avg_wer
 
if __name__ == "__main__":
    # input csv path, output csv path
    CSV_PATHS = [
        ["/export/fs06/shuan148/asr-research/cpd_pl/qwen_results/qwen_pl_24hrs_judged_70b_full_1664735.csv", "/export/fs06/shuan148/asr-research/cpd_pl/qwen_results/qwen_pl_24hrs_judged_70b_full_normalized_1664735.csv", "qwen judged 70b"],
        ["/export/fs06/shuan148/asr-research/cpd_pl/qwen_results/qwen_pl_24hrs_judged_aggressive_full_1664839.csv", "/export/fs06/shuan148/asr-research/cpd_pl/qwen_results/qwen_pl_24hrs_judged_aggressive_full_normalized_1664839.csv", "qwen judged aggressive"]
    ]
 
    summary_rows = []
    for path in CSV_PATHS:
        global_wer, avg_sample_wer, pre_global_wer, pre_avg_wer = reprocess(path[0], path[1])
        summary_rows.append({"type": path[2], 
                             "average wer": pre_global_wer,
                             "average per-sample wer": pre_avg_wer, 
                             "normalized average wer": global_wer,
                             "normalized average per-sample wer": avg_sample_wer, 
                             "original": path[0], 
                             "normalized": path[1]})
    
    # overall summary csv path 
    summary_path = "/export/fs06/shuan148/asr-research/csv_normalization/results/summary_june11.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["type", "average wer", "average per-sample wer", "normalized average wer", "normalized average per-sample wer", "original", "normalized"])
        writer.writeheader()
        writer.writerows(summary_rows)
