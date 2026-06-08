import pandas as pd
from collections import Counter

input_csv_file = "/export/fs06/shuan148/asr-research/cpd_pl/log_prob/qwen/qwen_train_pl_24hr.csv"
output_csv_file = "/export/fs06/shuan148/asr-research/cpd_pl/log_prob/qwen/qwen_train_pl_24hr_log_prob_filtered.csv"

df = pd.read_csv(input_csv_file, na_filter=False)

df = df[df['avg_logprob'] > -1]

def has_excessive_ngrams(text, max_repeats=8):
    if not isinstance(text, str) or text.strip() == "":
        return False
    words = text.split()
    for n in range(1, 6):
        ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        counts = Counter(ngrams)
        if any(count > max_repeats for count in counts.values()):
            return True
    return False

df = df[~df['text'].apply(has_excessive_ngrams)]

print(f"Remaining samples: {len(df)}")
total_duration = df['duration_s'].sum()
print(f"Total duration: {total_duration / 3600:.2f} hours")

df.to_csv(output_csv_file, index=False)
print(f"Saved to: {output_csv_file}")