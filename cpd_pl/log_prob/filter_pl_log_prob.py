import pandas as pd
from collections import Counter

# Path to the input CSV file
input_csv_file = "/export/fs06/shuan148/asr-research/cpd_pl/log_prob/whisper/whisper_train_pl_24hr.csv"

# Paths to save the resulting CSV files
low_10_percent_csv_file = "/export/fs06/shuan148/asr-research/cpd_pl/log_prob/whisper/whisper_train_pl_24hr_top_90.csv"
low_60_percent_csv_file = "/export/fs06/shuan148/asr-research/cpd_pl/log_prob/whisper/whisper_train_pl_24hr_top_40.csv"

# Load the CSV file with empty cells treated as empty strings
df = pd.read_csv(input_csv_file, na_filter=False)

# --- Step 1: Filter Rows Based on Criteria ---
# Filter rows where logprob >= -1
df = df[df['avg_logprob'] >= -1] #filter out the samples with logprob < -1

# Define a function to check for excessive n-gram repetitions
def has_excessive_ngrams(text, max_repeats=8):
    """Returns True if any n-gram (n=1 to 5) repeats more than `max_repeats` times."""
    if not isinstance(text, str) or text.strip() == "":
        return False  # Skip if text is empty or not a string
    
    words = text.split()  # Split transcript into words
    for n in range(1, 6):  # Check for n-grams of size 1 to 5
        ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        counts = Counter(ngrams)
        if any(count > max_repeats for count in counts.values()):
            return True  # Excessive n-gram repetition found
    return False  # No excessive repetition

# Filter out rows where n-gram repetitions exceed the threshold
df = df[~df['text'].apply(lambda x: has_excessive_ngrams(x))]

# --- Step 2: Rank Rows By LogProb and Discard Percentages ---
# Sort rows by LogProb in ascending order
df = df.sort_values(by="avg_logprob", ascending=True)

# Calculate the number of rows to discard for each percentage
num_rows = len(df)
discard_10_percent = int(num_rows * 0.10)
discard_60_percent = int(num_rows * 0.60)

# Retain the top percentages of rows (discard the bottom percentages)
top_90_percent_df = df.iloc[discard_10_percent:]  # Discard the bottom 10%
top_40_percent_df = df.iloc[discard_60_percent:]  # Discard the bottom 60%

# --- Step 3: Save the Resulting DataFrames to CSV ---
top_90_percent_df.to_csv(low_10_percent_csv_file, index=False)
top_40_percent_df.to_csv(low_60_percent_csv_file, index=False)

# Print summary
print(f"Filtered CSV saved with top 90% rows: {low_10_percent_csv_file}")
print(f"Filtered CSV saved with top 40% rows: {low_60_percent_csv_file}")