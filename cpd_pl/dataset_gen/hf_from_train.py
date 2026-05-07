from datasets import Dataset
import pandas as pd

df = pd.read_csv("/export/fs06/shuan148/asr-research/cpd_pl/whisper_csv/train/train_pl_24hr_gold_5hr.csv")
dataset = Dataset.from_pandas(df)
dataset.save_to_disk("/export/fs06/shuan148/asr-research/cpd_pl/whisper_datasets/train/train_pl_24hr_gold_5hr")