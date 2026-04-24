from qwen_asr import Qwen3ASRModel
import torch

save_path = "/export/fs06/shuan148/asr-research/cpd_pl/models/qwen3-asr-1.7b"

model = Qwen3ASRModel.from_pretrained("Qwen/Qwen3-ASR-1.7B")
model.save_pretrained(save_path)

print(f"Saved to {save_path}")
