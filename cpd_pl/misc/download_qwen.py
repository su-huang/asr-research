from qwen_asr import Qwen3ASRModel
import torch

save_path = "/export/fs06/shuan148/asr-research/cpd_pl/models/qwen3-asr-1.7b"

use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-1.7B",
    dtype=torch.bfloat16 if use_bf16 else torch.float16,
)

model.save_pretrained(save_path)
model.processor.save_pretrained(save_path)
print(f"Saved to {save_path}")