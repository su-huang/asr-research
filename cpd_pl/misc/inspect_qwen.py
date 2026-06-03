# import inspect
# from qwen_asr import Qwen3ASRModel

# print(inspect.getsource(Qwen3ASRModel.transcribe))
# print(inspect.getsource(Qwen3ASRModel._infer_asr))
# print(inspect.getsource(Qwen3ASRModel._infer_asr_transformers))
# print(inspect.signature(Qwen3ASRModel.transcribe))
# print(Qwen3ASRModel.__mro__)

import inspect
import torch
import qwen_asr
from transformers import AutoModel

model = AutoModel.from_pretrained("Qwen/Qwen3-ASR-1.7B", device_map="cpu")
print(type(model))
print(inspect.signature(model.forward))