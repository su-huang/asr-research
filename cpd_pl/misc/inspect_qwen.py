import inspect
from qwen_asr import Qwen3ASRModel

# print(inspect.getsource(Qwen3ASRModel.transcribe))
# print(inspect.getsource(Qwen3ASRModel._infer_asr))
print(inspect.getsource(Qwen3ASRModel._infer_asr_transformers))