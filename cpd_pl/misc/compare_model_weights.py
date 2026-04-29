import torch
from transformers import AutoModelForSpeechSeq2Seq

m1 = AutoModelForSpeechSeq2Seq.from_pretrained("/export/fs06/shuan148/asr-research/cpd_pl/models/whisper/pl-24hrs-full")
m2 = AutoModelForSpeechSeq2Seq.from_pretrained("/export/fs06/shuan148/asr-research/cpd_pl/models/whisper/gold-full")

for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
    if not torch.allclose(p1, p2, atol=1e-6):
        print(f"DIFFER: {n1}")
        break
else:
    print("Models are IDENTICAL — training had no effect or loaded same checkpoint")
