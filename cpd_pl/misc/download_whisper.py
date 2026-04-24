from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v3").save_pretrained("/export/fs06/shuan148/asr-research/cpd_pl/models/whisper-large-v3")
AutoProcessor.from_pretrained("openai/whisper-large-v3").save_pretrained("/export/fs06/shuan148/asr-research/cpd_pl/models/whisper-large-v3")
