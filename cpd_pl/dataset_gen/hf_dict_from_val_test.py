from datasets import DatasetDict, load_from_disk

validation = load_from_disk("/export/fs06/shuan148/asr-research/cpd_pl/whisper_datasets/val/val_gold_1.25hr")
test = load_from_disk("/export/fs06/shuan148/asr-research/cpd_pl/whisper_datasets/test/test_gold_2.25hr")

eval_dict = DatasetDict({
    "validation": validation,
    "test": test
})

eval_dict.save_to_disk("/export/fs06/shuan148/asr-research/cpd_pl/whisper_datasets/hf_dict/hf_dict_val_gold_1.25hr_test_gold_2.25hr")
