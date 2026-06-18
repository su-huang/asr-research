import numpy as np
from scipy import stats
#whisper significance testing
BASELINE_WER = 0.3690  # OTS whisper large-v3 zeroshot

EXPERIMENTS = [
    {"exp name": "Whisper Oracle FT (WER<=0.2)", "WERs from the runs": [0.3576, 0.3485, 0.3491 ]},
    {"exp name": "Whisper log-prob FT", "WERs from the runs": [0.3766, 0.3736, 0.3769]},
    {"exp name": "Whisper LLM-judge FT", "WERs from the runs": [0.3701, 0.3661, 0.3694]},
]

# #Qwen significance testing
# BASELINE_WER = 0.3611  # OTS whisper large-v3 zeroshot

# EXPERIMENTS = [
#     {"exp name": "Qwen Oracle FT (WER<=0.2)", "WERs from the runs": [0.3339, 0.3244, 0.3244]},
#     {"exp name": "Qwen LLM-judge FT", "WERs from the runs": [0.3485, 0.3517, 0.3485]},
#     {"exp name": "Qwen Unfiltered FT", "WERs from the runs": [0.3578, 0.3588, 0.3625]}
# ]

print(f"Baseline WER: {BASELINE_WER:.4f}\n")
print(f"{'Experiment':<30} {'Mean WER':>10} {'Delta %':>8} {'t':>8} {'p':>12} {'sig':>5}")
print("-" * 80)

all_means = []

for exp in EXPERIMENTS:
    wers = np.array(exp["WERs from the runs"])
    t_stat, p_val = stats.ttest_1samp(wers, popmean=BASELINE_WER)
    mean = wers.mean()
    delta = (mean - BASELINE_WER)/BASELINE_WER
    direction = "better" if delta < 0 else "worse"
    sig = "*" if p_val < 0.05 else "ns"
    print(f"{exp['exp name']:<30} {mean:>10.4f} {delta:>+8.4f} {t_stat:>8.4f} {p_val:>12.4e} {sig:>5}  ({direction})")
    all_means.append(mean)

avg_across_experiments = np.mean(all_means)
avg_delta = avg_across_experiments - BASELINE_WER
direction = "better" if avg_delta < 0 else "worse"
print("-" * 80)
