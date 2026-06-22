import numpy as np
from scipy import stats

# #whisper significance testing
# BASELINE_WER = 0.3690  # OTS whisper large-v3 zeroshot

# EXPERIMENTS = [
#     {"exp name": "Whisper Oracle FT (WER<=0.2)", "WERs from the runs": [0.3576, 0.3485, 0.3491 ]},
#     {"exp name": "Whisper Unfiltered FT", "WERs from the runs": [0.3900, 0.3896, 0.3772]},
#     {"exp name": "Whisper log-prob FT", "WERs from the runs": [0.3766, 0.3736, 0.3769]},
#     {"exp name": "Whisper LLM-judge FT", "WERs from the runs": [0.3701, 0.3661, 0.3694]},
# ]

# Qwen significance testing
BASELINE_WER = 0.3611  # OTS qwen zeroshot

EXPERIMENTS = [
    {"exp name": "Qwen Oracle FT (WER<=0.2)", "WERs from the runs": [0.3339, 0.3244, 0.3244]},
    {"exp name": "Qwen Unfiltered FT", "WERs from the runs": [0.3578, 0.3588, 0.3625]}, 
    {"exp name": "Qwen log-prob FT", "WERs from the runs": [0.3571, 0.3571, 0.3544]},
    {"exp name": "Qwen LLM-judge FT", "WERs from the runs": [0.3485, 0.3517, 0.3485]},

]

# 1. Dynamically find and extract the "Unfiltered FT" control group data
control_exp = next((e for e in EXPERIMENTS if "Unfiltered FT" in e["exp name"]), None)
if control_exp is None:
    raise ValueError("Could not find 'Unfiltered FT' experiment in the list.")

control_wers = np.array(control_exp["WERs from the runs"])
control_mean = control_wers.mean()

print(f"Baseline WER: {BASELINE_WER:.4f}")
print(f"Control ('Unfiltered FT') Mean WER: {control_mean:.4f}\n")

# Table Header updated for clarity
print(f"{'Experiment':<28} | {'Vs Baseline':^33} | {'Vs Unfiltered FT Control':^22}")
print(f"{'':<28} | {'Mean':>7} {'Delta%':>8} {'p-val':>9} {'sig':>5} | {'p-val':>9} {'sig':>5} {'dir':>5}")
print("-" * 95)

all_means = []

for exp in EXPERIMENTS:
    wers = np.array(exp["WERs from the runs"])
    mean = wers.mean()
    all_means.append(mean)
    
    # --- 1. Significance vs. Deterministic Baseline (1-sample t-test) ---
    t_stat_base, p_val_base = stats.ttest_1samp(wers, popmean=BASELINE_WER)
    delta_base = (mean - BASELINE_WER) / BASELINE_WER
    sig_base = "*" if p_val_base < 0.05 else "ns"
    
    # --- 2. Significance vs. Unfiltered FT Control (2-sample Welch's t-test) ---
    if exp["exp name"] == control_exp["exp name"]:
        # Skip comparing the control group to itself
        p_val_ctrl_str = "-"
        sig_ctrl = "-"
        dir_ctrl = "-"
    else:
        # equal_var=False applies Welch's t-test modification
        t_stat_ctrl, p_val_ctrl = stats.ttest_ind(wers, control_wers, equal_var=False)
        
        # Optional: For Bonferroni correction, change 0.05 to (0.05 / 3) -> 0.0167
        sig_ctrl = "*" if p_val_ctrl < 0.05 else "ns"
        p_val_ctrl_str = f"{p_val_ctrl:.3e}"
        dir_ctrl = "better" if mean < control_mean else "worse"

    # Print nicely formatted row
    print(f"{exp['exp name']:<28} | {mean:>7.4f} {delta_base:>+8.3f} {p_val_base:>9.2e} {sig_base:>5} | {p_val_ctrl_str:>9} {sig_ctrl:>5} {dir_ctrl:>5}")

print("-" * 95)