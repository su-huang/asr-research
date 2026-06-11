import argparse
from collections import defaultdict
import jiwer
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_word_error_stats(references, hypotheses):
    """Analyzes word-level transcription errors.

    Args:
        references (list of str): The ground truth sentences.
        hypotheses (list of str): The predicted transcripts.
    """
    word_stats = defaultdict(lambda: {"correct": 0, "substitution": 0, "deleted": 0})
    insertions = defaultdict(int)

    for ref, hyp in zip(references, hypotheses):
        alignment = jiwer.process_words(ref, hyp)
        ref_words = ref.split()
        hyp_words = hyp.split()

        for op in alignment.alignments[0]:
            op_type = op.type

            if op_type == "equal":
                for i in range(op.ref_start_idx, op.ref_end_idx):
                    word = ref_words[i]
                    word_stats[word]["correct"] += 1

            elif op_type == "substitute":
                for i in range(op.ref_start_idx, op.ref_end_idx):
                    word = ref_words[i]
                    word_stats[word]["substitution"] += 1

            elif op_type == "delete":
                for i in range(op.ref_start_idx, op.ref_end_idx):
                    word = ref_words[i]
                    word_stats[word]["deleted"] += 1

            elif op_type == "insert":
                for i in range(op.hyp_start_idx, op.hyp_end_idx):
                    word = hyp_words[i]
                    insertions[word] += 1

    return dict(word_stats), dict(insertions)


def stats_to_df(stats):
    rows = []
    for word, c in stats.items():
        incorrect = c["substitution"] + c["deleted"]
        total = c["correct"] + incorrect
        rows.append(
            {
                "word": word,
                "correct": c["correct"],
                "substitution": c["substitution"],
                "deleted": c["deleted"],
                "incorrect": incorrect,
                "incorrect_pct": incorrect / total if total > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows).set_index("word")


def main():
    parser = argparse.ArgumentParser(description="Analyze and compare ASR word error statistics.")

    # File paths (Condensed to single lines)
    parser.add_argument("--ots_input", type=str, default="/export/fs06/shuan148/asr-research/cpd_pl/qwen_results/qwen_ots_1601734.csv", help="Path to OTS model transcriptions CSV.")
    parser.add_argument("--ft_input", type=str, default="/export/fs06/shuan148/asr-research/cpd_pl/qwen_results/qwen_pl_24hrs_judged_full_1636066.csv", help="Path to fine-tuned model transcriptions CSV.")
    parser.add_argument("--ots_stats_output", type=str, default="/export/fs06/shuan148/asr-research/cpd_pl/misc/analysis_results/qwen_ots_word_error_stats.csv", help="Path to save OTS word error stats CSV.")
    parser.add_argument("--ft_stats_output", type=str, default="/export/fs06/shuan148/asr-research/cpd_pl/misc/analysis_results/qwen_ft_word_error_stats.csv", help="Path to save FT word error stats CSV.")
    parser.add_argument("--diff_output", type=str, default="/export/fs06/shuan148/asr-research/cpd_pl/misc/analysis_results/qwen_ots_vs_llm-judged_word_error_pct_diff.csv", help="Path to save comparison difference CSV.")
    parser.add_argument("--plot_output", type=str, default="/export/fs06/shuan148/asr-research/cpd_pl/misc/analysis_results/incorrect_pct_diff_qwen-ots-vs-llm-judged-distribution.png", help="Path to save distribution plot image.")

    # Column names (Condensed to single lines)
    parser.add_argument("--ots_gt_col", type=str, default="norm_ground_truth", help="Column name for OTS ground truth transcriptions.")
    parser.add_argument("--ots_pred_col", type=str, default="norm_prediction", help="Column name for OTS predicted transcriptions.")
    parser.add_argument("--ft_gt_col", type=str, default="gt_norm", help="Column name for Fine-tuned ground truth transcriptions.")
    parser.add_argument("--ft_pred_col", type=str, default="pred_norm", help="Column name for Fine-tuned predicted transcriptions.")

    args = parser.parse_args()

    # ── OTS model ────────────────────────────────────────────────────────────
    print(f"Loading OTS transcriptions from: {args.ots_input}")
    qwen_ots = pd.read_csv(args.ots_input)

    refs_ots = qwen_ots[args.ots_gt_col].fillna("").astype(str).tolist()
    hyps_ots = qwen_ots[args.ots_pred_col].fillna("").astype(str).tolist()
    stats_ots, _ = get_word_error_stats(refs_ots, hyps_ots)
    df_ots = stats_to_df(stats_ots)

    df_ots.to_csv(args.ots_stats_output)
    print(f"OTS word error stats saved to: {args.ots_stats_output}")

    # ── PL finetuned model ───────────────────────────────────────────────────
    print(f"Loading FT transcriptions from: {args.ft_input}")
    qwen_ft = pd.read_csv(args.ft_input)

    refs_ft = qwen_ft[args.ft_gt_col].fillna("").astype(str).tolist()
    hyps_ft = qwen_ft[args.ft_pred_col].fillna("").astype(str).tolist()
    stats_ft, _ = get_word_error_stats(refs_ft, hyps_ft)
    df_ft = stats_to_df(stats_ft)

    df_ft.to_csv(args.ft_stats_output)
    print(f"FT word error stats saved to: {args.ft_stats_output}")

    # ── Difference in incorrect_pct ──────────────────────────────────────────
    df_diff = df_ots.merge(df_ft, left_index=True, right_index=True, how="inner", suffixes=("_ots", "_ft"))
    df_diff["incorrect_pct_diff"] = df_diff["incorrect_pct_ft"] - df_diff["incorrect_pct_ots"]
    df_diff["word_count"] = df_diff["correct_ots"] + df_diff["incorrect_ots"]
    total_words = df_diff["word_count"].sum()
    df_diff["word_freq"] = df_diff["word_count"] / total_words
    df_diff["weighted_diff"] = df_diff["incorrect_pct_diff"] * (df_diff["word_freq"])
    df_diff = df_diff.sort_values("weighted_diff", ascending=True)

    df_diff.to_csv(args.diff_output)
    print(f"Incorrect pct diff saved to: {args.diff_output}")

    # ── Plot distribution of incorrect_pct_diff ──────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df_diff["weighted_diff"], bins=40, edgecolor="black", color="steelblue")
    ax.axvline(0, color="red", linestyle="--", linewidth=1, label="no change")
    ax.set_xlabel("(pct FT incorrect - pct OTS incorrect) * word freq")
    ax.set_ylabel("Number of words")
    ax.set_title("Distribution of change in incorrect % per word (FT vs OTS)")
    ax.legend()
    plt.tight_layout()

    plt.savefig(args.plot_output, dpi=150)
    print(f"Plot saved to: {args.plot_output}")


if __name__ == "__main__":
    main()