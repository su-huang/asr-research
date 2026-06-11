import argparse
import pandas as pd
import matplotlib.pyplot as plt

BINS = [0, 1, 2, 3, 4, 5, 10, 30]
BIN_LABELS = ["0-1s", "1-2s", "2-3s", "3-4s", "4-5s", "5-10s", "10-30s"]


def main(args):
    df = pd.read_csv(args.input_csv, na_filter=False)
    df = df[df["duration_s"] > 0]

    counts = []
    for i in range(len(BINS) - 1):
        lo, hi = BINS[i], BINS[i + 1]
        count = ((df["duration_s"] >= lo) & (df["duration_s"] < hi)).sum()
        counts.append(count)

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(BIN_LABELS, counts, color="#4472C4")

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 50,
            f"{count:,}",
            ha="center", va="bottom", fontsize=10
        )

    ax.set_xlabel("Duration")
    ax.set_ylabel("Count")
    ax.set_title(args.title)
    ax.set_ylim(0, max(counts) * 1.15)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Plot saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output",    default="duration_distribution.png")
    parser.add_argument("--title",     default="Duration Distribution")
    args = parser.parse_args()
    main(args)