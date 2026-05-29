from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = REPO_ROOT / "experiments" / "results_table.csv"
FIGURES_DIR = REPO_ROOT / "figures"

MODE_ORDER = ["easy", "shifted", "noisy"]
MODE_LABELS = {
    "easy": "Easy",
    "shifted": "Shifted",
    "noisy": "Noisy",
}


def load_results() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_PATH)
    return df.copy()


def select_bs512_30k(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[(df["n_samples"] == 30000) & (df["batch_size"] == 512)].copy()
    sub["mode"] = pd.Categorical(sub["mode"], categories=MODE_ORDER, ordered=True)
    return sub.sort_values("mode")


def plot_retrieval_bridge(df: pd.DataFrame) -> None:
    sub = select_bs512_30k(df)

    metrics = [
        ("val_recall@1", "Recall@1"),
        ("val_recall@10", "Recall@10"),
        ("val_recall@50", "Recall@50"),
    ]

    x = np.arange(len(MODE_ORDER))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8.8, 5.2))

    for i, (col, label) in enumerate(metrics):
        values = []
        for mode in MODE_ORDER:
            row = sub[sub["mode"] == mode]
            values.append(float(row[col].iloc[0]) if len(row) else np.nan)

        ax.bar(x + (i - 1) * width, values, width=width, label=label)

    ax.set_title("CXR-text retrieval bridge strength")
    ax.set_ylabel("Validation recall")
    ax.set_xticks(x)
    ax.set_xticklabels([MODE_LABELS[m] for m in MODE_ORDER])
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=8, frameon=True)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cxr_text_retrieval_bridge.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_alignment_similarity(df: pd.DataFrame) -> None:
    sub = select_bs512_30k(df)

    x = np.arange(len(MODE_ORDER))
    pos_sim = []
    lift50 = []

    for mode in MODE_ORDER:
        row = sub[sub["mode"] == mode]
        pos_sim.append(float(row["val_pos_sim_mean"].iloc[0]) if len(row) else np.nan)
        lift50.append(float(row["val_lift@50"].iloc[0]) if len(row) else np.nan)

    fig, ax1 = plt.subplots(figsize=(8.8, 5.2))

    ax1.bar(x - 0.18, pos_sim, width=0.36, label="Positive-pair similarity")
    ax1.set_ylabel("Cosine similarity")
    ax1.set_xticks(x)
    ax1.set_xticklabels([MODE_LABELS[m] for m in MODE_ORDER])
    ax1.grid(axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(x + 0.18, lift50, marker="o", linewidth=2.0, label="Lift@50")
    ax2.set_ylabel("Lift@50 over random")

    ax1.set_title("Alignment similarity vs retrieval lift")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right", fontsize=8, frameon=True)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cxr_text_alignment_similarity.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_batch_compute_tradeoff(df: pd.DataFrame) -> None:
    sub = df[(df["mode"] == "shifted") & (df["n_samples"] == 30000)].copy()
    sub = sub.sort_values("batch_size")

    if sub.empty:
        raise RuntimeError("No shifted 30k batch-size comparison rows found.")

    labels = [f"Batch {int(v)}" for v in sub["batch_size"].tolist()]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))

    axes[0].bar(x, sub["val_recall@50"].astype(float))
    axes[0].set_title("Recall@50")
    axes[0].set_ylabel("Validation Recall@50")

    axes[1].bar(x, sub["epoch_time_sec"].astype(float))
    axes[1].set_title("Epoch time")
    axes[1].set_ylabel("Seconds")

    axes[2].bar(x, sub["peak_cuda_memory_gb"].astype(float))
    axes[2].set_title("Peak CUDA memory")
    axes[2].set_ylabel("GB")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Batch-size tradeoff on shifted 30k setting", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cxr_text_batch_compute_tradeoff.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_results()

    plot_retrieval_bridge(df)
    plot_alignment_similarity(df)
    plot_batch_compute_tradeoff(df)

    print("Saved figures:")
    print(FIGURES_DIR / "cxr_text_retrieval_bridge.png")
    print(FIGURES_DIR / "cxr_text_alignment_similarity.png")
    print(FIGURES_DIR / "cxr_text_batch_compute_tradeoff.png")


if __name__ == "__main__":
    main()