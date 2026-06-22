#!/usr/bin/env python3
"""Generate fig10_dwa_weight_trajectory.png."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FIGSIZE = (9, 7)
DPI = 300

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "results" / "TopoFreeRL" / "logs"
OUTPUT_PATH = Path(__file__).resolve().parent / "fig10_dwa_weight_trajectory.png"

REGION = "Server1_Trap"
SEEDS = [42, 43, 44, 45, 46]
DWA_START = 3


plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 26,
        "axes.labelsize": 28,
        "axes.titlesize": 30,
        "axes.titleweight": "normal",
        "axes.linewidth": 2.0,
        "axes.edgecolor": "black",
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "legend.fontsize": 22,
        "lines.linewidth": 4.0,
        "ytick.color": "black",
        "xtick.color": "black",
        "axes.labelcolor": "black",
        "mathtext.fontset": "stix",
    }
)


def load_seed(seed):
    path = LOG_DIR / f"LATEST_{REGION}_seed{seed}" / "training_data.npz"
    if not path.exists():
        return None
    data = np.load(path)
    return {
        "seed": seed,
        "weights": data["weights_hist"],
    }


def main():
    data_list = [load_seed(seed) for seed in SEEDS]
    data_list = [item for item in data_list if item is not None]
    if not data_list:
        raise RuntimeError(f"No DWA training data found in {LOG_DIR}")

    n_epochs = data_list[0]["weights"].shape[0]
    epochs = np.arange(1, n_epochs + 1)
    freeze_epoch = int(n_epochs * 0.8)

    weights_all = np.stack([item["weights"] for item in data_list], axis=0)
    weights_mean = weights_all.mean(axis=0)
    weights_std = weights_all.std(axis=0)

    weight_colors = ["#C0392B", "#2471A3", "#27AE60"]
    weight_labels = [r"$\omega_L$ (Latency)", r"$\omega_C$ (Cost)", r"$\omega_R$ (SLA/Risk)"]
    weight_styles = ["-", "--", "-."]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for dim in range(3):
        ax.plot(
            epochs,
            weights_mean[:, dim],
            color=weight_colors[dim],
            linestyle=weight_styles[dim],
            lw=3.5,
            label=weight_labels[dim],
            zorder=4,
        )
        ax.fill_between(
            epochs,
            weights_mean[:, dim] - weights_std[:, dim],
            weights_mean[:, dim] + weights_std[:, dim],
            color=weight_colors[dim],
            alpha=0.12,
            zorder=2,
        )

    ax.axvline(DWA_START, color="#888", ls=":", lw=2, alpha=0.8)
    ax.axvline(freeze_epoch, color="#888", ls=":", lw=2, alpha=0.8)
    ax.axvspan(freeze_epoch, n_epochs, alpha=0.07, color="#95A5A6", label="Weights frozen (80%)")

    ax.annotate(
        "Init: [0.45, 0.40, 0.15]",
        xy=(1, 0.45),
        xytext=(6, 0.47),
        fontsize=18,
        color="#555",
        fontstyle="italic",
        arrowprops=dict(arrowstyle="->", color="#888", lw=1.5),
    )
    ax.text(DWA_START + 1, 0.205, "DWA active", fontsize=18, color="#555", fontstyle="italic")
    ax.text(freeze_epoch + 1, 0.205, "Frozen", fontsize=18, color="#555", fontstyle="italic")

    ax.set_xlabel("Training Epoch")
    ax.set_ylabel("DWA Weight")
    ax.set_xlim(1, n_epochs)
    ax.set_ylim(0.10, 0.57)
    ax.legend(fontsize=20, loc="upper right", ncol=2, framealpha=0.9)

    fig.tight_layout(pad=0.5)
    fig.savefig(OUTPUT_PATH, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
