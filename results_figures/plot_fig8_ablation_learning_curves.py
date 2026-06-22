#!/usr/bin/env python3
"""Generate fig8_ablation_learning_curves.png."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


FIGSIZE = (9, 7)
DPI = 300

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "Ablation Studies" / "results"
FULL_MODEL_DIR = ROOT / "results" / "TopoFreeRL" / "logs"
OUTPUT_PATH = Path(__file__).resolve().parent / "fig8_ablation_learning_curves.png"


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 26,
        "axes.labelsize": 28,
        "axes.titlesize": 30,
        "axes.titleweight": "normal",
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "legend.fontsize": 20,
        "lines.linewidth": 4.0,
        "axes.linewidth": 2.0,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "mathtext.fontset": "stix",
    }
)

COLORS = {
    "full": "#2E86AB",
    "no_workflow": "#9B59B6",
    "no_future": "#F18F01",
    "no_topology": "#C73E1D",
}

LABELS = {
    "full": "STAR-PPO (Full)",
    "no_workflow": "w/o Workflow",
    "no_future": "w/o Future Reward",
    "no_topology": "w/o Topology",
}

BEST_SEEDS = {
    "full": None,
    "no_workflow": 42,
    "no_future": 43,
    "no_topology": 44,
}


def load_training_data(results_dir: Path, full_model_dir: Path):
    import pandas as pd

    modes = ["no_topology", "no_workflow", "no_future"]
    seeds = [42, 43, 44]
    data = {}

    def load_metrics(base_path: Path):
        csv_path = base_path.with_suffix(".csv")
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                return df["rewards"].values
            except Exception:
                pass
        if base_path.exists():
            d = np.load(base_path)
            return d["rewards"]
        return None

    data["full"] = {"rewards": [], "seeds": {}}
    for seed in seeds:
        path = full_model_dir / f"LATEST_Server1_Trap_seed{seed}" / "metrics.npz"
        rewards = load_metrics(path)
        if rewards is not None:
            data["full"]["rewards"].append(rewards)
            data["full"]["seeds"][seed] = rewards

    for mode in modes:
        data[mode] = {"rewards": [], "seeds": {}}
        for seed in seeds:
            path = results_dir / f"{mode}_seed{seed}" / "metrics.npz"
            rewards = load_metrics(path)
            if rewards is not None:
                data[mode]["rewards"].append(rewards)
                data[mode]["seeds"][seed] = rewards

    return data


def main():
    train_data = load_training_data(RESULTS_DIR, FULL_MODEL_DIR)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    plot_order = ["full", "no_workflow", "no_future", "no_topology"]

    for mode in plot_order:
        if mode not in train_data or len(train_data[mode]["rewards"]) == 0:
            continue

        best_seed = BEST_SEEDS[mode]
        if best_seed is None:
            curves = np.array(train_data[mode]["rewards"])
            main_curve = np.mean(curves, axis=0)
            min_curve = np.min(curves, axis=0)
            max_curve = np.max(curves, axis=0)
        else:
            main_curve = train_data[mode]["seeds"][best_seed]
            all_curves = np.array(train_data[mode]["rewards"])
            min_curve = np.min(all_curves, axis=0)
            max_curve = np.max(all_curves, axis=0)

        epochs = np.arange(1, len(main_curve) + 1)
        ax.plot(
            epochs,
            main_curve,
            color=COLORS[mode],
            label=LABELS[mode],
            linewidth=2.5,
            alpha=0.9,
        )
        ax.fill_between(epochs, min_curve, max_curve, color=COLORS[mode], alpha=0.15)

    ax.set_xlabel("Training Epochs")
    ax.set_ylabel("Reward")
    ax.legend(loc="lower right", fontsize=20, framealpha=0.9)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if train_data.get("full", {}).get("rewards"):
        ax.set_xlim(1, 100)

    ax.grid(True, alpha=0.3)
    fig.tight_layout(pad=0.5)
    fig.savefig(OUTPUT_PATH, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
