#!/usr/bin/env python3
"""Generate fig9_ablation_tradeoff_scatter.png."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FIGSIZE = (9, 7)
DPI = 300

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "Ablation Studies" / "results"
OUTPUT_PATH = Path(__file__).resolve().parent / "fig9_ablation_tradeoff_scatter.png"


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

MARKERS = {
    "full": "o",
    "no_workflow": "s",
    "no_future": "^",
    "no_topology": "D",
}


def load_inference_data(results_dir: Path):
    path = results_dir / "ablation_inference_results.npz"
    if not path.exists():
        return None

    d = np.load(path)
    data = {}
    modes = ["full", "no_workflow", "no_future", "no_topology"]

    for mode in modes:
        lat_key = f"{mode}_avg_latencies"
        cost_key = f"{mode}_avg_costs"
        if lat_key in d and cost_key in d:
            data[mode] = {
                "latencies": d[lat_key],
                "costs": d[cost_key],
                "avg_latency": np.mean(d[lat_key]),
                "std_latency": np.std(d[lat_key]),
                "avg_cost": np.mean(d[cost_key]),
                "std_cost": np.std(d[cost_key]),
            }

    return data


def confidence_ellipse(ax, x, y, color, n_std=1.0, fill_alpha=0.18, edge_alpha=0.5):
    from matplotlib.patches import Ellipse

    if len(x) < 2:
        return None

    mean_x, mean_y = np.mean(x), np.mean(y)
    cov = np.cov(x, y)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width = 2 * n_std * np.sqrt(eigenvalues[0])
    height = 2 * n_std * np.sqrt(eigenvalues[1])

    ellipse = Ellipse(
        (mean_x, mean_y),
        width=width,
        height=height,
        angle=angle,
        facecolor=color,
        edgecolor=color,
        alpha=fill_alpha,
        linewidth=2,
        linestyle="-",
    )
    ax.add_patch(ellipse)

    border = Ellipse(
        (mean_x, mean_y),
        width=width,
        height=height,
        angle=angle,
        facecolor="none",
        edgecolor=color,
        alpha=edge_alpha,
        linewidth=1.5,
        linestyle="-",
    )
    ax.add_patch(border)
    return ellipse


def pareto_front_indices(costs, latencies):
    n = len(costs)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if costs[j] <= costs[i] and latencies[j] <= latencies[i]:
                if costs[j] < costs[i] or latencies[j] < latencies[i]:
                    is_pareto[i] = False
                    break
    return np.where(is_pareto)[0]


def main():
    inference_data = load_inference_data(RESULTS_DIR)
    if not inference_data:
        raise RuntimeError(f"No inference data found in {RESULTS_DIR}")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    modes = ["full", "no_workflow", "no_future", "no_topology"]
    n_seeds = None

    for mode in modes:
        if mode not in inference_data:
            continue

        d = inference_data[mode]
        costs = d["costs"]
        lats = d["latencies"]
        n_seeds = len(costs)
        avg_lat = np.mean(lats)
        avg_cost = np.mean(costs)
        legend_label = f"{LABELS[mode]} (Avg: {avg_lat:.0f}, ${avg_cost:.3f})"

        confidence_ellipse(ax, costs, lats, COLORS[mode], n_std=1.0, fill_alpha=0.18, edge_alpha=0.5)

        for i, (cost, lat) in enumerate(zip(costs, lats)):
            label = legend_label if i == 0 else None
            ax.scatter(
                cost,
                lat,
                marker=MARKERS[mode],
                color=COLORS[mode],
                s=160,
                alpha=0.85,
                zorder=5,
                label=label,
                edgecolors="black",
                linewidths=1.2,
            )

    if "no_future" in inference_data:
        nf = inference_data["no_future"]
        nf_costs = nf["costs"]
        nf_lats = nf["latencies"]
        pareto_idx = pareto_front_indices(nf_costs, nf_lats)

        if len(pareto_idx) >= 2:
            sort_order = np.argsort(nf_costs[pareto_idx])
            pareto_idx = pareto_idx[sort_order]
            ax.plot(
                nf_costs[pareto_idx],
                nf_lats[pareto_idx],
                color=COLORS["no_future"],
                linewidth=1.8,
                linestyle="--",
                alpha=0.6,
                zorder=4,
            )

        pareto_cx = np.mean(nf_costs[pareto_idx])
        pareto_cy = np.min(nf_lats[pareto_idx])
        ax.annotate(
            "myopic Pareto\n(high SLA viol.)",
            xy=(pareto_cx, pareto_cy),
            xytext=(pareto_cx + 0.012, pareto_cy - 60),
            fontsize=14,
            fontstyle="italic",
            color=COLORS["no_future"],
            arrowprops=dict(arrowstyle="->", color=COLORS["no_future"], lw=1.2, alpha=0.7),
            zorder=6,
        )

    if "full" in inference_data:
        full = inference_data["full"]
        full_costs = full["costs"]
        full_lats = full["latencies"]
        best_idx = np.argmin(full_costs + full_lats / 1000.0)
        ax.annotate(
            "best scalarized",
            xy=(full_costs[best_idx], full_lats[best_idx]),
            xytext=(full_costs[best_idx] + 0.012, full_lats[best_idx] + 50),
            fontsize=14,
            fontstyle="italic",
            color=COLORS["full"],
            arrowprops=dict(arrowstyle="->", color=COLORS["full"], lw=1.2, alpha=0.7),
            zorder=6,
        )

    ax.set_xlabel("Cost ($)")
    ax.set_ylabel("Latency (ms)")
    ax.legend(loc="upper left", fontsize=14, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.margins(x=0.16, y=0.14)

    seed_note = f"Each marker = one independent run ({n_seeds} seeds per variant)."
    ax.text(
        0.98,
        0.02,
        seed_note,
        transform=ax.transAxes,
        fontsize=11,
        ha="right",
        va="bottom",
        fontstyle="italic",
        color="#555555",
    )

    fig.tight_layout(pad=0.5)
    fig.savefig(OUTPUT_PATH, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
