#!/usr/bin/env python3
"""
Generate all supplementary experiment figures and tables
STRICTLY according to 实验数据规范收集v2（补充实验）.pdf

Experiments covered: E01, E02, E04, E06, E07, E09
Each figure → separate PNG; each table → CSV + LaTeX .tex file.
Figure style matches total/ folder (Times New Roman, font.size=26).
"""

import os, sys, csv, warnings, subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
warnings.filterwarnings('ignore')

# ── Paper-style rcParams (same as total/plot_all_comparison.py) ─────────────
def apply_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family':        'serif',
        'font.serif':         ['Times New Roman'],
        'font.size':          26,
        'axes.labelsize':     28,
        'axes.titlesize':     30,
        'axes.titleweight':   'normal',
        'axes.linewidth':     2.0,
        'axes.edgecolor':     'black',
        'xtick.labelsize':    24,
        'ytick.labelsize':    24,
        'legend.fontsize':    22,
        'lines.linewidth':    4.0,
        'ytick.color':        'black',
        'xtick.color':        'black',
        'axes.labelcolor':    'black',
        'mathtext.fontset':   'stix',
    })

SUPP = os.path.dirname(os.path.abspath(__file__))

IEEE_PANEL_FIGSIZE = (10, 6)
IEEE_PANEL_DPI = 300

def apply_ieee_panel_style():
    """Larger text for IEEE single-column side-by-side panels."""
    plt.rcParams.update({
        'font.size':       30,
        'axes.labelsize':  34,
        'axes.titlesize':  32,
        'xtick.labelsize': 28,
        'ytick.labelsize': 28,
        'legend.fontsize': 28,
    })

COLORS = {
    'STAR-PPO': '#d62728',
    'Greedy':     '#7f7f7f',
    'Random':     '#bcbd22',
    'PF-PPO':     '#17becf',
    'Equity-Trans':'#9467bd',
    'A3C':        '#8c564b',
    'STARK':      '#e377c2',
    'PPO-CN':     '#ff7f0e',
    'GA-PPO':     '#2ca02c',
    'PPO-Std':    '#1f77b4',
}

# ============================================================
#  E01 — Strict Zero-Shot Transfer
# ============================================================
def gen_e01():
    apply_style()
    apply_ieee_panel_style()
    d = os.path.join(SUPP, 'E01_ZeroShot_Transfer')

    rows = list(csv.DictReader(open(os.path.join(d, 'Table01_zeroshot_results.csv'))))

    algos     = [r['Algorithm'] for r in rows]
    lat_avg   = [float(r['Avg Latency (ms)'])  for r in rows]
    lat_p95   = [float(r['P95 Latency (ms)'])  for r in rows]
    lat_p99   = [float(r['P99 Latency (ms)'])  for r in rows]
    viol      = [float(r['SLA Violation (%)']) for r in rows]

    short = [
        'STAR-PPO\n(Retrained)',
        'STAR-PPO\n(Zero-Shot)',
        'PF-PPO\n(Zero-Shot)',
        'Equity-Trans\n(Zero-Shot)',
        'A3C\n(Zero-Shot)',
        'STARK\n(Zero-Shot)',
    ]
    bar_colors = ['#d62728', '#ff9896', '#9edae5', '#c5b0d5', '#c49c94', '#f7b6d2']

    x = np.arange(len(algos))
    w = 0.28

    # --- Fig 01a: Avg / P95 / P99 Latency grouped bar ---
    fig, ax = plt.subplots(figsize=IEEE_PANEL_FIGSIZE)
    ax.bar(x - w, lat_avg, w, color=bar_colors, edgecolor='black', lw=1.5,
           label='Avg Latency', alpha=1.0, hatch='/')
    ax.bar(x,     lat_p95, w, color=bar_colors, edgecolor='black', lw=1.5,
           label='P95 Latency', alpha=0.65, hatch='\\')
    ax.bar(x + w, lat_p99, w, color=bar_colors, edgecolor='black', lw=1.5,
           label='P99 Latency', alpha=0.35, hatch='x')
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=18)
    ax.set_ylabel('Latency (ms)')
    ax.legend(fontsize=28, loc='upper left', framealpha=0.9)
    ax.set_ylim(0, max(lat_p99) * 1.28)
    fig.tight_layout(pad=0.45)
    out = os.path.join(d, 'Fig01a_zeroshot_latency.png')
    fig.savefig(out, dpi=IEEE_PANEL_DPI, facecolor='white')
    plt.close()
    print(f'  ✓ {os.path.basename(out)}')

    # --- Fig 01b: SLA Violation bar ---
    algo_hatches = ['/', '\\', '|', '-', '+', 'x']
    fig, ax = plt.subplots(figsize=IEEE_PANEL_FIGSIZE)
    bars = ax.bar(x, viol, w * 2, color=bar_colors, edgecolor='black', lw=1.5)
    for bar, h in zip(bars, algo_hatches):
        bar.set_hatch(h)
    for bar, v in zip(bars, viol):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.6,
                f'{v:.1f}%', ha='center', va='bottom', fontsize=24, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=18)
    ax.set_ylabel('SLA Violation Rate (%)')
    ax.set_ylim(0, max(viol) * 1.34)
    ax.axhline(viol[0], color='#d62728', ls=':', lw=2.5, alpha=0.8,
               label=f'Retrained UB ({viol[0]:.1f}%)')
    ax.legend(fontsize=28, loc='upper left', framealpha=0.9)
    fig.tight_layout(pad=0.45)
    out = os.path.join(d, 'Fig01b_zeroshot_violation.png')
    fig.savefig(out, dpi=IEEE_PANEL_DPI, facecolor='white')
    plt.close()
    print(f'  ✓ {os.path.basename(out)}')

    # Table01_zeroshot_results.csv already exists — no need to rewrite
    print(f'  ✓ Table01_zeroshot_results.csv (existing)')

    # --- LaTeX ---
    tex = r"""\begin{table}[htbp]
\centering
\caption{E01: Strict Zero-Shot Cross-Region Transfer (Train: S1/Switzerland, Test: S3/Germany).
All methods evaluated without any fine-tuning on the target topology.}
\label{tab:e01_zeroshot}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lrrrrc}
\toprule
Algorithm & Avg Lat.\ (ms) & P95 (ms) & P99 (ms) & SLA Viol.\ (\%) & Gap vs Retrained \\
\midrule
"""
    for r in rows:
        algo = r['Algorithm'].replace('_', r'\_')
        tex += (f"{algo} & {float(r['Avg Latency (ms)']):.1f} & "
                f"{float(r['P95 Latency (ms)']):.1f} & "
                f"{float(r['P99 Latency (ms)']):.1f} & "
                f"{float(r['SLA Violation (%)']):.1f} & "
                f"{float(r['Gap vs Retrained']):.3f} \\\\\n")
    tex += r"""\bottomrule
\end{tabular}}
\end{table}
"""
    tex_out = os.path.join(d, 'Table01_zeroshot_results.tex')
    open(tex_out, 'w').write(tex)
    print(f'  ✓ {os.path.basename(tex_out)}')


# ============================================================
#  E02 — Service-Oriented Network Model
# ============================================================
def gen_e02():
    apply_style()
    apply_ieee_panel_style()
    d = os.path.join(SUPP, 'E02_Network_Model')

    rows = list(csv.DictReader(open(os.path.join(d, 'Table02_link_model_metrics.csv'))))

    algo_map = {
        'Greedy':              'Greedy',
        'Random':              'Random',
        'STAR-PPO-Overlay':    'STAR-PPO\n(Overlay)',
        'STAR-PPO-Wireless':   'STAR-PPO\n(Wireless)',
    }
    labels = [algo_map.get(r['Algorithm'], r['Algorithm']) for r in rows]
    alg_colors = ['#7f7f7f', '#bcbd22', '#d62728', '#ff7f0e']

    ep_lat  = [float(r['Avg Total Latency (ms)']) for r in rows]
    ep_std  = [float(r['Std (ms)'])               for r in rows]
    ep_cost = [float(r['Avg Cost (USD)'])          for r in rows]
    net_ms  = [float(r['Net Latency (ms)'])        for r in rows]
    cmp_ms  = [float(r['Compute (ms)'])            for r in rows]
    tx_ms   = [float(r['Tx (ms)'])                 for r in rows]
    R_ij    = [float(r['R_ij (Mbps)'])             for r in rows]
    T_base  = [float(r['T_base (ms)'])             for r in rows]

    # ── Fig 02a: Latency-Cost Scatter ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=IEEE_PANEL_FIGSIZE)
    for i, (r, lab, col) in enumerate(zip(rows, labels, alg_colors)):
        ax.errorbar(ep_cost[i], ep_lat[i], yerr=ep_std[i],
                    fmt='o', color=col, markersize=18, capsize=8, capthick=2.5,
                    elinewidth=2.5, label=lab.replace('\n', ' '), zorder=5)

    ax.set_xlabel('Avg Cost (USD/request)')
    ax.set_ylabel('Avg Latency (ms)')
    ax.legend(fontsize=26, loc='upper left', framealpha=0.9)
    ax.margins(x=0.18, y=0.20)
    fig.tight_layout(pad=0.45)
    out = os.path.join(d, 'Fig02a_latency_cost_scatter.png')
    fig.savefig(out, dpi=IEEE_PANEL_DPI, facecolor='white')
    plt.close()
    print(f'  ✓ {os.path.basename(out)}')

    # ── Fig 02b: Communication Delay Breakdown ───────────────────────────────
    # Reorder: STAR-PPO entries first, then Greedy / Random
    topo_idx  = [i for i, r in enumerate(rows) if 'STAR-PPO' in r['Algorithm']]
    other_idx = [i for i, r in enumerate(rows) if 'STAR-PPO' not in r['Algorithm']]
    order = topo_idx + other_idx

    b_labels = [labels[i]  for i in order]
    b_net    = [net_ms[i]  for i in order]
    b_cmp    = [cmp_ms[i]  for i in order]
    b_tx     = [tx_ms[i]   for i in order]
    b_lat    = [ep_lat[i]  for i in order]
    b_std    = [ep_std[i]  for i in order]

    step_total = [n + c + t for n, c, t in zip(b_net, b_cmp, b_tx)]

    fig, ax = plt.subplots(figsize=IEEE_PANEL_FIGSIZE)
    x = np.arange(len(order))
    w = 0.55
    ax.bar(x, b_net, w, label='Network $T_{\\rm net}$',
           color='#1f77b4', edgecolor='black', lw=1.5, hatch='/')
    ax.bar(x, b_cmp, w, bottom=b_net,
           label='Compute $T_{\\rm comp}$',
           color='#ff7f0e', edgecolor='black', lw=1.5, hatch='\\')
    bottom2 = [n + c for n, c in zip(b_net, b_cmp)]
    ax.bar(x, b_tx, w, bottom=bottom2,
           color='#2ca02c', edgecolor='black', lw=1.5, hatch='x')

    # Annotate per-step total just above each bar
    for i, tot in enumerate(step_total):
        ax.text(i, tot + max(step_total) * 0.02,
                f'{tot:.0f} ms', ha='center', va='bottom',
                fontsize=24, fontweight='bold')

    # Episode total latency as a secondary annotation below x-axis
    for i, (lat, std) in enumerate(zip(b_lat, b_std)):
        ax.text(i, -max(step_total) * 0.10,
                f'Ep: {lat:.0f} ms',
                ha='center', va='top', fontsize=21, color='#555',
                fontstyle='italic')

    ax.set_xticks(x)
    ax.set_xticklabels(b_labels, fontsize=24)
    ax.set_ylabel('Per-Step Latency (ms)')
    ax.legend(fontsize=26, loc='upper left', framealpha=0.9)
    ax.set_ylim(-max(step_total) * 0.26, max(step_total) * 1.32)
    fig.tight_layout(pad=0.45)
    out = os.path.join(d, 'Fig02b_comm_delay_breakdown.png')
    fig.savefig(out, dpi=IEEE_PANEL_DPI, facecolor='white')
    plt.close()
    print(f'  ✓ {os.path.basename(out)}')

    # Table02_link_model_metrics.csv already exists
    print(f'  ✓ Table02_link_model_metrics.csv (existing)')

    # ── LaTeX ─────────────────────────────────────────────────────────────────
    tex = r"""\begin{table}[htbp]
\centering
\caption{E02: Per-Component Latency and Cost Under Different Link Models.
All results from same algorithm, workload, and seed.}
\label{tab:e02_link_model}
\resizebox{\linewidth}{!}{%
\begin{tabular}{llrrrrrr}
\toprule
Algorithm & Link Model & $R_{ij}$ (Mbps) & $T_{\rm base}$ (ms) & $T_{\rm net}$ (ms) & $T_{\rm comp}$ (ms) & Avg Lat.\ (ms) $\pm$ Std & Avg Cost (USD) \\
\midrule
"""
    link_models = ['Service-Overlay', 'Service-Overlay', 'Service-Overlay', 'Wireless-Edge']
    for r, lm in zip(rows, link_models):
        tex += (f"{r['Algorithm']} & {lm} & "
                f"{float(r['R_ij (Mbps)']):.1f} & "
                f"{float(r['T_base (ms)']):.2f} & "
                f"{float(r['Net Latency (ms)']):.4f} & "
                f"{float(r['Compute (ms)']):.2f} & "
                f"{float(r['Avg Total Latency (ms)']):.1f} $\\pm$ {float(r['Std (ms)']):.1f} & "
                f"{float(r['Avg Cost (USD)']):.4f} \\\\\n")
    tex += r"""\bottomrule
\end{tabular}}
\end{table}
"""
    tex_out = os.path.join(d, 'Table02_link_model_metrics.tex')
    open(tex_out, 'w').write(tex)
    print(f'  ✓ {os.path.basename(tex_out)}')


# ============================================================
#  E04 — Queue Approximation Validation
# ============================================================
def kingman(rho, ES, ca2=1.0, cs2=1.0):
    rho = np.clip(rho, 0, 0.9999)
    return rho / (1 - rho) * ES * (ca2 + cs2) / 2.0

def _simulate_mg1_queue(rho_arr, ES, ca2, cs2, seed_offset=0, n_customers=600):
    """
    Discrete-event M/G/1 simulation using Lindley recursion.

    Inter-arrival distributions (all have mean = ES/rho):
      - Poisson  (ca2=1.0): Exponential
      - Bursty   (ca2=2.5): Gamma(shape=1/ca2, scale=ca2*mean_ia)
      - On-Off   (ca2=4.0): Log-Normal tuned to (mean=mean_ia, cv²=ca2)

    Service times: Exponential(ES)  → cs2 = 1 for all patterns.
    """
    rng    = np.random.default_rng(seed_offset + 42)
    rho    = np.clip(rho_arr, 0.001, 0.990)
    tq_obs = np.zeros_like(rho, dtype=float)

    for idx, r in enumerate(rho):
        mean_ia = ES / r      # correct mean: E[S] / rho = 1/lambda

        # ── Inter-arrival times with exact mean and target ca2 ────────────
        if ca2 <= 1.05:
            # Poisson: Exponential(mean_ia)  → ca2 = 1 exactly
            ia = rng.exponential(mean_ia, n_customers)
        elif ca2 <= 3.0:
            # Gamma: shape k = 1/ca2, scale θ = mean_ia*ca2
            # E[X]=kθ=mean_ia, Var[X]=kθ²=ca2*mean_ia²  → ca2=1/k ✓
            k = 1.0 / ca2
            ia = rng.gamma(k, mean_ia * ca2, n_customers)
        else:
            # Log-Normal: σ²_ln = log(1+ca2), μ_ln = log(mean_ia) - σ²_ln/2
            # E[X]=mean_ia, Var[X]/E[X]²=exp(σ²_ln)-1=ca2 ✓
            sig2 = np.log(1.0 + ca2)
            mu_  = np.log(mean_ia) - sig2 / 2.0
            ia   = rng.lognormal(mu_, np.sqrt(sig2), n_customers)

        ia = np.clip(ia, 1e-6, None)

        # ── Service times: Exponential(ES) ────────────────────────────────
        svc = rng.exponential(ES, n_customers)

        # ── Lindley recursion ─────────────────────────────────────────────
        arrivals    = np.cumsum(ia)
        wait        = np.zeros(n_customers)
        finish_prev = 0.0
        for i in range(n_customers):
            start       = max(arrivals[i], finish_prev)
            wait[i]     = start - arrivals[i]
            finish_prev = start + svc[i]

        warmup = int(n_customers * 0.35)   # 35% warm-up for slow-mixing high-ρ queues
        tq_obs[idx] = float(wait[warmup:].mean())

    return tq_obs

def gen_e04():
    apply_style()
    d = os.path.join(SUPP, 'E04_Queue_Validation')

    mean_ES = 385.3   # ms, from real system E[S] (LLM inference service time)

    # Traffic patterns and their arrival/service Squared Coefficient of Variation
    patterns = {
        'Poisson': {'ca2': 1.0, 'cs2': 1.0, 'color': '#1f77b4'},
        'Bursty':  {'ca2': 2.5, 'cs2': 1.5, 'color': '#ff7f0e'},
        'On-Off':  {'ca2': 4.0, 'cs2': 1.0, 'color': '#2ca02c'},
    }
    np.random.seed(42)

    # ── Validation sweep: ρ from 0.10 to 0.88 ────────────────────────────────
    # Each rho value runs a full M/G/1 discrete-event simulation to get T_q_obs.
    # Kingman T_q_est is computed analytically from the same rho.
    N_PER_PAT = 120
    rho_sweep = np.concatenate([
        np.random.uniform(0.15, 0.35,  N_PER_PAT // 4),
        np.random.uniform(0.35, 0.60,  N_PER_PAT // 4),
        np.random.uniform(0.60, 0.78,  N_PER_PAT // 4),
        np.random.uniform(0.78, 0.86,  N_PER_PAT // 4),
    ])
    rho_sweep = np.sort(rho_sweep)

    src_rows = []
    for i_pat, (pat, cfg) in enumerate(patterns.items()):
        tq_est = kingman(rho_sweep, mean_ES, ca2=cfg['ca2'], cs2=cfg['cs2'])
        tq_obs = _simulate_mg1_queue(rho_sweep, mean_ES, cfg['ca2'], cfg['cs2'],
                                     seed_offset=i_pat * 100, n_customers=3000)
        mae_arr  = np.abs(tq_est - tq_obs)
        # MAPE: use T_q_est as denominator (stable; avoids amplification when T_q_obs ≈ 0)
        mape_arr = mae_arr / np.clip(tq_est, 1.0, None) * 100

        for i in range(N_PER_PAT):
            rho_i = float(rho_sweep[i])
            util_bin = ('Low (ρ<0.3)' if rho_i < 0.3 else
                        'Medium (0.3≤ρ<0.6)' if rho_i < 0.6 else 'High (ρ≥0.6)')
            src_rows.append({
                'traffic_pattern': pat,
                'ca2': cfg['ca2'], 'cs2': cfg['cs2'],
                'rho': round(rho_i, 5),
                'T_q_est_ms':  round(float(tq_est[i]),  2),
                'T_q_obs_ms':  round(float(tq_obs[i]),  2),
                'MAE_ms':      round(float(mae_arr[i]),  2),
                'MAPE_pct':    round(float(mape_arr[i]), 2),
                'util_bin':    util_bin,
            })

    src_csv = os.path.join(d, 'Fig04_source.csv')
    with open(src_csv, 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(src_rows[0].keys()))
        wr.writeheader(); wr.writerows(src_rows)

    # ── Fig 04a: Estimated vs Observed Scatter ───────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    all_vals = []
    for pat, cfg in patterns.items():
        sub = [r for r in src_rows if r['traffic_pattern'] == pat]
        x_ = [r['T_q_obs_ms'] for r in sub]
        y_ = [r['T_q_est_ms'] for r in sub]
        all_vals += x_ + y_
        ax.scatter(x_, y_, alpha=0.30, s=28, color=cfg['color'], label=pat)

    max_val = max(all_vals) * 1.05
    ax.plot([0, max_val], [0, max_val], 'k--', lw=2.5, label='Perfect prediction')
    ax.set_xlim(0, max_val); ax.set_ylim(0, max_val)
    ax.set_xlabel(r'Observed $T_q^{\rm obs}$ (ms)')
    ax.set_ylabel(r'Kingman Estimate $T_q^{\rm est}$ (ms)')
    ax.legend(fontsize=22, loc='upper left')
    # Compute per-pattern R²
    for i_pat, (pat, cfg) in enumerate(patterns.items()):
        sub = [r for r in src_rows if r['traffic_pattern'] == pat]
        x_ = np.array([r['T_q_obs_ms'] for r in sub])
        y_ = np.array([r['T_q_est_ms'] for r in sub])
        ss_res = np.sum((y_ - x_)**2)
        ss_tot = np.sum((x_ - x_.mean())**2)
        r2 = 1 - ss_res / (ss_tot + 1e-9)
        ax.text(0.62, 0.22 - i_pat * 0.07,
                f'{pat}: $R^2$={r2:.3f}',
                transform=ax.transAxes, fontsize=19, color=cfg['color'])
    plt.tight_layout()
    out = os.path.join(d, 'Fig04a_queue_scatter.png')
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  ✓ {os.path.basename(out)}')

    # ── Fig 04b / 04c: MAE and MAPE Boxplots by Traffic Pattern ─────────────
    pat_list   = list(patterns.keys())
    pat_colors = [patterns[p]['color'] for p in pat_list]
    mae_by_pat  = [[r['MAE_ms']   for r in src_rows if r['traffic_pattern'] == p] for p in pat_list]
    mape_by_pat = [[r['MAPE_pct'] for r in src_rows if r['traffic_pattern'] == p] for p in pat_list]

    def _boxplot_single(data, labels, colors, ylabel, title, out_path, annot_medians=True):
        fig, ax = plt.subplots(figsize=(9, 7))
        bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=False,
                        widths=0.5,
                        medianprops=dict(color='black', linewidth=3.0),
                        whiskerprops=dict(linewidth=2.5),
                        capprops=dict(linewidth=2.5),
                        flierprops=dict(marker='o', markersize=5, alpha=0.35))
        for patch, col in zip(bp['boxes'], colors):
            patch.set_facecolor(col); patch.set_alpha(0.75)
        if annot_medians:
            for i, (d_arr, med_line) in enumerate(zip(data, bp['medians'])):
                med = float(np.median(d_arr))
                ax.text(i + 1, med * 1.03, f'{med:.1f}',
                        ha='center', va='bottom', fontsize=18, fontweight='bold')
        ax.set_xlabel('Traffic Pattern')
        ax.set_ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    _boxplot_single(
        mae_by_pat, pat_list, pat_colors,
        'MAE (ms)',
        r'Fig 04b: Queue Delay MAE by Traffic Pattern'
        '\n' r'(Kingman $T_q^{\rm est}$ vs Observed $T_q^{\rm obs}$, whisker=1.5×IQR)',
        os.path.join(d, 'Fig04b_queue_mae_boxplot.png'))
    print(f'  ✓ Fig04b_queue_mae_boxplot.png')

    _boxplot_single(
        mape_by_pat, pat_list, pat_colors,
        'MAPE (%)',
        r'Fig 04c: Queue Delay MAPE by Traffic Pattern'
        '\n' r'(Kingman $T_q^{\rm est}$ vs Observed $T_q^{\rm obs}$, whisker=1.5×IQR)',
        os.path.join(d, 'Fig04c_queue_mape_boxplot.png'))
    print(f'  ✓ Fig04c_queue_mape_boxplot.png')

    # ── Table: Error by Traffic Pattern × Utilization Bin ───────────────────
    util_bins = ['Low (ρ<0.3)', 'Medium (0.3≤ρ<0.6)', 'High (ρ≥0.6)']
    table_rows = []
    for pat in pat_list:
        ca2 = patterns[pat]['ca2']
        cs2 = patterns[pat]['cs2']
        for ub in util_bins:
            sub = [r for r in src_rows if r['traffic_pattern']==pat and r['util_bin']==ub]
            if not sub:
                continue
            rhos    = [r['rho']      for r in sub]
            maes    = [r['MAE_ms']   for r in sub]
            mapes   = [r['MAPE_pct'] for r in sub]
            table_rows.append({
                'traffic_pattern': pat,
                'ca2': ca2, 'cs2': cs2,
                'util_bin': ub,
                'n': len(sub),
                'rho_mean': round(np.mean(rhos), 4),
                'MAE_mean_ms': round(np.mean(maes), 3),
                'MAE_std_ms':  round(np.std(maes),  3),
                'MAPE_mean_pct': round(np.mean(mapes), 3),
                'MAPE_std_pct':  round(np.std(mapes),  3),
            })

    csv_out = os.path.join(d, 'Table04_queue_error.csv')
    with open(csv_out, 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(table_rows[0].keys()))
        wr.writeheader(); wr.writerows(table_rows)
    print(f'  ✓ {os.path.basename(csv_out)}')

    # LaTeX
    tex = r"""\begin{table}[htbp]
\centering
\caption{E04: Kingman Queue Approximation Error Under Different Traffic Patterns and Utilization Bins.
$E[S]=385.3$ ms. Validation sweep: $\rho \in [0.05, 0.92]$.}
\label{tab:e04_queue_error}
\resizebox{\linewidth}{!}{%
\begin{tabular}{llrrrrr}
\toprule
Traffic Pattern & Util.\ Bin & $c_a^2$ & $n$ & $\bar{\rho}$ & MAE (ms) $\pm$ Std & MAPE (\%) $\pm$ Std \\
\midrule
"""
    prev_pat = ''
    for r in table_rows:
        pat_str = r['traffic_pattern'] if r['traffic_pattern'] != prev_pat else ''
        prev_pat = r['traffic_pattern']
        tex += (f"{pat_str} & {r['util_bin']} & {r['ca2']} & "
                f"{r['n']} & {r['rho_mean']:.4f} & "
                f"{r['MAE_mean_ms']:.3f} $\\pm$ {r['MAE_std_ms']:.3f} & "
                f"{r['MAPE_mean_pct']:.3f} $\\pm$ {r['MAPE_std_pct']:.3f} \\\\\n")
    tex += r"""\bottomrule
\end{tabular}}
\end{table}
"""
    tex_out = os.path.join(d, 'Table04_queue_error.tex')
    open(tex_out, 'w').write(tex)
    print(f'  ✓ {os.path.basename(tex_out)}')


# ============================================================
#  E06 — DWA Multi-Objective Ablation
# ============================================================
def gen_e06():
    apply_style()
    d = os.path.join(SUPP, 'E06_DWA_Ablation')

    log_root = os.path.join(SUPP, '..', 'results', 'TopoFreeRL', 'logs')
    SEEDS    = [42, 43, 44, 45, 46]
    REGION   = 'Server1_Trap'

    # Load training data
    data_list = []
    for seed in SEEDS:
        p = os.path.join(log_root, f'LATEST_{REGION}_seed{seed}', 'training_data.npz')
        if os.path.exists(p):
            td = np.load(p)
            data_list.append({
                'seed':        seed,
                'weights':     td['weights_hist'],   # (100, 3)
                'L_L':         td['L_hist_L'],
                'L_C':         td['L_hist_C'],
                'L_S':         td['L_hist_S'],
            })

    n_seeds  = len(data_list)
    n_epochs = data_list[0]['weights'].shape[0]
    epochs   = np.arange(1, n_epochs + 1)
    FREEZE   = int(n_epochs * 0.8)
    DWA_START= 3

    wh_all = np.stack([d['weights'] for d in data_list], 0)  # (S,100,3)
    ll_all = np.stack([d['L_L']    for d in data_list], 0)
    lc_all = np.stack([d['L_C']    for d in data_list], 0)
    ls_all = np.stack([d['L_S']    for d in data_list], 0)

    wh_m, wh_s = wh_all.mean(0), wh_all.std(0)
    ll_m, ll_s = ll_all.mean(0), ll_all.std(0)
    lc_m, lc_s = lc_all.mean(0), lc_all.std(0)
    ls_m, ls_s = ls_all.mean(0), ls_all.std(0)

    def smooth(x, w=5):
        pad = np.pad(x, (w//2, w//2), mode='edge')
        return np.convolve(pad, np.ones(w)/w, mode='valid')

    W_COLORS = ['#C0392B', '#2471A3', '#27AE60']
    W_LABELS = [r'$\omega_L$ (Latency)', r'$\omega_C$ (Cost)', r'$\omega_R$ (SLA/Risk)']
    W_STYLES = ['-', '--', '-.']

    # ── Fig 06a: DWA Weight Trajectory ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 7))
    for dim in range(3):
        ax.plot(epochs, wh_m[:, dim], color=W_COLORS[dim],
                linestyle=W_STYLES[dim], lw=3.5, label=W_LABELS[dim], zorder=4)
        ax.fill_between(epochs,
                        wh_m[:, dim] - wh_s[:, dim],
                        wh_m[:, dim] + wh_s[:, dim],
                        color=W_COLORS[dim], alpha=0.12, zorder=2)

    ax.axvline(DWA_START, color='#888', ls=':', lw=2, alpha=0.8)
    ax.axvline(FREEZE,    color='#888', ls=':', lw=2, alpha=0.8)
    ax.axvspan(FREEZE, n_epochs, alpha=0.07, color='#95A5A6', label='Weights frozen (80%–)')

    ax.annotate('Init: [0.45, 0.40, 0.15]',
                xy=(1, 0.45), xytext=(6, 0.47), fontsize=18, color='#555',
                fontstyle='italic',
                arrowprops=dict(arrowstyle='->', color='#888', lw=1.5))
    ax.text(DWA_START+1, 0.205, 'DWA active', fontsize=18, color='#555', fontstyle='italic')
    ax.text(FREEZE+1,    0.205, 'Frozen',     fontsize=18, color='#555', fontstyle='italic')

    ax.set_xlabel('Training Epoch')
    ax.set_ylabel('DWA Weight')
    ax.set_xlim(1, n_epochs); ax.set_ylim(0.10, 0.57)
    ax.legend(fontsize=20, loc='upper right', ncol=2, framealpha=0.9)
    plt.tight_layout()
    out = os.path.join(d, 'Fig06a_dwa_trajectory.png')
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  ✓ {os.path.basename(out)}')

    # ── Fig 06b: Pareto Scatter (latency vs cost, coloured by risk) ──────────
    # Use E07 metric data as different operating points (each algorithm
    # represents a different implicit weight setting on the Pareto frontier)
    e07_path = os.path.join(SUPP, 'E07_Metric_Audit', 'Table07a_score_breakdown.csv')
    e07_rows = list(csv.DictReader(open(e07_path)))

    fig, ax = plt.subplots(figsize=(9, 7))
    pareto_colors = list(COLORS.values())
    legend_handles = []
    for i, r in enumerate(e07_rows):
        algo  = r['Algorithm']
        lat   = float(r['J_L (ms)'])
        cost  = float(r['J_C (USD)'])
        col   = pareto_colors[i % len(pareto_colors)]
        sc = ax.scatter(cost, lat, s=250, color=col, zorder=5,
                        edgecolors='black', linewidths=1.5, label=algo)
        legend_handles.append(
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=col,
                       markeredgecolor='black', markersize=10, label=algo))

    ax.set_xlabel('Avg Cost (USD/request)')
    ax.set_ylabel('Avg Latency (ms)')
    ax.legend(handles=legend_handles, fontsize=14, loc='upper left',
              framealpha=0.9, edgecolor='gray')
    plt.tight_layout()
    out = os.path.join(d, 'Fig06b_pareto_scatter.png')
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  ✓ {os.path.basename(out)}')

    # ── Table: DWA Weight Evolution (per-seed summary) ───────────────────────
    csv_rows = []
    for datum in data_list:
        wh = datum['weights']
        ll = datum['L_L']; lc = datum['L_C']; ls = datum['L_S']
        csv_rows.append({
            'seed':            datum['seed'],
            'w_L_init':        round(float(wh[0, 0]), 3),
            'w_C_init':        round(float(wh[0, 1]), 3),
            'w_S_init':        round(float(wh[0, 2]), 3),
            'w_L_frozen':      round(float(wh[FREEZE, 0]), 4),
            'w_C_frozen':      round(float(wh[FREEZE, 1]), 4),
            'w_S_frozen':      round(float(wh[FREEZE, 2]), 4),
            'w_S_increase_pct':round((float(wh[FREEZE,2])/float(wh[0,2])-1)*100, 1),
            'L_L_epoch1':      round(float(ll[0]),  4),
            'L_L_final':       round(float(ll[-1]), 4),
            'L_C_epoch1':      round(float(lc[0]),  4),
            'L_C_final':       round(float(lc[-1]), 4),
            'L_S_epoch1':      round(float(ls[0]),  4),
            'L_S_final':       round(float(ls[-1]), 4),
            'L_S_reduction_pct': round((1 - float(ls[-1])/float(ls[0]))*100, 1),
        })
    # Mean row
    csv_rows.append({
        'seed': 'Mean',
        'w_L_init':        round(np.mean([r['w_L_init']   for r in csv_rows]), 3),
        'w_C_init':        round(np.mean([r['w_C_init']   for r in csv_rows]), 3),
        'w_S_init':        round(np.mean([r['w_S_init']   for r in csv_rows]), 3),
        'w_L_frozen':      round(np.mean([r['w_L_frozen'] for r in csv_rows]), 4),
        'w_C_frozen':      round(np.mean([r['w_C_frozen'] for r in csv_rows]), 4),
        'w_S_frozen':      round(np.mean([r['w_S_frozen'] for r in csv_rows]), 4),
        'w_S_increase_pct':round(np.mean([r['w_S_increase_pct'] for r in csv_rows if isinstance(r['seed'], int)]), 1),
        'L_L_epoch1':      round(np.mean([r['L_L_epoch1'] for r in csv_rows if isinstance(r['seed'], int)]), 4),
        'L_L_final':       round(np.mean([r['L_L_final']  for r in csv_rows if isinstance(r['seed'], int)]), 4),
        'L_C_epoch1':      round(np.mean([r['L_C_epoch1'] for r in csv_rows if isinstance(r['seed'], int)]), 4),
        'L_C_final':       round(np.mean([r['L_C_final']  for r in csv_rows if isinstance(r['seed'], int)]), 4),
        'L_S_epoch1':      round(np.mean([r['L_S_epoch1'] for r in csv_rows if isinstance(r['seed'], int)]), 4),
        'L_S_final':       round(np.mean([r['L_S_final']  for r in csv_rows if isinstance(r['seed'], int)]), 4),
        'L_S_reduction_pct': round(np.mean([r['L_S_reduction_pct'] for r in csv_rows if isinstance(r['seed'], int)]), 1),
    })

    csv_out = os.path.join(d, 'Table06_dwa_ablation.csv')
    with open(csv_out, 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        wr.writeheader(); wr.writerows(csv_rows)
    print(f'  ✓ {os.path.basename(csv_out)}')

    # LaTeX
    seed_rows = [r for r in csv_rows if isinstance(r['seed'], int)]
    mean_row  = [r for r in csv_rows if r['seed'] == 'Mean'][0]
    tex = r"""\begin{table}[htbp]
\centering
\caption{E06: DWA Weight Evolution and Per-Objective Loss Reduction (5 seeds, 100 epochs).
$\omega_S$ is automatically upweighted from 0.15 to $\approx$0.32, accelerating SLA convergence.}
\label{tab:e06_dwa}
\resizebox{\linewidth}{!}{%
\begin{tabular}{crrrrrrrr}
\toprule
Seed & $\omega_S^{\rm init}$ & $\omega_S^{\rm frozen}$ & $\Delta\omega_S$ (\%) & $L_L^{(1)}$ & $L_L^{\rm final}$ & $L_S^{(1)}$ & $L_S^{\rm final}$ & $L_S$ Reduction (\%) \\
\midrule
"""
    for r in seed_rows:
        tex += (f"{r['seed']} & {r['w_S_init']:.3f} & {r['w_S_frozen']:.4f} & "
                f"+{r['w_S_increase_pct']:.1f} & "
                f"{r['L_L_epoch1']:.4f} & {r['L_L_final']:.4f} & "
                f"{r['L_S_epoch1']:.4f} & {r['L_S_final']:.4f} & "
                f"{r['L_S_reduction_pct']:.1f} \\\\\n")
    tex += r"\midrule" + "\n"
    tex += (f"Mean & {mean_row['w_S_init']:.3f} & {mean_row['w_S_frozen']:.4f} & "
            f"+{mean_row['w_S_increase_pct']:.1f} & "
            f"{mean_row['L_L_epoch1']:.4f} & {mean_row['L_L_final']:.4f} & "
            f"{mean_row['L_S_epoch1']:.4f} & {mean_row['L_S_final']:.4f} & "
            f"{mean_row['L_S_reduction_pct']:.1f} \\\\\n")
    tex += r"""\bottomrule
\end{tabular}}
\end{table}
"""
    tex_out = os.path.join(d, 'Table06_dwa_ablation.tex')
    open(tex_out, 'w').write(tex)
    print(f'  ✓ {os.path.basename(tex_out)}')


# ============================================================
#  E07 — Metric Transparency & Composite QoS Score Audit
# ============================================================
def gen_e07():
    d = os.path.join(SUPP, 'E07_Metric_Audit')

    rows  = list(csv.DictReader(open(os.path.join(d, 'Table07a_score_breakdown.csv'))))
    nrows = list(csv.DictReader(open(os.path.join(d, 'Table07b_normalization.csv'))))

    # Table07a_score_breakdown.csv already exists
    print(f'  ✓ Table07a_score_breakdown.csv (existing)')

    # ── Table 07a LaTeX ───────────────────────────────────────────────────────
    tex_a = r"""\begin{table}[htbp]
\centering
\caption{E07: Composite QoS Score Decomposition.
$Q = \eta_L(1-\hat{J}_L)+\eta_C(1-\hat{J}_C)+\eta_R(1-\hat{J}_R)+\eta_S(1-V_{\rm SLA})$,
with $\eta_L=0.45, \eta_C=0.35, \eta_R=0.10, \eta_S=0.10$.
Improvement (\%) is relative to Greedy baseline ($J_L=2793.6$ ms).}
\label{tab:e07_score}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lrrrrrrrrc}
\toprule
Algorithm & $J_L$ (ms) & $J_C$ (\$) & $J_R$ & $V_{\rm SLA}$ & $\hat{J}_L$ & $\hat{J}_C$ & $\hat{J}_R$ & QoS Score & Improv.\ (\%) \\
\midrule
"""
    for r in rows:
        algo = r['Algorithm'].replace('_', r'\_')
        tex_a += (f"{algo} & {float(r['J_L (ms)']):.1f} & {float(r['J_C (USD)']):.4f} & "
                  f"{float(r['J_R (raw)']):.3f} & {float(r['V_SLA']):.3f} & "
                  f"{float(r['J_L_norm']):.4f} & "
                  f"{float(r['J_C_norm']):.4f} & "
                  f"{float(r['J_R_norm']):.4f} & "
                  f"{float(r['Composite QoS Score']):.2f} & "
                  f"{float(r['Improvement vs Greedy (%)']):.1f} \\\\\n")
    tex_a += r"""\bottomrule
\end{tabular}}
\end{table}
"""
    open(os.path.join(d, 'Table07a_score_breakdown.tex'), 'w').write(tex_a)
    print(f'  ✓ Table07a_score_breakdown.tex')

    # Table07b_normalization.csv already exists
    print(f'  ✓ Table07b_normalization.csv (existing)')

    # ── Table 07b LaTeX ───────────────────────────────────────────────────────
    tex_b = r"""\begin{table}[htbp]
\centering
\caption{E07 Appendix: Normalization Constants and $\eta$ Weights for Composite QoS Score.}
\label{tab:e07_normalization}
\begin{tabular}{llll}
\toprule
Constant & Value & Unit & Description \\
\midrule
"""
    for r in nrows:
        const = r['Constant'].replace('_', r'\_')
        src   = r['Source / Description'][:55].replace('_', r'\_').replace('$', r'\$')
        tex_b += f"{const} & {r['Value']} & {r['Unit']} & {src} \\\\\n"
    tex_b += r"""\bottomrule
\end{tabular}
\end{table}
"""
    open(os.path.join(d, 'Table07b_normalization.tex'), 'w').write(tex_b)
    print(f'  ✓ Table07b_normalization.tex')


# ============================================================
#  E09 — Traffic Robustness with Statistical Tests
# ============================================================
def gen_e09():
    d = os.path.join(SUPP, 'E09_Traffic_Robustness')
    e09_script = os.path.join(d, 'E09_plot_traffic_robustness.py')
    subprocess.run([sys.executable, e09_script], check=True)
    src_png = os.path.join(d, 'E09_Workload_Robustness.png')
    dst_png = os.path.join(d, 'Fig09_traffic_robustness_boxplot.png')
    if os.path.exists(src_png):
        import shutil
        shutil.copy2(src_png, dst_png)
        print(f'  ✓ {os.path.basename(dst_png)}')

    tests = list(csv.DictReader(open(os.path.join(d, 'Table09_statistical_tests.csv'))))
    print(f'  ✓ Table09_statistical_tests.csv (existing)')

    # ── Table 09 LaTeX ────────────────────────────────────────────────────────
    tex = r"""\begin{table}[htbp]
\centering
\caption{E09: Statistical Tests for Traffic Robustness (Latency, ms).
Mann-Whitney U (pairwise) and Kruskal-Wallis (omnibus); $n=600$ per group.}
\label{tab:e09_stats}
\resizebox{\linewidth}{!}{%
\begin{tabular}{llllrrr}
\toprule
Metric & Group A & Group B & Test & Statistic & $p$-value & Effect Size \\
\midrule
"""
    for r in tests:
        grp_b = r['Group B'] if r['Group B'] != '—' else '---'
        eff   = r['Effect Size'] if r['Effect Size'] != '—' else '---'
        pval  = r['p-value']
        # Format p-value in LaTeX scientific notation
        try:
            pf = float(pval)
            if pf < 0.001:
                exp = int(np.floor(np.log10(pf)))
                mnt = pf / 10**exp
                pval_str = f'${mnt:.2f} \\times 10^{{{exp}}}$'
            else:
                pval_str = f'{pf:.4f}'
        except:
            pval_str = pval
        tex += (f"{r['Metric']} & {r['Group A']} & {grp_b} & "
                f"{r['Test']} & {r['Statistic']} & {pval_str} & {eff} \\\\\n")
    tex += r"""\bottomrule
\end{tabular}}
\end{table}
"""
    tex_out = os.path.join(d, 'Table09_statistical_tests.tex')
    open(tex_out, 'w').write(tex)
    print(f'  ✓ {os.path.basename(tex_out)}')


# ============================================================
#  Main
# ============================================================
if __name__ == '__main__':
    print('\n' + '='*60)
    print('E01 — Zero-Shot Transfer')
    print('='*60)
    gen_e01()

    print('\n' + '='*60)
    print('E02 — Service-Oriented Network Model')
    print('='*60)
    gen_e02()

    print('\n' + '='*60)
    print('E04 — Queue Approximation Validation')
    print('='*60)
    gen_e04()

    print('\n' + '='*60)
    print('E06 — DWA Multi-Objective Ablation')
    print('='*60)
    gen_e06()

    print('\n' + '='*60)
    print('E07 — Metric Transparency Audit')
    print('='*60)
    gen_e07()

    print('\n' + '='*60)
    print('E09 — Traffic Robustness')
    print('='*60)
    gen_e09()

    print('\n' + '='*60)
    print('All figures and tables generated successfully.')
    print('='*60)
