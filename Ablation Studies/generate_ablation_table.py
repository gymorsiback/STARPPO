"""
从消融推理结果生成 ablation_table.md
包含 Composite QoS、SLA Violations、Risk 等新指标（system.tex §4.6）

用法:
  python generate_ablation_table.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from metrics import (
    J_L_REF as QOS_L_REF, J_C_REF as QOS_C_REF,
    SLA_MS as SLA_THRESHOLD, KAPPA_GEO, R_REF_SWITCHES,
    ETA_L, ETA_C, ETA_R, ETA_S,
    composite_qos as _qos_full,
)

RESULTS_PATH = os.path.join(os.path.dirname(__file__), 'results', 'ablation_inference_results.npz')
OUTPUT_PATH  = os.path.join(os.path.dirname(__file__), 'ablation_table.md')

VARIANT_NAMES = {
    'full':        'STAR-PPO (Full)',
    'no_workflow': 'w/o Workflow Awareness',
    'no_future':   'w/o Future Reward',
    'no_topology': 'w/o Topology Awareness',
}


def composite_qos(avg_lat, avg_cost, avg_switches, violations_frac):
    """Composite QoS Score — 标量版本，直接调用 metrics._qos_full 的逻辑"""
    lats  = np.array([avg_lat])
    costs = np.array([avg_cost])
    sw    = np.array([avg_switches])
    # 覆盖 _qos_full 中的 sla_violation 步骤，直接用传入的 violations_frac
    j_l = np.clip(avg_lat  / QOS_L_REF, 0.0, 1.0)
    j_c = np.clip(avg_cost / QOS_C_REF, 0.0, 1.0)
    j_r = np.clip(KAPPA_GEO * avg_switches / R_REF_SWITCHES, 0.0, 1.0)
    return 100.0 * (ETA_L*(1-j_l) + ETA_C*(1-j_c) + ETA_R*(1-j_r) + ETA_S*(1-violations_frac))


def main():
    if not os.path.exists(RESULTS_PATH):
        print(f"[ERROR] Ablation results not found: {RESULTS_PATH}")
        print("Please run run_ablation_inference.py first.")
        return

    d = np.load(RESULTS_PATH)
    modes = ['full', 'no_workflow', 'no_future', 'no_topology']

    rows = {}
    for mode in modes:
        key_lat  = f'{mode}_avg_latencies'
        key_cost = f'{mode}_avg_costs'
        key_viol = f'{mode}_avg_violations'
        key_sw   = f'{mode}_avg_switches'

        if key_lat not in d:
            continue

        lats      = d[key_lat]
        costs     = d[key_cost]
        viols_pct = d[key_viol] if key_viol in d else np.zeros_like(lats)
        swits     = d[key_sw]   if key_sw   in d else np.zeros_like(lats)

        avg_lat   = float(np.mean(lats))
        std_lat   = float(np.std(lats))
        avg_cost  = float(np.mean(costs))
        avg_viol  = float(np.mean(viols_pct))
        avg_sw    = float(np.mean(swits))
        qos       = composite_qos(avg_lat, avg_cost, avg_sw, avg_viol / 100.0)

        rows[mode] = {
            'avg_lat':   avg_lat,
            'std_lat':   std_lat,
            'avg_cost':  avg_cost,
            'violations': avg_viol,
            'avg_sw':    avg_sw,
            'qos':       qos,
            'n_seeds':   len(lats),
        }

    if not rows:
        print("[ERROR] No data loaded.")
        return

    # 基准：full model
    base = rows.get('full', list(rows.values())[0])

    lines = []
    lines.append("# Ablation Study Results")
    lines.append("")
    lines.append("> Composite QoS 公式: system.tex eq.(composite_qos_new)，")
    lines.append(f"> 固定归一化常数: L_ref={QOS_L_REF}ms, C_ref={QOS_C_REF}$, SLA={SLA_THRESHOLD:.0f}ms")
    lines.append(f"> 权重: η_L={ETA_L}, η_C={ETA_C}, η_R={ETA_R}, η_S={ETA_S}")
    lines.append("")
    lines.append("| Variant | AvgLatency (ms) | AvgCost ($) | Violations (%) | Composite QoS | Δ Latency | Δ Cost |")
    lines.append("|---------|-----------------|-------------|----------------|---------------|-----------|--------|")

    for mode in modes:
        if mode not in rows:
            continue
        r    = rows[mode]
        name = VARIANT_NAMES.get(mode, mode)

        lat_s  = f"{r['avg_lat']:.2f} ±{r['std_lat']:.1f}"
        cost_s = f"{r['avg_cost']:.4f}"
        viol_s = f"{r['violations']:.1f}"
        qos_s  = f"{r['qos']:.2f}"

        if mode == 'full':
            delta_lat  = "—"
            delta_cost = "—"
            name = f"**{name}**"
            qos_s = f"**{qos_s}**"
        else:
            dl = (r['avg_lat']  - base['avg_lat'])  / max(base['avg_lat'],  1e-9) * 100
            dc = (r['avg_cost'] - base['avg_cost']) / max(base['avg_cost'], 1e-9) * 100
            delta_lat  = f"{dl:+.1f}%"
            delta_cost = f"{dc:+.1f}%"

        lines.append(f"| {name} | {lat_s} | {cost_s} | {viol_s} | {qos_s} | {delta_lat} | {delta_cost} |")

    lines.append("")
    lines.append("## 指标说明")
    lines.append("")
    lines.append("- **Δ Latency / Δ Cost**: 相对 Full Model 的百分比变化（正值 = 性能下降）")
    lines.append("- **Violations (%)**: SLA 违约率（端到端延迟 > 3000ms 的比例），对应 system.tex 中 V_SLA")
    lines.append("- **Composite QoS**: 归一化综合质量评分（越高越好），使用固定归一化常数")

    content = "\n".join(lines)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"Saved to {OUTPUT_PATH}")
    print()
    try:
        print(content)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or 'utf-8'
        print(content.encode(encoding, errors='replace').decode(encoding, errors='replace'))


if __name__ == '__main__':
    main()
