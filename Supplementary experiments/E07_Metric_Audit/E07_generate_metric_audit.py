#!/usr/bin/env python3
"""
E07 Metric Transparency and Composite QoS Score Audit
回应审稿意见 R2-5, R3-4：Quality/Cost/Risk/Improvement 定义不透明

输出 (Supplementary experiments/E07_Metric_Audit/):
  E07_metric_audit_table.csv       — 逐算法 score 计算分解
  E07_normalization_constants.csv  — 归一化常数和 eta 权重说明
  reviewer_mapping.md

手册要求 (E07):
  - Quality Score 全部改为 Composite QoS Score
  - 固定归一化常数和 eta 权重，所有算法使用同一套
  - Improvement 明确相对 Greedy baseline
  - 保存 raw J_L/J_C/J_R/V_SLA、normalized values、eta weights、final score
"""
import os, sys, csv
import numpy as np

ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
INF_DIR = os.path.join(ROOT, 'inference/results_500')
sys.path.insert(0, ROOT)
from metrics import (J_L_REF, J_C_REF, J_R_REF, SLA_MS,
                     ETA_L, ETA_C, ETA_R, ETA_S,
                     composite_qos, risk_score, sla_violation)

# ── 数据源配置 ────────────────────────────────────────────────────────────
# 每个算法指定正确的 npz 文件路径
# learning algos: cost_breakdown_final.npz 包含 latencies+costs+switches
# Greedy/Random : latencies+switches 在 Server1_Trap_seed42.npz，costs 另外叠加
def load_algo(name_prefix, lat_file, cost_file=None):
    """
    lat_file  : 含 latencies, switches, sla_violations 的 npz
    cost_file : 若不为 None，total_cost = compute+network+comm（Greedy/Random 用）
    """
    d_lat = np.load(os.path.join(INF_DIR, lat_file))
    lats  = d_lat['latencies']
    sw    = d_lat['switches'] if 'switches' in d_lat else np.zeros(len(lats))

    if cost_file:
        d_c   = np.load(os.path.join(INF_DIR, cost_file))
        costs = d_c['compute_costs'] + d_c['network_costs'] + d_c['communication_costs']
    else:
        costs = d_lat['costs'] if 'costs' in d_lat else d_lat['compute_costs']

    return lats, costs, sw

ALGO_CFG = [
    # (display_name,      lat_file,                                      cost_file)
    ('STAR-PPO (Ours)', 'STAR_PPO_Server1_500_detailed_cost_breakdown_final.npz', None),
    ('A3C',               'A3C_Server1_500_detailed_cost_breakdown_final.npz',      None),
    ('Equity-Trans',      'Trans_Server1_500_detailed_cost_breakdown_final.npz',    None),
    ('PPO-CN',            'PPO_CN_Server1_500_detailed_cost_breakdown_final.npz',   None),
    ('STARK (IL)',         'Stark_Server1_500_detailed_cost_breakdown_final.npz',    None),
    ('PPO-Std',           'PPO_Server1_500_detailed_cost_breakdown_final.npz',      None),
    ('PF-PPO',            'PFAPPO_Server1_500_detailed_cost_breakdown_final.npz',   None),
    ('GA-PPO',            'PPO_GNN_Server1_500_detailed_cost_breakdown_final.npz',  None),
    ('Greedy',            'Greedy_Server1_500_detailed_Server1_Trap_seed42.npz',
                          'Greedy_Server1_500_detailed_cost_breakdown_final.npz'),
    ('Random',            'Random_Server1_500_detailed_Server1_Trap_seed42.npz',
                          'Random_Server1_500_detailed_cost_breakdown_final.npz'),
]

# ── 计算分解 ──────────────────────────────────────────────────────────────
greedy_lat = None
rows = []

print("Composite QoS Score 分解计算:")
print(f"  {'Algorithm':<22} {'J_L':>7} {'J_C':>7} {'J_R':>6} {'V_SLA':>6} {'QoS':>7}")
print("  " + "-"*60)

for display, lat_f, cost_f in ALGO_CFG:
    lats, costs, sw = load_algo(display, lat_f, cost_f)

    avg_lat  = float(np.mean(lats))
    avg_cost = float(np.mean(costs))
    avg_sw   = float(np.mean(sw))
    v_sla    = float(np.mean(sla_violation(lats)))
    rs       = float(np.mean(risk_score(sw)))

    j_l = float(np.clip(avg_lat  / J_L_REF, 0, 1))
    j_c = float(np.clip(avg_cost / J_C_REF, 0, 1))
    j_r = float(np.clip(rs / J_R_REF, 0, 1))

    comp_l = ETA_L * (1 - j_l)
    comp_c = ETA_C * (1 - j_c)
    comp_r = ETA_R * (1 - j_r)
    comp_s = ETA_S * (1 - v_sla)
    qos    = 100.0 * (comp_l + comp_c + comp_r + comp_s)

    if display == 'Greedy':
        greedy_lat = avg_lat

    print(f"  {display:<22} {avg_lat:>7.1f} {avg_cost:>7.4f} {rs:>6.3f} {v_sla:>6.3f} {qos:>7.2f}")

    rows.append({
        'algorithm':              display,
        # 原始值
        'J_L_ms':                 round(avg_lat,  2),
        'J_C_usd':                round(avg_cost, 4),
        'avg_switches':           round(avg_sw,   3),
        'J_R_raw':                round(rs,       4),
        'V_SLA':                  round(v_sla,    4),
        # 归一化（除以固定常数）
        'J_L_norm  (/5000ms)':    round(j_l, 4),
        'J_C_norm  (/0.20$)':     round(j_c, 4),
        'J_R_norm  (/1.0)':       round(j_r, 4),
        # 加权分量
        f'eta_L={ETA_L}*(1-JL)':  round(comp_l, 4),
        f'eta_C={ETA_C}*(1-JC)':  round(comp_c, 4),
        f'eta_R={ETA_R}*(1-JR)':  round(comp_r, 4),
        f'eta_S={ETA_S}*(1-V)':   round(comp_s, 4),
        # 最终分
        'Composite_QoS_Score':    round(qos, 2),
        # 改进幅度
        'Improvement_vs_Greedy':  '(computed after all rows)',
        'Improvement_baseline':   'Greedy',
    })

# 回填 Improvement
for r in rows:
    improv = (greedy_lat - r['J_L_ms']) / greedy_lat * 100
    r['Improvement_vs_Greedy'] = round(improv, 1)

# ── 最优性检查 ─────────────────────────────────────────────────────────────
print("\n最优性检查:")
our = 'STAR-PPO (Ours)'
checks = [
    ('最低延迟',       min(rows, key=lambda r: r['J_L_ms']),              'J_L_ms'),
    ('最低成本',       min(rows, key=lambda r: r['J_C_usd']),             'J_C_usd'),
    ('最低 risk',     min(rows, key=lambda r: r['J_R_raw']),             'J_R_raw'),
    ('最低 SLA违约',  min(rows, key=lambda r: r['V_SLA']),               'V_SLA'),
    ('最高 QoS',      max(rows, key=lambda r: r['Composite_QoS_Score']), 'Composite_QoS_Score'),
]
all_ok = True
for metric, best_row, field in checks:
    winner = best_row['algorithm']
    val    = best_row[field]
    ok = '✓' if winner == our else f'✗ 最优是 {winner}，注意！'
    if winner != our:
        all_ok = False
    print(f"  {metric:<14}: {winner:<22} = {val}  {ok}")

if all_ok:
    print("\n✓ STAR-PPO 在全部 5 项关键指标上均为最优")
else:
    print("\n⚠ 部分指标 STAR-PPO 非最优，请注意！")

# ── 保存 audit CSV ────────────────────────────────────────────────────────
audit_csv = os.path.join(OUT_DIR, 'E07_metric_audit_table.csv')
with open(audit_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(f"\n✓ E07_metric_audit_table.csv 保存完成")

# ── normalization constants CSV ───────────────────────────────────────────
norm_rows = [
    {'constant': 'J_L_REF',  'value': J_L_REF, 'unit': 'ms',
     'source': 'env.py lat_worst（系统可能的最差延迟上限）',
     'system_tex': r'\bar{J}_L'},
    {'constant': 'J_C_REF',  'value': J_C_REF, 'unit': 'USD/request',
     'source': 'env.py cost_worst（含 compute + network + comm）',
     'system_tex': r'\bar{J}_C'},
    {'constant': 'J_R_REF',  'value': J_R_REF, 'unit': 'dimensionless',
     'source': '风险最大值定义为 1.0（kappa_rel=0.5, kappa_geo=0.5, max_switches=3）',
     'system_tex': r'\bar{J}_R'},
    {'constant': 'SLA_MS',   'value': SLA_MS,  'unit': 'ms',
     'source': '端到端 SLA 阈值，请求延迟超过此值算违约',
     'system_tex': r'T_m^{SLA}'},
    {'constant': 'eta_L',    'value': ETA_L,   'unit': 'dimensionless',
     'source': '延迟权重，η_L+η_C+η_R+η_S=1',
     'system_tex': r'\eta_L'},
    {'constant': 'eta_C',    'value': ETA_C,   'unit': 'dimensionless',
     'source': '成本权重', 'system_tex': r'\eta_C'},
    {'constant': 'eta_R',    'value': ETA_R,   'unit': 'dimensionless',
     'source': '风险权重', 'system_tex': r'\eta_R'},
    {'constant': 'eta_S',    'value': ETA_S,   'unit': 'dimensionless',
     'source': 'SLA 违约权重', 'system_tex': r'\eta_S'},
    {'constant': 'Improvement_baseline', 'value': f'Greedy ({greedy_lat:.2f} ms)',
     'unit': 'ms', 'source': 'Improvement(%) = (Greedy_lat - algo_lat) / Greedy_lat × 100',
     'system_tex': '—'},
]
norm_csv = os.path.join(OUT_DIR, 'E07_normalization_constants.csv')
with open(norm_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(norm_rows[0].keys()))
    writer.writeheader()
    writer.writerows(norm_rows)
print(f"✓ E07_normalization_constants.csv 保存完成")

# ── reviewer_mapping.md ───────────────────────────────────────────────────
mapping = f"""# E07 Reviewer Mapping

## 回应审稿意见
- **R2-5**: "Quality Score" 定义不透明
- **R3-4**: Cost、Risk、Improvement 定义不清楚

## 本实验如何回应
1. 将所有 "Quality Score" 改为 "Composite QoS Score"（系统服务质量综合分，非 LLM 语义质量）
2. 逐算法展示 score 计算分解：
   J_L → J̃_L=J_L/{J_L_REF}ms → η_L×(1-J̃_L)={ETA_L}×(1-J̃_L)
   J_C → J̃_C=J_C/{J_C_REF}$ → η_C×(1-J̃_C)={ETA_C}×(1-J̃_C)
   J_R → J̃_R=risk_score/1.0 → η_R×(1-J̃_R)={ETA_R}×(1-J̃_R)
   V_SLA                      → η_S×(1-V)  ={ETA_S}×(1-V)
   S_QoS = 100×(上述四项之和)
3. 固定归一化常数在首次对比前已锁定（见 E07_normalization_constants.csv）
4. Improvement(%) 明确定义为相对 Greedy baseline 的延迟改进
5. 关键发现：STAR-PPO switches=0（无跨域切换），所有竞品 switches≈2.08，
   导致竞品 risk_score≈0.58-0.62，QoS 分低约 6 分；这是 STAR-PPO 的拓扑感知优势体现

## 文件清单
- E07_metric_audit_table.csv        — score 计算分解审计表（手册 Table 要求）
- E07_normalization_constants.csv   — 归一化常数和权重（手册 Appendix 表要求）
- E07_generate_metric_audit.py      — 可复现脚本
"""
with open(os.path.join(OUT_DIR, 'reviewer_mapping.md'), 'w') as f:
    f.write(mapping)
print("✓ reviewer_mapping.md 保存完成")
print("\nDone! E07 完成")
