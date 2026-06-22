"""
统一评估指标模块 — 与 system.tex §4.6–§4.7 严格对齐

所有推理脚本应 import 本模块中的函数，不得在各脚本中自行实现。

固定常数（在首次对比前锁定，不得每次重调）:
  J̄_L  = 5000 ms      — env.py lat_worst
  J̄_C  = 0.20 $       — env.py cost_worst（含 network）
  T_SLA = 3000 ms     — 端到端 SLA 阈值
  η_L/C/R/S           — Composite QoS 权重

网络成本参数（与 run_500_inference.py 一致）:
  BYTES_PER_TOKEN     = 4
  NETWORK_COST_PER_MS = 0.00015
  COMM_COST_PER_MS    = 0.00015
"""

import numpy as np

# ── 固定归一化常数 ────────────────────────────────────────────────
J_L_REF       = 5000.0    # ms   — 对应 system.tex J̄_L
J_C_REF       = 0.20      # $    — 对应 system.tex J̄_C
J_R_REF       = 1.0       # 风险归一化分母
SLA_MS        = 3000.0    # ms   — T_m^SLA，对应 system.tex eq.(sla_violation_new)

# ── Composite QoS 权重（η_L + η_C + η_R + η_S = 1）──────────────
ETA_L = 0.45
ETA_C = 0.35
ETA_R = 0.10
ETA_S = 0.10

# ── 风险权重（κ_rel, κ_geo）─────────────────────────────────────
KAPPA_REL         = 0.5
KAPPA_GEO         = 0.5
R_REF_SWITCHES    = 3.0   # 跨域切换次数参考值（归一化分母）

# ── 网络成本参数 ──────────────────────────────────────────────────
BYTES_PER_TOKEN     = 4
NETWORK_COST_PER_MS = 0.00015
COMM_COST_PER_MS    = 0.00015


# ── 每步成本计算 ──────────────────────────────────────────────────

def network_cost(network_ms: float, tokens: int) -> float:
    """网络传输成本（对应 system.tex J_C 中 γ_ij * D_{m,p,u} 项）"""
    data_kb = (tokens * BYTES_PER_TOKEN) / 1024.0
    return data_kb * network_ms * NETWORK_COST_PER_MS


def comm_cost(tokens: int, network_ms: float) -> float:
    """中间结果传输成本（predecessor → successor 数据传输）"""
    intermediate_tokens = tokens * 0.3
    data_kb = (intermediate_tokens * BYTES_PER_TOKEN) / 1024.0
    return data_kb * network_ms * COMM_COST_PER_MS


# ── episode 聚合指标 ──────────────────────────────────────────────

def sla_violation(latencies: np.ndarray) -> np.ndarray:
    """
    逐 episode SLA 违约标志（0/1）
    system.tex eq.(sla_violation_new)：I(L_m > T_m^SLA)
    """
    return (np.asarray(latencies) > SLA_MS).astype(np.float32)


def risk_score(switches: np.ndarray, use_rel: bool = True) -> np.ndarray:
    """
    逐 episode 风险得分 J_R（system.tex eq.(j_risk_new)）

    κ_geo 分量：跨域切换次数 / R_REF_SWITCHES
    κ_rel 分量：用 switches 作为可靠性风险代理（无 per-step p_n_rel 时）
    """
    geo = KAPPA_GEO * np.clip(np.asarray(switches) / R_REF_SWITCHES, 0.0, 1.0)
    rel = KAPPA_REL * np.clip(np.asarray(switches) / R_REF_SWITCHES, 0.0, 1.0) if use_rel else 0.0
    return np.clip(geo + rel, 0.0, 1.0)


def composite_qos(
    latencies: np.ndarray,
    total_costs: np.ndarray,
    switches: np.ndarray,
) -> float:
    """
    Composite QoS Score（0–100，越高越好）
    system.tex eq.(composite_qos_new)：
      S_QoS = 100 * [η_L(1-J̃_L) + η_C(1-J̃_C) + η_R(1-J̃_R) + η_S(1-V_SLA)]
    """
    lat   = np.asarray(latencies, dtype=float)
    cost  = np.asarray(total_costs, dtype=float)
    sw    = np.asarray(switches, dtype=float)

    j_l = np.clip(np.mean(lat)  / J_L_REF, 0.0, 1.0)
    j_c = np.clip(np.mean(cost) / J_C_REF, 0.0, 1.0)
    j_r = float(np.mean(risk_score(sw)))
    v   = float(np.mean(sla_violation(lat)))

    return 100.0 * (
        ETA_L * (1.0 - j_l)
        + ETA_C * (1.0 - j_c)
        + ETA_R * (1.0 - j_r)
        + ETA_S * (1.0 - v)
    )


# ── 推理结果标准字段打包 ─────────────────────────────────────────

def pack_episode_results(
    latencies,
    compute_costs,
    network_costs_arr,
    comm_costs_arr,
    rewards,
    switches,
    inference_times,
) -> dict:
    """
    返回标准化的 episode 结果字典，统一所有推理脚本的 npz 输出格式。
    extra 字段由调用方按需 merge。
    """
    lat  = np.asarray(latencies,        dtype=np.float32)
    comp = np.asarray(compute_costs,    dtype=np.float32)
    net  = np.asarray(network_costs_arr,dtype=np.float32)
    comm = np.asarray(comm_costs_arr,   dtype=np.float32)
    rew  = np.asarray(rewards,          dtype=np.float32)
    sw   = np.asarray(switches,         dtype=np.float32)
    inf  = np.asarray(inference_times,  dtype=np.float32)

    total = comp + net + comm
    return {
        "latencies":           lat,
        "compute_costs":       comp,
        "network_costs":       net,
        "communication_costs": comm,
        "costs":               total,   # total cost，与 giant_table 一致
        "rewards":             rew,
        "switches":            sw,
        "inference_times":     inf,
        "sla_violations":      sla_violation(lat),
        "risk_scores":         risk_score(sw),
    }


# ── 快速摘要打印 ──────────────────────────────────────────────────

def print_summary(algo: str, d: dict) -> None:
    lat   = d["latencies"]
    total = d["costs"]
    sw    = d["switches"]
    viol  = np.mean(d["sla_violations"]) * 100
    qos   = composite_qos(lat, total, sw)
    print(
        f"  {algo:<10} lat={np.mean(lat):.0f}ms  "
        f"cost=${np.mean(total):.4f}  "
        f"SLA_viol={viol:.1f}%  "
        f"QoS={qos:.2f}"
    )
