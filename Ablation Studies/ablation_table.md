# Ablation Study Results

> Composite QoS 公式: system.tex eq.(composite_qos_new)，
> 固定归一化常数: L_ref=5000.0ms, C_ref=0.2$, SLA=3000ms
> 权重: η_L=0.45, η_C=0.35, η_R=0.1, η_S=0.1

| Variant | AvgLatency (ms) | AvgCost ($) | Violations (%) | Composite QoS | Δ Latency | Δ Cost |
|---------|-----------------|-------------|----------------|---------------|-----------|--------|
| **STAR-PPO (Full)** | 2122.00 ±19.5 | 0.0750 | 0.0 | **67.78** | — | — |
| w/o Workflow Awareness | 2319.74 ±32.1 | 0.0993 | 0.0 | 61.75 | +9.3% | +32.3% |
| w/o Future Reward | 2223.18 ±38.4 | 0.0913 | 0.0 | 64.02 | +4.8% | +21.7% |
| w/o Topology Awareness | 2418.54 ±57.6 | 0.1081 | 0.0 | 59.31 | +14.0% | +44.2% |

## 指标说明

- **Δ Latency / Δ Cost**: 相对 Full Model 的百分比变化（正值 = 性能下降）
- **Violations (%)**: SLA 违约率（端到端延迟 > 3000ms 的比例），对应 system.tex 中 V_SLA
- **Composite QoS**: 归一化综合质量评分（越高越好），使用固定归一化常数