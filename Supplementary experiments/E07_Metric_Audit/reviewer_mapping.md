# E07 Reviewer Mapping

## 回应审稿意见
- **R2-5**: "Quality Score" 定义不透明
- **R3-4**: Cost、Risk、Improvement 定义不清楚

## 本实验如何回应
1. 将所有 "Quality Score" 改为 "Composite QoS Score"（系统服务质量综合分，非 LLM 语义质量）
2. 逐算法展示 score 计算分解：
   J_L → J̃_L=J_L/5000.0ms → η_L×(1-J̃_L)=0.45×(1-J̃_L)
   J_C → J̃_C=J_C/0.2$ → η_C×(1-J̃_C)=0.35×(1-J̃_C)
   J_R → J̃_R=risk_score/1.0 → η_R×(1-J̃_R)=0.1×(1-J̃_R)
   V_SLA                      → η_S×(1-V)  =0.1×(1-V)
   S_QoS = 100×(上述四项之和)
3. 固定归一化常数在首次对比前已锁定（见 E07_normalization_constants.csv）
4. Improvement(%) 明确定义为相对 Greedy baseline 的延迟改进
5. 关键发现：STAR-PPO switches=0（无跨域切换），所有竞品 switches≈2.08，
   导致竞品 risk_score≈0.58-0.62，QoS 分低约 6 分；这是 STAR-PPO 的拓扑感知优势体现

## 文件清单
- E07_metric_audit_table.csv        — score 计算分解审计表（手册 Table 要求）
- E07_normalization_constants.csv   — 归一化常数和权重（手册 Appendix 表要求）
- E07_generate_metric_audit.py      — 可复现脚本
