# 图表与代码对照说明

本文档说明 MOE111 仓库的代码结构，以及每张论文图/表由哪个 Python 脚本生成、数据从哪里来、输出文件放在哪里。

---

## 1. 代码怎么读（整体结构）

```
MOE111/
├── TopoFreeRL/              # 本文方法：训练 + 推理
├── PFAPPO/ PPO_algorithm/ … # 各 baseline 算法（train.py / inference.py / plot_batch.py）
├── env.py                   # 仿真环境（读 data1/ 下 CSV）
├── metrics.py               # 统一指标：QoS、SLA、风险分、成本口径
├── data1/                   # 输入数据集（servers.csv、tasks.csv、network_links.csv 等）
│
├── inference/               # 主实验推理入口（生成 .npz 原始结果）
│   ├── run_500_inference.py      → inference/results_500/
│   ├── run_1000_inference.py     → inference/results_1000/
│   ├── run_2000_inference.py     → inference/results_2000/
│   ├── run_adaptability_inference.py
│   └── run_workflow_length_inference.py
│
├── results/                 # 各算法训练日志（metrics.npz、training_data.npz）
│   └── TopoFreeRL/logs/LATEST_Server1_Trap_seed*/
│
├── results_figures/         # 主实验出图 + 性能大表
│   ├── plot_all_comparison.py
│   ├── plot_cdf.py
│   ├── plot_pareto.py
│   └── generate_giant_table.py 等
│
├── Ablation Studies/        # 消融实验
├── Generalization Experiments/  # 泛化实验
├── Supplementary experiments/   # 补充实验 E01–E10
│
└── data_for_plot/           # 已整理好的图表源数据
    ├── chart_data/
    └── supple_chart_data/
```

### 典型数据流

```
训练:  {Algo}/train.py  →  results/{Algo}/logs/*.npz
推理:  inference/run_*_inference.py  →  inference/results_*/{Algo}_*.npz
出图:  results_figures/plot_*.py  →  results_figures/*.png
表格:  results_figures/generate_giant_table.py  →  results_figures/giant_table.md
```

请在项目根目录 `MOE111/` 下运行所有脚本。`results_figures/` 里的绘图脚本默认把 PNG 输出回 `results_figures/` 目录。

---

## 2. 主实验图表（results_figures/）

### Comparison_Reward / Comparison_Latency / Comparison_Cost

- **脚本**：`results_figures/plot_all_comparison.py`
- **输出**：
  - `results_figures/Comparison_Reward.png`
  - `results_figures/Comparison_Latency.png`
  - `results_figures/Comparison_Cost.png`
- **数据**：`results/{Algo}/logs/LATEST_Server1_Trap_seed{42,43,44}/metrics.npz` 或 `training_data.npz`；Greedy / Random / Stark 用推理 npz 画水平基线
- **命令**：

```bash
python results_figures/plot_all_comparison.py
python results_figures/plot_all_comparison.py --dataset Server2_Trap
```

一张脚本出三张图：Reward 曲线、Latency 曲线、Cost 曲线。

---

### Pareto_Server1_Trap

- **脚本**：`results_figures/plot_pareto.py`
- **输出**：`results_figures/Pareto_Server1_Trap.png`
- **数据**：`inference/results_500/*_trainseed42.npz`（中心点）、`*_trainseed44.npz`（方差）；Greedy / Random 用 `*_infseed*.npz`
- **命令**：

```bash
python results_figures/plot_pareto.py --server1
```

---

### Latency_CDF_Server1_Trap

- **脚本**：`results_figures/plot_cdf.py`
- **输出**：`results_figures/Latency_CDF_Server1_Trap.png`
- **数据**：`inference/results_500/{Algo}_*Server1_Trap*seed42.npz` 里的 `latencies`
- **命令**：

```bash
python results_figures/plot_cdf.py --server1
```

---

### Workflow_Length_Analysis

- **脚本**：`results_figures/plot_workflow_length_analysis.py`
- **输出**：`results_figures/Workflow_Length_Analysis.png`（另有 `Workflow_Length_Analysis_Line.png`）
- **数据**：`inference/results_500/{Algo}_workflow_{2,3,5}steps.npz`
- **前置**：先跑 `inference/run_workflow_length_inference.py`
- **命令**：

```bash
python results_figures/plot_workflow_length_analysis.py
```

---

### Cost_Breakdown_Server1_Trap

- **脚本**：`results_figures/plot_cost_breakdown.py`
- **输出**：`results_figures/Cost_Breakdown_Server1_Trap.png`
- **数据**：原始 npz 里的 compute cost + `*_cost_breakdown_final.npz` 里的 network / comm cost
- **命令**：

```bash
python results_figures/plot_cost_breakdown.py --server1
```

---

### Adaptability_Test

- **脚本**：`results_figures/plot_adaptability.py`
- **输出**：`results_figures/Adaptability_Test.png`
- **数据**：`results_figures/adaptability_results/{Algo}_adaptability.npz`
- **前置**：先跑 `inference/run_adaptability_inference.py`
- **命令**：

```bash
python results_figures/plot_adaptability.py
```

---

### Scalability_Analysis

- **脚本**：`results_figures/plot_scalability.py`
- **输出**：`results_figures/Scalability_Analysis.png`
- **数据**：500 / 1000 / 2000 三档 STAR_PPO 推理 npz；1500 节点可读 `results_figures/scalability_data/inference_data.npz`
- **命令**：

```bash
python results_figures/plot_scalability.py
```

---

### giant_table.md（性能大表）

- **脚本**：`results_figures/generate_giant_table.py`
- **输出**：`results_figures/giant_table.md`
- **数据**：`inference/results_500/{Algo}_*Server1_Trap*.npz` + `*_cost_breakdown_final.npz`
- **命令**：

```bash
python results_figures/generate_giant_table.py
python results_figures/generate_giant_table.py --dataset Server2_Trap
```

---

## 3. 消融实验（Ablation Studies/）

### A1_Learning_Curves

- **脚本**：`Ablation Studies/plot_ablation.py`
- **输出**：`Ablation Studies/figures/A1_Learning_Curves.png`
- **数据**：
  - Full model：`results/TopoFreeRL/logs/LATEST_Server1_Trap_seed*/metrics.npz`
  - 消融变体：`Ablation Studies/results/{no_topology,no_workflow,no_future}_seed*/metrics.npz`

### A2_Tradeoff_Scatter

- **脚本**：同上 `plot_ablation.py`
- **输出**：`Ablation Studies/figures/A2_Tradeoff_Scatter.png`
- **数据**：`Ablation Studies/results/ablation_inference_results.npz`

### A3_DWA_Weight_Trajectory

- **说明**：与补充实验 E06 的 DWA 权重轨迹是同一类图
- **等价脚本**：
  - `Supplementary experiments/E06_DWA_Ablation/E06_dwa_ablation.py` → `E06_DWA_Trajectory.png`
  - `Supplementary experiments/generate_all_figures.py` 的 `gen_e06()` → `Fig06a_dwa_trajectory.png`
- **数据**：`results/TopoFreeRL/logs/LATEST_Server1_Trap_seed*/training_data.npz` 中的 `weights_hist`

**消融出图命令**（需先跑 `Ablation Studies/run_ablation_inference.py`）：

```bash
python Ablation Studies/plot_ablation.py
```

---

## 4. 泛化实验

### DAG_Generalization_Radar

- **脚本**：`Generalization Experiments/Generalization to Unseen DAGs/plot_dag_generalization.py`
- **输出**：同目录下 `DAG_Generalization_Radar.png`（还会生成 `DAG_Generalization.png` 柱状图）
- **数据**：`dag_generalization_results.npz`
- **命令**：

```bash
python "Generalization Experiments/Generalization to Unseen DAGs/run_dag_inference.py"
python "Generalization Experiments/Generalization to Unseen DAGs/plot_dag_generalization.py"
```

---

## 5. 补充实验 E01–E10

### 批量出图命令

E01 / E02 / E04 / E06 / E07 / E09：

```bash
python "Supplementary experiments/generate_all_figures.py"
```

E08 / E10：

```bash
python "Supplementary experiments/generate_E03_E05_E08_E10.py"
```

---

### E01 — Zero-Shot Transfer

**Fig01a_zeroshot_latency**
- 脚本：`generate_all_figures.py` → `gen_e01()`
- 输出：`Supplementary experiments/E01_ZeroShot_Transfer/Fig01a_zeroshot_latency.png`
- 数据：读 `Table01_zeroshot_results.csv`

**Fig01b_zeroshot_violation**
- 脚本：同上
- 输出：`Fig01b_zeroshot_violation.png`
- 数据：同上 CSV

**Table01_zeroshot_results**
- 脚本：同上（生成 `.tex`；`.csv` 通常已存在）
- 输出：`.csv` + `.tex`
- 原始推理 npz：`Generalization Experiments/Zero-Shot Scalability Transfer/*.npz`
- 独立脚本（出另一张图 `E01_Scalability_Transfer.png`）：`E01_plot_zeroshot_transfer.py`
- 推理入口：`run_zeroshot_inference.py`

---

### E02 — Service-Oriented Network Model

**Fig02a_latency_cost_scatter**
- 脚本：`generate_all_figures.py` → `gen_e02()`
- 输出：`E02_Network_Model/Fig02a_latency_cost_scatter.png`
- 数据：`Table02_link_model_metrics.csv`

**Fig02b_comm_delay_breakdown**
- 脚本：同上
- 输出：`Fig02b_comm_delay_breakdown.png`
- 数据：同上 CSV

**Table02_link_model_metrics**
- 脚本：同上，或 `E02_service_overlay_eval.py` / `E02_plot_figures.py` 产出 CSV
- 输出：`.csv` + `.tex`
- 训练/评估：`E02_train_overlay.py` + `E02_service_overlay_eval.py`

---

### E03 — Shared Resource Contention

**Fig03a_group_utilization / Fig03b_contention_latency / Fig03c_contention_violation / Fig03d_contention_qnet**

- 脚本：`generate_E03_E05_E08_E10.py` → `gen_e03()`
- 输出目录：`Supplementary experiments/E03_Resource_Contention/`
- 数据：从 `Fig03_source.csv` 读取（副本在 `data_for_plot/supple_chart_data/E03/`）

**Table03_contention_results**

- 脚本：同上
- 输出：`.csv`（已有）+ 由 CSV 重新生成 `.tex`

---

### E04 — Queue Validation

**Fig04a_queue_scatter / Fig04b_queue_mae_boxplot / Fig04c_queue_mape_boxplot**
- 脚本：`generate_all_figures.py` → `gen_e04()`（三张独立 PNG）
- 或：`E04_Queue_Validation/E04_queue_validation.py`（出组合图 `E04_queue_validation.png`）
- 输出目录：`E04_Queue_Validation/`
- 数据：Kingman M/G/1 队列仿真，`Fig04_source.csv`

**Table04_queue_error**
- 脚本：同上
- 输出：`.csv` + `.tex`

---

### E05 — Temporal Reliability

**Fig05a_failure_timeseries**

- 脚本：`generate_E03_E05_E08_E10.py` → `gen_e05()`
- 输出目录：`Supplementary experiments/E05_Temporal_Reliability/`
- 数据：从 `Fig05_source.csv` 读取（副本在 `data_for_plot/supple_chart_data/E05/`）

**Table05_reliability_results**

- 脚本：同上
- 输出：`.csv`（已有）+ 由 CSV 重新生成 `.tex`

---

### E06 — DWA Ablation

**Fig06a_dwa_trajectory**
- 脚本：`generate_all_figures.py` → `gen_e06()`，或 `E06_dwa_ablation.py`
- 输出：`Fig06a_dwa_trajectory.png` / `E06_DWA_Trajectory.png`
- 数据：`results/TopoFreeRL/logs/LATEST_Server1_Trap_seed*/training_data.npz`

**Fig06b_pareto_scatter**
- 脚本：`generate_all_figures.py` → `gen_e06()`
- 输出：`Fig06b_pareto_scatter.png`
- 数据：读 `E07/Table07a_score_breakdown.csv`

**Table06_dwa_ablation**
- 脚本：同上
- 输出：`.csv` + `.tex`

独立运行：

```bash
python "Supplementary experiments/E06_DWA_Ablation/E06_dwa_ablation.py"
```

---

### E07 — Metric Audit

**Table07a_score_breakdown**
- 数据脚本：`E07_Metric_Audit/E07_generate_metric_audit.py` → 生成 `.csv`
- LaTeX：`generate_all_figures.py` → `gen_e07()` → 生成 `.tex`
- 数据：`inference/results_500/` 各算法 npz

**Table07b_normalization**
- 脚本：同上
- 内容：归一化常数 L_ref / C_ref / R_ref 与 η 权重说明

---

### E08 — Workflow Topology

**Fig08a_workflow_latency / Fig08b_workflow_cost / Fig08c_latency_cost_shift**
- 脚本：`generate_E03_E05_E08_E10.py` → `gen_e08()`
- 输出：`E08_Workflow_Topology/Fig08*.png`
- 数据：`inference/results_500/{Algo}_workflow_{2,3,5}steps.npz`

**Table08_ablation**
- 脚本：同上
- 输出：`.csv` + `.tex`

---

### E09 — Traffic Robustness

**Fig09_traffic_robustness_boxplot**
- 脚本：`E09_Traffic_Robustness/E09_plot_traffic_robustness.py`（`gen_e09()` 会调用它并复制为 Fig09 命名）
- 输出：`E09_Workload_Robustness.png` / `Fig09_traffic_robustness_boxplot.png`
- 数据：`Generalization Experiments/Robustness against Workload Patterns/workload_pattern_results.npz`

**Table09_statistical_tests**
- 脚本：`E09_plot_traffic_robustness.py` + `gen_e09()` 生成 `.tex`
- 输出：`.csv` + `.tex`（Mann-Whitney U / Kruskal-Wallis 检验）

推理入口：`run_workload_inference.py`

单独重跑：

```bash
python "Supplementary experiments/E09_Traffic_Robustness/E09_plot_traffic_robustness.py"
```

---

### E10 — Scalability & Decision Time

**Fig10a_decision_time / Fig10b_latency_scalability**
- 脚本：`generate_E03_E05_E08_E10.py` → `gen_e10()`
- 输出：`E10_Scalability_DecisionTime/Fig10*.png`
- 数据：500 / 1000 / 2000 三档 `inference/results_*/` 各算法 npz

**Table10_scalability**
- 脚本：同上
- 输出：`.csv` + `.tex`

---

## 6. 主实验完整运行顺序

先推理：

```bash
cd MOE111
python inference/run_500_inference.py
python inference/run_adaptability_inference.py
python inference/run_workflow_length_inference.py
python inference/run_1000_inference.py
python inference/run_2000_inference.py
```

再出图：

```bash
python results_figures/generate_giant_table.py
python results_figures/plot_cdf.py --server1
python results_figures/plot_pareto.py --server1
python results_figures/plot_cost_breakdown.py --server1
python results_figures/plot_all_comparison.py
python results_figures/plot_workflow_length_analysis.py
python results_figures/plot_adaptability.py
python results_figures/plot_scalability.py
```

---

## 7. 已整理的图表源数据（data_for_plot/）

若只需查看数值、不必重跑推理：

- `data_for_plot/chart_data/` — 主实验各图对应的 npz（按图名整理）
- `data_for_plot/supple_chart_data/E01/` … `E10/` — 补充实验 csv / tex / npz
- `data_for_plot/supple_chart_data/MANIFEST.json` — 原始路径与复制后文件名的映射

---

## 8. 一张图对应一条命令（速查）

**Comparison_Reward / Comparison_Latency / Comparison_Cost**
→ `python results_figures/plot_all_comparison.py`

**Pareto_Server1_Trap**
→ `python results_figures/plot_pareto.py --server1`

**Latency_CDF_Server1_Trap**
→ `python results_figures/plot_cdf.py --server1`

**Workflow_Length_Analysis**
→ `python results_figures/plot_workflow_length_analysis.py`

**Cost_Breakdown_Server1_Trap**
→ `python results_figures/plot_cost_breakdown.py --server1`

**Adaptability_Test**
→ `python results_figures/plot_adaptability.py`

**Scalability_Analysis**
→ `python results_figures/plot_scalability.py`

**giant_table.md**
→ `python results_figures/generate_giant_table.py`

**A1_Learning_Curves + A2_Tradeoff_Scatter**
→ `python Ablation Studies/plot_ablation.py`

**DAG_Generalization_Radar**
→ `python "Generalization Experiments/Generalization to Unseen DAGs/plot_dag_generalization.py"`

**E01–E07 / E09 补充实验**
→ `python "Supplementary experiments/generate_all_figures.py"`

**E03 / E05 / E08 / E10 补充实验**
→ `python "Supplementary experiments/generate_E03_E05_E08_E10.py"`

**E09 单独重跑**
→ `python "Supplementary experiments/E09_Traffic_Robustness/E09_plot_traffic_robustness.py"`

---

## 9. 注意事项

1. 所有脚本请在 `MOE111/` 根目录运行，否则 `results_figures/`、`inference/results_500/` 等相对路径会失效。

2. 以下脚本会一次生成多张图或多张表：
   - `plot_all_comparison.py` → 3 张 Comparison 图
   - `plot_ablation.py` → A1 + A2
   - `generate_all_figures.py` → E01 / E02 / E04 / E06 / E07 / E09
   - `generate_E03_E05_E08_E10.py` → E03 四图 + E05 一图 + E08 三图 + E10 两图 + 对应表格

3. 指标口径统一在 `metrics.py`；改 QoS、SLA、成本定义时优先改这个文件。
