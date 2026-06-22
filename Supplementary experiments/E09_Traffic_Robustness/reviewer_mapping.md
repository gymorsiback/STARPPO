# E09 Reviewer Mapping

## 回应审稿意见
- **R1-7**: Workload robustness boxplot 缺少样本量、分位数、异常值定义和统计检验

## 本实验如何回应
1. 箱线图每组标注 n（样本量）和 IQR 数值
2. 明确 whisker=1.5×IQR，outlier 定义为超出 whisker 的点
3. 对 Uniform vs Poisson/Bursty/On-Off 进行 Mann-Whitney U 双侧检验，显著性以 * / ** / *** 标注
4. Kruskal-Wallis 检验验证四组整体差异
5. 所有检验结果（test_name、statistic、p_value、effect_size、n）保存在 E09_statistical_tests.csv

## 文件清单
- E09_Workload_Robustness.png      — 增强版箱线图（替代原 Workload_Robustness.png）
- E09_statistical_tests.csv       — 统计检验结果（手册 statistical_tests.csv 规范）
- E09_sample_stats.csv            — 各组描述性统计（n/median/IQR/P95/P99/n_outliers）
- E09_plot_traffic_robustness.py  — 可复现脚本
