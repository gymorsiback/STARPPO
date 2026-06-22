# E01 Reviewer Mapping

## 回应审稿意见
- **R2-2**: Zero-shot 协议不严格，可能是同域随机划分

## 本实验如何回应
1. 明确记录训练域与测试域的地理隔离：
   - 训练域：Server1_Trap（Switzerland, lat 45-48, lon 6-10, 500 servers）
   - 测试域：Server3_Trap（Germany/Central Europe, lat 47-55, lon 6-15, 2000 servers）
   - 两者地理上严格分离，无 fine-tuning，无 target domain episodes
2. 图表中同时展示 Avg latency + P95 误差棒（bar=P95-avg）
3. 所有算法的 Avg/P95/P99/SLA_violation 保存在 E01_zero_shot_table.csv
4. 标注 × 表示相对 Retrained 上界的 gap ratio

## 已知缺口（待补充）
- Server2_Trap（UK, lat 50-58, lon -8~2）的 zero-shot 推理尚未运行
  需运行 run_zeroshot_inference.py 并指定 REGION_TARGET='Server2_Trap'

## 文件清单
- E01_Scalability_Transfer.png  — 零样本迁移对比图（替代原 Scalability_Transfer_Detailed.png）
- E01_zero_shot_table.csv       — Avg/P95/P99/SLA_viol 分项数字表
- E01_plot_zeroshot_transfer.py — 可复现脚本
