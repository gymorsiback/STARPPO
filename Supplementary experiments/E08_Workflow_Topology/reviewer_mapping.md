# E08 Reviewer Mapping

## Addresses: R1-6, R3-1, R2-1
Proves workflow routing + topology telemetry is effective combination, not rebranding.

### E08 Response:
- Uses real 2/3/5-step workflow inference results for all algorithms
- Shows STAR-PPO maintains superior latency and cost scaling vs workflow length
- Ablation table uses PPO-Std (w/o Workflow), PPO-CN (w/o Topology), A3C (w/o Future Reward)
  as proxies demonstrating each component contributes to performance
- Latency-cost scatter shows STAR-PPO dominates Pareto frontier at all workflow lengths
