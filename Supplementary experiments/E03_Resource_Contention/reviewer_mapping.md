# E03 Reviewer Mapping

## Addresses: R1-1
Shared physical resources and link contention were not modeled in the original submission.

### E03 Response:
- Constructs shared resource groups by region/backhaul/cloud-egress
- Enforces sum(W_ij) <= W_max constraint (with_contention condition)
- Records group_utilization, R_ij degradation, Q_net, latency, violation_rate per window
- Compares no_contention vs with_contention conditions quantitatively
