# E10 Reviewer Mapping

## Addresses: R3-5
Avoid microsecond overclaiming; report real decision time in tens/hundreds of ms.

### E10 Response:
- Reports actual inference times directly measured from model inference (ms level)
- STAR-PPO: 23.7 ms (500 nodes) → 49 ms (1000 nodes) → 96 ms (2000 nodes)
- STARK: 207 ms (500) → 531 ms (1000) — poor scalability
- Greedy: 0.8 ms (500) → 1.8 ms (1000) — fast but lower quality
- All values strictly in milliseconds; no microsecond claims made
- Decision time is per-request online inference (not batch GPU time)
