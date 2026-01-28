<div align="center">

# STAR-PPO: Spatiotemporal-Aware Reinforcement Learning for MoE Inference Scheduling

<!-- Badges -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework](https://img.shields.io/badge/PyTorch-1.12%2B-ee4c2c.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

</div>

---

## 📝 Abstract

This repository is the official implementation of **STAR-PPO**, a novel scheduling framework designed for **Mixture-of-Experts (MoE)** inference in geo-distributed edge-cloud environments. 

STAR-PPO addresses the critical challenge of routing complex, multi-step inference workflows across heterogeneous servers with unstable network connections. By integrating **spatiotemporal awareness** into the reinforcement learning state space and employing **Dynamic Weight Adjustment (DWA)**, STAR-PPO achieves a superior balance between latency, inference cost, and service reliability, specifically robustness against "network traps" (high-latency links).

---

## 🚀 News

- Code and sample dataset (Server1_Trap) released.

---

## 📂 Project Structure

```
.
├── STAR_PPO/               # [Ours] STAR-PPO Algorithm Implementation
│   ├── agent.py            # PPO Agent with Spatiotemporal Awareness
│   ├── model.py            # Actor-Critic Networks
│   └── train.py            # Training Loop
├── PFAPPO/                 # [Baseline] PF-PPO Implementation
├── PPO_algorithm/          # [Baseline] Standard PPO Implementation
├── Stark_Scheduler/        # [Baseline] STARK Implementation
├── env.py                  # Simulation Environment (Edge-Cloud MoE)
├── data1/                  # Dataset Directory
│   └── Server1_Trap/       # Provided Sample Dataset (500 Nodes)
├── total/                  # Visualization & Analysis Scripts
└── requirements.txt        # Dependencies
```

---

## 💾 Dataset: Edge-Cloud MoE Bench

We introduce a high-fidelity benchmark dataset for edge-cloud MoE inference scheduling. 

### Why this dataset is valuable?
Unlike synthetic traces used in prior works, this dataset captures the complexity of real-world edge computing:
1.  **Heterogeneity**: Diverse server compute capabilities and cost models.
2.  **Geo-Distribution**: Latency based on physical distance (Haversine formula).
3.  **Adversarial "Traps"**: Specific network links exhibit stochastic high latency/packet loss, simulating real-world network jitter. This is critical for testing robustness.
4.  **MoE Workflows**: Tasks require sequential processing by specific expert models distributed across the topology.

### Availability
We provide the **`Server1_Trap` (500 servers)** scale as a sample for reproducibility.

> **Note**: The full dataset includes larger scales (1000 and 2000 servers) with varying densities and trap configurations. Due to the size and commercial value of the full topology data, only the sample is open-sourced. 
>
> Researchers requiring the **complete dataset** for comparison benchmarks should contact: **gymorsiback@tju.edu.cn**

---

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/anonymous/STAR-PPO.git
   cd STAR-PPO
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## ⚡ Quick Start

### 1. Training

To train STAR-PPO on the provided sample dataset:

```bash
python STAR_PPO/train.py \
    --data data1 \
    --regions Server1_Trap \
    --epochs 100 \
    --device cuda
```

You can also train baseline algorithms (e.g., PF-PPO, Standard PPO):

```bash
# Train PF-PPO
python PFAPPO/train.py --data data1 --regions Server1_Trap --epochs 100
```

### 2. Inference & Evaluation

Load the trained model to evaluate performance:

```bash
python STAR_PPO/inference.py \
    --data data1 \
    --regions Server1_Trap \
    --model results/STAR_PPO/models/LATEST_actor.pt \
    --episodes 500
```

### 3. Visualization

Generate comparison figures (Learning Curves, Cost Breakdown, Latency CDFs):

```bash
# Generates all figures in the 'total/' folder
python total/plot_all_comparison.py
```

---

## 📊 Results

Our method consistently outperforms baselines in terms of reward, latency, and cost efficiency, especially in "Trap" environments.

*(Visualization results are generated in the `total/` directory)*

---

## 📧 Contact

For any questions regarding the code or to request the **full dataset**, please email:  
**gymorsiback@tju.edu.cn**
