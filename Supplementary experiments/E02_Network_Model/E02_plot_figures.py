#!/usr/bin/env python3
"""
E02 Service-Oriented Network Model — 规范图表生成
================================================
按手册 E02 要求生成两张独立图：
  Fig 1: E02_latency_cost_scatter.png   — latency-cost scatter (per episode)
  Fig 2: E02_comm_delay_breakdown.png   — communication delay breakdown

收集数据字段（按手册）：
  R_ij, T_base, tx_ms (D/R), compute_ms, queue_ms,
  network_ms, total_latency_ms, cost, transfer_bytes
"""
import os, sys, random
import numpy as np
import torch
import pandas as pd

sys.path.insert(0, '.')
sys.path.insert(0, './Supplementary experiments/E02_Network_Model')
from env import WorkflowDataset
from TopoFreeRL.inference import compute_resource_weights
from utils import haversine_km
from E02_train_overlay import OverlayAugmentedEnv

DATA_ROOT      = './data1'
REGION         = 'Server1_Trap'
OVERLAY_MODELS = './results/TopoFreeRL_Overlay/models'
ORIG_MODELS    = './results/TopoFreeRL/models'
OUT_DIR        = os.path.dirname(os.path.abspath(__file__))
SEEDS          = [42, 43, 44]
EPISODES       = 200
DEVICE         = 'cpu'
BITS_PER_TOKEN = 32
DEFAULT_BW     = 100.0   # Mbps


# ─── 扩展 Env：step 时记录完整分项 ─────────────────────────────────────────
class DetailedOverlayEnv(OverlayAugmentedEnv):
    def reset(self, task):
        s = super().reset(task)
        self.step_details = []   # per-step detailed records
        return s

    def step_detailed(self, action_idx):
        mi     = self.actions[action_idx]
        server = self.servers[mi.server_id]
        _, req_id, _ = self.cur_steps[self.step_idx]
        if req_id is not None and req_id in self.ds.req_tokens:
            in_tok, out_tok = self.ds.req_tokens[req_id]
        else:
            size = float(self.cur_task['TaskSize'])
            in_tok, out_tok = int(0.6*size), int(0.4*size)
        tokens         = in_tok + out_tok
        transfer_bytes = tokens * (BITS_PER_TOKEN // 8)   # bytes

        # identify link params BEFORE calling step()
        if self.step_idx == 0:
            d_km = haversine_km(self.cur_task['TaskLongitude'],
                                self.cur_task['TaskLatitude'],
                                server.lon, server.lat)
            if self.trap_latency > 0 and mi.server_id in self.trap_server_ids:
                t_base  = (self.trap_packet_loss_prob * self.trap_bad_latency +
                           (1-self.trap_packet_loss_prob) * self.trap_good_latency)
                r_ij    = DEFAULT_BW
            else:
                t_base  = max(d_km, 0.1) * 0.005
                r_ij    = DEFAULT_BW
        else:
            if self.prev_server_id:
                prev = self.servers[self.prev_server_id]
                d_km = haversine_km(prev.lon, prev.lat, server.lon, server.lat)
                link_key = (self.prev_server_id, mi.server_id)
                involves_trap = (self.prev_server_id in self.trap_server_ids or
                                 mi.server_id in self.trap_server_ids)
                if involves_trap and self.trap_latency > 0:
                    t_base = (self.trap_packet_loss_prob * self.trap_bad_latency +
                              (1-self.trap_packet_loss_prob) * self.trap_good_latency)
                    r_ij   = DEFAULT_BW
                elif link_key in self.link_latency:
                    t_base = self.link_latency[link_key]
                    r_ij   = self.link_bandwidth.get(link_key, DEFAULT_BW)
                else:
                    t_base = max(d_km, 0.1) * 0.005
                    r_ij   = DEFAULT_BW
            else:
                t_base = 0.5; r_ij = DEFAULT_BW

        tx_ms = (transfer_bytes * 8) / (r_ij * 1e6) * 1000.0   # D/R in ms

        # call the overlay step
        _, (rL, rC, rS), done, info = self.step(action_idx)

        detail = dict(
            t_base_ms     = t_base,
            r_ij_mbps     = r_ij,
            tx_ms         = tx_ms,
            network_ms    = info['network_ms'],
            compute_ms    = info['compute_ms'],
            queue_ms      = info['queue_ms'],
            total_lat_ms  = info['latency_ms'],
            cost          = info['cost'],
            transfer_bytes= transfer_bytes,
            server_id     = info['server_id'],
        )
        self.step_details.append(detail)
        return done, info, detail


# ─── 推理函数 ────────────────────────────────────────────────────────────────
def run_algo(env, ds, algo_name, model_path=None, seed=42):
    """返回 episodes list，每个 episode = {latency, cost, steps:[detail]}"""
    from TopoFreeRL.agent import StarPPOAgent
    N = 500
    w_dwa = np.array([0.45, 0.40, 0.15], np.float32)

    if model_path:
        agent = StarPPOAgent(state_dim=10, num_servers=N, device=DEVICE)
        agent.actor.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        agent.actor.eval()
        server_ids = sorted(env.servers.keys())[:N]
        sid2idx    = {sid: i for i, sid in enumerate(server_ids)}

    episodes = []
    for ep in range(EPISODES):
        np.random.seed(seed + ep); random.seed(seed + ep)
        task = ds.tasks[ep % len(ds.tasks)]
        env.reset(task)
        if model_path:
            s_vec = env.get_augmented_state(dwa_weights=w_dwa)
        done = False; ep_lat = ep_cost = 0.

        while not done:
            valid = env.available_actions()
            if not valid: break

            # select action
            if algo_name == 'Random':
                action_idx = random.choice(valid)
            elif algo_name == 'Greedy':
                best_a, best_est = None, 1e18
                for a in valid:
                    mi  = env.actions[a]
                    srv = env.servers[mi.server_id]
                    _, req_id, _ = env.cur_steps[env.step_idx]
                    if req_id and req_id in env.ds.req_tokens:
                        it, ot = env.ds.req_tokens[req_id]
                    else:
                        sz = float(env.cur_task['TaskSize'])
                        it, ot = int(0.6*sz), int(0.4*sz)
                    toks = it + ot
                    est_compute = (toks / max(srv.normalized_compute,1e-6) / env.base_speed_tps) * 1000.0
                    est_queue   = max(0., env.busy_until[mi.server_id] - env.current_time_ms)
                    if best_a is None or (est_compute + est_queue) < best_est:
                        best_est, best_a = est_compute + est_queue, a
                action_idx = best_a
            else:
                # TopoFreeRL (overlay or wireless-trained)
                r_w = compute_resource_weights(env, dwa_weights=w_dwa)
                if len(r_w) < N:
                    r_w = np.pad(r_w, (0, N-len(r_w)))
                s_t = torch.tensor(s_vec,  dtype=torch.float32).unsqueeze(0)
                w_t = torch.tensor(r_w,    dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    logits   = agent.actor(s_t, w_t)
                    rw       = torch.tensor(r_w)
                    enhanced = logits.squeeze(0) + 1.0*(rw-rw.mean())/(rw.std()+1e-6)*1.5
                idx2act = {}; vi = []
                for a in valid:
                    sid = env.actions[a].server_id
                    if sid in sid2idx:
                        i = sid2idx[sid]
                        idx2act.setdefault(i, a); vi.append(i)
                action_idx = idx2act.get(max(vi, key=lambda i: enhanced[i].item()), valid[0]) if vi else valid[0]

            done, info, detail = env.step_detailed(action_idx)
            ep_lat  += detail['total_lat_ms']
            ep_cost += detail['cost']
            if model_path:
                s_vec = env.get_augmented_state(dwa_weights=w_dwa)

        episodes.append(dict(algo=algo_name, seed=seed, ep=ep,
                             latency=ep_lat, cost=ep_cost,
                             steps=list(env.step_details)))
    return episodes


# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    matplotlib.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 22,
        'axes.labelsize': 24,
        'axes.titlesize': 24,
        'axes.titleweight': 'normal',
        'axes.linewidth': 2.0,
        'axes.edgecolor': 'black',
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 18,
        'lines.linewidth': 2.5,
        'ytick.color': 'black',
        'xtick.color': 'black',
        'axes.labelcolor': 'black',
        'mathtext.fontset': 'stix',
    })

    print("=== E02 规范图表生成 ===")
    ds  = WorkflowDataset(DATA_ROOT, split='test', regions=[REGION])
    env = DetailedOverlayEnv(ds, device=DEVICE)

    all_eps = []

    configs = [
        ('STAR-PPO-Overlay', OVERLAY_MODELS),
        ('STAR-PPO-Wireless', ORIG_MODELS),
        ('Greedy', None),
        ('Random', None),
    ]

    for algo, mdir in configs:
        for seed in SEEDS:
            mpath = os.path.join(mdir, f'LATEST_{REGION}_seed{seed}_final.pt') if mdir else None
            if mdir and not os.path.exists(mpath):
                print(f"  skip {algo} seed={seed} (no model)")
                continue
            print(f"  {algo} seed={seed} ...")
            eps = run_algo(env, ds, algo, mpath, seed)
            all_eps.extend(eps)

    print("  推理完成，开始出图...")

    # ── 整理数据 ──────────────────────────────────────────────────────────
    # per-episode
    ep_df = pd.DataFrame([{
        'algo': e['algo'], 'seed': e['seed'],
        'latency_ms': e['latency'], 'cost_usd': e['cost'],
        'n_steps': len(e['steps']),
    } for e in all_eps])

    # per-step
    step_rows = []
    for e in all_eps:
        for s in e['steps']:
            step_rows.append({**s, 'algo': e['algo'], 'seed': e['seed']})
    step_df = pd.DataFrame(step_rows)

    # 保存详细 CSV（手册要求的数据字段）
    agg = step_df.groupby('algo').agg(
        mean_R_ij_mbps  = ('r_ij_mbps',  'mean'),
        mean_T_base_ms  = ('t_base_ms',  'mean'),
        mean_tx_ms      = ('tx_ms',      'mean'),
        mean_network_ms = ('network_ms', 'mean'),
        mean_compute_ms = ('compute_ms', 'mean'),
        mean_queue_ms   = ('queue_ms',   'mean'),
        mean_total_ms   = ('total_lat_ms','mean'),
        mean_cost       = ('cost',       'mean'),
        mean_bytes      = ('transfer_bytes','mean'),
    ).reset_index()
    ep_agg = ep_df.groupby('algo').agg(
        ep_lat_mean = ('latency_ms','mean'),
        ep_lat_std  = ('latency_ms','std'),
        ep_cost_mean= ('cost_usd',  'mean'),
        ep_cost_std = ('cost_usd',  'std'),
    ).reset_index()
    table = agg.merge(ep_agg, on='algo')
    table_path = os.path.join(OUT_DIR, 'E02_detailed_metrics.csv')
    table.to_csv(table_path, index=False, float_format='%.4f')
    print(f"  Table saved → {table_path}")
    print(table[['algo','mean_R_ij_mbps','mean_T_base_ms','mean_tx_ms',
                 'mean_compute_ms','ep_lat_mean','ep_cost_mean']].to_string(index=False))

    # ── 图 1: latency-cost scatter ─────────────────────────────────────────
    algo_colors = {
        'STAR-PPO-Overlay':  '#1565C0',
        'STAR-PPO-Wireless': '#42A5F5',
        'Greedy':              '#4CAF50',
        'Random':              '#9E9E9E',
    }
    algo_markers = {
        'STAR-PPO-Overlay':  'o',
        'STAR-PPO-Wireless': 's',
        'Greedy':              '^',
        'Random':              'x',
    }
    algo_labels = {
        'STAR-PPO-Overlay':  'STAR-PPO-Overlay (Ours)',
        'STAR-PPO-Wireless': 'STAR-PPO-Wireless',
        'Greedy':              'Greedy',
        'Random':              'Random',
    }

    fig1, ax1 = plt.subplots(figsize=(9, 7))
    for algo in ['STAR-PPO-Overlay', 'STAR-PPO-Wireless', 'Greedy', 'Random']:
        sub = ep_df[ep_df['algo'] == algo]
        if sub.empty: continue
        ax1.scatter(sub['cost_usd'], sub['latency_ms'],
                    c=algo_colors[algo], marker=algo_markers[algo],
                    alpha=0.45, s=28, label=algo_labels[algo])
        # mean point (larger, filled)
        ax1.scatter(sub['cost_usd'].mean(), sub['latency_ms'].mean(),
                    c=algo_colors[algo], marker=algo_markers[algo],
                    s=180, edgecolors='black', linewidths=1.2, zorder=5)

    ax1.set_xlabel('Episode Cost (USD)')
    ax1.set_ylabel('Episode Latency (ms)')
    ax1.set_title('Service-Overlay Model: Latency-Cost Trade-off')
    ax1.legend(loc='upper right', fontsize=18)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.text(0.02, 0.98,
             'Large markers = mean\nService-Overlay model: $T_{comm}=T_{base}+D/R$',
             transform=ax1.transAxes, ha='left', va='top', fontsize=14,
             bbox=dict(facecolor='white', edgecolor='#ccc', alpha=0.85, pad=3))
    plt.tight_layout()
    fig1_path = os.path.join(OUT_DIR, 'E02_latency_cost_scatter.png')
    fig1.savefig(fig1_path, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print(f"  Fig1 saved → {fig1_path}")

    # ── 图 2: communication delay breakdown ──────────────────────────────────
    fig2, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig2.suptitle('Service-Overlay Model: Communication Delay Breakdown', fontsize=24)

    # Panel A: 堆叠柱 — 每步延迟分项 (T_base / tx / compute / queue)
    ax = axes[0]
    ax.set_title('(A) Per-Step Latency Components')
    plot_algos = ['STAR-PPO-Overlay', 'STAR-PPO-Wireless', 'Greedy', 'Random']
    plot_labels_short = ['STAR-PPO\n(Overlay)', 'STAR-PPO\n(Wireless)', 'Greedy', 'Random']
    bar_colors = [algo_colors[a] for a in plot_algos]
    x = np.arange(len(plot_algos))

    def get_mean(algo, col):
        r = table[table['algo'] == algo]
        return float(r[col].iloc[0]) if not r.empty else 0.

    t_base_m  = [get_mean(a, 'mean_T_base_ms')  for a in plot_algos]
    tx_m      = [get_mean(a, 'mean_tx_ms')       for a in plot_algos]
    compute_m = [get_mean(a, 'mean_compute_ms')  for a in plot_algos]
    queue_m   = [get_mean(a, 'mean_queue_ms')    for a in plot_algos]

    b1 = ax.bar(x, t_base_m,  color='#EF9A9A', label='$T_{base}$ (propagation)')
    b2 = ax.bar(x, tx_m,      bottom=t_base_m, color='#42A5F5', label='$D/R_{ij}$ (transmission)')
    bot2 = [a+b for a,b in zip(t_base_m, tx_m)]
    b3 = ax.bar(x, compute_m, bottom=bot2, color='#66BB6A', label='Compute')
    bot3 = [a+b for a,b in zip(bot2, compute_m)]
    b4 = ax.bar(x, queue_m,   bottom=bot3, color='#FFA726', label='Queue wait')

    ax.set_xticks(x); ax.set_xticklabels(plot_labels_short)
    ax.set_ylabel('Mean Per-Step Latency (ms)')
    ax.legend(loc='upper right')
    for i, (tb, tx, cp, qu) in enumerate(zip(t_base_m, tx_m, compute_m, queue_m)):
        total = tb+tx+cp+qu
        ax.text(i, total+1, f'{total:.1f}', ha='center', va='bottom', fontsize=16, fontweight='bold')

    # Panel B: 网络延迟成分占比 饼/堆叠条 — 只看网络部分
    ax = axes[1]
    ax.set_title('(B) Network Delay Components\n(T_base vs. Transmission D/R)')

    # 对比 service-overlay vs wireless 的网络延迟来源
    # Wireless 的网络延迟来自 Shannon 模型 (我们记录的是 network_ms)
    # Service-overlay 分解为 T_base + tx
    net_algos  = ['STAR-PPO-Overlay', 'STAR-PPO-Wireless']
    net_labels = ['Service-Overlay\n(T_base + D/R)', 'Wireless\n(Shannon)']
    net_colors = ['#1565C0', '#42A5F5']
    x2 = np.arange(len(net_algos))

    tbase_vals = [get_mean(a, 'mean_T_base_ms')  for a in net_algos]
    tx_vals    = [get_mean(a, 'mean_tx_ms')       for a in net_algos]
    net_vals   = [get_mean(a, 'mean_network_ms')  for a in net_algos]

    # For wireless, decompose as: propagation (estimated) + Shannon transmission
    # net_ms(wireless) = channel_latency + link_latency_ms
    # We show net_ms as a single bar for wireless since T_base/tx not separately tracked
    b_tbase = ax.bar(x2, tbase_vals, color='#EF9A9A', label='$T_{base}$ (propagation/RTT)')
    b_tx    = ax.bar(x2, tx_vals,    bottom=tbase_vals, color='#42A5F5', label='$D/R_{ij}$ (transmission)')
    # For wireless, show total net_ms as a different color bar overlaid
    ax.bar([1], [net_vals[1]], color='#90CAF9', alpha=0.7, label='Shannon capacity model', width=0.4)

    ax.set_xticks(x2); ax.set_xticklabels(net_labels)
    ax.set_ylabel('Mean Per-Step Network Latency (ms)')
    ax.legend()

    # Annotate values
    for i, (tb, tx, nv) in enumerate(zip(tbase_vals, tx_vals, net_vals)):
        total = max(tb+tx, nv)
        ax.text(i, total+0.03, f'{nv:.2f}ms\ntotal', ha='center', va='bottom', fontsize=16, fontweight='bold')

    ax.text(0.02, 0.98,
            'Service-overlay model decomposes\nnetwork latency into:\n'
            '  • $T_{base}$: link propagation/RTT\n'
            '  • $D/R_{ij}$: data transmission',
            transform=ax.transAxes, ha='left', va='top', fontsize=14,
            bbox=dict(facecolor='#FFFDE7', edgecolor='#F9A825', alpha=0.9, pad=4))

    plt.tight_layout()
    fig2_path = os.path.join(OUT_DIR, 'E02_comm_delay_breakdown.png')
    fig2.savefig(fig2_path, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f"  Fig2 saved → {fig2_path}")

    print("\n=== E02 图表生成完成 ===")
    print(f"  1. {fig1_path}")
    print(f"  2. {fig2_path}")
    print(f"  Table: {table_path}")


if __name__ == '__main__':
    main()
