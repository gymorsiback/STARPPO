#!/usr/bin/env python3
"""
E02  Service-Oriented Network Model Validation
================================================
系统模型(system.tex §2.1)声明采用 service-overlay 链路:
    T_comm(i,j) = T_ij^base + D / R_ij
其中:
  T_ij^base = BaseLatencyMs (传播 + 路由开销, 来自 network_links.csv)
  D         = tokens × 32 bits
  R_ij      = BandwidthMbps × 1e6 (bps)

当前 env 使用 Shannon 无线信道 (随机衰落), 不直接利用 BandwidthMbps。
本实验:
  1. 用 service-overlay 公式替换无线信道模型, 在 Server1_Trap 上重新评估
  2. 对比四种算法: TopoFreeRL, Greedy-Overlay, Greedy-BW, Random
  3. 量化带宽感知对路由决策的影响 (链路带宽利用率、通信延迟组件)
  4. 生成 E02_service_overlay.png + E02_service_overlay_table.csv
"""
import os, sys, random, math
import numpy as np
import torch
import pandas as pd

sys.path.insert(0, '.')
from env import WorkflowDataset, WorkflowMoEEnv
from TopoFreeRL.env_augmented import AugmentedWorkflowEnv
from TopoFreeRL.inference import compute_resource_weights
from utils import haversine_km

# ─── 路径 ──────────────────────────────────────────────────────────────────
DATA_ROOT = './data1'
REGION    = 'Server1_Trap'
OUT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = './results/TopoFreeRL/models'
SEEDS     = [42, 43, 44]
EPISODES  = 200
DEVICE    = 'cpu'

# bits per token assumption
BITS_PER_TOKEN = 32          # 1 token ≈ 4 bytes = 32 bits
DEFAULT_BW_MBPS = 100.0      # fallback (first hop: user→server), assumed wired
PROP_SPEED_MS_PER_KM = 0.005 # ~1.5x speed of light in fiber

# ─── Service-Overlay 通信延迟函数 ──────────────────────────────────────────
def overlay_comm_ms(src_id, dst_id, tokens, link_latency, link_bandwidth, d_km):
    """
    T_comm = T_base(src→dst) + D / R(src→dst)
    若无链路记录则退化为距离传播 + 默认带宽
    """
    data_bits = tokens * BITS_PER_TOKEN
    key = (src_id, dst_id)
    if key in link_latency:
        t_base = link_latency[key]         # ms
        bw_mbps = link_bandwidth.get(key, DEFAULT_BW_MBPS)
    else:
        # 无直连链路记录 → 用距离估算传播 + 默认高带宽
        t_base = max(d_km, 0.1) * PROP_SPEED_MS_PER_KM
        bw_mbps = DEFAULT_BW_MBPS
    tx_ms = data_bits / (bw_mbps * 1e6) * 1000.0  # ms
    return t_base + tx_ms, t_base, tx_ms, bw_mbps


# ─── Service-Overlay 包装 Env ─────────────────────────────────────────────
class ServiceOverlayEnv(AugmentedWorkflowEnv):
    """
    覆盖 env 中的网络延迟计算, 使用 service-overlay 公式。
    继承 AugmentedWorkflowEnv 以保留 get_augmented_state() 功能。
    同时记录每步延迟分量。
    """
    def reset(self, task):
        state = super().reset(task)
        self.step_records = []          # list of dicts per step
        return state

    def _overlay_network_ms(self, src_id, dst_id, tokens, d_km):
        comm, t_base, t_tx, bw = overlay_comm_ms(
            src_id, dst_id, tokens,
            self.link_latency, self.link_bandwidth, d_km)
        return comm, t_base, t_tx, bw

    def estimate_step_overlay(self, action_idx):
        """Service-overlay 估计延迟 (不修改环境状态)"""
        mi = self.actions[action_idx]
        server = self.servers[mi.server_id]
        _, req_id, _ = self.cur_steps[self.step_idx]
        if req_id is not None and req_id in self.ds.req_tokens:
            in_tok, out_tok = self.ds.req_tokens[req_id]
        else:
            size = float(self.cur_task['TaskSize'])
            in_tok, out_tok = int(0.6*size), int(0.4*size)
        tokens = in_tok + out_tok

        if self.step_idx == 0:
            d_km = haversine_km(self.cur_task['TaskLongitude'],
                                self.cur_task['TaskLatitude'],
                                server.lon, server.lat)
            # 第一跳: 用户→服务器，无 src_id
            data_bits = tokens * BITS_PER_TOKEN
            t_base = max(d_km, 0.1) * PROP_SPEED_MS_PER_KM
            t_tx   = data_bits / (DEFAULT_BW_MBPS * 1e6) * 1000.0
            net_ms = t_base + t_tx
        else:
            if self.prev_server_id is not None:
                prev = self.servers[self.prev_server_id]
                d_km = haversine_km(prev.lon, prev.lat, server.lon, server.lat)
                net_ms, _, _, _ = self._overlay_network_ms(
                    self.prev_server_id, mi.server_id, tokens, d_km)
            else:
                d_km = 1.0
                data_bits = tokens * BITS_PER_TOKEN
                net_ms = d_km * PROP_SPEED_MS_PER_KM + data_bits/(DEFAULT_BW_MBPS*1e6)*1000.0

        speed_tps = max(server.normalized_compute, 1e-6) * self.base_speed_tps
        compute_ms = (tokens / speed_tps) * 1000.0
        queue_ms   = max(0.0, self.busy_until[server.server_id] - self.current_time_ms)
        return net_ms + compute_ms + queue_ms, net_ms, compute_ms, queue_ms

    def step_overlay(self, action_idx):
        """执行 step，但网络延迟用 service-overlay 公式; 返回 (state,reward,done,info,record)"""
        mi = self.actions[action_idx]
        server = self.servers[mi.server_id]
        _, req_id, _ = self.cur_steps[self.step_idx]
        if req_id is not None and req_id in self.ds.req_tokens:
            in_tok, out_tok = self.ds.req_tokens[req_id]
        else:
            size = float(self.cur_task['TaskSize'])
            in_tok, out_tok = int(0.6*size), int(0.4*size)
        tokens = in_tok + out_tok

        # --- overlay comm latency ---
        if self.step_idx == 0:
            d_km = haversine_km(self.cur_task['TaskLongitude'],
                                self.cur_task['TaskLatitude'],
                                server.lon, server.lat)
            data_bits = tokens * BITS_PER_TOKEN
            t_base = max(d_km, 0.1) * PROP_SPEED_MS_PER_KM
            t_tx   = data_bits / (DEFAULT_BW_MBPS * 1e6) * 1000.0
            net_ms = t_base + t_tx
            bw_used = DEFAULT_BW_MBPS
        else:
            if self.prev_server_id is not None:
                prev = self.servers[self.prev_server_id]
                d_km = haversine_km(prev.lon, prev.lat, server.lon, server.lat)
                net_ms, t_base, t_tx, bw_used = self._overlay_network_ms(
                    self.prev_server_id, mi.server_id, tokens, d_km)
            else:
                d_km = 1.0
                data_bits = tokens * BITS_PER_TOKEN
                t_base = d_km * PROP_SPEED_MS_PER_KM
                t_tx   = data_bits / (DEFAULT_BW_MBPS * 1e6) * 1000.0
                net_ms = t_base + t_tx
                bw_used = DEFAULT_BW_MBPS

        speed_tps  = max(server.normalized_compute, 1e-6) * self.base_speed_tps
        compute_ms = (tokens / speed_tps) * 1000.0
        queue_ms   = max(0.0, self.busy_until[server.server_id] - self.current_time_ms)
        step_lat   = net_ms + compute_ms + queue_ms

        # cost / switch (same as base env)
        cost = (tokens / 1000.0) * mi.cost_per_token * server.cost_multiplier
        switch_ms = 0.0
        if self.prev_server_id is not None:
            if self.servers[self.prev_server_id].region != server.region:
                switch_ms = self.lambda_switch

        # update env state (replicate base env step logic)
        self.busy_until[server.server_id] = max(
            self.busy_until[server.server_id], self.current_time_ms) + compute_ms
        self.current_time_ms += step_lat + switch_ms
        self.ep_latency.append(step_lat)
        self.ep_cost.append(cost)
        if switch_ms > 0:
            self.ep_switches += 1
        self.ep_total_latency += step_lat + switch_ms
        self.server_latency_history[server.server_id] = step_lat
        self.prev_server_id = mi.server_id
        self.step_idx += 1

        record = dict(net_ms=net_ms, compute_ms=compute_ms,
                      queue_ms=queue_ms, bw_mbps=bw_used, step_lat=step_lat)
        self.step_records.append(record)

        done = (self.step_idx >= len(self.cur_steps))
        state = None if done else self._get_state()
        info  = {'ep_total_latency': self.ep_total_latency,
                 'ep_cost': sum(self.ep_cost),
                 'ep_switches': self.ep_switches}
        reward = -(step_lat / 1000.0)
        return state, reward, done, info, record


# ─── TopoFreeRL 推理 ───────────────────────────────────────────────────────
def run_topofreerl(env, ds, model_path, seed):
    """
    TopoFreeRL (StarPPO) inference under service-overlay model.
    Actor input: state(10-dim) + resource_weights(500-dim)
    """
    from TopoFreeRL.agent import StarPPOAgent
    NUM_SERVERS_MODEL = 500
    agent = StarPPOAgent(state_dim=10, num_servers=NUM_SERVERS_MODEL, device=DEVICE)
    agent.actor.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    agent.actor.eval()

    server_ids = sorted(env.servers.keys())[:NUM_SERVERS_MODEL]
    sid2idx    = {sid: i for i, sid in enumerate(server_ids)}
    w_dwa = np.array([0.45, 0.40, 0.15], np.float32)

    records = []
    for ep in range(EPISODES):
        np.random.seed(seed + ep)
        task = ds.tasks[ep % len(ds.tasks)]
        env.reset(task)
        done = False
        ep_lat = 0.

        while not done:
            valid = env.available_actions()
            if not valid:
                break

            # 10-dim augmented state (topology-aware)
            s_vec = env.get_augmented_state(dwa_weights=w_dwa)
            # 500-dim resource weights
            r_weights = compute_resource_weights(env, dwa_weights=w_dwa)
            if len(r_weights) < NUM_SERVERS_MODEL:
                padded = np.zeros(NUM_SERVERS_MODEL, np.float32)
                padded[:len(r_weights)] = r_weights
                r_weights = padded

            s_t = torch.tensor(s_vec,     dtype=torch.float32, device=DEVICE).unsqueeze(0)
            w_t = torch.tensor(r_weights, dtype=torch.float32, device=DEVICE).unsqueeze(0)

            with torch.no_grad():
                logits = agent.actor(s_t, w_t)
                # resource-weight guidance (same as inference.py)
                rw = torch.tensor(r_weights, device=DEVICE)
                rw_norm = (rw - rw.mean()) / (rw.std() + 1e-6)
                enhanced = logits.squeeze(0) + 1.0 * rw_norm * 1.5

            # map best server index → action
            idx2act = {}
            valid_server_indices = []
            for a in valid:
                sid = env.actions[a].server_id
                if sid in sid2idx:
                    i = sid2idx[sid]
                    idx2act.setdefault(i, a)
                    valid_server_indices.append(i)

            if not idx2act:
                action_idx = valid[0]
            else:
                # restrict to valid servers
                best_i = max(valid_server_indices,
                             key=lambda i: enhanced[i].item())
                action_idx = idx2act[best_i]

            _, _, done, _, rec = env.step_overlay(action_idx)
            ep_lat += rec['step_lat']

        records.append({'algo': 'STAR-PPO', 'seed': seed,
                        'ep': ep, 'latency': ep_lat, 'steps': len(env.step_records),
                        'step_records': list(env.step_records)})
    return records


# ─── Greedy-Overlay 基线 ──────────────────────────────────────────────────
def run_greedy_overlay(env, ds, seed):
    """每步选择 service-overlay 估计延迟最小的 action"""
    records = []
    for ep in range(EPISODES):
        np.random.seed(seed + ep)
        task = ds.tasks[ep % len(ds.tasks)]
        env.reset(task)
        done = False; ep_lat = 0.
        while not done:
            valid = env.available_actions()
            if not valid: break
            best_a, best_lat = None, 1e18
            for a in valid:
                lat, net, com, q = env.estimate_step_overlay(a)
                if lat < best_lat:
                    best_lat, best_a = lat, a
            _, _, done, _, rec = env.step_overlay(best_a)
            ep_lat += rec['step_lat']
        records.append({'algo': 'Greedy-Overlay', 'seed': seed,
                        'ep': ep, 'latency': ep_lat, 'steps': len(env.step_records),
                        'step_records': list(env.step_records)})
    return records


# ─── Greedy-BW 基线 (带宽感知贪心) ────────────────────────────────────────
def run_greedy_bw(env, ds, seed):
    """先选高带宽链路，带宽相同时按估计延迟排"""
    records = []
    for ep in range(EPISODES):
        np.random.seed(seed + ep)
        task = ds.tasks[ep % len(ds.tasks)]
        env.reset(task)
        done = False; ep_lat = 0.
        while not done:
            valid = env.available_actions()
            if not valid: break
            # pick best BW then best latency
            best_a, best_key = None, (1e18, 1e18)
            for a in valid:
                sid = env.actions[a].server_id
                if env.prev_server_id is not None:
                    key = (env.prev_server_id, sid)
                    bw = env.link_bandwidth.get(key, DEFAULT_BW_MBPS)
                else:
                    bw = DEFAULT_BW_MBPS
                lat_est, _, _, _ = env.estimate_step_overlay(a)
                ckey = (-bw, lat_est)  # higher BW better → negative
                if ckey < best_key:
                    best_key, best_a = ckey, a
            _, _, done, _, rec = env.step_overlay(best_a)
            ep_lat += rec['step_lat']
        records.append({'algo': 'Greedy-BW', 'seed': seed,
                        'ep': ep, 'latency': ep_lat, 'steps': len(env.step_records),
                        'step_records': list(env.step_records)})
    return records


# ─── Random 基线 ─────────────────────────────────────────────────────────
def run_random(env, ds, seed):
    records = []
    for ep in range(EPISODES):
        np.random.seed(seed + ep); random.seed(seed + ep)
        task = ds.tasks[ep % len(ds.tasks)]
        env.reset(task)
        done = False; ep_lat = 0.
        while not done:
            valid = env.available_actions()
            if not valid: break
            a = random.choice(valid)
            _, _, done, _, rec = env.step_overlay(a)
            ep_lat += rec['step_lat']
        records.append({'algo': 'Random', 'seed': seed,
                        'ep': ep, 'latency': ep_lat, 'steps': len(env.step_records),
                        'step_records': list(env.step_records)})
    return records


# ─── 无线模型对照 (原 env.step) ───────────────────────────────────────────
def run_wireless_topofreerl(env_aug, ds, model_path, seed):
    """用原始 env.step (无线信道模型) 运行 TopoFreeRL, 仅收集总延迟。
    env_aug 须为 AugmentedWorkflowEnv 实例以获取 10-dim state。
    """
    from TopoFreeRL.agent import StarPPOAgent
    NUM_SERVERS_MODEL = 500
    agent = StarPPOAgent(state_dim=10, num_servers=NUM_SERVERS_MODEL, device=DEVICE)
    agent.actor.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    agent.actor.eval()

    server_ids = sorted(env_aug.servers.keys())[:NUM_SERVERS_MODEL]
    sid2idx    = {sid: i for i, sid in enumerate(server_ids)}
    w_dwa = np.array([0.45, 0.40, 0.15], np.float32)

    lats = []
    for ep in range(EPISODES):
        np.random.seed(seed + ep)
        task = ds.tasks[ep % len(ds.tasks)]
        env_aug.reset(task)
        done = False; ep_lat = 0.
        s_vec = env_aug.get_augmented_state(dwa_weights=w_dwa)
        while not done:
            valid = env_aug.available_actions()
            if not valid: break
            r_weights = compute_resource_weights(env_aug, dwa_weights=w_dwa)
            if len(r_weights) < NUM_SERVERS_MODEL:
                padded = np.zeros(NUM_SERVERS_MODEL, np.float32)
                padded[:len(r_weights)] = r_weights
                r_weights = padded
            s_t = torch.tensor(s_vec,     dtype=torch.float32, device=DEVICE).unsqueeze(0)
            w_t = torch.tensor(r_weights, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                logits = agent.actor(s_t, w_t)
                rw = torch.tensor(r_weights, device=DEVICE)
                rw_norm = (rw - rw.mean()) / (rw.std() + 1e-6)
                enhanced = logits.squeeze(0) + 1.0 * rw_norm * 1.5
            idx2act = {}
            valid_idxs = []
            for a in valid:
                sid = env_aug.actions[a].server_id
                if sid in sid2idx:
                    i = sid2idx[sid]
                    idx2act.setdefault(i, a)
                    valid_idxs.append(i)
            if not idx2act:
                action_idx = valid[0]
            else:
                best_i = max(valid_idxs, key=lambda i: enhanced[i].item())
                action_idx = idx2act[best_i]
            _, (r_L, r_C, r_S), done, info = env_aug.step(action_idx)
            ep_lat += info['latency_ms']   # 实际延迟(ms), 非 reward
            s_vec = env_aug.get_augmented_state(dwa_weights=w_dwa)
        lats.append(ep_lat)
    return lats


# ─── Main ─────────────────────────────────────────────────────────────────
def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    import matplotlib
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

    print("=== E02 Service-Oriented Network Model Validation ===")
    print(f"Region: {REGION}  Episodes: {EPISODES}  Seeds: {SEEDS}")

    # 加载数据集
    ds  = WorkflowDataset(DATA_ROOT, split='test', regions=[REGION])
    env = ServiceOverlayEnv(ds, device=DEVICE)

    all_records = []

    # --- TopoFreeRL (service-overlay) ---
    for seed in SEEDS:
        mpath = os.path.join(MODELS_DIR, f'LATEST_{REGION}_seed{seed}_final.pt')
        if not os.path.exists(mpath):
            print(f"  [WARN] 模型不存在: {mpath}")
            continue
        print(f"  STAR-PPO seed={seed} ...")
        recs = run_topofreerl(env, ds, mpath, seed)
        all_records.extend(recs)

    # --- Greedy-Overlay ---
    for seed in SEEDS:
        print(f"  Greedy-Overlay seed={seed} ...")
        recs = run_greedy_overlay(env, ds, seed)
        all_records.extend(recs)

    # --- Greedy-BW ---
    for seed in SEEDS:
        print(f"  Greedy-BW seed={seed} ...")
        recs = run_greedy_bw(env, ds, seed)
        all_records.extend(recs)

    # --- Random ---
    for seed in SEEDS:
        print(f"  Random seed={seed} ...")
        recs = run_random(env, ds, seed)
        all_records.extend(recs)

    # --- TopoFreeRL under Wireless model (对照) ---
    env_aug = AugmentedWorkflowEnv(ds, device=DEVICE)   # original wireless env
    wireless_lats = []
    for seed in SEEDS:
        mpath = os.path.join(MODELS_DIR, f'LATEST_{REGION}_seed{seed}_final.pt')
        if not os.path.exists(mpath):
            continue
        print(f"  STAR-PPO-Wireless seed={seed} ...")
        lats = run_wireless_topofreerl(env_aug, ds, mpath, seed)
        wireless_lats.extend(lats)

    print("  推理完成，开始分析...")

    # ─── 统计汇总 ────────────────────────────────────────────────────────
    algo_lats   = {}
    algo_nets   = {}
    algo_comps  = {}
    algo_queues = {}
    algo_bws    = {}

    for r in all_records:
        a = r['algo']
        if a not in algo_lats:
            algo_lats[a]   = []
            algo_nets[a]   = []
            algo_comps[a]  = []
            algo_queues[a] = []
            algo_bws[a]    = []
        algo_lats[a].append(r['latency'])
        for sr in r['step_records']:
            algo_nets[a].append(sr['net_ms'])
            algo_comps[a].append(sr['compute_ms'])
            algo_queues[a].append(sr['queue_ms'])
            algo_bws[a].append(sr['bw_mbps'])

    algos = ['STAR-PPO', 'Greedy-Overlay', 'Greedy-BW', 'Random']
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9E9E9E']
    labels = ['STAR-PPO\n(Ours)', 'Greedy-\nOverlay', 'Greedy-BW', 'Random']

    # ─── CSV ─────────────────────────────────────────────────────────────
    rows = []
    for a in algos:
        if a not in algo_lats: continue
        lats = algo_lats[a]
        nets = algo_nets[a]
        rows.append({
            'Algorithm': a,
            'Mean_Total_Latency_ms': np.mean(lats),
            'Std_Total_Latency_ms':  np.std(lats),
            'P95_Total_Latency_ms':  np.percentile(lats, 95),
            'Mean_Net_ms':           np.mean(nets),
            'Mean_Compute_ms':       np.mean(algo_comps[a]),
            'Mean_Queue_ms':         np.mean(algo_queues[a]),
            'Pct_50Mbps_Links':      100*np.mean([b >= 50. for b in algo_bws[a]]),
        })
    # Add wireless comparison
    if wireless_lats:
        rows.append({
            'Algorithm': 'STAR-PPO-Wireless',
            'Mean_Total_Latency_ms': np.mean(wireless_lats),
            'Std_Total_Latency_ms':  np.std(wireless_lats),
            'P95_Total_Latency_ms':  np.percentile(wireless_lats, 95),
            'Mean_Net_ms': np.nan, 'Mean_Compute_ms': np.nan,
            'Mean_Queue_ms': np.nan, 'Pct_50Mbps_Links': np.nan,
        })
    df_out = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, 'E02_service_overlay_table.csv')
    df_out.to_csv(csv_path, index=False, float_format='%.2f')
    print(f"\n[E02] CSV saved → {csv_path}")
    print(df_out[['Algorithm','Mean_Total_Latency_ms','Std_Total_Latency_ms',
                  'Mean_Net_ms','Pct_50Mbps_Links']].to_string(index=False))

    # ─── 绘图 ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle('E02: Service-Oriented Network Model Validation', fontsize=24)

    # Panel A: 总延迟对比（service-overlay vs wireless，TopoFreeRL 专项）
    ax = axes[0]
    ax.set_title('(A) Network Model Comparison\n(STAR-PPO)')
    overlay_lats_tf = algo_lats.get('STAR-PPO', [])
    data_A = [overlay_lats_tf, wireless_lats] if wireless_lats else [overlay_lats_tf]
    labels_A = ['Service-Overlay', 'Wireless (Shannon)'] if wireless_lats else ['Service-Overlay']
    colors_A = ['#2196F3', '#FF7043']
    bp = ax.boxplot(data_A, patch_artist=True, widths=0.4,
                    medianprops=dict(color='white', linewidth=2.5))
    for patch, c in zip(bp['boxes'], colors_A):
        patch.set_facecolor(c); patch.set_alpha(0.8)
    ax.set_xticks(range(1, len(labels_A)+1))
    ax.set_xticklabels(labels_A)
    ax.set_ylabel('Total Episode Latency (ms)')
    ax.set_xlabel('Network Model')
    # annotate medians
    for i, d in enumerate(data_A):
        med = np.median(d)
        ax.text(i+1, med, f'{med:.0f}ms', ha='center', va='bottom', fontsize=16,
                fontweight='bold', fontsize=16, color='white',
                bbox=dict(facecolor=colors_A[i], edgecolor='none', alpha=0.8, pad=1))

    # Panel B: 延迟成分分解（算法对比，堆叠柱）
    ax = axes[1]
    ax.set_title('(B) Latency Component Breakdown\n(Service-Overlay Model)')
    x_pos = np.arange(len(algos))
    net_means   = [np.mean(algo_nets.get(a, [0]))   for a in algos]
    comp_means  = [np.mean(algo_comps.get(a, [0]))  for a in algos]
    queue_means = [np.mean(algo_queues.get(a, [0])) for a in algos]
    b1 = ax.bar(x_pos, net_means,   color='#42A5F5', label='Network (T_base+D/R)')
    b2 = ax.bar(x_pos, comp_means,  bottom=net_means, color='#66BB6A', label='Compute')
    b3 = ax.bar(x_pos, queue_means, bottom=[n+c for n,c in zip(net_means,comp_means)],
                color='#FFA726', label='Queue')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Mean Per-Step Latency (ms)')
    ax.legend(loc='upper right')
    ax.set_xlabel('Algorithm')
    # total annotation
    for i, (n, c, q) in enumerate(zip(net_means, comp_means, queue_means)):
        ax.text(i, n+c+q+0.5, f'{n+c+q:.1f}', ha='center', va='bottom', fontsize=16)

    # Panel C: 带宽利用率（高带宽链路占比）
    ax = axes[2]
    ax.set_title('(C) High-Bandwidth Link Preference\n(≥50 Mbps, %)')
    bw50_pcts = [100*np.mean([b >= 50. for b in algo_bws.get(a, [0])]) for a in algos]
    bars = ax.bar(x_pos, bw50_pcts, color=colors, alpha=0.85)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel('% Steps Using ≥50 Mbps Links')
    ax.set_ylim(0, 105)
    ax.axhline(80.3, linestyle='--', color='gray', linewidth=2.0, label='Avg availability (80.3%)')
    ax.legend()
    for bar, pct in zip(bars, bw50_pcts):
        ax.text(bar.get_x()+bar.get_width()/2, pct+0.5, f'{pct:.1f}%',
                ha='center', va='bottom', fontsize=16, fontweight='bold')
    ax.set_xlabel('Algorithm')

    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, 'E02_service_overlay.png')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\n[E02] 图已保存 → {fig_path}")

    # ─── 结果汇报 ─────────────────────────────────────────────────────────
    print("\n=== E02 结果汇报 ===")
    for row in rows:
        a = row['Algorithm']
        mu = row['Mean_Total_Latency_ms']
        std = row['Std_Total_Latency_ms']
        net = row['Mean_Net_ms']
        bw50 = row['Pct_50Mbps_Links']
        net_s  = f"{net:.1f}ms"  if not (isinstance(net,  float) and np.isnan(net))  else 'N/A'
        bw50_s = f"{bw50:.1f}%" if not (isinstance(bw50, float) and np.isnan(bw50)) else 'N/A'
        print(f"  {a:25s}: {mu:7.1f} ± {std:6.1f} ms  |  net={net_s}  |  50Mbps={bw50_s}")

    # 最优检查
    overlay_algos = {a: np.mean(algo_lats[a]) for a in algos if a in algo_lats}
    best_algo = min(overlay_algos, key=overlay_algos.get)
    tf_lat    = overlay_algos.get('STAR-PPO', 1e9)
    if best_algo != 'STAR-PPO':
        print(f"\n[WARN] STAR-PPO ({tf_lat:.1f}ms) 不是最优! 最优: {best_algo} ({overlay_algos[best_algo]:.1f}ms)")
    else:
        print(f"\n[OK] STAR-PPO ({tf_lat:.1f}ms) 是 service-overlay 模型下的最优算法.")


if __name__ == '__main__':
    main()
