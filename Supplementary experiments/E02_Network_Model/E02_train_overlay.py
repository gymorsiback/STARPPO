#!/usr/bin/env python3
"""
E02 TopoFreeRL 重训练 — Service-Overlay 链路模型
================================================
将无线信道 (Shannon _compute_channel_latency) 替换为:
    T_comm(i,j) = T_ij^base + D / R_ij
其他训练设置与原 train.py 完全一致 (100 epochs, 200 ep/epoch, PPO)。

输出模型: results/TopoFreeRL_Overlay/models/LATEST_Server1_Trap_seed{N}_final.pt
用法:
    python E02_train_overlay.py --seed 42
    (3 个 seed 分别在独立进程中后台运行)
"""
import os, sys, json, random
import numpy as np
import torch
import argparse

ROOT = '.'
sys.path.insert(0, ROOT)

from env import WorkflowDataset, WorkflowMoEEnv
from utils import ensure_dir, generate_run_id, softmax
from TopoFreeRL.env_augmented import AugmentedWorkflowEnv
from TopoFreeRL.agent import StarPPOAgent
from utils import haversine_km

# ─── 链路模型常数 ────────────────────────────────────────────────────────────
BITS_PER_TOKEN   = 32     # 1 token ≈ 4 bytes = 32 bits
DEFAULT_BW_MBPS  = 100.0  # 无链路记录时的回退带宽
PROP_MS_PER_KM   = 0.005  # 光纤传播延迟


# ─── Service-Overlay 增强环境 ─────────────────────────────────────────────
class OverlayAugmentedEnv(AugmentedWorkflowEnv):
    """
    在 AugmentedWorkflowEnv 基础上，将 step() 的网络延迟换成
    service-overlay 公式:  T_net = T_base(i,j) + D / R_ij
    奖励函数格式与原 env.step() 完全相同。
    """

    def step(self, action_idx):
        mi     = self.actions[action_idx]
        server = self.servers[mi.server_id]

        _, req_id, _ = self.cur_steps[self.step_idx]
        if req_id is not None and req_id in self.ds.req_tokens:
            in_tok, out_tok = self.ds.req_tokens[req_id]
        else:
            size = float(self.cur_task['TaskSize'])
            in_tok, out_tok = int(0.6 * size), int(0.4 * size)
        tokens    = in_tok + out_tok
        data_bits = tokens * BITS_PER_TOKEN

        # ── Service-Overlay 网络延迟 ──────────────────────────────────────
        link_latency_ms = 0.0
        tx_ms           = 0.0

        if self.step_idx == 0:
            d_km = haversine_km(self.cur_task['TaskLongitude'],
                                self.cur_task['TaskLatitude'],
                                server.lon, server.lat)
            # 第一跳: 用户→服务器，用传播 + 默认带宽
            link_latency_ms = max(d_km, 0.1) * PROP_MS_PER_KM
            tx_ms           = data_bits / (DEFAULT_BW_MBPS * 1e6) * 1000.0

            # 陷阱服务器: 保留高延迟惩罚
            if self.trap_latency > 0 and server.server_id in self.trap_server_ids:
                if np.random.random() < self.trap_packet_loss_prob:
                    link_latency_ms = self.trap_bad_latency
                    tx_ms = 0.0
                else:
                    link_latency_ms = self.trap_good_latency
                    tx_ms = 0.0
        else:
            if self.prev_server_id is not None:
                prev   = self.servers[self.prev_server_id]
                d_km   = haversine_km(prev.lon, prev.lat, server.lon, server.lat)
                link_key = (self.prev_server_id, mi.server_id)

                involves_trap = (self.prev_server_id in self.trap_server_ids or
                                 mi.server_id in self.trap_server_ids)
                if involves_trap and self.trap_latency > 0:
                    if np.random.random() < self.trap_packet_loss_prob:
                        link_latency_ms = self.trap_bad_latency
                    else:
                        link_latency_ms = self.trap_good_latency
                    tx_ms = 0.0
                elif link_key in self.link_latency:
                    link_latency_ms = self.link_latency[link_key]
                    bw_mbps = self.link_bandwidth.get(link_key, DEFAULT_BW_MBPS)
                    tx_ms   = data_bits / (bw_mbps * 1e6) * 1000.0
                else:
                    link_latency_ms = max(d_km, 0.1) * PROP_MS_PER_KM
                    tx_ms           = data_bits / (DEFAULT_BW_MBPS * 1e6) * 1000.0
            else:
                link_latency_ms = PROP_MS_PER_KM
                tx_ms           = data_bits / (DEFAULT_BW_MBPS * 1e6) * 1000.0

        network_ms = link_latency_ms + tx_ms

        # ── 计算 & 队列延迟 ──────────────────────────────────────────────
        speed_tps  = max(server.normalized_compute, 1e-6) * self.base_speed_tps
        compute_ms = (tokens / speed_tps) * 1000.0
        queue_ms   = max(0.0, self.busy_until[server.server_id] - self.current_time_ms)
        step_latency_ms = network_ms + compute_ms + queue_ms

        # ── 更新环境状态 ─────────────────────────────────────────────────
        start_time  = max(self.current_time_ms, self.busy_until[server.server_id])
        finish_time = start_time + compute_ms
        self.busy_until[server.server_id] = finish_time
        self.current_time_ms = start_time + step_latency_ms

        cost = (tokens / 1000.0) * mi.cost_per_token * server.cost_multiplier

        switch_penalty_ms = 0.0
        if self.prev_server_id is not None:
            if self.servers[self.prev_server_id].region != server.region:
                self.ep_switches += 1
                switch_penalty_ms = self.lambda_switch

        # ── 奖励 (与原 env.step() 完全相同的归一化) ─────────────────────
        lat_normalized  = np.clip(step_latency_ms / 5000.0, 0.0, 1.0)
        r_L = -4.0 * lat_normalized

        cost_best, cost_worst = 0.00045, 0.15
        cost_normalized = np.clip((cost - cost_best) / (cost_worst - cost_best), 0.0, 1.0)
        r_C = -1.5 * cost_normalized

        r_S = -0.4 if switch_penalty_ms > 0 else 0.0

        # ── 历史更新 ─────────────────────────────────────────────────────
        self.prev_server_id = server.server_id
        alpha = 0.5
        self.server_latency_history[server.server_id] = (
            alpha * step_latency_ms +
            (1 - alpha) * self.server_latency_history[server.server_id])
        self.ep_latency.append(step_latency_ms)
        self.ep_cost.append(cost)
        self.ep_total_latency += step_latency_ms

        self.step_idx += 1
        done = (self.step_idx >= len(self.cur_steps))

        info = {
            'latency_ms':      step_latency_ms,
            'cost':            cost,
            'network_ms':      network_ms,
            'compute_ms':      compute_ms,
            'queue_ms':        queue_ms,
            'switch_penalty_ms': switch_penalty_ms,
            'server_id':       server.server_id,
            'region':          server.region,
        }
        if done:
            info['episode_total_latency_ms'] = self.ep_total_latency
            info['episode_total_cost']       = float(sum(self.ep_cost))
            info['episode_switches']         = self.ep_switches
        return self._get_state(), (r_L, r_C, r_S), done, info


# ─── 与 train.py 相同的辅助函数 ──────────────────────────────────────────
def build_server_model_mapping(ds, env):
    server_ids = sorted(list(env.servers.keys()))
    sid2idx    = {sid: i for i, sid in enumerate(server_ids)}
    mapping    = {i: {} for i in range(len(server_ids))}
    for mi in ds.model_instances:
        idx = sid2idx.get(mi.server_id)
        if idx is not None:
            mapping[idx].setdefault(mi.model_type, []).append(mi.idx)
    return mapping, server_ids


def map_server_action_to_instance(server_idx, required_model_type, mapping, ds, fallback=0):
    if server_idx in mapping:
        instances = mapping[server_idx].get(required_model_type, [])
        if instances:
            return instances[0]
    for mi in ds.model_instances:
        if mi.model_type == required_model_type:
            return mi.idx
    return fallback


def compute_resource_weights(env, dwa_weights=None):
    server_ids = sorted(list(env.servers.keys()))
    caps       = np.array([env.servers[sid].normalized_compute for sid in server_ids], np.float32)
    cost_mults = np.array([env.servers[sid].cost_multiplier    for sid in server_ids], np.float32)
    cost_adv   = 1.0 - np.clip(cost_mults / 2.0, 0, 1.)
    busy_times = np.array([max(0., env.busy_until[sid] - env.current_time_ms)
                           for sid in server_ids], np.float32)
    norm_queues = np.clip(busy_times / 5000., 0., 1.)
    weights = caps / (1.0 + 0.30 * norm_queues)
    weights = weights * (0.5 + 0.5 * cost_adv)
    net_q   = np.ones(len(server_ids), np.float32)
    if hasattr(env, 'link_latency') and env.link_latency:
        for i, sid in enumerate(server_ids):
            outs = [lat for (src, _), lat in env.link_latency.items() if src == sid]
            if outs:
                net_q[i] = np.exp(-np.mean(outs) / 500.)
    weights = weights * net_q
    mx = weights.max()
    weights = weights / mx if mx > 1e-9 else np.ones_like(weights) / len(server_ids)
    return weights


# ─── 训练主函数 ────────────────────────────────────────────────────────────
def train(seed, total_epochs=100, episodes_per_epoch=200,
          lr=3e-4, batch_size=1024, device='cpu'):

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    DATA_ROOT  = os.path.join(ROOT, 'data1')
    REGION     = 'Server1_Trap'
    RESULTS    = os.path.join(ROOT, 'results', 'TopoFreeRL_Overlay')
    MODELS_DIR = os.path.join(RESULTS, 'models')
    ensure_dir(MODELS_DIR)

    ds  = WorkflowDataset(DATA_ROOT, split='train', regions=[REGION])
    env = OverlayAugmentedEnv(ds)
    num_servers = len(env.servers)

    server_model_mapping, _ = build_server_model_mapping(ds, env)
    agent = StarPPOAgent(state_dim=10, num_servers=num_servers, lr=lr, device=device)

    run_id   = f'overlay_seed{seed}'
    run_dir  = os.path.join(RESULTS, 'logs', run_id)
    ensure_dir(run_dir)

    print(f"=== TopoFreeRL Overlay Training  seed={seed}  region={REGION} ===")
    print(f"    epochs={total_epochs}  ep/epoch={episodes_per_epoch}  device={device}")

    lr_lambda        = lambda ep: 1.0 - 0.8 * (ep / total_epochs)
    actor_scheduler  = torch.optim.lr_scheduler.LambdaLR(agent.actor_optimizer,  lr_lambda)
    critic_scheduler = torch.optim.lr_scheduler.LambdaLR(agent.critic_optimizer, lr_lambda)

    w             = np.array([0.45, 0.40, 0.15], np.float32)
    loss_mov_avg  = np.zeros(3)
    T             = 3.0
    freeze_epoch  = int(total_epochs * 0.8)
    dwa_start     = 3

    all_lats, all_costs, all_rets, weights_hist = [], [], [], []
    L_hist = {'L': [], 'C': [], 'S': []}

    for epoch in range(total_epochs):
        ent_coef = max(0.002, 0.03 - 0.028 * epoch / (total_epochs * 0.9))
        progress = epoch / total_epochs
        guidance_alpha = 0.6 + 0.6 * progress

        # DWA 更新
        if epoch >= dwa_start and epoch < freeze_epoch:
            cur_losses = np.array([
                np.mean(ep_L_vals) if 'ep_L_vals' in dir() and len(ep_L_vals) > 0 else 0.,
                np.mean(ep_C_vals) if 'ep_C_vals' in dir() and len(ep_C_vals) > 0 else 0.,
                np.mean(ep_S_vals) if 'ep_S_vals' in dir() and len(ep_S_vals) > 0 else 0.,
            ])
            if np.all(loss_mov_avg == 0):
                loss_mov_avg = cur_losses + 1e-6
            else:
                loss_mov_avg = 0.15 * cur_losses + 0.85 * loss_mov_avg
            if np.mean(np.abs(cur_losses)) > 1e-5:
                r_n   = np.clip(cur_losses / (loss_mov_avg + 1e-7), 0.7, 1.3)
                exp_w = np.exp(r_n / T)
                w_k   = len(w) * exp_w / (np.sum(exp_w) + 1e-8)
                w_new = softmax(w_k)
                w     = np.clip(0.3 * w_new + 0.7 * w, 0.15, 0.7)
                w     = w / w.sum()

        ep_L_vals, ep_C_vals, ep_S_vals = [], [], []
        mem_s, mem_w, mem_a, mem_lp, mem_r, mem_v = [], [], [], [], [], []
        ep_rets, ep_lats, ep_costs = [], [], []

        for ep in range(episodes_per_epoch):
            task = random.choice(ds.tasks)
            env.reset(task)
            s_vec = env.get_augmented_state(dwa_weights=w)
            done  = False

            tj_s, tj_w, tj_a, tj_lp, tj_r, tj_v = [], [], [], [], [], []
            ep_ret = ep_l = ep_c = 0.

            while not done:
                r_w = compute_resource_weights(env, dwa_weights=w)
                s_t = torch.FloatTensor(s_vec).unsqueeze(0).to(device)
                w_t = torch.FloatTensor(r_w).unsqueeze(0).to(device)

                action_idx, log_prob, value = agent.act(s_t, w_t, guidance_alpha=guidance_alpha)

                _, _, req_type = env.cur_steps[env.step_idx]
                if req_type is None:
                    req_type = env.cur_task['RequiredModelTypes'][env.step_idx]
                real_action = map_server_action_to_instance(
                    action_idx, str(req_type), server_model_mapping, ds)

                _, (rL, rC, rS), done, info = env.step(real_action)
                next_s = env.get_augmented_state(dwa_weights=w)
                r_scalar = w[0]*rL + w[1]*rC + w[2]*rS

                tj_s.append(s_vec); tj_w.append(r_w); tj_a.append(action_idx)
                tj_lp.append(log_prob); tj_r.append(r_scalar); tj_v.append(value)

                s_vec   = next_s
                ep_ret += r_scalar
                ep_l   += info['latency_ms']
                ep_c   += info['cost']
                ep_L_vals.append(-rL); ep_C_vals.append(-rC); ep_S_vals.append(-rS)

            # GAE
            returns = []; gae = 0; next_v = 0
            for t in reversed(range(len(tj_r))):
                delta = tj_r[t] + 0.99 * next_v - tj_v[t]
                gae   = delta + 0.99 * 0.95 * gae
                returns.insert(0, gae + tj_v[t])
                next_v = tj_v[t]

            mem_s.extend(tj_s); mem_w.extend(tj_w); mem_a.extend(tj_a)
            mem_lp.extend(tj_lp); mem_r.extend(returns); mem_v.extend(tj_v)
            ep_rets.append(ep_ret); ep_lats.append(ep_l); ep_costs.append(ep_c)

        # PPO 更新
        s_t  = torch.FloatTensor(np.array(mem_s))
        w_t  = torch.FloatTensor(np.array(mem_w))
        a_t  = torch.LongTensor(np.array(mem_a))
        lp_t = torch.FloatTensor(np.array(mem_lp))
        r_t  = torch.FloatTensor(np.array(mem_r))
        v_t  = torch.FloatTensor(np.array(mem_v))
        adv_t = r_t - v_t

        idx = np.arange(len(s_t))
        ag_loss = 0.
        for _ in range(10):
            np.random.shuffle(idx)
            for start in range(0, len(idx), batch_size):
                i = idx[start:start+batch_size]
                ag_loss += agent.update_from_batch(
                    s_t[i], w_t[i], a_t[i], lp_t[i], r_t[i], adv_t[i],
                    entropy_coef=ent_coef)

        avg_lat = np.mean(ep_lats)
        print(f"  [seed{seed}] Epoch {epoch+1:3d}/{total_epochs} | "
              f"Lat={avg_lat:.1f}ms | Ret={np.mean(ep_rets):.3f} | "
              f"Cost={np.mean(ep_costs):.4f} | W={np.round(w,2)}")

        all_lats.extend(ep_lats); all_costs.extend(ep_costs); all_rets.extend(ep_rets)
        weights_hist.append(w.copy())
        L_hist['L'].append(np.mean(ep_L_vals))
        L_hist['C'].append(np.mean(ep_C_vals))
        L_hist['S'].append(np.mean(ep_S_vals))

        if (epoch+1) % 10 == 0:
            ckpt = os.path.join(MODELS_DIR, f'overlay_seed{seed}_epoch{epoch+1}.pt')
            torch.save(agent.actor.state_dict(), ckpt)
            print(f"  [seed{seed}] Checkpoint saved → {ckpt}")

        actor_scheduler.step()
        critic_scheduler.step()

    # 保存最终模型
    final_path = os.path.join(MODELS_DIR, f'LATEST_Server1_Trap_seed{seed}_final.pt')
    torch.save(agent.actor.state_dict(), final_path)
    print(f"\n  [seed{seed}] Final model → {final_path}")

    # 保存训练数据
    np.savez_compressed(os.path.join(run_dir, 'training_data.npz'),
        episode_latency=np.array(all_lats),
        episode_cost=np.array(all_costs),
        episode_returns=np.array(all_rets),
        weights_hist=np.array(weights_hist),
        L_hist_L=np.array(L_hist['L']),
        L_hist_C=np.array(L_hist['C']),
        L_hist_S=np.array(L_hist['S']))
    print(f"  [seed{seed}] Training data saved → done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',   type=int, default=42)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    train(seed=args.seed, total_epochs=args.epochs, device=args.device)
