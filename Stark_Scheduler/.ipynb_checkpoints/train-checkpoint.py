import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env import WorkflowDataset, WorkflowMoEEnv
from utils import ensure_dir, generate_run_id
from Stark_Scheduler.model import StarkScheduler
from Stark_Scheduler.dataset import OnlineExpertDataset, ExpertPolicy, collate_fn


def evaluate_model(model, env, dataset, device, num_episodes=30, seed=None):
    import random as py_random
    
    model.eval()

    if seed is not None:
        py_random.seed(seed)
        np.random.seed(seed)
    
    latencies = []
    costs = []
    
    tasks = dataset.tasks
    expert_data = OnlineExpertDataset(env)

    eval_tasks = np.random.choice(tasks, size=min(num_episodes, len(tasks)), replace=False)
    
    with torch.no_grad():
        for i, task in enumerate(eval_tasks):
            
            state_dict = env.reset(task)
            done = False
            ep_latency = 0
            ep_cost = 0
            
            while not done:
                available = env.available_actions()
                if not available:
                    break
                    
                task_feat, server_feats, _, avail_mask = expert_data.extract_structured_state(env, action_idx=available[0])
  
                task_feat_t = torch.FloatTensor(task_feat).unsqueeze(0).to(device)
                server_feats_t = torch.FloatTensor(server_feats).unsqueeze(0).to(device)
                avail_mask_t = torch.FloatTensor(avail_mask).unsqueeze(0).to(device)
                
                logits = model(task_feat_t, server_feats_t)
  
                masked_logits = logits.clone()
                masked_logits[avail_mask_t == 0] = -1e9
                
                server_idx = torch.argmax(masked_logits, dim=1).item()

                server_ids = sorted(list(env.servers.keys()))
                target_server_id = server_ids[server_idx]

                action = available[0]
                for act_idx in available:
                    mi = env.actions[act_idx]
                    if mi.server_id == target_server_id:
                        action = act_idx
                        break
 
                next_state, rewards, done, info = env.step(action)
                ep_latency += info['latency_ms']
                ep_cost += info['cost']
                state_dict = next_state
                
            latencies.append(ep_latency)
            costs.append(ep_cost)
    
    model.train()
    return np.mean(latencies), np.mean(costs)


def train(args):
    if args.run_idx > 0:
        print(f"\n{'='*60}")
        print(f"[Run {args.run_idx}/{args.total_runs}] Starting training...")
        print(f"{'='*60}")

    print(f"Loading dataset from {args.data_path}...")
    regions = getattr(args, 'regions', ['Server2'])
    print(f"Regions: {regions}")
    dataset = WorkflowDataset(args.data_path, split='train', regions=regions)
    env = WorkflowMoEEnv(dataset)
    
    print(f"Using device: {args.device}")
    print(f"Dataset loaded: {len(dataset.tasks)} tasks, {len(env.servers)} servers") 

    dummy_task_data = np.random.choice(dataset.tasks)
    env.reset(dummy_task_data)
    dummy_task_feat, dummy_server_feats, _, _ = OnlineExpertDataset(env).extract_structured_state(env, action_idx=0)
    task_dim = dummy_task_feat.shape[0]
    num_servers = dummy_server_feats.shape[0]
    server_dim = dummy_server_feats.shape[1]
    
    print(f"Task Dim: {task_dim}, Server Dim: {server_dim}, Num Servers: {num_servers}")
    
    model = StarkScheduler(
        task_dim=task_dim,
        server_dim=server_dim,
        num_servers=num_servers,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_layers,
        num_decoder_layers=args.num_layers
    ).to(args.device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    expert_data = OnlineExpertDataset(env, steps_per_epoch=args.steps_per_epoch)
 
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"{generate_run_id()}_{timestamp}"
  
    output_dir = getattr(args, 'output_dir', None)
    if output_dir is not None:
        log_dir = os.path.join(output_dir, 'logs')
        model_dir = os.path.join(output_dir, 'models')
    else:
        log_dir = os.path.join('results', 'Stark_Scheduler', 'logs')
        model_dir = os.path.join('results', 'Stark_Scheduler', 'models')
    ensure_dir(log_dir)
    ensure_dir(model_dir)
    
    loss_history = []
    acc_history = []
    latency_history = []
    cost_history = []
 
    best_latency = float('inf')
    best_model_state = None
    best_epoch = 0
    
    print(f"Starting training (Supervised Learning with Expert: Weighted Best-Fit)...")
    print(f"Run ID: {run_id}")
    print(f"Total Epochs: {args.epochs}")
    print("-" * 60)
    
    start_time = time.time()
 
    avg_latency_init, avg_cost_init = evaluate_model(model, env, dataset, args.device, num_episodes=30, seed=None)
    latency_history.append(avg_latency_init)
    cost_history.append(avg_cost_init)
    loss_history.append(5.0)  
    acc_history.append(2.0)   
    print(f"Epoch 0 (untrained): Lat={avg_latency_init:.1f}ms, Cost=${avg_cost_init:.4f}")
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
    
        expert_data.generate_epoch_data()
        
        train_loader = DataLoader(
            expert_data, 
            batch_size=args.batch_size, 
            shuffle=True, 
            collate_fn=collate_fn
        )
 
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        for task_feats, server_feats, labels, avail_masks in train_loader:
            task_feats = task_feats.to(args.device)
            server_feats = server_feats.to(args.device)
            labels = labels.to(args.device)
            avail_masks = avail_masks.to(args.device)
            
            optimizer.zero_grad()
            
            logits = model(task_feats, server_feats)
     
            masked_logits = logits.clone()
            masked_logits[avail_masks == 0] = -1e9
            
            loss = criterion(masked_logits, labels)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
   
            preds = torch.argmax(masked_logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
        avg_loss = epoch_loss / len(train_loader)
        accuracy = correct / total * 100.0
        
        loss_history.append(avg_loss)
        acc_history.append(accuracy)
        
        scheduler.step()
   
        avg_latency, avg_cost = evaluate_model(model, env, dataset, args.device, num_episodes=30, seed=None)
        latency_history.append(avg_latency)
        cost_history.append(avg_cost)
  
        if avg_latency < best_latency:
            best_latency = avg_latency
            best_epoch = epoch + 1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        epoch_time = time.time() - epoch_start
   
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            star = " *BEST*" if (epoch + 1) == best_epoch else ""
            if args.run_idx > 0:
                print(f"[Run {args.run_idx}/{args.total_runs}] Epoch {epoch+1}/{args.epochs}: "
                      f"Loss={avg_loss:.4f}, Acc={accuracy:.2f}%, "
                      f"Lat={avg_latency:.1f}ms, Cost=${avg_cost:.4f}{star}")
            else:
                print(f"Epoch {epoch+1}/{args.epochs}: Loss={avg_loss:.4f}, Acc={accuracy:.2f}%, "
                      f"Lat={avg_latency:.1f}ms, Cost=${avg_cost:.4f}{star}")
 
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(model_dir, f"{run_id}_epoch_{epoch+1:04d}.pt")
            torch.save(model.state_dict(), save_path)
  
    print(f"\n>>> Best model was at epoch {best_epoch} with latency {best_latency:.1f}ms")
    final_path = os.path.join(model_dir, f"{run_id}_final.pt")
    torch.save(best_model_state, final_path)
    print(f">>> Saved best model (epoch {best_epoch}, latency={best_latency:.1f}ms) as final.pt")
    
    total_time = time.time() - start_time
    print("-" * 60)
    print(f"Training completed in {total_time:.1f}s. Model saved to {final_path}")
  
    metrics_path = os.path.join(log_dir, f"metrics_{run_id}.npz")
    np.savez(metrics_path, 
             loss=loss_history, 
             accuracy=acc_history,
             latency=latency_history,
             cost=cost_history,
             best_epoch=best_epoch,
             best_latency=best_latency)
    print(f"Training metrics saved to {metrics_path}")

    epochs = np.arange(1, len(latency_history) + 1)
 
    def smooth(data, window=5):
        if len(data) < window:
            return data
        cumsum = np.cumsum(np.insert(data, 0, 0))
        ma = (cumsum[window:] - cumsum[:-window]) / window
        start_ma = [cumsum[i+1]/(i+1) for i in range(window-1)]
        return np.concatenate((start_ma, ma))
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color_lat = '#E07070'   
    color_cost = '#4CAF50'  
    
    latency_arr = np.array(latency_history)
    cost_arr = np.array(cost_history)
    
    window = max(3, len(latency_arr) // 20)
    l_smooth = smooth(latency_arr, window)
    c_smooth = smooth(cost_arr, window)
 
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Latency (ms)', color=color_lat, fontsize=12, fontweight='bold')
    
    ax1.plot(epochs, l_smooth, color=color_lat, linewidth=1.2)
    ax1.plot(epochs, latency_arr, color=color_lat, linewidth=0.5, alpha=0.3)
    
    ax1.tick_params(axis='y', labelcolor=color_lat)
    ax1.grid(alpha=0.3, linestyle='-')
    
    l_margin = (latency_arr.max() - latency_arr.min()) * 0.1
    ax1.set_ylim(latency_arr.min() - l_margin, latency_arr.max() + l_margin)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Cost ($)', color=color_cost, fontsize=12, fontweight='bold')
    
    ax2.plot(epochs, c_smooth, color=color_cost, linewidth=1.2)
    ax2.plot(epochs, cost_arr, color=color_cost, linewidth=0.5, alpha=0.3)
    
    ax2.tick_params(axis='y', labelcolor=color_cost)
    
    c_margin = (cost_arr.max() - cost_arr.min()) * 0.1
    ax2.set_ylim(cost_arr.min() - c_margin, cost_arr.max() + c_margin)
    
    ax1.set_title('Performance Metrics Evolution', fontsize=14, fontweight='bold')
    
    plt.savefig(os.path.join(log_dir, f"training_curve_{run_id}.png"), dpi=150, bbox_inches='tight')
    print(f"Training curve saved to {log_dir}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--epochs', type=int, default=50) 
    parser.add_argument('--steps_per_epoch', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--run_idx', type=int, default=0, help='Current run index (for batch training)')
    parser.add_argument('--total_runs', type=int, default=1, help='Total number of runs (for batch training)')
    parser.add_argument('--regions', type=str, nargs='+', default=['Server2'],
                        help='Regions to train on')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for models and logs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    train(args)
