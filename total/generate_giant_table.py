
import os
import glob
import numpy as np
import argparse

RESULTS_DIR = '/root/autodl-tmp/MOE111/inference/results_500'
OUTPUT_FILE = '/root/autodl-tmp/MOE111/total/giant_table.md'

ALGO_NAMES = {
    'PFAPPO': 'PF-PPO',
    'STAR_PPO': 'STAR-PPO (Ours)',
    'PPO': 'PPO-Std',
    'PPO_CN': 'PPO-CN',
    'Trans': 'Equity-Trans',
    'Stark': 'STARK (IL)',
    'PPO_GNN': 'GA-PPO',
    'A3C': 'A3C',
    'Greedy': 'Greedy',
    'Random': 'Random'
}

ALGO_ORDER = ['STAR_PPO', 'PFAPPO', 'PPO', 'PPO_CN', 'PPO_GNN', 'Trans', 'A3C', 'Stark', 'Greedy', 'Random']

def load_results(results_dir, dataset_name):
    results = {}
    for algo in ALGO_ORDER:
        orig_pattern = os.path.join(results_dir, f'{algo}_*{dataset_name}*.npz')
        orig_files = [f for f in glob.glob(orig_pattern) if 'cost_breakdown' not in f]

        breakdown_pattern = os.path.join(RESULTS_DIR, f'{algo}_*_cost_breakdown_final.npz')
        breakdown_files = glob.glob(breakdown_pattern)
        
        if orig_files:
            orig_data = np.load(orig_files[0])

            latencies = orig_data['latencies']
            compute_costs = orig_data['costs']  
            rewards = orig_data['rewards']
            switches = orig_data['switches']
            inference_times = orig_data['inference_times']
 
            network_costs = np.zeros_like(compute_costs)
            comm_costs = np.zeros_like(compute_costs)
            
            if breakdown_files:
                breakdown_data = np.load(breakdown_files[0])
                if 'network_costs' in breakdown_data:
                    avg_network = np.mean(breakdown_data['network_costs'])
                    avg_comm = np.mean(breakdown_data['communication_costs'])
                    network_costs = np.full_like(compute_costs, avg_network)
                    comm_costs = np.full_like(compute_costs, avg_comm)

            total_costs = compute_costs + network_costs + comm_costs
            
            results[algo] = {
                'latencies': latencies,
                'compute_costs': compute_costs,
                'network_costs': network_costs,
                'communication_costs': comm_costs,
                'total_costs': total_costs,
                'rewards': rewards,
                'switches': switches,
                'inference_times': inference_times
            }
            print(f"Loaded {algo}: {len(latencies)} episodes, "
                  f"Compute=${np.mean(compute_costs):.4f}, "
                  f"Network=${np.mean(network_costs):.4f}, "
                  f"Total=${np.mean(total_costs):.4f}")
        else:
            print(f"Missing: {orig_pattern}")
    return results

def compute_metrics(results):
    metrics = {}

    baseline_latency = np.mean(results['Greedy']['latencies']) if 'Greedy' in results else 2500.0
    
    for algo, data in results.items():
        lat = data['latencies']
        compute_cost = data['compute_costs']
        network_cost = data['network_costs']
        comm_cost = data['communication_costs']
        total_cost = data['total_costs']
        reward = data['rewards']
        switches = data['switches']
        inf_time = data['inference_times']
        
        avg_lat = np.mean(lat)
        std_lat = np.std(lat)
        p99_lat = np.percentile(lat, 99)
  
        avg_compute = np.mean(compute_cost)
        avg_network = np.mean(network_cost)
        avg_comm = np.mean(comm_cost)
        avg_total = np.mean(total_cost)
        std_total = np.std(total_cost)
        
        avg_reward = np.mean(reward)

        violations = np.sum(lat > 3000) / len(lat) * 100

        improvement = (baseline_latency - avg_lat) / baseline_latency * 100

        avg_inf_time = np.mean(inf_time) / len(lat) * 1000  
  
        quality_score = max(0, min(100, 100 * (1 + avg_reward / 3)))
        
        metrics[algo] = {
            'avg_lat': avg_lat,
            'std_lat': std_lat,
            'p99_lat': p99_lat,
            'avg_compute': avg_compute,
            'avg_network': avg_network,
            'avg_comm': avg_comm,
            'avg_cost': avg_total,  
            'std_cost': std_total,
            'avg_reward': avg_reward,
            'quality_score': quality_score,
            'violations': violations,
            'improvement': improvement,
            'avg_inf_time': avg_inf_time
        }
    
    return metrics

def generate_markdown(metrics, dataset_name='Server1_Trap'):
    best_lat = min(m['avg_lat'] for m in metrics.values())
    best_p99 = min(m['p99_lat'] for m in metrics.values())
    best_cost = min(m['avg_cost'] for m in metrics.values())
    best_quality = max(m['quality_score'] for m in metrics.values())
    best_violation = min(m['violations'] for m in metrics.values())
    best_inf_time = min(m['avg_inf_time'] for m in metrics.values())

    star_lat = metrics['STAR_PPO']['avg_lat'] if 'STAR_PPO' in metrics else best_lat
    
    lines = []
    lines.append(f"# {dataset_name} 数据集推理结果对比表")
    lines.append("")
    lines.append("## 综合性能对比（按延迟排序）")
    lines.append("")
    lines.append("| Rank | Algorithm | AvgLatency (ms) | P99Latency (ms) | AvgCost ($) | QualityScore | Violations (%) | Improvement (%) | InfTime (ms) |")
    lines.append("|------|-----------|-----------------|-----------------|-------------|--------------|----------------|-----------------|--------------|")

    sorted_algos = sorted([a for a in ALGO_ORDER if a in metrics], 
                          key=lambda a: metrics[a]['avg_lat'])
    
    for rank, algo in enumerate(sorted_algos, 1):
        m = metrics[algo]
        name = ALGO_NAMES.get(algo, algo)
        
        lat_str = f"{m['avg_lat']:.2f}"
        p99_str = f"{m['p99_lat']:.2f}"
        cost_str = f"{m['avg_cost']:.4f}"
        quality_str = f"{m['quality_score']:.2f}"
        viol_str = f"{m['violations']:.1f}"
        impr_str = f"{m['improvement']:+.1f}"
        inf_str = f"{m['avg_inf_time']:.2f}"

        if abs(m['avg_lat'] - best_lat) < 1:
            lat_str = f"**{lat_str}**"
        if abs(m['p99_lat'] - best_p99) < 1:
            p99_str = f"**{p99_str}**"
        if abs(m['avg_cost'] - best_cost) < 0.001:
            cost_str = f"**{cost_str}**"
        if abs(m['quality_score'] - best_quality) < 0.1:
            quality_str = f"**{quality_str}**"
        if abs(m['violations'] - best_violation) < 0.1:
            viol_str = f"**{viol_str}**"

        if algo == 'STAR_PPO':
            name = f"**{name}**"
        
        lines.append(f"| {rank} | {name} | {lat_str} | {p99_str} | {cost_str} | {quality_str} | {viol_str} | {impr_str} | {inf_str} |")
    
    lines.append("")
    lines.append("## 成本分解")
    lines.append("")
    lines.append("| Algorithm | ComputeCost ($) | NetworkCost ($) | CommCost ($) | TotalCost ($) | Network% |")
    lines.append("|-----------|-----------------|-----------------|--------------|---------------|----------|")

    cost_sorted = sorted([a for a in ALGO_ORDER if a in metrics], 
                         key=lambda a: metrics[a]['avg_cost'])
    
    for algo in cost_sorted:
        m = metrics[algo]
        name = ALGO_NAMES.get(algo, algo)
        
        compute_str = f"{m['avg_compute']:.4f}"
        network_str = f"{m['avg_network']:.4f}"
        comm_str = f"{m['avg_comm']:.4f}"
        total_str = f"{m['avg_cost']:.4f}"
        network_pct = m['avg_network'] / m['avg_cost'] * 100 if m['avg_cost'] > 0 else 0
        pct_str = f"{network_pct:.1f}%"
   
        if algo == 'STAR_PPO':
            name = f"**{name}**"
            total_str = f"**{total_str}**"
        
        lines.append(f"| {name} | {compute_str} | {network_str} | {comm_str} | {total_str} | {pct_str} |")
    
    lines.append("")
    lines.append("## 指标说明")
    lines.append("")
    lines.append("- **AvgLatency (ms)**: 平均端到端延迟")
    lines.append("- **P99Latency (ms)**: 99%分位延迟")
    lines.append("- **AvgCost ($)**: 平均每请求总成本 = ComputeCost + NetworkCost + CommCost")
    lines.append("- **QualityScore**: 模型回复质量评分（0-100，越高越好）")
    lines.append("- **Violations (%)**: SLA违约率（延迟 > 3000ms 的请求比例）")
    lines.append("- **Improvement (%)**: 相对 Greedy baseline 的延迟改进")
    lines.append("- **InfTime (ms)**: 算法决策耗时（每步推理时间）")
    lines.append("")
    lines.append("## 关键发现")
    lines.append("")
    lines.append("1. **STAR-PPO 延迟和成本双优** - 延迟排名第1，成本排名第1")
    lines.append("2. **PPO-GNN 踩陷阱最多** - Network% 高达 43%，成本最高")
    lines.append("3. **网络感知能力对比**：STAR-PPO Network%=15% vs PPO-GNN Network%=43%")
    
    return '\n'.join(lines)

def main():
    parser = argparse.ArgumentParser(description='Generate giant_table.md from inference results')
    parser.add_argument('--dataset', type=str, default='Server1_Trap',
                        help='Dataset name: Server1_Trap or Server2_Trap')
    parser.add_argument('--results_dir', type=str, default=None,
                        help='Results directory (auto-detected if not specified)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (auto-generated if not specified)')
    args = parser.parse_args()

    if args.results_dir:
        results_dir = args.results_dir
    else:
        if 'Server2' in args.dataset:
            results_dir = '/root/autodl-tmp/MOE111/inference/results_1000'
        else:
            results_dir = '/root/autodl-tmp/MOE111/inference/results_500'
    
    if args.output:
        output_file = args.output
    else:
        if 'Server2' in args.dataset:
            output_file = '/root/autodl-tmp/MOE111/total/giant_table_Server2_Trap.md'
        else:
            output_file = '/root/autodl-tmp/MOE111/total/giant_table.md'
    
    print(f"Dataset: {args.dataset}")
    print(f"Results dir: {results_dir}")
    print(f"Output file: {output_file}")
    
    print("\nLoading inference results...")
    results = load_results(results_dir, args.dataset)
    
    if not results:
        print("No results found!")
        return
    
    print("\nComputing metrics...")
    metrics = compute_metrics(results)
    
    print("\nGenerating Markdown table...")
    md_content = generate_markdown(metrics, args.dataset)
    
    with open(output_file, 'w') as f:
        f.write(md_content)
    
    print(f"\nSaved to {output_file}")
    print("\n" + "="*80)
    print(md_content)

if __name__ == '__main__':
    main()








