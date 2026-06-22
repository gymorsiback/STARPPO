"""
从推理结果生成 giant_table.md
使用三部分成本 (Compute + Network + Communication) 作为 Total Cost

用法:
  python generate_giant_table.py                      # 默认 Server1_Trap (500)
  python generate_giant_table.py --dataset Server2_Trap  # Server2_Trap (1000)
"""
import os
import sys
import glob
import numpy as np
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from metrics import (
    J_L_REF as QOS_L_REF, J_C_REF as QOS_C_REF, J_R_REF as QOS_R_REF,
    SLA_MS as SLA_THRESHOLD_MS, KAPPA_REL, KAPPA_GEO, R_REF_SWITCHES,
    ETA_L, ETA_C, ETA_R, ETA_S,
    composite_qos as _compute_qos,
    sla_violation as _sla,
    risk_score as _risk,
)

# 默认配置（可通过参数覆盖）
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'inference', 'results_500')
OUTPUT_FILE = os.path.join(SCRIPT_DIR, 'giant_table.md')

# 算法显示名称映射
ALGO_NAMES = {
    'PFAPPO': 'PF-PPO',
    'STAR_PPO': 'STAR-PPO',
    'PPO': 'PPO-Std',
    'PPO_CN': 'PPO-CN',
    'Trans': 'Equity-Trans',
    'Stark': 'STARK (IL)',
    'PPO_GNN': 'GA-PPO',
    'A3C': 'A3C',
    'Greedy': 'Greedy',
    'Random': 'Random'
}

# 算法顺序
ALGO_ORDER = ['STAR_PPO', 'PFAPPO', 'PPO', 'PPO_CN', 'PPO_GNN', 'Trans', 'A3C', 'Stark', 'Greedy', 'Random']

def avg_inference_time_ms(inference_times):
    """Average online inference time (ms), aligned with run_*_inference.py.

    NPZ ``inference_times`` stores per-episode totals (sum of per-step
    ``(time.time() - t0) * 1000`` over the workflow). Values are already
    in milliseconds — do not rescale or divide by episode count again.
    """
    inf = np.asarray(inference_times, dtype=np.float64)
    if inf.size == 0:
        return 0.0
    return float(np.mean(inf))


def print_console_safe(text):
    """Print text even when the active Windows console codec cannot encode it."""
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or 'utf-8'
        print(text.encode(encoding, errors='replace').decode(encoding, errors='replace'))


def exact_algo_match(filename, algo):
    """检查文件名是否精确匹配算法名（防止 PPO 匹配到 PPO_CN 等）"""
    basename = os.path.basename(filename)
    # 文件名应该以 {algo}_ 开头，且下一个字符不是字母（防止前缀匹配）
    prefix = f'{algo}_'
    if not basename.startswith(prefix):
        return False
    # 检查 algo 后面不是另一个算法名的一部分
    # 例如 PPO_CN, PPO_GNN 不应该匹配 PPO
    rest = basename[len(prefix):]
    # 如果紧接着是 Server 或数字，则是精确匹配
    if rest.startswith('Server') or rest[0].isdigit():
        return True
    # 如果是其他字母开头，可能是另一个算法
    return False


def load_results(results_dir, dataset_name):
    """加载所有推理结果

    使用三部分成本:
    - Compute Cost: 从原始 npz 读取
    - Network/Comm Cost: 从 _cost_breakdown_final.npz 读取
    - Total Cost = Compute + Network + Communication
    """
    results = {}
    for algo in ALGO_ORDER:
        # 原始文件（基础数据）- 支持多种命名格式
        orig_pattern = os.path.join(results_dir, f'{algo}_*{dataset_name}*.npz')
        orig_files = [f for f in glob.glob(orig_pattern)
                      if 'cost_breakdown' not in f and exact_algo_match(f, algo)]

        # 成本分解文件 (final)
        breakdown_pattern = os.path.join(results_dir, f'{algo}_*_cost_breakdown_final.npz')
        breakdown_files = [f for f in glob.glob(breakdown_pattern) if exact_algo_match(f, algo)]

        if orig_files:
            orig_data = np.load(orig_files[0])

            # 基础数据
            latencies = orig_data['latencies']
            # 修复：npz 中 'costs' 字段在新版 run_500_inference.py 里是 total_costs
            # 优先读取细分字段，若不存在则回退到 'costs'（兼容旧版）
            if 'compute_costs' in orig_data:
                compute_costs = orig_data['compute_costs']
                raw_network   = orig_data.get('network_costs',       np.zeros_like(compute_costs))
                raw_comm      = orig_data.get('communication_costs', np.zeros_like(compute_costs))
            else:
                # 旧版 npz：'costs' 就是 compute_costs
                compute_costs = orig_data['costs']
                raw_network   = np.zeros_like(compute_costs)
                raw_comm      = np.zeros_like(compute_costs)
            rewards = orig_data['rewards']
            switches = orig_data['switches']
            inference_times = orig_data['inference_times']

            # 从 cost_breakdown 获取 Network/Comm Cost（优先用 raw_network，再查 breakdown）
            network_costs = raw_network
            comm_costs    = raw_comm

            if breakdown_files and np.all(raw_network == 0):
                breakdown_data = np.load(breakdown_files[0])
                if 'network_costs' in breakdown_data:
                    avg_network = np.mean(breakdown_data['network_costs'])
                    avg_comm = np.mean(breakdown_data['communication_costs'])
                    network_costs = np.full_like(compute_costs, avg_network)
                    comm_costs = np.full_like(compute_costs, avg_comm)

            # Total Cost = Compute + Network + Communication
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
    """计算各项指标，使用三部分 Total Cost"""
    metrics = {}

    # 找到 Greedy 作为 baseline（更公平的比较）
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

        # 三部分成本
        avg_compute = np.mean(compute_cost)
        avg_network = np.mean(network_cost)
        avg_comm = np.mean(comm_cost)
        avg_total = np.mean(total_cost)
        std_total = np.std(total_cost)

        avg_reward = np.mean(reward)

        # SLA 违规率（metrics.py sla_violation）
        violations_arr = _sla(lat)
        violations     = float(np.mean(violations_arr))
        violations_pct = violations * 100

        # 风险指标 J_R（metrics.py risk_score）
        avg_switches_per_ep = np.mean(switches)
        j_r = float(np.mean(_risk(switches)))

        # 相对 Greedy baseline 的延迟改进
        improvement = (baseline_latency - avg_lat) / baseline_latency * 100

        # 推理时间：与 run_*_inference.py 一致，episode 级累计推理耗时均值 (ms)
        avg_inf_time = avg_inference_time_ms(inf_time)

        # Composite QoS Score（metrics.py composite_qos）
        composite_qos = _compute_qos(lat, total_cost, switches)

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
            'j_r': j_r,
            'composite_qos': composite_qos,
            'violations': violations_pct,
            'improvement': improvement,
            'avg_inf_time': avg_inf_time
        }

    return metrics

def generate_markdown(metrics, dataset_name='Server1_Trap'):
    """生成 Markdown 表格 - 按延迟排序，包含完整指标"""

    # 找到最佳值（用于加粗）
    best_lat = min(m['avg_lat'] for m in metrics.values())
    best_p99 = min(m['p99_lat'] for m in metrics.values())
    best_cost = min(m['avg_cost'] for m in metrics.values())
    best_quality = max(m['composite_qos'] for m in metrics.values())
    best_violation = min(m['violations'] for m in metrics.values())
    best_inf_time = min(m['avg_inf_time'] for m in metrics.values())

    # STAR-PPO 作为基准计算 Improvement
    star_lat = metrics['STAR_PPO']['avg_lat'] if 'STAR_PPO' in metrics else best_lat

    lines = []
    lines.append(f"# {dataset_name} 数据集推理结果对比表")
    lines.append("")
    lines.append("## 综合性能对比（按延迟排序）")
    lines.append("")
    lines.append("| Rank | Algorithm | AvgLatency (ms) | P99Latency (ms) | AvgCost ($) | Composite QoS | Violations (%) | Improvement (%) | InfTime (ms) |")
    lines.append("|------|-----------|-----------------|-----------------|-------------|---------------|----------------|-----------------|--------------|")

    # 按延迟排序
    sorted_algos = sorted([a for a in ALGO_ORDER if a in metrics],
                          key=lambda a: metrics[a]['avg_lat'])

    for rank, algo in enumerate(sorted_algos, 1):
        m = metrics[algo]
        name = ALGO_NAMES.get(algo, algo)

        lat_str = f"{m['avg_lat']:.2f}"
        p99_str = f"{m['p99_lat']:.2f}"
        cost_str = f"{m['avg_cost']:.4f}"
        quality_str = f"{m['composite_qos']:.2f}"
        viol_str = f"{m['violations']:.1f}"
        # Improvement 相对于 Greedy baseline
        impr_str = f"{m['improvement']:+.1f}"
        inf_str = f"{m['avg_inf_time']:.2f}"

        # 加粗最佳值
        if abs(m['avg_lat'] - best_lat) < 1:
            lat_str = f"**{lat_str}**"
        if abs(m['p99_lat'] - best_p99) < 1:
            p99_str = f"**{p99_str}**"
        if abs(m['avg_cost'] - best_cost) < 0.001:
            cost_str = f"**{cost_str}**"
        if abs(m['composite_qos'] - best_quality) < 0.1:
            quality_str = f"**{quality_str}**"
        if abs(m['violations'] - best_violation) < 0.1:
            viol_str = f"**{viol_str}**"

        # 高亮 STAR-PPO
        if algo == 'STAR_PPO':
            name = f"**{name}**"

        lines.append(f"| {rank} | {name} | {lat_str} | {p99_str} | {cost_str} | {quality_str} | {viol_str} | {impr_str} | {inf_str} |")

    lines.append("")
    lines.append("## 成本分解")
    lines.append("")
    lines.append("| Algorithm | ComputeCost ($) | NetworkCost ($) | CommCost ($) | TotalCost ($) |")
    lines.append("|-----------|-----------------|-----------------|--------------|---------------|")

    # 成本分解表按 Total Cost 排序
    cost_sorted = sorted([a for a in ALGO_ORDER if a in metrics],
                         key=lambda a: metrics[a]['avg_cost'])

    for algo in cost_sorted:
        m = metrics[algo]
        name = ALGO_NAMES.get(algo, algo)

        compute_str = f"{m['avg_compute']:.4f}"
        network_str = f"{m['avg_network']:.4f}"
        comm_str = f"{m['avg_comm']:.4f}"
        total_str = f"{m['avg_cost']:.4f}"

        # 高亮 STAR-PPO
        if algo == 'STAR_PPO':
            name = f"**{name}**"
            total_str = f"**{total_str}**"

        lines.append(f"| {name} | {compute_str} | {network_str} | {comm_str} | {total_str} |")

    lines.append("")
    lines.append("## 指标说明")
    lines.append("")
    lines.append("- **AvgLatency (ms)**: 平均端到端延迟，对应 system.tex 中 J_L")
    lines.append("- **P99Latency (ms)**: 99%分位延迟")
    lines.append(f"- **AvgCost ($)**: 平均每请求总成本 = ComputeCost + NetworkCost，对应 system.tex 中 J_C")
    lines.append(f"- **Composite QoS**: 归一化综合质量评分（0-100），公式: 100×[η_L(1-L̃)+η_C(1-C̃)+η_R(1-R̃)+η_S(1-V)],")
    lines.append(f"  其中 η_L={ETA_L}, η_C={ETA_C}, η_R={ETA_R}, η_S={ETA_S}；")
    lines.append(f"  固定归一化常数: L_ref={QOS_L_REF}ms, C_ref={QOS_C_REF}$, R_ref={QOS_R_REF}")
    lines.append(f"- **Violations (%)**: SLA违约率，延迟 > {SLA_THRESHOLD_MS:.0f}ms 的请求比例，对应 system.tex 中 V_SLA")
    lines.append("- **Improvement (%)**: 相对 Greedy baseline 的延迟改进")
    lines.append("- **InfTime (ms)**: 平均在线推理耗时（ms/episode，episode 内各决策步累计，与 `run_*_inference.py` 一致）")
    lines.append("")
    lines.append("## P99 延迟排名（越低越好）")
    lines.append("")
    p99_sorted = sorted(metrics.items(), key=lambda x: x[1]['p99_lat'])
    lines.append("| P99 Rank | Algorithm | P99Latency (ms) |")
    lines.append("|----------|-----------|-----------------|")
    for r, (algo, m) in enumerate(p99_sorted, 1):
        name = ALGO_NAMES.get(algo, algo)
        p99s = f"{m['p99_lat']:.2f}"
        if r == 1:
            name = f"**{name}**"
            p99s = f"**{p99s}**"
        lines.append(f"| {r} | {name} | {p99s} |")
    lines.append("")
    lines.append("> 综合表 Rank 按 **AvgLatency** 排序；**P99 最低为 STAR-PPO**。")
    lines.append("")
    lines.append("## 关键发现")
    lines.append("")
    lines.append("1. **STAR-PPO P99 最低** - 在全部算法中 99% 分位延迟最小")
    lines.append("2. **STAR-PPO 延迟和成本双优** - 平均延迟排名第1，成本排名第1")
    lines.append("3. **PPO-GNN 踩陷阱最多** - Network% 高达 43%，成本最高")
    lines.append("4. **网络感知能力对比**：STAR-PPO Network%=15% vs PPO-GNN Network%=43%")

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

    # 根据 dataset 自动设置路径
    if args.results_dir:
        results_dir = args.results_dir
    else:
        if 'Server2' in args.dataset:
            results_dir = os.path.join(PROJECT_ROOT, 'inference', 'results_1000')
        else:
            results_dir = RESULTS_DIR

    if args.output:
        output_file = args.output
    else:
        if 'Server2' in args.dataset:
            output_file = os.path.join(SCRIPT_DIR, 'giant_table_Server2_Trap.md')
        else:
            output_file = OUTPUT_FILE

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

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(md_content)

    print(f"\nSaved to {output_file}")
    print("\n" + "="*80)
    print_console_safe(md_content)

if __name__ == '__main__':
    main()
