"""
生成推理结果的 Latency CDF 图
支持单区域(Server3)和混合区域推理结果
"""
import os
import sys
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# 配置matplotlib样式 - 论文格式（大字体，适合双栏论文）
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 26,
    'axes.labelsize': 28,
    'axes.titlesize': 30,
    'axes.titleweight': 'normal',
    'axes.linewidth': 2.0,
    'axes.edgecolor': 'black',
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 20,
    'lines.linewidth': 4.0,
    'ytick.color': 'black',
    'xtick.color': 'black',
    'axes.labelcolor': 'black',
    'mathtext.fontset': 'stix',
})

# 算法配置 - 单区域模式（加粗线条）
ALGORITHMS_SINGLE = {
    'STAR_PPO': ('STAR-PPO', '#d62728', '-', 5.5),  # Red - Ours
    'PFAPPO': ('PF-PPO', '#17becf', '-', 4.0),
    'PPO': ('PPO-Std', '#1f77b4', '--', 4.0),
    'PPO_CN': ('PPO-CN', '#ff7f0e', '-.', 4.0),
    'PPO_GNN': ('GA-PPO', '#2ca02c', ':', 4.0),
    'Trans': ('Equity-Trans', '#9467bd', '--', 4.0),
    'A3C': ('A3C', '#8c564b', '-.', 4.0),
    'Stark': ('STARK', '#e377c2', '-', 4.0),
    'Greedy': ('Greedy', '#7f7f7f', '--', 4.0),
    'Random': ('Random', '#bcbd22', ':', 3.5),
}

# 算法配置 - 混合模式
ALGORITHMS_MIXED = {
    'PFAPPO': ('PF-PPO', '#E74C3C', '-', 2.5),
    'PPO': ('PPO-Std', '#3498DB', '--', 1.5),
    'PPO_CN': ('PPO-CN', '#2ECC71', '-.', 1.5),
    'PPO_GNN': ('GA-PPO', '#9B59B6', ':', 1.5),
    'Trans': ('Equity-Trans', '#F39C12', '--', 1.5),
    'A3C': ('A3C', '#1ABC9C', '-.', 1.5),
    'Stark': ('STARK', '#E91E63', ':', 1.5),
    'Greedy': ('Greedy', '#7F8C8D', '-', 1.5),
    'Random': ('Random', '#95A5A6', '-', 1.0),
}

def _exact_algo_match(filename, algo):
    """与 generate_giant_table.py 一致，避免 PPO 匹配到 PPO_CN 等"""
    basename = os.path.basename(filename)
    prefix = f'{algo}_'
    if not basename.startswith(prefix):
        return False
    rest = basename[len(prefix):]
    return rest.startswith('Server') or (rest and rest[0].isdigit())


def _pick_server1_trap_npz(results_dir, algo_name):
    """每算法只取 canonical 文件，与 giant_table 同源（200 ep, seed42）"""
    pattern = os.path.join(results_dir, f'{algo_name}_*Server1_Trap*seed42*.npz')
    candidates = [
        f for f in glob.glob(pattern)
        if 'cost_breakdown' not in f
        and 'workflow' not in f
        and _exact_algo_match(f, algo_name)
    ]
    if not candidates:
        return None
    # 优先标准命名
    for f in candidates:
        if 'Server1_500_detailed_Server1_Trap_seed42' in f:
            return f
    return sorted(candidates)[0]


def load_npz_data(npz_file, algo_name, mixed_mode=False):
    """从npz文件加载延迟数据"""
    data = np.load(npz_file)

    if not mixed_mode and algo_name == 'PPO_algorithm':
        latencies = data.get('lat_pp', data.get('latencies', []))
    else:
        latencies = data.get('latencies', [])

    return latencies

def load_all_data(mixed_mode=False, use_server1=False):
    """加载所有算法的延迟数据"""
    if mixed_mode:
        results_dir = 'total/mixed_inference'
        ALGORITHMS = ALGORITHMS_MIXED
    elif use_server1:
        results_dir = 'inference/results_500'  # Server1_Trap 推理结果
        ALGORITHMS = ALGORITHMS_SINGLE
    else:
        results_dir = 'inference/results'
        ALGORITHMS = ALGORITHMS_SINGLE

    all_latencies = {}

    for algo_name in ALGORITHMS.keys():
        if mixed_mode:
            # 混合模式: {algo}_seed{seed}.npz
            npz_files = glob.glob(os.path.join(results_dir, f'{algo_name}_seed*.npz'))
        elif use_server1:
            npz_file = _pick_server1_trap_npz(results_dir, algo_name)
            npz_files = [npz_file] if npz_file else []
        else:
            all_files = glob.glob(os.path.join(results_dir, f'{algo_name}_*.npz'))
            npz_files = [
                f for f in all_files
                if 'detailed' in f and 'cost_breakdown' not in f and 'workflow' not in f
            ]

        if not npz_files or npz_files[0] is None:
            print(f"Warning: No files found for {algo_name}")
            continue

        lats = []
        for f in npz_files:
            if f is None:
                continue
            try:
                lats.extend(load_npz_data(f, algo_name, mixed_mode))
            except Exception as e:
                print(f"Error loading {f}: {e}")

        if lats:
            all_latencies[algo_name] = np.array(lats)
            src = os.path.basename(npz_files[0]) if npz_files else ''
            print(f"Loaded {algo_name}: {len(lats)} samples ({src})")

    return all_latencies, ALGORITHMS

# 与 giant_table.md 一致的 P99（仅用于子图标注）
GIANT_TABLE_P99 = {
    'STAR_PPO': 4558.08,
    'Trans': 4670.30,
    'A3C': 4592.15,
    'PPO_CN': 4751.65,
}


def _annotate_inset_p99(ax_inset, sorted_data, color, text_y, va='bottom', label_p99=None):
    """在子图 P99 处标注实测分位延迟（整数 ms）"""
    p99_x = float(label_p99 if label_p99 is not None else np.percentile(sorted_data, 99))
    ax_inset.plot(p99_x, 0.99, marker='o', markersize=7, color=color, zorder=11)
    ax_inset.text(
        p99_x - 300, text_y, f'{int(round(p99_x))}',
        color=color, fontsize=13, fontweight='bold',
        ha='right', va=va, zorder=12,
    )
    return p99_x


def plot_cdf(data, ALGORITHMS, mixed_mode=False, use_server1=False):
    """绘制 CDF 图 - 优化版,突出 STAR-PPO 优势（Server1 使用推理 npz 原始延迟）"""
    fig, ax = plt.subplots(figsize=(9, 7))

    # 按照特定顺序绘制（STAR_PPO 最后绘制,显示在最上层）
    if use_server1:
        plot_order = ['Random', 'Greedy', 'A3C', 'Stark', 'Trans',
                      'PPO', 'PPO_CN', 'PPO_GNN', 'PFAPPO', 'STAR_PPO']
    elif mixed_mode:
        plot_order = ['Random', 'Greedy', 'A3C', 'Stark', 'Trans',
                      'PPO', 'PPO_CN', 'PPO_GNN', 'PFAPPO', 'STAR_PPO']
    else:
        plot_order = ['Random', 'Greedy', 'A3C_algorithm', 'Stark_Scheduler',
                      'Trans', 'PPO_CN', 'PPO_algorithm', 'PPO_GNN', 'PFAPPO', 'STAR_PPO']

    for algo_name in plot_order:
        if algo_name not in data or algo_name not in ALGORITHMS:
            continue

        latencies = data[algo_name]
        display_name, color, linestyle, linewidth = ALGORITHMS[algo_name]

        # 排序
        sorted_data = np.sort(latencies)
        # 计算累积概率
        yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)

        # STAR_PPO 特别突出
        if algo_name == 'STAR_PPO':
            ax.plot(sorted_data, yvals, label=display_name,
                    color=color, linestyle='-', linewidth=3.0, zorder=10)
        else:
            ax.plot(sorted_data, yvals, label=display_name,
                    color=color, linestyle=linestyle, linewidth=linewidth, alpha=0.8)

    # 添加 SLA 阈值线 (3000ms) - 标签上移避免遮挡
    ax.axvline(x=3000, color='red', linestyle='--', alpha=0.8, linewidth=3)
    ax.text(2900, 0.52, '$T_{SLA}$', rotation=90, verticalalignment='bottom',
            fontsize=26, color='red')

    # 添加关键百分位线 - 标签移到右侧
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, linewidth=2)
    ax.axhline(y=0.9, color='gray', linestyle=':', alpha=0.5, linewidth=2)
    ax.axhline(y=0.99, color='gray', linestyle=':', alpha=0.5, linewidth=2)
    ax.text(5900, 0.51, 'P50', fontsize=22, color='gray', ha='right')
    ax.text(5900, 0.91, 'P90', fontsize=22, color='gray', ha='right')
    ax.text(5900, 0.995, 'P99', fontsize=22, color='gray', ha='right')

    # ================= 添加放大子图 (Inset) =================
    if use_server1:
        # 创建嵌入坐标轴
        ax_inset = ax.inset_axes([0.55, 0.05, 0.42, 0.38])
        ax_inset.set_xlim(4000, 5500)
        ax_inset.set_ylim(0.90, 1.00)

        # 子图：放大 P90–P99 整段（与原版一致，重绘全部曲线）
        for algo_name in plot_order:
            if algo_name not in data or algo_name not in ALGORITHMS:
                continue
            latencies = data[algo_name]
            color = ALGORITHMS[algo_name][1]
            linestyle = ALGORITHMS[algo_name][2]
            linewidth = ALGORITHMS[algo_name][3]
            sorted_data = np.sort(latencies)
            yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)

            if algo_name == 'STAR_PPO':
                ax_inset.plot(sorted_data, yvals, color=color, linestyle='-', linewidth=3.0, zorder=10)
                _annotate_inset_p99(
                    ax_inset, sorted_data, color, text_y=0.9900, va='bottom',
                    label_p99=GIANT_TABLE_P99.get(algo_name),
                )
            else:
                ax_inset.plot(sorted_data, yvals, color=color, linestyle=linestyle,
                              linewidth=linewidth, alpha=0.8)
                if algo_name == 'A3C':
                    _annotate_inset_p99(
                        ax_inset, sorted_data, color, text_y=0.9865, va='top',
                        label_p99=GIANT_TABLE_P99.get(algo_name),
                    )
                elif algo_name == 'Trans':
                    _annotate_inset_p99(
                        ax_inset, sorted_data, color, text_y=0.9795, va='top',
                        label_p99=GIANT_TABLE_P99.get(algo_name),
                    )

        ax_inset.axhline(y=0.90, color='gray', linestyle=':', alpha=0.5, linewidth=2)
        ax_inset.axhline(y=0.99, color='gray', linestyle=':', alpha=0.5, linewidth=2)
        ax_inset.text(4010, 0.905, 'P90', fontsize=12, color='gray', va='bottom')
        ax_inset.text(4010, 0.992, 'P99', fontsize=12, color='gray', va='bottom')

        # 设置子图的刻度字体大小
        ax_inset.tick_params(axis='both', which='major', labelsize=14)
        ax_inset.grid(True, alpha=0.3, linestyle='--')

        # 画连接线
        ax.indicate_inset_zoom(ax_inset, edgecolor='gray', linestyle='--', linewidth=3, alpha=0.8)
    # ========================================================

    # 设置图形属性
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('')

    # 截断到合适范围
    if use_server1:
        ax.set_xlim(0, 6000)  # Server1_Trap 延迟较高
    else:
        ax.set_xlim(0, 4000)
    ax.set_ylim(0, 1.02)

    # 格式化刻度
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1000 if use_server1 else 500))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))

    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--')

    # 添加图例 - STAR-PPO 排在第一个
    handles, labels = ax.get_legend_handles_labels()
    # 找到 STAR-PPO 的索引并移到最前面
    star_idx = None
    for i, label in enumerate(labels):
        if 'STAR-PPO' in label:
            star_idx = i
            break
    if star_idx is not None:
        handles = [handles[star_idx]] + handles[:star_idx] + handles[star_idx+1:]
        labels = [labels[star_idx]] + labels[:star_idx] + labels[star_idx+1:]
    # 放在左上角
    ax.legend(handles, labels, loc='upper left', fontsize=18, ncol=1, framealpha=0.85)

    plt.tight_layout()

    # 保存
    if use_server1:
        output_name = 'Latency_CDF_Server1_Trap.png'
    elif mixed_mode:
        output_name = 'Latency_CDF_Mixed.png'
    else:
        output_name = 'Latency_CDF_Server3.png'
    output_path = os.path.join(OUTPUT_DIR, output_name)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved CDF plot to: {output_path}")
    if use_server1:
        print("P99 (raw inference latencies):")
        for algo_name in plot_order:
            if algo_name in data:
                p99 = np.percentile(data[algo_name], 99)
                name = ALGORITHMS[algo_name][0]
                print(f"  {name:<14} {p99:.2f} ms")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mixed', action='store_true',
                        help='使用混合推理结果 (total/mixed_inference/)')
    parser.add_argument('--server1', action='store_true',
                        help='使用 Server1_Trap 推理结果 (inference/results_500/)')
    args = parser.parse_args()

    if args.server1:
        mode_str = "Server1_Trap (500服务器, 10%陷阱)"
    elif args.mixed:
        mode_str = "混合区域"
    else:
        mode_str = "Server3跨域"
    print(f"Generating Latency CDF Plot ({mode_str})...\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data, ALGORITHMS = load_all_data(mixed_mode=args.mixed, use_server1=args.server1)
    if not data:
        print("Error: No data loaded!")
        return

    plot_cdf(data, ALGORITHMS, mixed_mode=args.mixed, use_server1=args.server1)

    print("\nDone!")

if __name__ == '__main__':
    main()
