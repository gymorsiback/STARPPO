"""
可扩展性分析图表 - 双 Y 轴折线图
展示 STAR-PPO 在不同网络规模下的计算效率与调度性能
500/1000/2000 优先从 NPZ 文件读取，1500 从文件读取
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from metrics import sla_violation as _sla, composite_qos as _qos

# 设置字体 - 论文格式（大字体，适合双栏论文）
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
    'mathtext.fontset': 'stix',
    'axes.unicode_minus': False,
})

DATA_DIR = os.path.join(SCRIPT_DIR, 'scalability_data')

# 真实数据（500/1000/2000节点）
REAL_DATA = {
    500: {
        'inference_time': 23.91,
        'inf_std': 10.11,
        'latency': 2020.0,
        'lat_std': 80.4,
    },
    1000: {
        'inference_time': 49.03,
        'inf_std': 11.41,
        'latency': 2080.0,
        'lat_std': 60.5,
    },
    2000: {
        'inference_time': 96.44,
        'inf_std': 23.87,
        'latency': 2146.2,
        'lat_std': 90.2,
    },
}


NPZ_MAP = {
    500:  'inference/results_500/STAR_PPO_Server1_500_detailed_infseed100.npz',
    1000: 'inference/results_1000/STAR_PPO_Server2_Trap_seed42.npz',
    2000: 'inference/results_2000/STAR_PPO_Server3_Trap_seed42.npz',
}
BASE_DIR = PROJECT_ROOT


def load_scalability_data():
    """优先从 NPZ 文件加载，回退到 REAL_DATA 硬编码值"""
    scales = [500, 1000, 1500, 2000]

    inference_times, inf_stds, latencies, lat_stds = [], [], [], []

    for scale in scales:
        npz_path = os.path.join(BASE_DIR, NPZ_MAP.get(scale, ''))
        if scale in NPZ_MAP and os.path.exists(npz_path):
            d = np.load(npz_path)
            lat  = d['latencies']
            itm  = d.get('inference_times', np.zeros_like(lat))
            n = len(lat)
            inference_times.append(float(np.mean(itm)))
            inf_stds.append(float(np.std(itm) / np.sqrt(n)))   # SEM
            latencies.append(float(np.mean(lat)))
            lat_stds.append(float(np.std(lat) / np.sqrt(n)))   # SEM
            viol = float(np.mean(_sla(lat))) * 100
            print(f"Scale {scale} (NPZ): InfTime={np.mean(itm):.2f}ms, "
                  f"Latency={np.mean(lat):.2f}±{np.std(lat):.2f}ms, SLA_viol={viol:.1f}%")
        elif scale in REAL_DATA:
            data = REAL_DATA[scale]
            inference_times.append(data['inference_time'])
            inf_stds.append(data['inf_std'])
            latencies.append(data['latency'])
            lat_stds.append(data['lat_std'])
            print(f"Scale {scale} (硬编码): InfTime={data['inference_time']:.2f}ms, "
                  f"Latency={data['latency']:.2f}ms")
        else:
            inf_path = os.path.join(DATA_DIR, 'inference_data.npz')
            if os.path.exists(inf_path):
                file_data = np.load(inf_path)
                itm = file_data['inference_times']
                lat = file_data['latencies']
                inference_times.append(np.mean(itm))
                inf_stds.append(np.std(itm))
                latencies.append(np.mean(lat))
                lat_stds.append(np.std(lat))
                print(f"Scale {scale} (文件): InfTime={np.mean(itm):.2f}ms, Latency={np.mean(lat):.2f}ms")
            else:
                print(f"Warning: no data for scale {scale}, using zeros")
                inference_times.append(0); inf_stds.append(0)
                latencies.append(0); lat_stds.append(0)

    return scales, inference_times, inf_stds, latencies, lat_stds


def plot_scalability():
    """绘制可扩展性分析双 Y 轴图"""

    # 加载数据
    scales, inference_times, inf_stds, latencies, lat_stds = load_scalability_data()

    # 创建图表
    fig, ax1 = plt.subplots(figsize=(9, 7))

    # 颜色定义
    color_inf = '#2E86AB'  # 蓝色 - 推理时间
    color_lat = '#E94F37'  # 红色 - 时延

    # 左 Y 轴 - Inference Time
    ax1.set_xlabel('Network Scale (Number of Servers)')
    ax1.set_ylabel('Inference Time (ms)', color=color_inf)

    line1 = ax1.plot(scales, inference_times, 'o-', color=color_inf, linewidth=4.0,
                     markersize=12, label='Inference Time')

    ax1.tick_params(axis='y', labelcolor=color_inf)

    # 设置 x 轴刻度
    ax1.set_xticks(scales)
    ax1.set_xticklabels([str(s) for s in scales])

    # 右 Y 轴 - Average Latency
    ax2 = ax1.twinx()
    ax2.set_ylabel('Average Latency (ms)', color=color_lat)

    line2 = ax2.plot(scales, latencies, 's--', color=color_lat, linewidth=4.0,
                     markersize=12, label='Average Latency')

    ax2.tick_params(axis='y', labelcolor=color_lat)

    # 设置 Y 轴范围
    ax1.set_ylim(0, 120)
    ax2.set_ylim(1800, 2400)

    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=20, framealpha=0.9)

    # 网格线
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)

    # 封闭边框
    for spine in ax1.spines.values():
        spine.set_visible(True)
    for spine in ax2.spines.values():
        spine.set_visible(True)

    plt.tight_layout()

    # 保存图表
    output_path = os.path.join(SCRIPT_DIR, 'Scalability_Analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n图表已保存: {output_path}")
    plt.close()


if __name__ == '__main__':
    plot_scalability()
