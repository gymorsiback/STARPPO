
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 26,
    'axes.labelsize': 28,
    'axes.titlesize': 30,
    'axes.titleweight': 'normal',
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 20,
    'lines.linewidth': 4.0,
    'mathtext.fontset': 'stix',
    'axes.unicode_minus': False,
})

DATA_DIR = 'total/scalability_data'

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


def load_scalability_data():
    scales = [500, 1000, 1500, 2000]
    
    inference_times = []
    inf_stds = []
    latencies = []
    lat_stds = []
    
    for scale in scales:
        if scale in REAL_DATA:
            data = REAL_DATA[scale]
            inference_times.append(data['inference_time'])
            inf_stds.append(data['inf_std'])
            latencies.append(data['latency'])
            lat_stds.append(data['lat_std'])
            print(f"Scale {scale} (真实): InfTime={data['inference_time']:.2f}±{data['inf_std']:.2f}ms, "
                  f"Latency={data['latency']:.2f}±{data['lat_std']:.2f}ms")
        else:
            inf_path = os.path.join(DATA_DIR, 'inference_data.npz')
            if os.path.exists(inf_path):
                file_data = np.load(inf_path)
                inf_time = file_data['inference_times']
                lat = file_data['latencies']
                
                inference_times.append(np.mean(inf_time))
                inf_stds.append(np.std(inf_time))
                latencies.append(np.mean(lat))
                lat_stds.append(np.std(lat))
                
                print(f"Scale {scale} (文件): InfTime={np.mean(inf_time):.2f}±{np.std(inf_time):.2f}ms, "
                      f"Latency={np.mean(lat):.2f}±{np.std(lat):.2f}ms")
            else:
                print(f"Warning: {inf_path} not found!")
                inference_times.append(0)
                inf_stds.append(0)
                latencies.append(0)
                lat_stds.append(0)
    
    return scales, inference_times, inf_stds, latencies, lat_stds


def plot_scalability():
    scales, inference_times, inf_stds, latencies, lat_stds = load_scalability_data()

    fig, ax1 = plt.subplots(figsize=(8, 8))

    color_inf = '#2E86AB'  
    color_lat = '#E94F37' 

    ax1.set_xlabel('Network Scale (Number of Servers)')
    ax1.set_ylabel('Inference Time (ms)', color=color_inf)
    
    line1 = ax1.plot(scales, inference_times, 'o-', color=color_inf, linewidth=4.0, 
                     markersize=12, label='Inference Time')
    ax1.fill_between(scales, 
                     np.array(inference_times) - np.array(inf_stds) * 0.3,
                     np.array(inference_times) + np.array(inf_stds) * 0.3,
                     color=color_inf, alpha=0.15)
    
    ax1.tick_params(axis='y', labelcolor=color_inf)

    ax1.set_xticks(scales)
    ax1.set_xticklabels([str(s) for s in scales])

    ax2 = ax1.twinx()
    ax2.set_ylabel('Average Latency (ms)', color=color_lat)
    
    line2 = ax2.plot(scales, latencies, 's--', color=color_lat, linewidth=4.0, 
                     markersize=12, label='Average Latency')
    ax2.fill_between(scales,
                     np.array(latencies) - np.array(lat_stds) * 0.3,
                     np.array(latencies) + np.array(lat_stds) * 0.3,
                     color=color_lat, alpha=0.15)
    
    ax2.tick_params(axis='y', labelcolor=color_lat)
 
    ax1.set_ylim(0, 120)
    ax2.set_ylim(1800, 2400)
 
    ax1.set_title('Scalability Analysis: Efficiency vs Performance', pad=15)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=20, framealpha=0.9)

    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)

    for spine in ax1.spines.values():
        spine.set_visible(True)
    for spine in ax2.spines.values():
        spine.set_visible(True)
    
    plt.tight_layout()

    output_path = 'total/Scalability_Analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n图表已保存: {output_path}")
    plt.close()


if __name__ == '__main__':
    os.chdir('/root/autodl-tmp/MOE111')
    plot_scalability()
