
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 26,
    'axes.labelsize': 28,
    'axes.titlesize': 30,
    'axes.titleweight': 'normal',
    'xtick.labelsize': 22,
    'ytick.labelsize': 24,
    'legend.fontsize': 20,
    'lines.linewidth': 4.0,
    'mathtext.fontset': 'stix',
})

RESULT_DIR = os.path.dirname(os.path.abspath(__file__))
result_path = os.path.join(RESULT_DIR, 'workload_pattern_results.npz')

data = np.load(result_path, allow_pickle=True)
patterns = ['Uniform', 'Poisson', 'Bursty', 'On-Off']

uniform_lats = data['uniform']
poisson_lats = data['poisson']
bursty_lats = data['bursty']
on_off_lats = data['on_off']

print("数据统计:")
for name, lats in [('Uniform', uniform_lats), ('Poisson', poisson_lats), 
                   ('Bursty', bursty_lats), ('On-Off', on_off_lats)]:
    print(f"  {name}: mean={np.mean(lats):.1f}, std={np.std(lats):.1f}, "
          f"median={np.median(lats):.1f}, P95={np.percentile(lats, 95):.1f}")

fig, ax = plt.subplots(figsize=(8, 8))

box_data = [uniform_lats, poisson_lats, bursty_lats, on_off_lats]
positions = [1, 2, 3, 4]
labels = ['Uniform\n(Training)', 'Poisson', 'Bursty', 'On-Off']

colors = ['#74c476', '#6baed6', '#a89078', '#9e9ac8']

bp = ax.boxplot(box_data, positions=positions, patch_artist=True,
                widths=0.6, showfliers=True, 
                flierprops={'marker': 'o', 'markersize': 4, 'alpha': 0.5})

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

for median in bp['medians']:
    median.set_color('black')
    median.set_linewidth(2)

means = [np.mean(d) for d in box_data]
ax.scatter(positions, means, marker='D', color='white', edgecolor='black', 
           s=80, zorder=5, label='Mean')

ax.set_xticks(positions)
ax.set_xticklabels(labels)
ax.set_ylabel('Latency (ms)')
ax.set_xlabel('Traffic Pattern')
ax.set_title('STAR-PPO Robustness: Latency Distribution')

ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.set_axisbelow(True)



plt.tight_layout()

output_path = os.path.join(RESULT_DIR, 'Workload_Robustness.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: {output_path}")

plt.close()
print("\nDone!")

