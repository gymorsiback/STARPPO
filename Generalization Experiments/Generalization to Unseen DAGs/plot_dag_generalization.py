
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
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 20,
    'lines.linewidth': 4.0,
    'mathtext.fontset': 'stix',
})

RESULT_DIR = os.path.dirname(os.path.abspath(__file__))
result_path = os.path.join(RESULT_DIR, 'dag_generalization_results.npz')

data = np.load(result_path, allow_pickle=True)
topologies = list(data['topologies'])
star_ppo_latencies = data['star_ppo']
greedy_latencies = data['greedy']
random_latencies = data['random']

print("加载的实验结果:")
for i, topo in enumerate(topologies):
    print(f"  {topo}: STAR-PPO={star_ppo_latencies[i]:.2f}, Greedy={greedy_latencies[i]:.2f}, Random={random_latencies[i]:.2f}")

algorithms = ['STAR_PPO', 'Greedy', 'Random']
colors = {'STAR_PPO': '#a6cee3', 'Greedy': '#b2df8a', 'Random': '#a89078'}
hatches = {'STAR_PPO': '/', 'Greedy': '...', 'Random': 'xx'}
labels = {'STAR_PPO': 'STAR-PPO (Ours)', 'Greedy': 'Greedy', 'Random': 'Random'}

latencies = {
    'STAR_PPO': star_ppo_latencies,
    'Greedy': greedy_latencies,
    'Random': random_latencies
}

fig, ax = plt.subplots(figsize=(8, 8))

x = np.arange(len(topologies))
width = 0.25

for i, algo in enumerate(algorithms):
    offset = (i - 1) * width
    bars = ax.bar(x + offset, latencies[algo], width, 
                  label=labels[algo], color=colors[algo], 
                  edgecolor='black', linewidth=1.2, hatch=hatches[algo])

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=15)

ax.set_xlabel('DAG Topology')
ax.set_ylabel('Average Latency (ms)')
ax.set_title('Generalization to Unseen DAG Topologies')
ax.set_xticks(x)
ax.set_xticklabels(topologies)
ax.legend(fontsize=18, loc='upper center', ncol=1, bbox_to_anchor=(0.5, 1.0))
ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.set_axisbelow(True)

ax.set_ylim(0, max(max(latencies['Random']), max(latencies['Greedy'])) * 1.15)

plt.tight_layout()
output_path = os.path.join(RESULT_DIR, 'DAG_Generalization.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n已保存: {output_path}")
plt.close()

improvements_vs_greedy = []
improvements_vs_random = []

for i in range(len(topologies)):
    star = star_ppo_latencies[i]
    greedy = greedy_latencies[i]
    random = random_latencies[i]
    
    imp_greedy = (greedy - star) / greedy * 100
    imp_random = (random - star) / random * 100
    
    improvements_vs_greedy.append(imp_greedy)
    improvements_vs_random.append(imp_random)

print("\n" + "="*60)
print("📊 实验结果汇总")
print("="*60)
print(f"\n{'拓扑':<12} {'STAR-PPO':<12} {'Greedy':<12} {'Random':<12} {'vs Greedy':<12} {'vs Random':<12}")
print("-"*72)

for i, topo in enumerate(topologies):
    star = star_ppo_latencies[i]
    greedy = greedy_latencies[i]
    random = random_latencies[i]
    imp_g = improvements_vs_greedy[i]
    imp_r = improvements_vs_random[i]
    print(f"{topo:<12} {star:<12.2f} {greedy:<12.2f} {random:<12.2f} {imp_g:>+.1f}%{'':<6} {imp_r:>+.1f}%")

print("\n✅ 关键发现:")
print(f"   - STAR-PPO 比 Greedy 平均提升: {np.mean(improvements_vs_greedy):.1f}%")
print(f"   - STAR-PPO 比 Random 平均提升: {np.mean(improvements_vs_random):.1f}%")
print(f"   - 无需重新训练，模型直接泛化到复杂 DAG 拓扑")
print("\nDone!")

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

categories = topologies
N = len(categories)

angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]  

star_ppo_norm = [random_latencies[i] / star_ppo_latencies[i] for i in range(N)]
greedy_norm = [random_latencies[i] / greedy_latencies[i] for i in range(N)]
random_norm = [1.0] * N  

star_ppo_norm += star_ppo_norm[:1]
greedy_norm += greedy_norm[:1]
random_norm += random_norm[:1]

ax.plot(angles, star_ppo_norm, 'o-', linewidth=2.5, label='STAR-PPO (Ours)', color='#2E86AB', markersize=8)
ax.fill(angles, star_ppo_norm, alpha=0.25, color='#2E86AB')

ax.plot(angles, greedy_norm, 's-', linewidth=2.5, label='Greedy', color='#28A745', markersize=8)
ax.fill(angles, greedy_norm, alpha=0.25, color='#28A745')

ax.plot(angles, random_norm, '^-', linewidth=2.5, label='Random', color='#DC3545', markersize=8)
ax.fill(angles, random_norm, alpha=0.25, color='#DC3545')

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)

ax.set_title('Generalization to Unseen DAG Topologies', 
             pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=18)

ax.set_ylim(0, max(max(star_ppo_norm), max(greedy_norm)) * 1.1)

ax.tick_params(axis='y', labelsize=14)

plt.tight_layout()
output_path = os.path.join(RESULT_DIR, 'DAG_Generalization_Radar.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"已保存: {output_path}")
plt.close()
