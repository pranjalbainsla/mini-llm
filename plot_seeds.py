"""
Strip plot of best val loss per seed. First tag = reference (mean line + ±1 std band);
extra tags are overlaid as points next to it.
Usage: python plot_seeds.py baseline [gqa mla ...]   -> plots/seed_strip.png
Needs: numpy, matplotlib (pip install matplotlib)
"""
import glob, json, sys
import numpy as np
import matplotlib.pyplot as plt

tags = sys.argv[1:] or ['baseline']

def load(tag):
    runs = [json.load(open(f)) for f in sorted(glob.glob(f'results/{tag}_s*.json'))]
    return np.array([r['best_val'] for r in runs]), runs

ref, ref_runs = load(tags[0])
mu, sd = ref.mean(), ref.std(ddof=1)
print(f"{tags[0]}: n={len(ref)} mean={mu:.4f} std={sd:.4f}  2*std={2*sd:.4f}")

fig, ax = plt.subplots(figsize=(1.6 + 1.4 * len(tags), 4.5))
ax.axhspan(mu - sd, mu + sd, color='tab:blue', alpha=0.18, label='baseline ±1 std')
ax.axhspan(mu - 2 * sd, mu + 2 * sd, color='tab:blue', alpha=0.07, label='baseline ±2 std (decision threshold)')
ax.axhline(mu, color='tab:blue', lw=1.5, label=f'baseline mean {mu:.4f}')
rng = np.random.default_rng(0)
for i, tag in enumerate(tags):
    v, _ = load(tag)
    ax.scatter(i + rng.uniform(-0.08, 0.08, len(v)), v, s=40, zorder=3,
               color='tab:blue' if i == 0 else f'C{i}', edgecolor='white')
    if i > 0:
        ax.hlines(v.mean(), i - 0.2, i + 0.2, color=f'C{i}', lw=2)
ax.set_xticks(range(len(tags))); ax.set_xticklabels(tags)
ax.set_xlim(-0.6, len(tags) - 0.4)
ax.set_ylabel('best val loss'); ax.legend(fontsize=8, loc='upper right')
fig.tight_layout(); fig.savefig('plots/seed_strip.png', dpi=150)
