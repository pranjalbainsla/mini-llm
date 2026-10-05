"""
Best val loss per seed. x = seed, y = best val loss. First tag = reference: its points,
mean line and ±1 / ±2 std bands. Extra tags are overlaid at the same seeds, so each
variant point sits next to the baseline point it is paired with.
Usage: python experiments/plot_seeds.py baseline [gqa mla ...]   -> plots/seed_strip.png
Needs: numpy, matplotlib (pip install matplotlib)
"""
import glob, json, sys
import numpy as np
import matplotlib.pyplot as plt

tags = sys.argv[1:] or ['baseline']

def load(tag):
    runs = [json.load(open(f)) for f in sorted(glob.glob(f'results/{tag}_s*.json'))]
    return np.array([r['init_seed'] for r in runs]), np.array([r['best_val'] for r in runs])

seeds, ref = load(tags[0])
mu, sd = ref.mean(), ref.std(ddof=1)
print(f"{tags[0]}: n={len(ref)} mean={mu:.4f} std={sd:.4f}  2*std={2*sd:.4f}")

fig, ax = plt.subplots(figsize=(6.5, 4.5))
lo, hi = seeds.min() - 0.6, seeds.max() + 0.6
ax.fill_between([lo, hi], mu - 2 * sd, mu + 2 * sd, color='tab:blue', alpha=0.07, label='±2 std (decision threshold)')
ax.fill_between([lo, hi], mu - sd, mu + sd, color='tab:blue', alpha=0.18, label='±1 std')
ax.hlines(mu, lo, hi, color='tab:blue', lw=1.5, label=f'{tags[0]} mean {mu:.4f}')
ax.scatter(seeds, ref, s=45, color='tab:blue', edgecolor='white', zorder=3, label=tags[0])
for i, tag in enumerate(tags[1:], start=1):
    s, v = load(tag)
    ax.scatter(s, v, s=55, marker='D', color=f'C{i}', edgecolor='white', zorder=4, label=tag)
ax.set_xticks(seeds); ax.set_xlim(lo, hi)
ax.set_xlabel('seed'); ax.set_ylabel('best val loss')
ax.legend(fontsize=8, loc='best')
fig.tight_layout(); fig.savefig('plots/seed_strip.png', dpi=150)
