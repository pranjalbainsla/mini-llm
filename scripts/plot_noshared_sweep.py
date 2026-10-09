"""Val loss vs granularity (E/k) with shared experts = 0, vs a dense SwiGLU reference,
plus per-expert load balance from the logged topk_idx loads.

Reads results/moe_{E}_{k}_s*.json and results/dense_s*.json.
Writes plots/noshared_granularity.png (or --out).

Usage: python scripts/plot_noshared_sweep.py [panels] [--out PATH]
  panels: any of a-f, e.g. "adf" (default "abcdef")
  a granularity vs dense   b val curves        c cost vs quality
  d final expert load      e load over training f loss vs imbalance

Load is logged per eval as a [n_layer, E] array of routing fractions (rows sum to 1).
We report it as E * load, so 1.0 = perfectly uniform, max = hottest expert, min = coldest.
"""
import argparse
import glob
import json

import matplotlib.pyplot as plt
import numpy as np

RUNGS = ["8_2", "16_4", "32_8"]
LABELS = [r.replace("_", "/") for r in RUNGS]


def load(pattern):
    return [json.load(open(f)) for f in sorted(glob.glob(pattern))]


def norm_load(entry):
    L = np.array(entry["load"])  # [n_layer, E]
    return L * L.shape[1]


moe = {r: load(f"results/moe_{r}_s*.json") for r in RUNGS}
dense = load("results/dense_s*.json")
assert all(moe[r] for r in RUNGS) and dense, "missing result files"

best = {r: np.array([d["best_val"] for d in runs]) for r, runs in moe.items()}
dense_best = np.array([d["best_val"] for d in dense])
ms = {r: np.mean([d["sec_per_iter"] for d in runs]) * 1e3 for r, runs in moe.items()}
dense_ms = np.mean([d["sec_per_iter"] for d in dense]) * 1e3

# final-eval load per run: [n_seeds, n_layer, E]
final = {r: np.stack([norm_load(d["history"][-1]) for d in moe[r]]) for r in RUNGS}

print(f"{'rung':<8}{'n':>3}{'mean':>9}{'std':>8}{'d vs dense':>12}{'ms/iter':>9}{'max load':>10}{'min load':>10}")
for r in RUNGS:
    v, f = best[r], final[r]
    print(f"{r:<8}{len(v):>3}{v.mean():>9.4f}{v.std(ddof=1):>8.4f}{v.mean() - dense_best.mean():>+12.4f}"
          f"{ms[r]:>9.0f}{f.max(-1).mean():>10.3f}{f.min(-1).mean():>10.3f}")
print(f"{'dense':<8}{len(dense_best):>3}{dense_best.mean():>9.4f}{dense_best.std(ddof=1):>8.4f}{0:>+12.4f}{dense_ms:>9.0f}")
print("(load = E*fraction at final eval, max/min over experts, averaged over layers and seeds; 1.0 = uniform)")


def panel_a(a):
    """(a) best val per seed, mean +- std, dense band"""
    x = np.arange(len(RUNGS))
    a.axhspan(dense_best.mean() - dense_best.std(ddof=1), dense_best.mean() + dense_best.std(ddof=1),
              color="gray", alpha=0.25, label="dense SwiGLU ±1 std")
    a.axhline(dense_best.mean(), color="gray", ls="--")
    for i, r in enumerate(RUNGS):
        a.scatter(np.full(len(best[r]), i), best[r], color="C0", alpha=0.6, zorder=3)
    a.errorbar(x, [best[r].mean() for r in RUNGS], [best[r].std(ddof=1) for r in RUNGS],
               fmt="o-", color="C0", capsize=4, label="MoE (no shared) mean ± std")
    a.set_xticks(x, LABELS); a.set_xlabel("E/k"); a.set_ylabel("best val loss")
    a.set_title("Granularity at fixed sparsity"); a.legend(fontsize=8)


def panel_b(b):
    """(b) mean val curves"""
    for i, r in enumerate(RUNGS):
        h = np.array([[p["val"] for p in d["history"]] for d in moe[r]])
        it = [p["iter"] for p in moe[r][0]["history"]]
        b.plot(it, h.mean(0), label=LABELS[i], color=f"C{i}")
    h = np.array([[p["val"] for p in d["history"]] for d in dense])
    b.plot([p["iter"] for p in dense[0]["history"]], h.mean(0), "k--", label="dense")
    b.set_ylim(1.55, 2.0); b.set_xlim(400, max(it))
    b.set_xlabel("iter"); b.set_ylabel("val loss (mean over seeds)"); b.set_title("Val curves"); b.legend()


def panel_c(c):
    """(c) cost vs quality"""
    for i, r in enumerate(RUNGS):
        c.errorbar(ms[r], best[r].mean(), best[r].std(ddof=1), fmt="o", color=f"C{i}", capsize=3, label=LABELS[i])
    c.errorbar(dense_ms, dense_best.mean(), dense_best.std(ddof=1), fmt="s", color="k", capsize=3, label="dense")
    c.set_xlabel("ms / iter"); c.set_ylabel("best val loss"); c.set_title("Cost vs quality"); c.legend()


def panel_d(d_):
    """(d) final max/min expert load per layer, mean over seeds (whiskers = seed range)"""
    n_layer = final[RUNGS[0]].shape[1]
    w = 0.8 / len(RUNGS)
    for i, r in enumerate(RUNGS):
        xs = np.arange(n_layer) + (i - (len(RUNGS) - 1) / 2) * w
        for stat, marker in ((final[r].max(-1), "^"), (final[r].min(-1), "v")):  # [seeds, layer]
            d_.errorbar(xs, stat.mean(0), [stat.mean(0) - stat.min(0), stat.max(0) - stat.mean(0)],
                        fmt=marker, color=f"C{i}", capsize=2, label=f"{LABELS[i]} {'max' if marker == '^' else 'min'}")
    d_.axhline(1.0, color="gray", ls="--", lw=1)
    d_.set_xticks(np.arange(n_layer), [f"L{l}" for l in range(n_layer)])
    d_.set_ylabel("E · load fraction"); d_.set_title("Final expert load (max ▲ / min ▼)")
    d_.legend(fontsize=7, ncol=2)


def panel_e(e):
    """(e) imbalance over training: max/min E*load, mean over layers & seeds"""
    for i, r in enumerate(RUNGS):
        it = [p["iter"] for p in moe[r][0]["history"]]
        L = np.stack([np.stack([norm_load(p) for p in d["history"]]) for d in moe[r]])  # [seed, t, layer, E]
        e.plot(it, L.max(-1).mean((0, 2)), "-", color=f"C{i}", label=f"{LABELS[i]} max")
        e.plot(it, L.min(-1).mean((0, 2)), ":", color=f"C{i}", label=f"{LABELS[i]} min")
    e.axhline(1.0, color="gray", ls="--", lw=1)
    e.set_xlabel("iter"); e.set_ylabel("E · load (mean over layers, seeds)")
    e.set_title("Load balance over training"); e.legend(fontsize=7, ncol=2)


def panel_f(f_):
    """(f) does imbalance explain loss? per-seed best val vs final max load"""
    for i, r in enumerate(RUNGS):
        f_.scatter(final[r].max(-1).mean(-1), best[r], color=f"C{i}", label=LABELS[i])
    f_.axhline(dense_best.mean(), color="gray", ls="--", label="dense mean")
    f_.set_xlabel("final max E·load (mean over layers)"); f_.set_ylabel("best val loss")
    f_.set_title("Loss vs imbalance (per seed)"); f_.legend(fontsize=8)


PANELS = {"a": panel_a, "b": panel_b, "c": panel_c, "d": panel_d, "e": panel_e, "f": panel_f}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("panels", nargs="?", default="abcdef",
                    help="which panels to draw, e.g. 'adf' (default: all)")
    ap.add_argument("--out", default="plots/noshared_granularity.png")
    args = ap.parse_args()

    keys = [p for p in args.panels.lower() if p in PANELS]
    assert keys, f"no valid panels in {args.panels!r}; choose from {''.join(PANELS)}"
    ncols = min(3, len(keys))
    nrows = -(-len(keys) // ncols)
    fig, ax = plt.subplots(nrows, ncols, figsize=(5.3 * ncols, 4.5 * nrows), squeeze=False)
    for a, k in zip(ax.flat, keys):
        PANELS[k](a)
    for a in ax.flat[len(keys):]:
        a.set_visible(False)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"wrote {args.out} (panels: {''.join(keys)})")
