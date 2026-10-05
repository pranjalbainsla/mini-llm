"""
One summary row per ablation tag, with the verdict from the settled decision rule, plus the plot.
Reference = results/baseline_s*.json (5 seeds); std is ddof=1.

  single seed (n=1):  |delta vs baseline mean| <= 1 std -> noise; > 2 std -> real; else borderline (run seeds 2-5)
  n=5 seeds:          real only if |mean paired delta| > 2 std AND the sign matches in >= 4 of 5 paired seeds
  n=2..4:             incomplete, finish the seeds

Usage: python experiments/summarize.py [tag ...]     (default: every tag except baseline)
       python experiments/summarize.py --no-plot
Prints a markdown table, then a paste-ready log entry skeleton per tag.
"""
import glob, json, os, re, subprocess, sys
import numpy as np

args = [a for a in sys.argv[1:] if not a.startswith('--')]
plot = '--no-plot' not in sys.argv


def load(tag):
    runs = [json.load(open(f)) for f in sorted(glob.glob(f'results/{tag}_s*.json'))
            if re.fullmatch(rf'results/{re.escape(tag)}_s\d+\.json', f)]
    return sorted(runs, key=lambda r: r['init_seed'])


base = load('baseline')
bv = np.array([r['best_val'] for r in base])
mu, sd = bv.mean(), bv.std(ddof=1)
base_by_seed = {r['init_seed']: r['best_val'] for r in base}

tags = args or sorted({re.sub(r'_s\d+\.json$', '', f.split('/')[-1]) for f in glob.glob('results/*_s*.json')} - {'baseline'})


def verdict(runs):
    n = len(runs)
    v = np.array([r['best_val'] for r in runs])
    if n == 1:
        d = v[0] - mu
        if abs(d) <= sd:
            return 'noise', d
        if abs(d) > 2 * sd:
            return ('real (better)' if d < 0 else 'real (worse)'), d
        return 'borderline -> run seeds 2-5', d
    paired = np.array([r['best_val'] - base_by_seed[r['init_seed']] for r in runs if r['init_seed'] in base_by_seed])
    d = paired.mean()
    if n < 5:
        return f'incomplete ({n}/5 seeds)', d
    agree = max((paired < 0).sum(), (paired > 0).sum())
    if abs(d) > 2 * sd and agree >= 4:
        return ('real (better)' if d < 0 else 'real (worse)'), d
    return 'noise (5-seed)', d


rows, entries = [], []
for tag in tags:
    runs = load(tag)
    if not runs:
        print(f'no results for {tag}')
        continue
    try:
        hyp = json.load(open(f'results/{tag}.hyp.json'))
    except FileNotFoundError:
        hyp = {}
    m = lambda k: np.mean([r[k] for r in runs])
    verd, d = verdict(runs)
    bi = int(np.round(m('best_iter')))
    cfg = ' '.join(hyp.get('args', [])) or '-'
    gpu = hyp.get('gpu', '?')
    n = len(runs)
    rows.append(f"| {tag} | {n} | {m('best_val'):.4f} | {d:+.4f} ({d / sd:+.1f} std) | {bi} | {m('gap_at_best'):.3f} | "
                f"{runs[0]['params']:,} | {m('sec_per_iter') * 1000:.1f} | {gpu} | {verd} |")
    entries.append(f"### {tag}\n* Hypothesis: {hyp.get('note') or '(none recorded)'}\n* Config: {cfg}, all else as baseline\n"
                   f"* Results ({'seed 1' if n == 1 else f'{n} seeds'}): best val {m('best_val'):.4f} vs baseline "
                   f"{mu:.4f} ({d:+.4f}, {d / sd:+.1f} std), best iter {bi}, gap {m('gap_at_best'):.3f}, "
                   f"{runs[0]['params']:,} params, {m('sec_per_iter') * 1000:.1f} ms/iter on {gpu}\n"
                   f"* Verdict: {verd}\n* Takeaway: \n")

print(f'baseline: n={len(bv)} mean={mu:.4f} std={sd:.4f} (1 std={sd:.4f}, 2 std={2 * sd:.4f})\n')
print('| tag | seeds | best val | delta | best iter | gap | params | ms/iter | gpu | verdict |')
print('|---|---|---|---|---|---|---|---|---|---|')
print('\n'.join(rows), '\n')
print('\n'.join(entries))
print('Note: ms/iter is only comparable between runs on the same GPU.')

if plot and tags:
    subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plot_seeds.py'), 'baseline', *[t for t in tags if load(t)]], check=False)
    print('plot -> plots/seed_strip.png')
