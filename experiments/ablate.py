"""
Resumable ablation queue for Colab. Runs each (tag, args) back to back via toy_train.py,
skips runs whose results/<tag>_s<seed>.json already exists, logs failures and moves on,
and mirrors results/ to Drive after every run (Colab wipes local disk with the session).

Before each tag's first run it writes results/<tag>.hyp.json (hypothesis + GPU + time), so the
hypothesis is on disk before any result exists. summarize.py reads it back.
Run everything from the repo root (paths like results/ and toy_train.py are cwd-relative).

Colab cell:
    from google.colab import drive; drive.mount('/content/drive')
    from experiments.ablate import run_queue
    run_queue([
        dict(tag='gqa', args=['--attention=gqa'], note='GQA ~ noise: fewer KV heads cost little at this scale'),
        dict(tag='swiglu', args=['--ffn=swiglu'], note='SwiGLU slightly better, <1 std'),
    ], drive='/content/drive/MyDrive/mini-llm')           # seed 1 screen
    # borderline ones only:  run_queue([...same dict...], seeds=[2, 3, 4, 5], drive=...)
Wrap the cell in %%time. After it: !python experiments/summarize.py
"""
import glob, json, os, shutil, subprocess, sys, time


def gpu_name():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return out.splitlines()[0] if out else 'cpu'
    except Exception:
        return 'cpu'


def sync_to_drive(drive):
    if not drive:
        return
    os.makedirs(f'{drive}/results', exist_ok=True)
    for f in glob.glob('results/*'):
        shutil.copy2(f, f'{drive}/results/')
    if os.path.isdir('plots'):
        os.makedirs(f'{drive}/plots', exist_ok=True)
        for f in glob.glob('plots/*'):
            shutil.copy2(f, f'{drive}/plots/')


def restore_from_drive(drive):
    """Pull back anything a wiped session lost, without overwriting local files."""
    if not drive or not os.path.isdir(f'{drive}/results'):
        return
    os.makedirs('results', exist_ok=True)
    for f in glob.glob(f'{drive}/results/*'):
        dst = os.path.join('results', os.path.basename(f))
        if not os.path.exists(dst):
            shutil.copy2(f, dst)


def run_queue(runs, seeds=(1,), drive=None, config='configs/base.py'):
    os.makedirs('results', exist_ok=True)
    restore_from_drive(drive)
    gpu, failed = gpu_name(), []
    for r in runs:
        tag, args = r['tag'], r.get('args', [])
        hyp = f'results/{tag}.hyp.json'
        if not os.path.exists(hyp):  # keep the original hypothesis on reruns for extra seeds
            with open(hyp, 'w') as f:
                json.dump({'tag': tag, 'args': args, 'note': r.get('note', ''), 'gpu': gpu,
                           'written': time.strftime('%Y-%m-%d %H:%M:%S')}, f, indent=1)
        for seed in seeds:
            name = f'{tag}_s{seed}'
            if os.path.exists(f'results/{name}.json'):
                print(f'skip {name} (done)')
                continue
            t0 = time.time()
            with open(f'results/{name}.log', 'w') as log:
                rc = subprocess.run([sys.executable, 'toy_train.py', config, f'--init_seed={seed}',
                                     f'--train_seed={seed}', f'--run_name={name}', *args],
                                    stdout=log, stderr=subprocess.STDOUT).returncode
            print(f'{"done" if rc == 0 else "FAILED rc=" + str(rc)} {name} ({time.time() - t0:.0f}s)')
            if rc != 0:
                failed.append(name)
            sync_to_drive(drive)
    if failed:
        print('failed runs (see results/<name>.log):', ', '.join(failed))
    return failed
