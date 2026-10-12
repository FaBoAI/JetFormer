#!/usr/bin/env python3
"""段階3の GPU の試し(1本ずつ・1プロセスずつ・続きから再開できる)

    cd ~/jetformer && nice -n 10 python -u /home/jetson/jetformer_reports/policy_zoo_20261012/smoke/run_smoke.py [方策 ...]

方策ごとに: tools/train.py(--max-steps 5・1 エポック、clean_driving_20261011、モデル B と同じ条件)→ ONNX(train.py が書く)
→ tools/build_engine.py(別のプロセス)→ engine_check.py(別のプロセス: TensorRT と PyTorch FP32 の差・速さ・走行の手順)。
学習結果は data/runs/_smoke/smoke-<slug>(走行・学習の一覧に出ない。終わったら消す)。出来たものは飛ばす。
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path('/home/jetson/jetformer')
HERE = Path(__file__).resolve().parent
SMOKE_RUNS = REPO / 'data' / 'runs' / '_smoke'
LOGS = HERE / 'logs'
PY = sys.executable
ORDER = ['jetformer_v0', 'act', 'tcp', 'msfusion', 'dinov2']
SLUG = {'jetformer_v0': 'v0', 'act': 'act', 'tcp': 'tcp', 'msfusion': 'msf', 'dinov2': 'dino'}
GPU_PATTERNS = ('tools/train.py', 'build_engine.py', 'policy_zoo.py', 'benchmark_model.py', 'engine_check.py',
                'rl_policies.py', 'rl_evaluate.py')


def mem_available_mb():
    for line in open('/proc/meminfo'):
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) // 1024
    return 0


def gpu_busy():
    me = os.getpid()
    out = subprocess.run(['ps', '-eo', 'pid,args'], capture_output=True, text=True).stdout.splitlines()[1:]
    busy = []
    for line in out:
        pid, _, args = line.strip().partition(' ')
        if int(pid) in (me, os.getppid()) or 'run_smoke.py' in args or 'grep' in args:
            continue
        if 'python' in args and any(p in args for p in GPU_PATTERNS):
            busy.append(line.strip()[:160])
    return busy


def wait_gpu_free():
    while True:
        busy = gpu_busy()
        if not busy:
            return
        print('GPU を使うプロセスが動いているので待ちます: %s' % busy, flush=True)
        time.sleep(30)


def step(name, cmd, log):
    wait_gpu_free()
    t = time.time()
    print('[%s] %s' % (time.strftime('%H:%M:%S'), ' '.join(map(str, cmd))), flush=True)
    with open(log, 'w', encoding='utf-8') as f:
        r = subprocess.run(['nice', '-n', '10'] + [str(c) for c in cmd], cwd=str(REPO), stdout=f,
                           stderr=subprocess.STDOUT)
    sec = round(time.time() - t, 1)
    print('  → 終了コード %d・%.0f 秒(%s)' % (r.returncode, sec, log.name), flush=True)
    return r.returncode, sec


def load(path):
    try:
        return json.load(open(path, encoding='utf-8'))
    except (OSError, ValueError):
        return None


def main(names):
    LOGS.mkdir(parents=True, exist_ok=True)
    SMOKE_RUNS.mkdir(parents=True, exist_ok=True)
    summary_path = HERE / 'summary.json'
    summary = load(summary_path) or {}
    for p in names:
        run = SMOKE_RUNS / ('smoke-%s' % SLUG[p])
        s = summary.get(p, {})
        res = load(run / 'result.json')
        if not (res and res.get('state') == 'done' and (run / 'jetformer.onnx').exists()):
            if run.exists():
                subprocess.run(['rm', '-rf', str(run)])
            if mem_available_mb() < 3500:
                print('空きメモリが %d MB(3500 MB 未満)。止めます' % mem_available_mb(), flush=True)
                return 1
            s['mem_available_before_train_mb'] = mem_available_mb()
            code, sec = step('train', [PY, '-u', 'tools/train.py', '--policy', p, '--data',
                                       'data/datasets/clean_driving_20261011', '--out', run, '--epochs', 1,
                                       '--max-steps', 5, '--batch', 64, '--workers', 4, '--stride', 3,
                                       '--no-command-input', '--impact-g', 2.5, '--seed', 0], LOGS / ('%s_train.log' % p))
            s['train'] = {'exit': code, 'wall_s': sec}
            res = load(run / 'result.json')
            if code != 0 or not res:
                summary[p] = s
                json.dump(summary, open(summary_path, 'w'), ensure_ascii=False, indent=1)
                continue
            s['train'].update({k: res.get(k) for k in ('state', 'best_val_us', 'params_m', 'peak_mem_mb', 'duration_s',
                                                       'train_samples', 'val_samples')})
            s['train']['history'] = res.get('history')
        engine = run / 'jetformer_fp16.engine'
        if not engine.exists():
            code, sec = step('engine', [PY, '-u', 'tools/build_engine.py', run / 'jetformer.onnx', engine],
                             LOGS / ('%s_engine.log' % p))
            s['engine'] = {'exit': code, 'wall_s': sec}
            try:
                import onnx
                m = onnx.load(str(run / 'jetformer.onnx'), load_external_data=False)
                s['onnx'] = {'nodes': len(m.graph.node), 'mb': round((run / 'jetformer.onnx').stat().st_size / 2 ** 20, 1)}
            except Exception as e:      # noqa: BLE001
                s['onnx'] = {'error': repr(e)}
            summary[p] = s
            json.dump(summary, open(summary_path, 'w'), ensure_ascii=False, indent=1)
            if code != 0:
                continue
        out = HERE / ('check_%s.json' % p)
        if not out.exists():
            code, sec = step('check', [PY, '-u', HERE / 'engine_check.py', run, out], LOGS / ('%s_check.log' % p))
            s['check'] = {'exit': code, 'wall_s': sec}
        summary[p] = s
        json.dump(summary, open(summary_path, 'w'), ensure_ascii=False, indent=1)
    print('おわり', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:] or ORDER))
