#!/usr/bin/env python3
"""方策の比較(段階4)の学習: 5つの方策 × 種 0・1 を1本ずつ学習し、エンジンまで作る(GPU、1プロセスずつ・続きから再開できる)

    cd ~/jetformer && nice -n 10 python -u /home/jetson/jetformer_reports/policy_zoo_20261012/train/zoo_train.py <小コマンド>

小コマンド(どれも出来たものは飛ばす。電源断のあとは同じコマンドをもう一度実行すれば続きから):
    probe   5つの方策を 1 エポックだけ学習して、1エポックの時間・書き出しの時間・メモリを測る(data/runs/_zoo_probe/、終わったら消す)
    plan    probe の時間から本体 10 本の見込みを出し、エポック数を決めて plan.json に1回だけ書く(あれば読むだけ)
    train   plan.json のとおり、種 0 の5本 → 種 1 の5本 の順に tools/train.py(last.pth があれば --resume)→ 別のプロセスで
            tools/build_engine.py。各学習結果に run.json(管理画面の一覧に出す)
    check   各学習結果のエンジンを、走行と同じ手順で読み(drive.model_info → TRTRunner → check_io → 空回し)、
            実フレーム 64 個で PyTorch FP32 との差と run_numpy の速さを測る(1本ずつ別のプロセス。engine_load_check.py)
    summary 学習の時間・曲線・結果を summary.json にまとめ、図を fig/ に書く(CPU)

spec.md §4.1 の条件(モデル B と同じ): clean_driving_20261011・stride 3・バッチ 64・lr 3e-4・workers 4・今の操作値なし・
衝突 2.5G の除外・既定の形(--policy-cfg なし)。名前は clean_driving_20261011-<方策>-s<種>-nocmd-stride3-e<エポック>。

GPU を使う手順の前に毎回、ほかに GPU を使うプロセス(学習・エンジン・評価の道具)が無いこと、走行中でないこと(管理画面の API)、
空きメモリ 3.5 GB 以上を確かめる(足りなければ待つ)。状態は state.json(tmp → 置き換え)。
"""

import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path('/home/jetson/jetformer')
HERE = Path(__file__).resolve().parent
REPORT = HERE.parent
LOGS = HERE / 'logs'
ENGINE_CHECK_DIR = HERE / 'engine_check'
STATE_PATH = HERE / 'state.json'
PLAN_PATH = HERE / 'plan.json'
RUNS = REPO / 'data' / 'runs'
PROBE_RUNS = RUNS / '_zoo_probe'
DATA = 'data/datasets/clean_driving_20261011'
DATASET = 'clean_driving_20261011'
PY = sys.executable
ORDER = ['jetformer_v0', 'act', 'tcp', 'msfusion', 'dinov2']
SEEDS = [0, 1]
EPOCHS_SPEC = 30
COMMON = ['--stride', '3', '--batch', '64', '--lr', '3e-4', '--workers', '4', '--no-command-input', '--impact-g', '2.5']
MIN_FREE_MB = 3500
# 予算(spec.md §4.2): 学習 190 分 + エンジン 9.7×2 分 + 起動・書き出し 10 分 = 219.4 分。見込みの 1.3 倍までは任意(§4.3)を削って吸収する
BUDGET_MIN = 190 + 19.4 + 10
BUDGET_TOLERANCE = 1.3
# エンジンの作成の秒数(段階3の試しの実測。plan の見込みに使う)
ENGINE_S = {'jetformer_v0': 72, 'act': 98, 'tcp': 102, 'msfusion': 207, 'dinov2': 109}
GPU_PATTERNS = ('tools/train.py', 'build_engine.py', 'policy_zoo.py', 'benchmark_model.py', 'engine_check.py',
                'engine_load_check.py', 'rl_policies.py', 'rl_evaluate.py', 'compare_models.py', 'tools/evaluate.py',
                'offline_rl.py', 'run_smoke.py')
# 学習・エンジンに使うコードの指紋(途中でコードが変わっていないかを報告に残す)
CODE_FILES = ['tools/train.py', 'tools/build_engine.py', 'jetformer/model.py', 'jetformer/train_data.py',
              'jetformer/trt_runtime.py'] + ['jetformer/policies/%s' % p for p in (
                  '__init__.py', 'runtime.py', 'blocks.py', 'jetformer_v0.py', 'act.py', 'tcp.py', 'tcp_labels.py',
                  'msfusion.py', 'dinov2.py', 'vit.py')]


def now():
    return time.strftime('%Y-%m-%d %H:%M:%S')


def log(msg):
    print('[%s] %s' % (now(), msg), flush=True)


def load(path, default=None):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def write_json(path, obj):
    tmp = str(path) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def code_fingerprint():
    h = hashlib.sha1()
    for rel in CODE_FILES:
        p = REPO / rel
        h.update(rel.encode())
        h.update(p.read_bytes() if p.exists() else b'<none>')
    return h.hexdigest()[:12]


def mem_available_mb():
    for line in open('/proc/meminfo'):
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) // 1024
    return 0


def gpu_busy():
    """GPU を使いそうなほかのプロセス(自分と子は除く)と、管理画面の走行・学習"""
    me = {os.getpid(), os.getppid()}
    out = subprocess.run(['ps', '-eo', 'pid,ppid,args'], capture_output=True, text=True).stdout.splitlines()[1:]
    busy = []
    for line in out:
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        pid, ppid, args = int(parts[0]), int(parts[1]), parts[2]
        if pid in me or ppid in me or 'zoo_train.py' in args or 'grep' in args:
            continue
        if 'python' in args and any(p in args for p in GPU_PATTERNS):
            busy.append(args[:160])
    for path in ('/api/drive/status', '/api/train/status'):
        try:
            with urllib.request.urlopen('http://127.0.0.1:8080' + path, timeout=3) as r:
                st = json.load(r)
            if st.get('running'):
                busy.append('管理画面 %s running' % path)
        except Exception:       # noqa: BLE001  管理画面が止まっていれば ps の確認だけ
            pass
    return busy


def wait_ready(min_free_mb=MIN_FREE_MB):
    """GPU が空いて空きメモリが足りるまで待つ(30 秒ごと)"""
    said = None
    while True:
        busy = gpu_busy()
        free = mem_available_mb()
        if not busy and free >= min_free_mb:
            return free
        why = 'GPU: %s' % busy if busy else '空きメモリ %d MB < %d MB' % (free, min_free_mb)
        if why != said:
            log('待ちます(%s)' % why)
            said = why
        time.sleep(30)


def run_step(cmd, logfile, append=False, header=None):
    free = wait_ready()
    t = time.time()
    log('%s(空き %d MB)' % (' '.join(map(str, cmd)), free))
    with open(logfile, 'a' if append else 'w', encoding='utf-8') as f:
        if header:
            f.write(header + '\n')
            f.flush()
        r = subprocess.run(['nice', '-n', '10'] + [str(c) for c in cmd], cwd=str(REPO), stdout=f,
                           stderr=subprocess.STDOUT)
    sec = round(time.time() - t, 1)
    log('  → 終了コード %d・%.0f 秒(%s)' % (r.returncode, sec, logfile))
    return r.returncode, sec, free


def run_name(policy, seed, epochs):
    return '%s-%s-s%d-nocmd-stride3-e%d' % (DATASET, policy, seed, epochs)


def run_done(run, epochs):
    res = load(run / 'result.json')
    return bool(res and res.get('state') == 'done' and res.get('epochs_done') == epochs
                and (run / 'jetformer.onnx').exists() and (run / 'model_config.json').exists()
                and (run / 'best.pth').exists() and (run / 'last.pth').exists())


def last_epoch(run):
    """last.pth の済んだエポック(無ければ 0、読めなければ -1)。この親プロセスで torch を読まない(メモリを学習に残す)ため、
    別のプロセス(CPU だけ)で読む"""
    p = run / 'last.pth'
    if not p.exists():
        return 0
    code = ('import torch,sys; ck=torch.load(sys.argv[1], map_location="cpu", weights_only=False); '
            'print(int(ck.get("epoch", 0)))')
    r = subprocess.run([PY, '-c', code, str(p)], capture_output=True, text=True,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
    try:
        return int(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return -1


# ---------------------------------------------------------------- probe
def cmd_probe(state):
    probe = state.setdefault('probe', {})
    PROBE_RUNS.mkdir(parents=True, exist_ok=True)
    for p in ORDER:
        if probe.get(p, {}).get('done'):
            continue
        run = PROBE_RUNS / p
        if run.exists():
            shutil.rmtree(run)
        code, sec, free = run_step([PY, '-u', 'tools/train.py', '--policy', p, '--data', DATA, '--out', run,
                                    '--epochs', 1, '--seed', 0] + COMMON, LOGS / ('probe_%s.log' % p))
        res = load(run / 'result.json') or {}
        hist = (res.get('history') or [{}])[0]
        txt = (LOGS / ('probe_%s.log' % p)).read_text(encoding='utf-8', errors='replace')
        probe[p] = {'done': code == 0 and res.get('state') == 'done', 'exit': code, 'wall_s': sec,
                    'mem_available_mb': free, 'epoch_s': hist.get('seconds'), 'duration_s': res.get('duration_s'),
                    'val_us_epoch1': hist.get('val_us'), 'train_us_epoch1': hist.get('train_us'),
                    'peak_mem_mb': res.get('peak_mem_mb'), 'params_m': res.get('params_m'),
                    'train_samples': res.get('train_samples'), 'val_samples': res.get('val_samples'),
                    'log_tail': txt.strip().splitlines()[-3:], 'code': code_fingerprint(), 'at': now()}
        # 1エポックの外の時間(起動・データの読み込み・ONNX の書き出し)= 全体 − エポック
        if hist.get('seconds') is not None:
            probe[p]['overhead_s'] = round(sec - hist['seconds'], 1)
        save_state(state)
        if not probe[p]['done']:
            log('probe %s が失敗しました。止めます' % p)
            return 1
        shutil.rmtree(run)
    if PROBE_RUNS.exists() and not any(PROBE_RUNS.iterdir()):
        PROBE_RUNS.rmdir()
    return 0


# ---------------------------------------------------------------- plan
def project(probe, epochs):
    """本体 10 本の見込み [分] {方策: 1本の分}・合計"""
    per = {}
    for p in ORDER:
        pr = probe[p]
        per[p] = (pr['overhead_s'] + epochs * pr['epoch_s'] + ENGINE_S[p] + 15) / 60.0
    return per, sum(per.values()) * len(SEEDS)


def cmd_plan(state):
    plan = load(PLAN_PATH)
    if plan:
        log('plan.json はもうあります: エポック %d(見込み %.0f 分)' % (plan['epochs'], plan['projected_min']))
        return 0
    probe = state.get('probe', {})
    if not all(probe.get(p, {}).get('done') for p in ORDER):
        raise SystemExit('先に probe を済ませてください')
    limit = BUDGET_MIN * BUDGET_TOLERANCE
    epochs = EPOCHS_SPEC
    per, total = project(probe, epochs)
    cut = None
    while total > limit and epochs > 1:       # 収まらなければ、種は2本のまま全方策のエポックを同じだけ減らす
        epochs -= 1
        per, total = project(probe, epochs)
        cut = 'エポックを %d → %d に一律で減らした(見込み %.0f 分 > 上限 %.0f 分)' % (EPOCHS_SPEC, epochs, total, limit)
    plan = {'epochs': epochs, 'seeds': SEEDS, 'order': [[p, s] for s in SEEDS for p in ORDER],
            'runs': {'%s/s%d' % (p, s): run_name(p, s, epochs) for s in SEEDS for p in ORDER},
            'per_run_min': {p: round(v, 1) for p, v in per.items()}, 'projected_min': round(total, 1),
            'budget_min': BUDGET_MIN, 'limit_min': round(limit, 1), 'cut': cut,
            'spec_estimate_min': {'jetformer_v0': 17, 'act': 18, 'tcp': 18, 'msfusion': 26, 'dinov2': 16},
            'common_args': COMMON, 'written_at': now(), 'code': code_fingerprint()}
    write_json(PLAN_PATH, plan)
    log('plan: エポック %d・見込み %.0f 分(予算 %.0f 分・上限 %.0f 分)%s' % (
        epochs, total, BUDGET_MIN, limit, ('・' + cut) if cut else ''))
    return 0


# ---------------------------------------------------------------- train
def write_run_json(run, policy, seed, epochs, created_at):
    """管理画面の学習タブ・走行タブの一覧に出すための run.json(管理画面の学習と同じ欄)"""
    from jetformer import policies      # torch を読まない
    meta = {'dataset': DATASET, 'stride': 3, 'epochs': epochs, 'batch': 64, 'lr': 0.0003, 'pretrained': True,
            'build_engine': True, 'command_input': False, 'mask_impacts': True, 'policy': policy,
            'policy_cfg': policies.get(policy).resolve({}), 'impact_g': 2.5, 'seed': seed,
            'name': run.name, 'created_at': created_at,
            'note': '方策の比較(policy zoo 2026-10-12)の学習。机上の比較用で、実車の A/B/A の前は走行の標準はモデル B のまま'}
    write_json(run / 'run.json', meta)


def cmd_train(state):
    plan = load(PLAN_PATH)
    if not plan:
        raise SystemExit('先に plan を済ませてください')
    sys.path.insert(0, str(REPO))
    epochs = plan['epochs']
    runs_state = state.setdefault('runs', {})
    for policy, seed in plan['order']:
        key = '%s/s%d' % (policy, seed)
        run = RUNS / plan['runs'][key]
        rs = runs_state.setdefault(key, {'run': run.name, 'policy': policy, 'seed': seed, 'attempts': []})
        # ---- 学習
        if not run_done(run, epochs):
            run.mkdir(parents=True, exist_ok=True)
            rs.setdefault('created_at', datetime.datetime.now().isoformat(timespec='seconds'))
            done_ep = last_epoch(run)
            resume = (run / 'last.pth').exists()
            cmd = [PY, '-u', 'tools/train.py', '--policy', policy, '--data', DATA, '--out', run, '--epochs', epochs,
                   '--seed', seed] + COMMON + (['--resume'] if resume else [])
            header = '==== %s %s(%s)====' % (now(), '再開: %d エポック目まで済み' % done_ep if resume else '開始',
                                             ' '.join(map(str, cmd[2:])))
            att = {'started': now(), 'resume_from_epoch': done_ep if resume else 0, 'code': code_fingerprint()}
            rs['attempts'].append(att)
            save_state(state)
            code, sec, free = run_step(cmd, run / 'train.log', append=True, header=header)
            att.update({'exit': code, 'wall_s': sec, 'ended': now(), 'mem_available_mb': free})
            save_state(state)
            if not run_done(run, epochs):
                log('%s の学習が終わっていません(終了コード %d)。止めます(もう一度実行すれば続きから)' % (key, code))
                return 1
        res = load(run / 'result.json')
        rs.update({'train_wall_s': round(sum(a.get('wall_s') or 0 for a in rs['attempts']), 1),
                   'best_val_us': res.get('best_val_us'), 'best_epoch': res.get('best_epoch'),
                   'peak_mem_mb': res.get('peak_mem_mb'), 'params_m': res.get('params_m'),
                   'epoch_s_mean': round(sum(h['seconds'] for h in res['history']) / len(res['history']), 1)})
        # ---- エンジン(別のプロセス。tmp → check_io → 置き換え)
        onnx, engine = run / 'jetformer.onnx', run / 'jetformer_fp16.engine'
        if not engine.exists() or engine.stat().st_mtime < onnx.stat().st_mtime:
            code, sec, free = run_step([PY, '-u', 'tools/build_engine.py', onnx, engine], run / 'engine.log')
            rs['engine'] = {'exit': code, 'wall_s': sec, 'at': now(), 'code': code_fingerprint()}
            save_state(state)
            if code != 0 or not engine.exists():
                log('%s のエンジンの作成に失敗しました(終了コード %d)。止めます' % (key, code))
                return 1
        if not (run / 'run.json').exists():
            write_run_json(run, policy, seed, epochs, rs['created_at'])
        rs['done'] = True
        save_state(state)
    log('学習・エンジン: 全部済みました')
    return 0


# ---------------------------------------------------------------- check
def cmd_check(state):
    plan = load(PLAN_PATH)
    ENGINE_CHECK_DIR.mkdir(parents=True, exist_ok=True)
    checks = state.setdefault('checks', {})
    for policy, seed in plan['order']:
        key = '%s/s%d' % (policy, seed)
        run = RUNS / plan['runs'][key]
        out = ENGINE_CHECK_DIR / ('%s.json' % run.name)
        if out.exists() or not (run / 'jetformer_fp16.engine').exists():
            continue
        code, sec, _ = run_step([PY, '-u', HERE / 'engine_load_check.py', run, out], LOGS / ('check_%s.log' % run.name))
        checks[key] = {'exit': code, 'wall_s': sec, 'at': now()}
        save_state(state)
    return 0


def save_state(state):
    state['updated'] = now()
    write_json(STATE_PATH, state)


def main(argv):
    if len(argv) != 2 or argv[1] not in ('probe', 'plan', 'train', 'check', 'summary'):
        raise SystemExit(__doc__)
    LOGS.mkdir(parents=True, exist_ok=True)
    os.chdir(REPO)
    state = load(STATE_PATH, {})
    if argv[1] == 'summary':
        from zoo_summary import main as summary_main      # noqa: E402  同じフォルダー
        return summary_main()
    return {'probe': cmd_probe, 'plan': cmd_plan, 'train': cmd_train, 'check': cmd_check}[argv[1]](state)


if __name__ == '__main__':
    sys.path.insert(0, str(HERE))
    sys.exit(main(sys.argv))
