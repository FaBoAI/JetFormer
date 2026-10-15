#!/usr/bin/env python3
"""sim_diverse_train_20261015 の学習の道具: protocol.json の腕(S-DIV・S-MOD・C-DIV 種 0・1、A4)を 1 本ずつ学習する
(GPU は 1 プロセスずつ・電源断のあとは同じコマンドで続きから)

    cd ~/jetformer && nohup /home/jetson/miniconda/envs/jetracer/bin/python -u \\
        experiments/20261015_sim_diverse_train/train/diverse_train.py train S-DIV/s0 \\
        >> /home/jetson/jetformer_reports/sim_diverse_train_20261015/train/logs/driver.log 2>&1 &
    python experiments/20261015_sim_diverse_train/train/diverse_train.py status          # 状態(train/state.json)
    python experiments/20261015_sim_diverse_train/train/diverse_train.py dryrun S-DIV/s0  # 学習のコマンドを出すだけ
    python experiments/20261015_sim_diverse_train/train/diverse_train.py bundle S-DIV/s0  # 一式だけ作り直す(アプリの写しが出来てから)
    python experiments/20261015_sim_diverse_train/train/diverse_train.py smoke S-DIV [--kill-after-epoch1]   # CPU の動作確認
    python experiments/20261015_sim_diverse_train/train/diverse_train.py post [S-DIV/s0 ...]  # 後片付けだけ(HTTP の読み込み・図・training.md)

学習の段(10/15 09:55〜)で足したこと:
  - train の引数 --workers N・--mem-cap M(protocol の外。使うと deviations.md に自動で書く)
  - train/PAUSE があれば、次の学習を始めずに待つ(学習の途中では止めない)。評価の段が lock big を使いたいときに置き、済んだら消す
  - 1 本ごとの後片付け post_run: Q1 の腕は http_load_check.py(口 8825)→ bundles/<名前>/http_load_check.json、
    そのあと training_report.py(図・表・training.md)

腕の順番は引数で決める(既定は S-DIV/s0 だけ)。時間割(protocol.json の time_plan)では、S-DIV の学習のあとに Firefox の評価
(lock big)が入るので、C-DIV・S-MOD は評価が済んでから別に `train C-DIV/s0 C-DIV/s1`・`train S-MOD/s0` で始める。

各 1 本ずつ:
  ① tools/train.py(protocol の arms_Q1.common_recipe・arms_Q2.C-DIV.recipe のコマンド。last.pth があれば --resume)
     jf-heavy --mem 3G --lock big --need-mb 3200、expandable_segments・THP なし(nothp.py)・MALLOC_ARENA_MAX 2・choom 1000、
     全体の空きメモリの見張り(200 MB 未満が 8 秒続いたら自分の学習だけ止めて、空くのを待って続きから)。
     workers: Q1 の腕は 1(メモリの上限で止まったら 0)、C-DIV は 2(止まったら 1。さらに 0 は protocol の外なので deviations.md に書く)
  ② run.json(管理画面の一覧。Q1 の腕は「シミュレーター専用(ジャンル 8)。実車では使わない」)
  ③ Q1 の腕だけ: export_bundle.py(CPU)→ check_bundle.py(シミュレーターの Python)。アプリの写し(レポートの sim/)がまだ無ければ飛ばし、
     あとで `bundle` で作る。C-DIV は一式を作らない(実機の保留データのオフラインの評価だけ。FP16 のエンジンも作らない)
出来たもの(result.json の state=done)は飛ばす。

10/14 の experiments/20261014_sim_dart_margin/train/dart_train.py の写しを、この作業の腕・名前(simdiv_20261015-<腕>-s<種>)・パス・
学習データの確かめ(prepare_check.json)に合わせて直したもの。メモリの工夫は 10/14 と同じ。
protocol.json の sha256 が protocol.sha256 と違えば止める。状態は train/state.json(tmp → 置き換え)。車は動かさない・PCA9685 に触らない。
"""

import datetime
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

REPO = Path('/home/jetson/jetformer')
HERE = Path(__file__).resolve().parent
EXP_REL = 'experiments/20261015_sim_diverse_train'
REPORT = Path('/home/jetson/jetformer_reports/sim_diverse_train_20261015')
TRAIN = REPORT / 'train'
LOGS = TRAIN / 'logs'
PROTOCOL = REPORT / 'protocol.json'
PROTOCOL_SHA = REPORT / 'protocol.sha256'
PREPARE_CHECK = REPORT / 'prepare_check.json'
DEVIATIONS = REPORT / 'deviations.md'
STATE_PATH = TRAIN / 'state.json'
A4_PLAN = TRAIN / 'a4_plan.json'           # DAgger の段が書く(元の腕・データ・初期値)
BUNDLES = REPORT / 'bundles'
SIM_COPY = REPORT / 'sim'
SMOKE = REPORT / 'smoke'
RUNS = REPO / 'data' / 'runs'
JF_HEAVY = '/home/jetson/bin/jf-heavy'
SIM_PY = '/home/jetson/jetformer/data/sim/venv/bin/python'     # 読むだけ(onnxruntime 1.31.0・numpy 2.2.6・opencv 4.13.0)
PY = '/home/jetson/miniconda/envs/jetracer/bin/python'        # 10/13〜10/14 の学習と同じ Python
PREFIX = 'simdiv_20261015-'
ARM_CODE = {'S-DIV': 'SDIV', 'S-MOD': 'SMOD', 'C-DIV': 'CDIV', 'A4': 'A4'}
Q1_ARMS = ('S-DIV', 'S-MOD', 'A4')
DEFAULT_ORDER = ['S-DIV/s0']
MIN_FREE_MB = 3200          # 学習を始める空きメモリ(10/13〜10/14 と同じ)
GPU_ENV = {'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True', 'MALLOC_ARENA_MAX': '2'}
DEFAULT_WORKERS = {'S-DIV': 1, 'S-MOD': 1, 'A4': 1, 'C-DIV': 2}
WORKERS_STEPS = {'S-DIV': {1: 0}, 'S-MOD': {1: 0}, 'A4': {1: 0}, 'C-DIV': {2: 1, 1: 0}}
WORKERS_IN_PROTOCOL = {'S-DIV': {1, 0}, 'S-MOD': {1, 0}, 'A4': {1, 0}, 'C-DIV': {2, 1}}
MAX_ATTEMPTS = 6
MAX_WATCHDOG_RETRIES = 40
GPU_PATTERNS = ('tools/train.py', 'build_engine.py', 'policy_zoo.py', 'benchmark_model.py', 'engine_check.py',
                'engine_load_check.py', 'rl_policies.py', 'rl_evaluate.py', 'compare_models.py', 'tools/evaluate.py',
                'offline_rl.py', 'run_smoke.py', 'zoo_train.py', 'sim_cotrain_eval', 'trtexec', 'offline_eval', 'q2_eval')
CODE_FILES = ['tools/train.py', 'jetformer/model.py', 'jetformer/train_data.py', 'jetformer/policies/__init__.py',
              'jetformer/policies/runtime.py', 'jetformer/policies/blocks.py', 'jetformer/policies/jetformer_v0.py']
NOTE = {
    'S-DIV': 'いろいろな走りの実験(sim_diverse_train_20261015)の S-DIV: 合格した強化学習の 5 モード(最速・速さいろいろ・'
             '絶対壁にぶつからない・離脱復活・経路いろいろ)の見本 75 シーン + DART(D1)30 シーン + 余裕の線(D2)30 シーン = 45 分。'
             'シミュレーター専用(ジャンル 8)。実車では使わない',
    'S-MOD': 'いろいろな走りの実験(sim_diverse_train_20261015)の S-MOD: 合格した強化学習の 5 モードの見本 75 シーンだけ(25 分)。'
             'シミュレーター専用(ジャンル 8)。実車では使わない',
    'C-DIV': 'いろいろな走りの実験(sim_diverse_train_20261015)の C-DIV: 実機 clean_driving_20261011 の 52 シーン + 合格した 5 モードの見本の'
             '学習の 60 シーン(36,000 フレーム)を混ぜて学習。机上の比較用で、実車の A/B/A の前は走行の標準はモデル B のまま'
             '(ジャンル 8 = シミュレーターの頭は実車では使わない)',
    'A4': 'いろいろな走りの実験(sim_diverse_train_20261015)の A4: 一番良い Q1 の腕に DAgger を 1 回(学習した JetFormer が行った状態に'
          '先生の操作を付けて足し、10 エポック続きを学習)。シミュレーター専用(ジャンル 8)。実車では使わない',
}


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


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def code_fingerprint():
    h = hashlib.sha1()
    for rel in CODE_FILES:
        p = REPO / rel
        h.update(rel.encode())
        h.update(p.read_bytes() if p.exists() else b'<none>')
    return h.hexdigest()[:12]


def add_deviation(text):
    """deviations.md に 1 行足す(protocol の外のことを自動で行ったとき)"""
    try:
        with open(DEVIATIONS, 'a', encoding='utf-8') as f:
            f.write('- %s(diverse_train.py が自動で記録)%s\n' % (now(), text))
    except OSError as e:
        log('deviations.md に書けませんでした: %s' % e)


def load_protocol():
    sha = sha256(PROTOCOL)
    want = PROTOCOL_SHA.read_text(encoding='utf-8').split()[0]
    if sha != want:
        raise SystemExit('protocol.json の sha256 %s が固定したもの %s と違います。止めます' % (sha, want))
    return json.load(open(PROTOCOL, encoding='utf-8')), sha


def split_key(key):
    arm, s = key.split('/')
    if arm not in ARM_CODE or not s.startswith('s'):
        raise SystemExit('腕・種の書き方が違います: %s(例 S-DIV/s0・C-DIV/s1)' % key)
    return arm, int(s[1:])


def run_name(arm, seed):
    return '%s%s-s%d' % (PREFIX, ARM_CODE[arm], seed)


def expected_run_name(proto, arm, seed):
    if arm in proto['arms_Q1'] and arm != 'A4':
        return proto['arms_Q1'][arm]['run_name'] if seed in proto['arms_Q1'][arm]['seeds'] else None
    if arm == 'A4':
        return proto['arms_Q1']['A4_DAgger']['run_name'] if seed == 0 else None
    names = proto['arms_Q2'][arm]['run_names']
    seeds = proto['arms_Q2'][arm]['seeds']
    return names[seeds.index(seed)] if seed in seeds else None


def data_of(arm):
    """腕の学習データのパス(REPO からの相対)。準備の段の確かめ(prepare_check.json)が合格で、そのあと変わっていないこと"""
    if arm == 'A4':
        plan = load(A4_PLAN) or {}
        p = plan.get('data')
        if not plan.get('ok') or not p or not (REPO / p / 'splits.json').exists():
            raise SystemExit('A4 のデータ(DAgger の段が書く train/a4_plan.json の data・ok)がありません')
        return p
    p = '%s/train/data/%s' % (EXP_REL, arm)
    chk = load(PREPARE_CHECK) or {}
    a = (chk.get('arms') or {}).get(arm) or {}
    if not chk.get('all_ok') or not a.get('ok'):
        raise SystemExit('%s の学習データの確かめ(prepare_check.json)が合格していません' % arm)
    for fn, want in (a.get('files_sha256') or {}).items():
        if sha256(REPO / p / fn) != want:
            raise SystemExit('%s の %s が確かめたあとで変わっています(prepare_check.py をやり直す)' % (arm, fn))
    if not a.get('files_sha256'):
        raise SystemExit('prepare_check.json に %s の files_sha256 がありません(prepare_check.py をやり直す)' % arm)
    return p


def command_of(proto, arm, seed, resume, workers, out=None):
    """protocol の作り方と同じ train.py の引数 → (引数のリスト, 学習結果の名前)"""
    name = run_name(arm, seed)
    want = expected_run_name(proto, arm, seed)
    if want != name:
        raise SystemExit('protocol.json の名前 %s と違います: %s(この腕・種は計画にありません?)' % (want, name))
    epochs, lr = ('10', '1e-4') if arm == 'A4' else ('30', '3e-4')
    args = ['tools/train.py', '--policy', 'jetformer_v0', '--data', data_of(arm), '--epochs', epochs, '--lr', lr,
            '--out', out or ('data/runs/' + name), '--seed', str(seed), '--stride', '3', '--batch', '64', '--workers', str(workers),
            '--no-command-input', '--impact-g', '2.5']
    if arm == 'A4':
        init = (load(A4_PLAN) or {}).get('init')
        if not init or not (REPO / init).exists():
            raise SystemExit('A4 の初期値(元の腕の best.pth)がありません: %s' % init)
        args += ['--init', init]
    if resume:
        args.append('--resume')
    return args, name


def arg_value(args, flag):
    return args[args.index(flag) + 1] if flag in args else None


def set_arg(args, flag, value):
    out = list(args)
    if flag in out:
        out[out.index(flag) + 1] = value
    else:
        out += [flag, value]
    return out


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
        if pid in me or ppid in me or 'diverse_train.py' in args or 'grep' in args:
            continue
        if ('python' in args or 'trtexec' in args) and any(p in args for p in GPU_PATTERNS):
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


def wait_gpu_free():
    said = None
    while True:
        busy = gpu_busy()
        if not busy:
            return mem_available_mb()
        why = 'GPU: %s' % busy
        if why != said:
            log('待ちます(%s)' % why)
            said = why
        time.sleep(30)


WATCH_MIN_AVAIL_MB = 200    # 見張り: 全体の空きメモリがこれを WATCH_SECONDS 秒続けて下回ったら、自分の学習を止める(10/14 と同じ)
WATCH_SECONDS = 8
LAST_WATCHDOG = {'killed': False}


def own_pids(pattern):
    out = subprocess.run(['ps', '-eo', 'pid,args'], capture_output=True, text=True).stdout.splitlines()[1:]
    return [int(l.split(None, 1)[0]) for l in out if pattern in l and 'diverse_train.py' not in l
            and len(l.split(None, 1)) > 1 and l.split(None, 1)[1].startswith(PY)]


def _watchdog(proc, pattern, stop):
    low = 0
    while not stop.is_set() and proc.poll() is None:
        low = low + 1 if mem_available_mb() < WATCH_MIN_AVAIL_MB else 0
        if low >= WATCH_SECONDS:
            pids = own_pids(pattern)
            log('見張り: 全体の空きメモリが %d MB 未満のまま %d 秒。自分の処理 %s を止めます' % (WATCH_MIN_AVAIL_MB, low, pids))
            for pid in pids:
                try:
                    os.kill(pid, 9)
                except OSError:
                    pass
            LAST_WATCHDOG['killed'] = True
            return
        stop.wait(1.0)


def wrap(cmd, mem, lock, need_mb, wrap_gpu=True):
    inner = [str(c) for c in cmd]
    if wrap_gpu:
        inner = ['choom', '-n', '1000', '--', PY, '-I', str(HERE / 'nothp.py')] + inner
    return [JF_HEAVY, '--mem', mem, '--lock', lock, '--need-mb', str(need_mb), '--', 'nice', '-n', '10'] + inner


def run_step(cmd, logfile, mem, lock='big', need_mb=MIN_FREE_MB, gpu=True, append=False, header=None, watch=None):
    """jf-heavy を通して 1 つのコマンドを動かす → (終了コード, 秒, 始めたときの空き MB)"""
    free = wait_gpu_free() if gpu else mem_available_mb()
    full = wrap(cmd, mem, lock, need_mb, wrap_gpu=gpu)
    t = time.time()
    log('%s(空き %d MB)' % (' '.join(full), free))
    LAST_WATCHDOG['killed'] = False
    with open(logfile, 'a' if append else 'w', encoding='utf-8') as f:
        if header:
            f.write(header + '\n')
            f.flush()
        proc = subprocess.Popen(full, cwd=str(REPO), stdout=f, stderr=subprocess.STDOUT,
                                env=dict(os.environ, PYTHONUNBUFFERED='1', **(GPU_ENV if gpu else {})))
        stop = threading.Event()
        th = None
        if watch:
            th = threading.Thread(target=_watchdog, args=(proc, watch, stop), daemon=True)
            th.start()
        code = proc.wait()
        stop.set()
        if th:
            th.join(5)
        if LAST_WATCHDOG['killed']:
            f.write('==== %s 見張りが止めました(全体の空きメモリ < %d MB が %d 秒)====\n' % (now(), WATCH_MIN_AVAIL_MB, WATCH_SECONDS))
    sec = round(time.time() - t, 1)
    log('  → 終了コード %d・%.0f 秒(%s)%s' % (code, sec, logfile, '・見張りが止めた' if LAST_WATCHDOG['killed'] else ''))
    return code, sec, free


def run_done(run, epochs):
    res = load(run / 'result.json')
    return bool(res and res.get('state') == 'done' and res.get('epochs_done') == epochs
                and (run / 'jetformer.onnx').exists() and (run / 'model_config.json').exists()
                and (run / 'best.pth').exists() and (run / 'last.pth').exists())


def last_epoch(run):
    """last.pth のエポック(無ければ 0、読めなければ −1 = 電源断で壊れた)"""
    p = run / 'last.pth'
    if not p.exists():
        return 0
    if p.stat().st_size == 0:
        return -1
    code = ('import torch,sys; ck=torch.load(sys.argv[1], map_location="cpu", weights_only=False); '
            'print(int(ck.get("epoch", 0)))')
    r = subprocess.run([PY, '-c', code, str(p)], capture_output=True, text=True,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
    try:
        return int(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return -1


def killed_by_memory(code, logfile):
    if code in (-9, 137):
        return True
    try:
        tail = Path(logfile).read_text(encoding='utf-8', errors='replace')[-6000:]
    except OSError:
        return False
    return 'killed by signal: Killed' in tail or 'Out of memory' in tail or 'MemoryError' in tail


def write_run_json(run, args, arm, seed, created_at, workers_used):
    sys.path.insert(0, str(REPO))
    from jetformer import policies      # torch を読まない
    meta = {'dataset': Path(arg_value(args, '--data')).name, 'stride': int(arg_value(args, '--stride')),
            'epochs': int(arg_value(args, '--epochs')), 'batch': int(arg_value(args, '--batch')),
            'lr': float(arg_value(args, '--lr')), 'pretrained': True, 'build_engine': False,
            'command_input': '--no-command-input' not in args, 'mask_impacts': True, 'policy': arg_value(args, '--policy'),
            'policy_cfg': policies.get(arg_value(args, '--policy')).resolve({}), 'impact_g': float(arg_value(args, '--impact-g')),
            'seed': seed, 'name': run.name, 'created_at': created_at, 'note': NOTE[arm],
            'experiment': 'sim_diverse_train_20261015', 'arm': arm, 'simulator_only': arm in Q1_ARMS,
            'workers_used': workers_used, 'train_env': GPU_ENV, 'data_path': arg_value(args, '--data')}
    if arm == 'A4':
        meta['init'] = arg_value(args, '--init')
    write_json(run / 'run.json', meta)


def save_state(state):
    state['updated'] = now()
    write_json(STATE_PATH, state)


def make_bundle(rs, key, name, run):
    """Q1 の腕のブリッジ用の一式(CPU)。アプリの写しが無ければ飛ばす → True(済み)/ False(後で)"""
    onnx = run / 'jetformer.onnx'
    bdir = BUNDLES / name
    bcheck = bdir / 'bundle_check.json'
    fresh = bcheck.exists() and bcheck.stat().st_mtime >= onnx.stat().st_mtime and (load(bcheck) or {}).get('ok')
    if not fresh:
        if not (SIM_COPY / 'bridge.py').exists() or not (SIM_COPY / 'dist').is_dir():
            log('%s: アプリの写し(%s)がまだありません。一式は後で `bundle %s` で作ります' % (key, SIM_COPY, key))
            rs['bundle'] = {'skipped': 'no app copy yet', 'at': now()}
            return False
        BUNDLES.mkdir(parents=True, exist_ok=True)
        code, sec, _ = run_step([PY, '-u', HERE / 'export_bundle.py', run, bdir], LOGS / ('%s.bundle.log' % name),
                                mem='1500M', lock='none', need_mb=1500, gpu=False)
        if code != 0:
            log('%s の一式の書き出しに失敗しました(終了コード %d)' % (key, code))
            rs['bundle'] = {'exit': code, 'at': now(), 'failed': 'export'}
            return False
        code, sec, _ = run_step([SIM_PY, '-I', '-B', HERE / 'check_bundle.py', bdir],
                                LOGS / ('%s.bundle.log' % name), mem='1500M', lock='none', need_mb=1000, gpu=False,
                                append=True, header='==== %s check_bundle.py(シミュレーターの Python)====' % now())
        rs['bundle'] = {'exit': code, 'at': now(), 'dir': str(bdir)}
        if code != 0 or not (load(bcheck) or {}).get('ok'):
            log('%s の一式の確認が通りませんでした(終了コード %d)' % (key, code))
            return False
    bc = load(bcheck)
    rs['bundle_check'] = {k: bc.get(k) for k in ('ok', 'zip', 'zip_sha256', 'onnx_vs_torch_max_us', 'defaults')}
    return True


def train_one(proto, state, key):
    arm, seed = split_key(key)
    runs_state = state.setdefault('runs', {})
    rs = runs_state.setdefault(key, {'arm': arm, 'seed': seed, 'attempts': [], 'workers': DEFAULT_WORKERS[arm]})
    rs.setdefault('workers', DEFAULT_WORKERS[arm])      # bundle の段で作った項目には workers が無い
    if 'workers' in OVERRIDE and not run_done(RUNS / run_name(arm, seed), 10 if arm == 'A4' else 30):
        if rs['workers'] != OVERRIDE['workers']:
            log('%s: workers を %d → %d にします(train の --workers)' % (key, rs['workers'], OVERRIDE['workers']))
        rs['workers'] = OVERRIDE['workers']
        if OVERRIDE['workers'] not in WORKERS_IN_PROTOCOL[arm]:
            add_deviation('%s: train の --workers %d で学習した(protocol は %s)' % (
                key, OVERRIDE['workers'], sorted(WORKERS_IN_PROTOCOL[arm], reverse=True)))
    args0, name = command_of(proto, arm, seed, resume=False, workers=rs['workers'])
    rs['run'] = name
    rs['data'] = arg_value(args0, '--data')
    run = REPO / arg_value(args0, '--out')
    epochs = int(arg_value(args0, '--epochs'))
    logfile = LOGS / ('%s.train.log' % name)
    # ---- ① 学習
    while not run_done(run, epochs):
        own = [a for a in rs['attempts'] if not a.get('watchdog_killed')]
        if len(own) >= MAX_ATTEMPTS or len(rs['attempts']) >= MAX_ATTEMPTS + MAX_WATCHDOG_RETRIES:
            log('%s: 学習を %d 回試しても終わりませんでした。止めます' % (key, len(rs['attempts'])))
            save_state(state)
            return False
        run.mkdir(parents=True, exist_ok=True)
        rs.setdefault('created_at', datetime.datetime.now().isoformat(timespec='seconds'))
        done_ep = last_epoch(run)
        if done_ep < 0:
            log('%s: last.pth が読めません(電源断で壊れた?)。退避して最初から' % key)
            os.replace(run / 'last.pth', run / ('last.pth.broken-%d' % int(time.time())))
            done_ep = 0
        resume = (run / 'last.pth').exists() and done_ep > 0
        workers = rs.get('workers', DEFAULT_WORKERS[arm])
        args, _ = command_of(proto, arm, seed, resume=resume, workers=workers)
        cmd = [PY, '-u'] + args
        header = '==== %s %s(workers %d・%s)====' % (
            now(), '再開: %d エポック目まで済み' % done_ep if resume else '開始', workers, ' '.join(args))
        att = {'started': now(), 'resume_from_epoch': done_ep if resume else 0, 'workers': workers,
               'code': code_fingerprint(), 'env': GPU_ENV}
        rs['attempts'].append(att)
        resume_capped = resume and any(a.get('resume_from_epoch') and a.get('exit') in (-9, 137) and not a.get('watchdog_killed')
                                       for a in rs['attempts'][:-1])
        mem_cap, need = ('3400M', 3500) if resume_capped else ('3G', MIN_FREE_MB)
        if OVERRIDE.get('mem_cap'):
            mem_cap, need = OVERRIDE['mem_cap'], max(MIN_FREE_MB, OVERRIDE['need_mb'])
            if mem_cap != '3G' and rs.get('mem_cap_override_logged') != mem_cap:
                add_deviation('%s: train の --mem-cap で、学習のメモリの上限を 3G → %s にした(workers %d で学習するため。'
                              '計算の結果は変わらない。全体の空きメモリの見張りと choom 1000 はそのまま)' % (key, mem_cap, workers))
                rs['mem_cap_override_logged'] = mem_cap
            resume_capped = False
        if resume_capped and not rs.get('mem_cap_deviation_logged'):
            add_deviation('%s: 続きからの学習がメモリの上限 3G で止まったので、上限を 3400M にした(10/14 の dart_train.py と同じ扱い。'
                          '計算の結果は変わらない)' % key)
            rs['mem_cap_deviation_logged'] = True
        att.update({'mem_cap': mem_cap, 'need_mb': need})
        save_state(state)
        code, sec, free = run_step(cmd, logfile, mem=mem_cap, need_mb=need, append=True, header=header,
                                   watch='--out %s ' % arg_value(args, '--out'))
        att.update({'exit': code, 'wall_s': sec, 'ended': now(), 'mem_available_mb': free,
                    'watchdog_killed': LAST_WATCHDOG['killed'], 'wrapped': 'choom 1000 + nothp'})
        if not run_done(run, epochs):
            steps = WORKERS_STEPS[arm]
            if LAST_WATCHDOG['killed']:
                log('%s: 全体の空きメモリが足りなくなったので止めました(自分の上限ではない)。空くのを待って続きから' % key)
                time.sleep(60)
            elif killed_by_memory(code, logfile) and workers in steps:
                rs['workers'] = steps[workers]
                att['memory_cap'] = True
                log('%s: メモリの上限で止まったようです。--workers %d で続きから' % (key, rs['workers']))
                if rs['workers'] not in WORKERS_IN_PROTOCOL[arm]:
                    add_deviation('%s: workers %d でもメモリの上限(3G)で止まったので workers %d にした(protocol は %s。'
                                  'workers は色の揺らしの乱数の流れだけを変え、計算の結果は変えない)'
                                  % (key, workers, rs['workers'], sorted(WORKERS_IN_PROTOCOL[arm], reverse=True)))
            elif code == 0:
                log('%s: 終了コード 0 なのに終わっていません(停止の指示?)。止めます' % key)
                save_state(state)
                return False
            else:
                log('%s の学習が終わっていません(終了コード %d)。もう一度続きから試します' % (key, code))
                time.sleep(30)
        save_state(state)
    res = load(run / 'result.json')
    hist = res['history']
    rs.update({'train_wall_s': round(sum(a.get('wall_s') or 0 for a in rs['attempts']), 1),
               'resumes': sum(1 for a in rs['attempts'] if a.get('resume_from_epoch')),
               'best_val_us': res.get('best_val_us'), 'best_epoch': res.get('best_epoch'),
               'last_val_us': hist[-1]['val_us'], 'peak_mem_mb': res.get('peak_mem_mb'),
               'train_samples': res.get('train_samples'), 'val_samples': res.get('val_samples'),
               'train_mode_counts': res.get('train_mode_counts'),
               'epoch1_s': hist[0]['seconds'], 'epoch_s_median': sorted(h['seconds'] for h in hist)[len(hist) // 2],
               'steer_center_us_trained': res.get('steer_center_us'),
               'throttle_neutral_us_trained': res.get('throttle_neutral_us'), 'train_done_at': rs.get('train_done_at') or now()})
    if not (run / 'train.log').exists() or (run / 'train.log').stat().st_size != logfile.stat().st_size:
        shutil.copyfile(logfile, run / 'train.log')
    if not (run / 'run.json').exists():
        write_run_json(run, args0, arm, seed, rs['created_at'], rs.get('workers', DEFAULT_WORKERS[arm]))
    save_state(state)
    # ---- ② ブリッジ用の一式(Q1 の腕だけ。CPU)
    if arm in Q1_ARMS:
        ok = make_bundle(rs, key, name, run)
        save_state(state)
        log('%s 済み: best %.1f µs(%d エポック目)・一式 %s' % (key, res['best_val_us'], res['best_epoch'],
                                                       'OK' if ok else '後で / NG(state.json の bundle)'))
    else:
        log('%s 済み: best %.1f µs(%d エポック目)。一式は作らない(実機の保留データの評価だけ)' % (
            key, res['best_val_us'], res['best_epoch']))
    rs['done'] = True
    save_state(state)
    return True


def parse_overrides(argv):
    """train の追加の引数(学習の段 2026-10-15 09:55 に足した。protocol の外なので、使うと deviations.md に書く):
      --workers N     この呼び出しの各腕の最初の workers(メモリの上限で止まれば WORKERS_STEPS のとおり下げる)
      --mem-cap M     jf-heavy の --mem(既定 3G)。--need-mb は M + 100 MB(3200 MB より小さくはしない)"""
    keys, ov = [], {}
    it = iter(argv)
    for a in it:
        if a == '--workers':
            ov['workers'] = int(next(it))
        elif a == '--mem-cap':
            ov['mem_cap'] = next(it)
            num = float(ov['mem_cap'][:-1]) * (1024 if ov['mem_cap'].upper().endswith('G') else 1)
            ov['need_mb'] = int(num) + 100
        else:
            keys.append(a)
    return keys, ov


OVERRIDE = {}


def cmd_train(state, argv):
    proto, sha = load_protocol()
    state['protocol_sha256'] = sha
    keys, ov = parse_overrides(argv)
    OVERRIDE.clear()
    OVERRIDE.update(ov)
    order = keys or DEFAULT_ORDER
    state.setdefault('orders', []).append({'at': now(), 'order': order, 'pid': os.getpid(), 'overrides': ov})
    for d in (LOGS, BUNDLES):
        d.mkdir(parents=True, exist_ok=True)
    log('順番: %s' % order)
    for key in order:
        wait_pause(key)
        if not train_one(proto, state, key):
            post_run(state, key)
            return 1
        post_run(state, key)
    save_state(state)
    log('学習: %s 済みました' % order)
    return 0


PAUSE = TRAIN / 'PAUSE'     # これがある間は、次の学習を始めない(学習の途中では止めない)。評価の段が lock big を使いたいときに置く


def wait_pause(key):
    said = False
    while PAUSE.exists():
        if not said:
            log('%s: %s があるので、次の学習を始めずに待ちます(消すと始めます)' % (key, PAUSE))
            said = True
        time.sleep(10)
    if said:
        log('%s: %s が消えたので始めます' % (key, PAUSE))


def http_check(name):
    """Q1 の腕の一式の ZIP を、ユーザーのブリッジの写しに HTTP で読み込ませる(http_load_check.py、口 8825、CPU)"""
    bdir = BUNDLES / name
    z = BUNDLES / ('%s.zip' % name)
    out = bdir / 'http_load_check.json'
    if not z.exists() or not (load(bdir / 'bundle_check.json') or {}).get('ok'):
        return None
    if out.exists() and out.stat().st_mtime >= z.stat().st_mtime and (load(out) or {}).get('ok') is not None:
        return (load(out) or {}).get('ok')
    tmp = bdir / 'http_load_check.json.tmp'
    cmd = wrap([SIM_PY, '-I', '-B', HERE / 'http_load_check.py', z], '1300M', 'none', 1000, wrap_gpu=False)
    log(' '.join(cmd))
    with open(tmp, 'w', encoding='utf-8') as fo, open(LOGS / ('%s.http_load.log' % name), 'w', encoding='utf-8') as fe:
        code = subprocess.run(cmd, cwd=str(REPO), stdout=fo, stderr=fe).returncode
    os.replace(tmp, out)
    ok = (load(out) or {}).get('ok')
    log('  → %s の HTTP の読み込み: 終了コード %d・ok %s' % (name, code, ok))
    return ok


def post_run(state, key):
    """1 本ごとの後片付け(CPU): Q1 の腕は HTTP の読み込みの確かめ、そのあと図と表と training.md(training_report.py)。
    失敗しても学習の順番は止めない"""
    try:
        arm, seed = split_key(key)
        rs = (state.get('runs') or {}).get(key) or {}
        if arm in Q1_ARMS and rs.get('done'):
            rs['http_load_ok'] = http_check(run_name(arm, seed))
            save_state(state)
        cmd = wrap([PY, '-I', HERE / 'training_report.py'], '900M', 'none', 900, wrap_gpu=False)
        with open(LOGS / 'training_report.log', 'a', encoding='utf-8') as f:
            f.write('==== %s(%s のあと)====\n' % (now(), key))
            f.flush()
            code = subprocess.run(cmd, cwd=str(REPO), stdout=f, stderr=subprocess.STDOUT).returncode
        log('%s のあと: training_report.py 終了コード %d' % (key, code))
    except Exception as e:      # noqa: BLE001  後片付けの失敗で学習の順番を止めない
        log('%s の後片付けで失敗しました: %r' % (key, e))


def cmd_post(state, keys):
    """後片付けだけ(HTTP の読み込みの確かめ・図と表と training.md)"""
    load_protocol()
    for key in keys or [k for k in TODAY_KEYS if k in (state.get('runs') or {})]:
        post_run(state, key)
    return 0


TODAY_KEYS = ['S-DIV/s0', 'C-DIV/s0', 'C-DIV/s1', 'S-MOD/s0', 'A4/s0']


def cmd_bundle(state, keys):
    load_protocol()
    for key in keys:
        arm, seed = split_key(key)
        if arm not in Q1_ARMS:
            log('%s: 一式は Q1 の腕だけ' % key)
            continue
        rs = state.setdefault('runs', {}).setdefault(key, {'arm': arm, 'seed': seed, 'attempts': []})
        name = run_name(arm, seed)
        run = RUNS / name
        if not (run / 'jetformer.onnx').exists():
            log('%s: 学習結果がまだありません' % key)
            continue
        ok = make_bundle(rs, key, name, run)
        save_state(state)
        log('%s の一式: %s' % (key, 'OK' if ok else 'NG / 後で'))
    return 0


def cmd_dryrun(state, keys):
    proto, _ = load_protocol()
    for key in keys or DEFAULT_ORDER:
        arm, seed = split_key(key)
        rs = (state.get('runs') or {}).get(key) or {}
        workers = rs.get('workers', DEFAULT_WORKERS[arm])
        args, name = command_of(proto, arm, seed, resume=False, workers=workers)
        print(' '.join(wrap([PY, '-u'] + args, '3G', 'big', MIN_FREE_MB)))
    return 0


def cmd_status(state, _keys):
    print(json.dumps(state, ensure_ascii=False, indent=1))
    return 0


def cmd_smoke(state, argv):
    """CPU の動作確認: 腕の本番と同じコマンドを、エポック 2・1 エポック 2 ステップ・バッチ 8・workers 0・CPU で動かす。
    --kill-after-epoch1: 1 エポック目の last.pth が出来たら kill -9(電源断のまね)→ 同じコマンドに --resume で続きを確かめる。
    出力は REPORT/smoke/runs/<名前>(data/runs に作らない)。終わったら結果を REPORT/smoke/<腕>.json に書いて、出力のフォルダーは消す"""
    proto, _ = load_protocol()
    arm = argv[0]
    kill = '--kill-after-epoch1' in argv
    mem = next((a.split('=', 1)[1] for a in argv if a.startswith('--mem=')), '1500M')
    seed = 0
    name = run_name(arm, seed) if arm != 'A4' else None
    out = SMOKE / 'runs' / name
    if out.exists():
        shutil.rmtree(out)           # 前の試みの残り
    out.mkdir(parents=True)
    (SMOKE / 'logs').mkdir(parents=True, exist_ok=True)
    args, _ = command_of(proto, arm, seed, resume=False, workers=DEFAULT_WORKERS[arm], out=str(out))
    real_args = list(args)
    args = set_arg(set_arg(set_arg(args, '--epochs', '2'), '--batch', '8'), '--workers', '0')
    args += ['--max-steps', '2']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONUNBUFFERED='1', **GPU_ENV)
    logfile = SMOKE / 'logs' / ('%s.smoke.log' % arm)
    res = {'arm': arm, 'started': now(), 'real_command': ' '.join(wrap([PY, '-u'] + real_args, '3G', 'big', MIN_FREE_MB)),
           'smoke_overrides': {'--epochs': 2, '--batch': 8, '--workers': 0, '--max-steps': 2, '--out': str(out),
                               'CUDA_VISIBLE_DEVICES': '', 'jf-heavy': '--mem %s --lock none' % mem},
           'kill_after_epoch1': kill, 'phases': []}

    def phase(extra, kill_now):
        cmd = wrap([PY, '-u'] + args + extra, mem, 'none', 1200)
        t = time.time()
        with open(logfile, 'a', encoding='utf-8') as f:
            f.write('==== %s %s ====\n' % (now(), ' '.join(cmd)))
            f.flush()
            start = f.tell()
            proc = subprocess.Popen(cmd, cwd=str(REPO), stdout=f, stderr=subprocess.STDOUT, env=env)
            killed = False
            while proc.poll() is None:
                time.sleep(0.2)
                if kill_now and not killed:
                    txt = logfile.read_text(encoding='utf-8', errors='replace')[start:]
                    if 'epoch   1 ' in txt:
                        pids = own_pids('--out %s ' % out)
                        for pid in pids:
                            os.kill(pid, signal.SIGKILL)
                        killed = True
                        f.write('==== %s 電源断のまね: kill -9 %s ====\n' % (now(), pids))
                        f.flush()
            code = proc.wait()
        txt = logfile.read_text(encoding='utf-8', errors='replace')[start:]
        return {'cmd_extra': extra, 'exit': code, 'seconds': round(time.time() - t, 1), 'killed': killed,
                'last_epoch_after': last_epoch(out),
                'lines': [l for l in txt.splitlines() if l.startswith(('学習 ', '衝突の前後', 'ジャンル別', 'epoch ', '再開', '保存',
                                                                        'Traceback', 'RuntimeError', 'SystemExit'))
                          or 'Error' in l][:20]}

    logfile.write_text('', encoding='utf-8')
    if kill:
        p1 = phase([], True)
        res['phases'].append(p1)
        p2 = phase(['--resume'], False)
        res['phases'].append(p2)
    else:
        res['phases'].append(phase([], False))
    r = load(out / 'result.json') or {}
    cfg = load(out / 'model_config.json') or {}
    res['result'] = {k: r.get(k) for k in ('state', 'epochs_done', 'train_samples', 'val_samples', 'best_val_us', 'best_epoch',
                                           'train_mode_counts', 'impact_masked_frames', 'steer_center_us',
                                           'throttle_neutral_us', 'dataset', 'seed')}
    res['history'] = r.get('history')
    res['model_config'] = {k: cfg.get(k) for k in ('num_modes', 'trained_modes', 'dataset', 'policy', 'command_input', 'inputs')}
    res['onnx_bytes'] = (out / 'jetformer.onnx').stat().st_size if (out / 'jetformer.onnx').exists() else None
    ok = (r.get('state') == 'done' and r.get('epochs_done') == 2 and cfg.get('num_modes') == 9 and res['onnx_bytes'])
    if kill:
        ok = ok and res['phases'][0]['killed'] and res['phases'][0]['last_epoch_after'] == 1 and any(
            l.startswith('再開: 1 エポック目まで済み') for l in res['phases'][1]['lines'])
    res['ok'] = bool(ok)
    res['ended'] = now()
    write_json(SMOKE / ('%s.json' % arm), res)
    shutil.rmtree(out)               # 動作確認の出力は残さない(ディスク)。結果は <腕>.json とログ
    log('smoke %s: %s' % (arm, 'OK' if res['ok'] else 'NG'))
    return 0 if res['ok'] else 1


def main(argv):
    cmds = {'train': cmd_train, 'status': cmd_status, 'bundle': cmd_bundle, 'dryrun': cmd_dryrun, 'smoke': cmd_smoke,
            'post': cmd_post}
    if len(argv) < 2 or argv[1] not in cmds:
        raise SystemExit(__doc__)
    os.chdir(REPO)
    TRAIN.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    if argv[1] == 'smoke':
        return cmd_smoke({}, argv[2:])
    state = load(STATE_PATH, {})
    return cmds[argv[1]](state, argv[2:])


if __name__ == '__main__':
    sys.exit(main(sys.argv))
