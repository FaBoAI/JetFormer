#!/usr/bin/env python3
"""sim_dart_20261014 の学習の段: protocol.json の arms(A1・A2、削らなければ A3)を 1 本ずつ学習し、
シミュレーターのブリッジ用の一式(bundle)を作る(GPU は 1 プロセスずつ・続きから再開できる)

    cd ~/jetformer && nohup /home/jetson/miniconda/envs/jetracer/bin/python -u \
        experiments/20261014_sim_dart_margin/train/dart_train.py train [A1/s0 A2/s0 ...] > ... &
    cd ~/jetformer && python experiments/20261014_sim_dart_margin/train/dart_train.py status

各 1 本ずつ:
  ① tools/train.py(protocol の arms.common_recipe のコマンド。last.pth があれば --resume)
     jf-heavy --mem 3G --lock big(ベストラップの学習と同じメモリの工夫: expandable_segments・THP なし・workers 2 → 1)
  ② run.json(管理画面の一覧。「シミュレーター専用(ジャンル 8)。実車では使わない」と書く)
  ③ export_bundle.py(CPU): ブリッジ用の一式(ONNX・直した model_config.json・io_spec.json・calibration.json・ZIP)と
     PyTorch の参照の出力 → check_bundle.py(シミュレーターの Python で、ユーザーのアダプターの検査と ONNX の一致)
  FP16 のエンジンは作らない(protocol の評価はブリッジの onnxruntime(CPU)で ONNX を動かす。エンジンは要らない)。
出来たもの(result.json の state=done・一式の確認)は飛ばす。電源断のあとは同じコマンドで続きから。

ベストラップの bestlap_train.py の写しを、この作業の腕・名前(simdart_20261014-<腕>-s<種>)・パスに合わせて直したもの。
protocol.json の sha256 が protocol.sha256 と違えば止める。状態は train/state.json(tmp → 置き換え)。車は動かさない・PCA9685 に触らない。
"""

import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

REPO = Path('/home/jetson/jetformer')
HERE = Path(__file__).resolve().parent
REPORT = Path('/home/jetson/jetformer_reports/sim_dart_20261014')
TRAIN = REPORT / 'train'
LOGS = TRAIN / 'logs'
PROTOCOL = REPORT / 'protocol.json'
PROTOCOL_SHA = REPORT / 'protocol.sha256'
READY = REPORT / 'recording' / 'READY.json'
STATE_PATH = TRAIN / 'state.json'
BUNDLES = REPORT / 'bundles'
RUNS = REPO / 'data' / 'runs'
JF_HEAVY = '/home/jetson/bin/jf-heavy'
SIM_PY = '/home/jetson/jetformer/data/sim/venv/bin/python'     # 読むだけ(onnxruntime 1.31.0・numpy 2.2.6・opencv 4.13.0)
PY = '/home/jetson/miniconda/envs/jetracer/bin/python'        # ベストラップの学習と同じ Python
PREFIX = 'simdart_20261014-'
DEFAULT_ORDER = ['A1/s0', 'A2/s0']
MIN_FREE_MB = 3200          # 学習を始める空きメモリ(ベストラップと同じ)
GPU_ENV = {'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True', 'MALLOC_ARENA_MAX': '2'}
DEFAULT_WORKERS = 2         # ベストラップの S-R 種 0 と同じ。メモリの上限で止まったら 1
MAX_ATTEMPTS = 6
MAX_WATCHDOG_RETRIES = 40
WORKERS_STEPS = {2: 1}
GPU_PATTERNS = ('tools/train.py', 'build_engine.py', 'policy_zoo.py', 'benchmark_model.py', 'engine_check.py',
                'engine_load_check.py', 'rl_policies.py', 'rl_evaluate.py', 'compare_models.py', 'tools/evaluate.py',
                'offline_rl.py', 'run_smoke.py', 'zoo_train.py', 'sim_cotrain_eval', 'trtexec', 'offline_eval')
CODE_FILES = ['tools/train.py', 'jetformer/model.py', 'jetformer/train_data.py', 'jetformer/policies/__init__.py',
              'jetformer/policies/runtime.py', 'jetformer/policies/blocks.py', 'jetformer/policies/jetformer_v0.py']
NOTE = {
    'A1': 'DART の実験(sim_dart_20261014)の A1: RL の最速の先生のきれいなデータ 20 分(A0)+ 先生の操作をわざと乱して線から外し、'
          '正解を先生の操作に付け替えたデータ 10 分(D1)。シミュレーター専用(ジャンル 8)。実車では使わない',
    'A2': 'DART の実験(sim_dart_20261014)の A2: 余裕を持たせた線(経路いろいろ c0 = +0.3)の RL の先生のきれいなデータ 10 分(D2)だけ。'
          'シミュレーター専用(ジャンル 8)。実車では使わない',
    'A3': 'DART の実験(sim_dart_20261014)の A3: 余裕を持たせた線のデータ(D2)+ その DART(D3)。シミュレーター専用(ジャンル 8)。実車では使わない',
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


def load_protocol():
    sha = sha256(PROTOCOL)
    want = PROTOCOL_SHA.read_text(encoding='utf-8').split()[0]
    if sha != want:
        raise SystemExit('protocol.json の sha256 %s が固定したもの %s と違います。止めます' % (sha, want))
    return json.load(open(PROTOCOL, encoding='utf-8')), sha


def data_of(arm):
    """腕の学習データのパス(REPO からの相対)。A1 は合わせたフォルダー(make_combined.py で作り、count_samples.py で確かめたもの)"""
    ready = load(READY) or {}
    if arm == 'A1':
        p = 'experiments/20261014_sim_dart_margin/train/data/A1'
        chk = load(TRAIN / 'count_A1.json') or {}
        if not chk.get('ok'):
            raise SystemExit('A1 の合わせたフォルダーのサンプル数の確かめ(train/count_A1.json)が合格していません')
        return p
    if arm == 'A2':
        name = (ready.get('D2') or {}).get('name')
        if not name or not (REPO / 'data' / 'datasets' / name / 'splits.json').exists():
            raise SystemExit('READY.json の D2 のデータセットがありません')
        return 'data/datasets/' + name
    if arm == 'A3':
        p = 'experiments/20261014_sim_dart_margin/train/data/A3'
        if not (REPO / p / 'splits.json').exists():
            raise SystemExit('A3 のデータ(D2 + D3)がありません(A3 は削った)')
        return p
    raise SystemExit('知らない腕 %s' % arm)


def run_name(arm, seed):
    return '%s%s-s%d' % (PREFIX, arm, seed)


def command_of(proto, arm, seed, resume, workers):
    """protocol の arms.common_recipe と同じ train.py の引数 → (引数のリスト, 学習結果の名前)"""
    name = run_name(arm, seed)
    want = proto['arms'][arm]['run_name'] if seed == 0 else None
    if want is not None and want != name:
        raise SystemExit('protocol.json の run_name %s と違います: %s' % (want, name))
    args = ['tools/train.py', '--policy', 'jetformer_v0', '--data', data_of(arm), '--epochs', '30', '--lr', '3e-4',
            '--out', 'data/runs/' + name, '--seed', str(seed), '--stride', '3', '--batch', '64', '--workers', str(workers),
            '--no-command-input', '--impact-g', '2.5']
    if resume:
        args.append('--resume')
    return args, name


def arg_value(args, flag):
    return args[args.index(flag) + 1] if flag in args else None


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
        if pid in me or ppid in me or 'dart_train.py' in args or 'grep' in args:
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


WATCH_MIN_AVAIL_MB = 200    # 見張り: 全体の空きメモリがこれを WATCH_SECONDS 秒続けて下回ったら、自分の GPU の処理を止める(ベストラップと同じ)
WATCH_SECONDS = 8
LAST_WATCHDOG = {'killed': False}


def _watchdog(proc, pattern, stop):
    low = 0
    while not stop.is_set() and proc.poll() is None:
        low = low + 1 if mem_available_mb() < WATCH_MIN_AVAIL_MB else 0
        if low >= WATCH_SECONDS:
            out = subprocess.run(['ps', '-eo', 'pid,args'], capture_output=True, text=True).stdout.splitlines()[1:]
            pids = [int(l.split(None, 1)[0]) for l in out if pattern in l and 'dart_train.py' not in l
                    and l.split(None, 1)[1].startswith(PY)]
            log('見張り: 全体の空きメモリが %d MB 未満のまま %d 秒。自分の処理 %s を止めます' % (WATCH_MIN_AVAIL_MB, low, pids))
            for pid in pids:
                try:
                    os.kill(pid, 9)
                except OSError:
                    pass
            LAST_WATCHDOG['killed'] = True
            return
        stop.wait(1.0)


def run_step(cmd, logfile, mem, lock='big', need_mb=MIN_FREE_MB, gpu=True, append=False, header=None, watch=None):
    """jf-heavy を通して 1 つのコマンドを動かす → (終了コード, 秒, 始めたときの空き MB)"""
    free = wait_gpu_free() if gpu else mem_available_mb()
    inner = [str(c) for c in cmd]
    if gpu:
        inner = ['choom', '-n', '1000', '--', PY, '-I', str(HERE / 'nothp.py')] + inner
    full = [JF_HEAVY, '--mem', mem, '--lock', lock, '--need-mb', str(need_mb), '--', 'nice', '-n', '10'] + inner
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
            'experiment': 'sim_dart_20261014', 'arm': arm, 'simulator_only': True, 'workers_used': workers_used,
            'train_env': GPU_ENV, 'data_path': arg_value(args, '--data')}
    write_json(run / 'run.json', meta)


def save_state(state):
    state['updated'] = now()
    write_json(STATE_PATH, state)


def train_one(proto, state, key):
    arm, s = key.split('/')
    seed = int(s[1:])
    runs_state = state.setdefault('runs', {})
    rs = runs_state.setdefault(key, {'arm': arm, 'seed': seed, 'attempts': [], 'workers': DEFAULT_WORKERS})
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
            return False
        run.mkdir(parents=True, exist_ok=True)
        rs.setdefault('created_at', datetime.datetime.now().isoformat(timespec='seconds'))
        done_ep = last_epoch(run)
        if done_ep < 0:
            log('%s: last.pth が読めません(電源断で壊れた?)。退避して最初から' % key)
            os.replace(run / 'last.pth', run / ('last.pth.broken-%d' % int(time.time())))
            done_ep = 0
        resume = (run / 'last.pth').exists() and done_ep > 0
        workers = rs.get('workers', DEFAULT_WORKERS)
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
        att.update({'mem_cap': mem_cap, 'need_mb': need})
        save_state(state)
        code, sec, free = run_step(cmd, logfile, mem=mem_cap, need_mb=need, append=True, header=header,
                                   watch='--out data/runs/%s ' % name)
        att.update({'exit': code, 'wall_s': sec, 'ended': now(), 'mem_available_mb': free,
                    'watchdog_killed': LAST_WATCHDOG['killed'], 'wrapped': 'choom 1000 + nothp'})
        if not run_done(run, epochs):
            if LAST_WATCHDOG['killed']:
                log('%s: 全体の空きメモリが足りなくなったので止めました(自分の上限ではない)。空くのを待って続きから' % key)
                time.sleep(60)
            elif killed_by_memory(code, logfile) and workers in WORKERS_STEPS:
                rs['workers'] = WORKERS_STEPS[workers]
                att['memory_cap'] = True
                log('%s: メモリの上限で止まったようです。--workers %d で続きから' % (key, rs['workers']))
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
               'epoch1_s': hist[0]['seconds'], 'epoch_s_median': sorted(h['seconds'] for h in hist)[len(hist) // 2],
               'steer_center_us_trained': res.get('steer_center_us'),
               'throttle_neutral_us_trained': res.get('throttle_neutral_us'), 'train_done_at': rs.get('train_done_at') or now()})
    if not (run / 'train.log').exists() or (run / 'train.log').stat().st_size != logfile.stat().st_size:
        shutil.copyfile(logfile, run / 'train.log')
    if not (run / 'run.json').exists():
        write_run_json(run, args0, arm, seed, rs['created_at'], rs.get('workers', DEFAULT_WORKERS))
    save_state(state)
    # ---- ② ブリッジ用の一式(CPU)と、シミュレーターの Python での確認
    onnx = run / 'jetformer.onnx'
    bdir = BUNDLES / name
    bcheck = bdir / 'bundle_check.json'
    fresh = bcheck.exists() and bcheck.stat().st_mtime >= onnx.stat().st_mtime and (load(bcheck) or {}).get('ok')
    if not fresh:
        if not (HERE / 'export_bundle.py').exists() or not (HERE / 'check_bundle.py').exists():
            log('%s: 一式の道具がまだありません。一式は後で作ります' % key)
            rs['bundle'] = {'skipped': 'no tool yet', 'at': now()}
            save_state(state)
            return True
        BUNDLES.mkdir(parents=True, exist_ok=True)
        code, sec, _ = run_step([PY, '-u', HERE / 'export_bundle.py', run, bdir], LOGS / ('%s.bundle.log' % name),
                                mem='1500M', lock='none', need_mb=1500, gpu=False)
        if code != 0:
            log('%s の一式の書き出しに失敗しました(終了コード %d)' % (key, code))
            rs['bundle'] = {'exit': code, 'at': now(), 'failed': 'export'}
            save_state(state)
            return True         # 学習は済んだので次の腕へ(一式は後で直す)
        code, sec, _ = run_step([SIM_PY, '-I', '-B', HERE / 'check_bundle.py', bdir],
                                LOGS / ('%s.bundle.log' % name), mem='1500M', lock='none', need_mb=1000, gpu=False,
                                append=True, header='==== %s check_bundle.py(シミュレーターの Python)====' % now())
        rs['bundle'] = {'exit': code, 'at': now(), 'dir': str(bdir)}
        save_state(state)
        if code != 0 or not (load(bcheck) or {}).get('ok'):
            log('%s の一式の確認が通りませんでした(終了コード %d)' % (key, code))
            return True
    bc = load(bcheck)
    rs['bundle_check'] = {k: bc.get(k) for k in ('ok', 'zip', 'zip_sha256', 'onnx_vs_torch_max_us', 'defaults')}
    rs['done'] = True
    save_state(state)
    log('%s 済み: best %.1f µs(%d エポック目)・一式 %s' % (key, res['best_val_us'], res['best_epoch'], 'OK' if bc.get('ok') else 'NG'))
    return True


def cmd_train(state, keys):
    proto, sha = load_protocol()
    state['protocol_sha256'] = sha
    state['ready'] = load(READY)
    order = keys or DEFAULT_ORDER
    state['order'] = order
    for d in (LOGS, BUNDLES):
        d.mkdir(parents=True, exist_ok=True)
    log('順番: %s' % order)
    for key in order:
        if not train_one(proto, state, key):
            return 1
    state['all_done'] = now()
    save_state(state)
    log('学習・一式: 全部済みました')
    return 0


def cmd_status(state, _keys):
    print(json.dumps(state, ensure_ascii=False, indent=1))
    return 0


def main(argv):
    if len(argv) < 2 or argv[1] not in ('train', 'status'):
        raise SystemExit(__doc__)
    os.chdir(REPO)
    state = load(STATE_PATH, {})
    return {'train': cmd_train, 'status': cmd_status}[argv[1]](state, argv[2:])


if __name__ == '__main__':
    sys.exit(main(sys.argv))
