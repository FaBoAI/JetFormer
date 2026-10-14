#!/usr/bin/env python3
"""RL ベストラップの段 D(JetFormer の学習): protocol.json の training_arms を 1 本ずつ学習し、FP16 のエンジンと
シミュレーターのブリッジ用の一式(bundle)を作る(GPU は 1 プロセスずつ・続きから再開できる)

    cd ~/jetformer && nohup python -u experiments/20261013_sim_rl_bestlap/train/bestlap_train.py train > ... &
    cd ~/jetformer && python experiments/20261013_sim_rl_bestlap/train/bestlap_train.py status

小コマンド:
    train   ORDER の順(S-R 種 0 → S-T 種 0 → S-R 種 1 → S-T 種 1。protocol の優先 1 → 2)に、各 1 本ずつ
            ① tools/train.py(protocol.json の train のコマンド。last.pth があれば resume のコマンド)
               jf-heavy --mem 3G --lock big(メモリの上限で止まったら同じ種を --workers 2 で続きから。protocol の oom_cap_fallback)
            ② tools/build_engine.py(FP16。jf-heavy --mem 2500M --lock big)
            ③ engine_load_check.py(走行タブと同じ手順でエンジンを読む・実フレーム 64 個で FP32 との差・速さ。同じ lock)
            ④ run.json(管理画面の一覧。「シミュレーター専用(ジャンル 8)。実車では使わない」と書く)と engine_check.json
            ⑤ export_bundle.py(CPU): ブリッジ用の一式(ONNX・直した model_config.json・io_spec.json・calibration.json・ZIP)と
               PyTorch の参照の出力 → check_bundle.py(シミュレーターの Python で、ユーザーのアダプターの検査と ONNX の一致)
            出来たもの(result.json の state=done・エンジン・確認の json・一式の確認)は飛ばす。電源断のあとは同じコマンドで続きから。
    status  今の状態を表示する(GPU なし)

学習結果の名前は、この段の指示の形 simrl_bestlap_20261013-<腕>-s<種>(protocol.json の run は simrl-bestlap-<腕>-s<種>。
deviations.md に記録)。コマンドの中身(データ・エポック・lr・間引き・バッチ・種・今の操作値なし・衝突 2.5G)は protocol のまま。
GPU を使う手順の前に毎回、ほかに GPU を使うプロセスが無いこと、管理画面が走行中・学習中でないことを確かめる(足りなければ待つ)。
protocol.json の sha256 が protocol.sha256 と違えば止める。状態は state.json(tmp → 置き換え)。車は動かさない・PCA9685 に触らない。
"""

import datetime
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path('/home/jetson/jetformer')
HERE = Path(__file__).resolve().parent
REPORT = Path('/home/jetson/jetformer_reports/sim_rl_bestlap_20261013')
TRAIN = REPORT / 'train'
LOGS = TRAIN / 'logs'
PROTOCOL = REPORT / 'protocol.json'
PROTOCOL_SHA = REPORT / 'protocol.sha256'
RL_DATASET_JSON = REPORT / 'rl_dataset.json'
STATE_PATH = TRAIN / 'state.json'
ENGINE_CHECK_DIR = TRAIN / 'engine_check'
BUNDLES = REPORT / 'bundles'
RUNS = REPO / 'data' / 'runs'
JF_HEAVY = '/home/jetson/bin/jf-heavy'
SIM_PY = '/home/jetson/jetformer/data/sim/venv/bin/python'     # 読むだけ(onnxruntime 1.31.0・numpy 2.2.6・opencv 4.13.0)
PY = sys.executable
PREFIX = 'simrl_bestlap_20261013-'
ARM_TAG = {'S-T': 'ST', 'S-R': 'SR', 'S-TR': 'STR'}
ORDER = [('S-R', 0), ('S-T', 0), ('S-R', 1), ('S-T', 1)]
PY_MARK = 'python -u '
MIN_FREE_MB = 3200          # 学習を始める空きメモリ(protocol は 3.5 GB。測った学習の固い分 約 2.8 GB・上限 3G に合わせて下げた。deviations.md)
ENGINE_FREE_MB = 2800       # エンジンの作成・確認を始める空きメモリ(上限 2500M)
# GPU の処理に渡す環境変数(数値は変わらない。CUDA のメモリの取り方と glibc の arena の数だけ)。
# 2026-10-14 03:24〜03:45: 上限 3G では workers 4・2・1 のどれもメモリの上限で止まった。memory.stat を測ると CUDA の確保(nvmap)が
# kernel として 1.8 GB 数えられていた。expandable_segments にすると 1.2 GB になり、workers 2 で上限 3G の中で 1 エポック・ONNX の書き出しまで通った
GPU_ENV = {'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True', 'MALLOC_ARENA_MAX': '2'}
DEFAULT_WORKERS = 2         # protocol の oom_cap_fallback の値。全部の腕で同じにする(色の揺らぎの乱数が workers の数で変わるため)
GATE_US = 2.5
MAX_ATTEMPTS = 6
MAX_WATCHDOG_RETRIES = 40   # 見張りで止めた回(別の作業とメモリを取り合った回)は MAX_ATTEMPTS に数えず、別にこの回数まで
WORKERS_STEPS = {4: 2, 2: 1}      # メモリの上限で止まったときの workers の下げ方(protocol の予備策は 2。2 でも止まれば 1)
GPU_PATTERNS = ('tools/train.py', 'build_engine.py', 'policy_zoo.py', 'benchmark_model.py', 'engine_check.py',
                'engine_load_check.py', 'rl_policies.py', 'rl_evaluate.py', 'compare_models.py', 'tools/evaluate.py',
                'offline_rl.py', 'run_smoke.py', 'zoo_train.py', 'sim_cotrain_eval', 'trtexec', 'offline_eval')
CODE_FILES = ['tools/train.py', 'tools/build_engine.py', 'jetformer/model.py', 'jetformer/train_data.py',
              'jetformer/trt_runtime.py', 'jetformer/drive.py', 'jetformer/policies/__init__.py',
              'jetformer/policies/runtime.py', 'jetformer/policies/blocks.py', 'jetformer/policies/jetformer_v0.py']
NOTE = {
    'S-T': 'RL ベストラップの実験(sim_rl_bestlap_20261013)の S-T: 教師(作者の RC 教師)のシミュレーターデータ 20 分だけで学習。'
           'シミュレーター専用(ジャンル 8)。実車では使わない',
    'S-R': 'RL ベストラップの実験(sim_rl_bestlap_20261013)の S-R: 強化学習(PPO)のベストラップのシミュレーターデータ 20 分だけで学習。'
           'シミュレーター専用(ジャンル 8)。実車では使わない',
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


def rl_dataset_name():
    d = load(RL_DATASET_JSON)
    if not d or not d.get('name') or not (REPO / 'data' / 'datasets' / d['name'] / 'splits.json').exists():
        raise SystemExit('rl_dataset.json の RL のデータセットがありません(記録・取り込みの段が先)')
    return d['name']


def run_name(arm, seed):
    return '%s%s-s%d' % (PREFIX, ARM_TAG[arm], seed)


def command_of(proto, arm, seed, resume, workers=None):
    """protocol.json のコマンド → (train.py の引数のリスト, 学習結果の名前)。--out だけこの段の名前に、
    --workers は oom の予備策のときだけ変える"""
    c = proto['training_arms'][arm]['commands'][str(seed)]
    if c['run'] != 'simrl-bestlap-%s-s%d' % (ARM_TAG[arm], seed):
        raise SystemExit('protocol.json の run の形が想定と違います: %s' % c['run'])
    text = c['resume' if resume else 'train'].replace('<RL_DATASET>', rl_dataset_name())
    if PY_MARK not in text or '<' in text:
        raise SystemExit('知らない形のコマンド: %s' % text)
    args = shlex.split(text.split(PY_MARK, 1)[1])
    name = run_name(arm, seed)
    i = args.index('--out')
    if args[i + 1] != 'data/runs/' + c['run']:
        raise SystemExit('protocol.json の run と --out が違います: %s / %s' % (c['run'], args[i + 1]))
    args[i + 1] = 'data/runs/' + name
    if workers is not None:
        args[args.index('--workers') + 1] = str(workers)
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
        if pid in me or ppid in me or 'bestlap_train.py' in args or 'grep' in args:
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
    """ほかの GPU の処理が無くなるまで待つ(30 秒ごと)。メモリは jf-heavy の --need-mb が待つ"""
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


WATCH_MIN_AVAIL_MB = 200   # 見張り: 全体の空きメモリがこれを WATCH_SECONDS 秒続けて下回ったら、自分の GPU の処理を止める(機械が固まるのを防ぐ)
WATCH_SECONDS = 8           # 05:19・05:46・05:56 に 250 MB・3 秒で止めた。空きは 220〜400 MB を行き来していたので 200 MB・8 秒にした
LAST_WATCHDOG = {'killed': False}


def _watchdog(proc, pattern, stop):
    """GPU の処理の間、全体の空きメモリを見張る。危なければ pattern を含む自分の処理(jetracer の python)を SIGKILL"""
    low = 0
    while not stop.is_set() and proc.poll() is None:
        if mem_available_mb() < WATCH_MIN_AVAIL_MB:
            low += 1
        else:
            low = 0
        if low >= WATCH_SECONDS:
            out = subprocess.run(['ps', '-eo', 'pid,args'], capture_output=True, text=True).stdout.splitlines()[1:]
            pids = [int(l.split(None, 1)[0]) for l in out if pattern in l and 'bestlap_train.py' not in l
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
    """jf-heavy を通して 1 つのコマンドを動かす → (終了コード, 秒, 始めたときの空き MB)。
    GPU の処理は choom 1000(全体のメモリが尽きたら最初に止められる側)+ nothp.py(THP を使わない)で包み、
    watch(止める処理を見分ける文字列)があれば全体の空きメモリを見張る"""
    import threading
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
    """last.pth の済んだエポック(無ければ 0、読めなければ -1)。この親プロセスで torch を読まない(メモリを学習に残す)"""
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
    """メモリの上限(jf-heavy の MemoryMax)で止まったらしいか: SIGKILL の終了コード、または DataLoader の worker が Killed"""
    if code in (-9, 137):
        return True
    try:
        tail = Path(logfile).read_text(encoding='utf-8', errors='replace')[-6000:]
    except OSError:
        return False
    return 'killed by signal: Killed' in tail or 'Out of memory' in tail or 'MemoryError' in tail


def write_run_json(run, args, arm, seed, created_at, workers_used):
    """管理画面の学習タブ・走行タブの一覧に出すための run.json(管理画面の学習と同じ欄 + 実験の印)"""
    sys.path.insert(0, str(REPO))
    from jetformer import policies      # torch を読まない
    meta = {'dataset': Path(arg_value(args, '--data')).name, 'stride': int(arg_value(args, '--stride')),
            'epochs': int(arg_value(args, '--epochs')), 'batch': int(arg_value(args, '--batch')),
            'lr': float(arg_value(args, '--lr')), 'pretrained': True, 'build_engine': True,
            'command_input': '--no-command-input' not in args, 'mask_impacts': True, 'policy': arg_value(args, '--policy'),
            'policy_cfg': policies.get(arg_value(args, '--policy')).resolve({}), 'impact_g': float(arg_value(args, '--impact-g')),
            'seed': seed, 'name': run.name, 'created_at': created_at, 'note': NOTE[arm],
            'experiment': 'sim_rl_bestlap_20261013', 'arm': arm, 'simulator_only': True, 'workers_used': workers_used,
            'train_env': GPU_ENV}
    write_json(run / 'run.json', meta)


def write_engine_check(run, src):
    """確認の json → <学習結果>/engine_check.json(走行タブ。10/13 の cotrain_train.py と同じ形)。
    確認の後にエンジンが変わっていたら写さない"""
    c = load(src)
    eng = run / 'jetformer_fp16.engine'
    if not c or not eng.exists():
        return None
    st = eng.stat()
    checked = datetime.datetime.strptime(c['checked_at'], '%Y-%m-%d %H:%M:%S').timestamp()
    if st.st_mtime > checked:
        return None
    acc, lat = c.get('accuracy') or {}, c.get('latency_ms') or {}
    out = {'ok': bool(c.get('all_gates')), 'accuracy_ok': bool(acc.get('gate_actions_2p5us')),
           'speed_ok': bool(lat.get('gate_run_numpy')), 'actions_max_us': acc.get('actions_max_us'),
           'actions_p99_us': acc.get('actions_p99_us'), 'gate_us': GATE_US,
           'run_numpy_median_ms': lat.get('run_numpy_median'), 'run_numpy_p99_ms': lat.get('run_numpy_p99'),
           'checked_at': c['checked_at'], 'note': 'シミュレーター専用(ジャンル 8)。実車では使わない',
           'engine': {'file': eng.name, 'size': st.st_size, 'mtime_ns': st.st_mtime_ns},
           'source': str(src), 'method': '実フレーム 64 個で TensorRT(run_numpy)と PyTorch FP32(CPU・best.pth)の actions の差。'
                                         '速さは 100 回の空回しの後 600 回(sim_rl_bestlap_20261013 の train/engine_load_check.py)'}
    write_json(run / 'engine_check.json', out)
    return out


def save_state(state):
    state['updated'] = now()
    write_json(STATE_PATH, state)


def train_one(proto, state, arm, seed):
    """1 本を学習・エンジン・確認・一式まで。失敗したら False"""
    key = '%s/s%d' % (arm, seed)
    args0, name = command_of(proto, arm, seed, resume=False)
    run = REPO / arg_value(args0, '--out')
    epochs = int(arg_value(args0, '--epochs'))
    runs_state = state.setdefault('runs', {})
    rs = runs_state.setdefault(key, {'run': name, 'arm': arm, 'seed': seed, 'attempts': [], 'workers': DEFAULT_WORKERS})
    logfile = LOGS / ('%s.train.log' % name)
    # ---- ① 学習
    while not run_done(run, epochs):
        own = [a for a in rs['attempts'] if not a.get('watchdog_killed') and a.get('exit', 0) is not None]
        if len(own) >= MAX_ATTEMPTS or len(rs['attempts']) >= MAX_ATTEMPTS + MAX_WATCHDOG_RETRIES:
            log('%s: 学習を %d 回試しても終わりませんでした(見張りで止めた回を含む)。止めます' % (key, len(rs['attempts'])))
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
        args, _ = command_of(proto, arm, seed, resume=resume, workers=None if workers == 4 else workers)
        cmd = [PY, '-u'] + args
        header = '==== %s %s(workers %d・%s)====' % (
            now(), '再開: %d エポック目まで済み' % done_ep if resume else '開始', workers, ' '.join(args))
        att = {'started': now(), 'resume_from_epoch': done_ep if resume else 0, 'workers': workers,
               'code': code_fingerprint(), 'env': GPU_ENV, 'mem_cap': '3G', 'need_mb': MIN_FREE_MB}
        rs['attempts'].append(att)
        save_state(state)
        # 再開は last.pth を GPU に読むので一時的に増える(05:48 に上限 3G で止まった)。前の再開が自分の上限で止まっていたら
        # 再開のときだけ上限 3400M・始める空き 3500 MB にする(deviations.md)
        resume_capped = resume and any(a.get('resume_from_epoch') and a.get('exit') in (-9, 137) and not a.get('watchdog_killed')
                                       for a in rs['attempts'][:-1])
        mem_cap, need = ('3400M', 3500) if resume_capped else ('3G', MIN_FREE_MB)
        att.update({'mem_cap': mem_cap, 'need_mb': need})
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
                log('%s: メモリの上限で止まったようです。--workers %d で続きから(protocol の予備策は 2)' % (key, rs['workers']))
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
               'throttle_neutral_us_trained': res.get('throttle_neutral_us')})
    if not (run / 'train.log').exists() or (run / 'train.log').stat().st_size != logfile.stat().st_size:
        shutil.copyfile(logfile, run / 'train.log')
    save_state(state)
    # ---- ② エンジン(別のプロセス。tools/build_engine.py は tmp → check_io → 置き換え)
    onnx, engine = run / 'jetformer.onnx', run / 'jetformer_fp16.engine'
    if not engine.exists() or engine.stat().st_mtime < onnx.stat().st_mtime:
        elog = LOGS / ('%s.engine.log' % name)
        code, sec, free = run_step([PY, '-u', 'tools/build_engine.py', onnx, engine], elog, mem='2500M', need_mb=ENGINE_FREE_MB,
                                   watch=str(engine))
        rs['engine'] = {'exit': code, 'wall_s': sec, 'at': now(), 'code': code_fingerprint()}
        save_state(state)
        if code != 0 or not engine.exists():
            log('%s のエンジンの作成に失敗しました(終了コード %d)。止めます' % (key, code))
            return False
        shutil.copyfile(elog, run / 'engine.log')
    if not (run / 'run.json').exists():
        write_run_json(run, args0, arm, seed, rs['created_at'], rs.get('workers', DEFAULT_WORKERS))
    # ---- ③ エンジンの確認(走行タブと同じ手順で読む。別のプロセス)
    out = ENGINE_CHECK_DIR / ('%s.json' % name)
    if not out.exists() or out.stat().st_mtime < engine.stat().st_mtime:
        code, sec, _ = run_step([PY, '-u', HERE / 'engine_load_check.py', run, out], LOGS / ('%s.engine_check.log' % name),
                                mem='2500M', need_mb=ENGINE_FREE_MB, watch=str(out))
        rs['check'] = {'exit': code, 'wall_s': sec, 'at': now()}
        save_state(state)
        if code != 0 or not out.exists():
            log('%s のエンジンの確認が失敗しました(終了コード %d)。止めます' % (key, code))
            return False
    ec = write_engine_check(run, out)
    rs['engine_check'] = ec and {k: ec[k] for k in ('ok', 'accuracy_ok', 'speed_ok', 'actions_max_us',
                                                     'run_numpy_median_ms', 'run_numpy_p99_ms')}
    save_state(state)
    # ---- ④ ブリッジ用の一式(CPU)と、シミュレーターの Python での確認
    bdir = BUNDLES / name
    bcheck = bdir / 'bundle_check.json'
    fresh = bcheck.exists() and bcheck.stat().st_mtime >= onnx.stat().st_mtime and (load(bcheck) or {}).get('ok')
    if not fresh:
        code, sec, _ = run_step([PY, '-u', HERE / 'export_bundle.py', run, bdir], LOGS / ('%s.bundle.log' % name),
                                mem='1500M', lock='none', need_mb=1500, gpu=False)
        if code != 0:
            log('%s の一式の書き出しに失敗しました(終了コード %d)。止めます' % (key, code))
            return False
        code, sec, _ = run_step([SIM_PY, '-I', '-B', HERE / 'check_bundle.py', bdir],
                                LOGS / ('%s.bundle.log' % name), mem='1500M', lock='none', need_mb=1000, gpu=False,
                                append=True, header='==== %s check_bundle.py(シミュレーターの Python)====' % now())
        rs['bundle'] = {'exit': code, 'at': now(), 'dir': str(bdir)}
        save_state(state)
        if code != 0 or not (load(bcheck) or {}).get('ok'):
            log('%s の一式の確認が通りませんでした(終了コード %d)。止めます' % (key, code))
            return False
    bc = load(bcheck)
    rs['bundle_check'] = {k: bc.get(k) for k in ('ok', 'zip', 'zip_sha256', 'onnx_vs_torch_max_us', 'defaults')}
    rs['done'] = True
    save_state(state)
    log('%s 済み: best %.1f µs(%d エポック目)・エンジンの確認 %s・一式 %s' % (
        key, res['best_val_us'], res['best_epoch'], 'OK' if ec and ec['ok'] else 'NG', 'OK' if bc.get('ok') else 'NG'))
    return True


def cmd_train(state):
    proto, sha = load_protocol()
    state['protocol_sha256'] = sha
    state['rl_dataset'] = rl_dataset_name()
    state.setdefault('order', ['%s/s%d' % k for k in ORDER])
    for d in (LOGS, ENGINE_CHECK_DIR, BUNDLES):
        d.mkdir(parents=True, exist_ok=True)
    for arm, seed in ORDER:
        if not train_one(proto, state, arm, seed):
            return 1
    state['all_done'] = now()
    save_state(state)
    log('学習・エンジン・確認・一式: 全部済みました')
    return 0


def cmd_status(state):
    print(json.dumps(state, ensure_ascii=False, indent=1))
    return 0


def main(argv):
    if len(argv) != 2 or argv[1] not in ('train', 'status'):
        raise SystemExit(__doc__)
    os.chdir(REPO)
    state = load(STATE_PATH, {})
    return {'train': cmd_train, 'status': cmd_status}[argv[1]](state)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
