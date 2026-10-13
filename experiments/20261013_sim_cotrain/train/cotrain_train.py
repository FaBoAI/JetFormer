#!/usr/bin/env python3
"""シミュレーター + 実機の混合学習(案 C)の学習の段: protocol.json の腕を 1 本ずつ学習し、エンジンを作って確かめる
(GPU、1 プロセスずつ・続きから再開できる)

    cd ~/jetformer && nice -n 10 python -u /home/jetson/jetformer_reports/sim_cotrain_20261013/train/cotrain_train.py train
    cd ~/jetformer && python /home/jetson/jetformer_reports/sim_cotrain_20261013/train/cotrain_train.py status

小コマンド:
    train   protocol.json の順(C 種 0 → C 種 1 → C+FT 種 0 → C+FT 種 1 → A+FT 種 0 → A+FT 種 1)に、各 1 本ずつ
            ① tools/train.py(protocol.json の train のコマンド。last.pth があれば resume のコマンド)
            ② 別のプロセスで tools/build_engine.py(FP16)
            ③ 別のプロセスで engine_load_check.py(走行タブと同じ手順でエンジンを読む・実フレーム 64 個で FP32 との差・速さ)
            ④ run.json(管理画面の一覧に出す)と engine_check.json(走行タブの確認の表示)を書く(CPU)
            出来たもの(result.json の state=done、エンジン、確認の json)は飛ばす。電源断のあとは同じコマンドで続きから。
    status  今の状態を表示する(GPU なし)

GPU を使う手順の前に毎回、ほかに GPU を使うプロセスが無いこと、管理画面が走行中・学習中でないこと、空きメモリ 3.5 GB 以上を
確かめる(足りなければ待つ)。protocol.json の sha256 が logs/protocol.log の最後の「固定しました」と違えば止める。
状態は state.json(tmp → 置き換え)。車は動かさない・PCA9685 に触らない。
"""

import datetime
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path('/home/jetson/jetformer')
HERE = Path(__file__).resolve().parent
REPORT = HERE.parent
LOGS = REPORT / 'logs'
PROTOCOL = REPORT / 'protocol.json'
PROTOCOL_LOG = LOGS / 'protocol.log'
STATE_PATH = HERE / 'state.json'
ENGINE_CHECK_DIR = HERE / 'engine_check'
RUNS = REPO / 'data' / 'runs'
PY = sys.executable
ORDER = [('C', 0), ('C', 1), ('C+FT', 0), ('C+FT', 1), ('A+FT', 0), ('A+FT', 1)]
CMD_PREFIX = 'cd ~/jetformer && nice -n 10 python -u '
MIN_FREE_MB = 3500
GATE_US = 2.5
GPU_PATTERNS = ('tools/train.py', 'build_engine.py', 'policy_zoo.py', 'benchmark_model.py', 'engine_check.py',
                'engine_load_check.py', 'rl_policies.py', 'rl_evaluate.py', 'compare_models.py', 'tools/evaluate.py',
                'offline_rl.py', 'run_smoke.py', 'zoo_train.py', 'sim_cotrain_eval', 'trtexec')
CODE_FILES = ['tools/train.py', 'tools/build_engine.py', 'jetformer/model.py', 'jetformer/train_data.py',
              'jetformer/trt_runtime.py', 'jetformer/drive.py', 'jetformer/policies/__init__.py',
              'jetformer/policies/runtime.py', 'jetformer/policies/blocks.py', 'jetformer/policies/jetformer_v0.py']
NOTE = {
    'C': 'シミュレーター+実機の混合学習(案 C、2026-10-13)の C(混ぜて学習)。机上の比較用で、実車の A/B/A の前は走行の標準はモデル B のまま'
         '(ジャンル 8 = シミュレーターの頭は実車では使わない)',
    'C+FT': 'シミュレーター+実機の混合学習(案 C、2026-10-13)の C+FT(C のあと実機だけで微調整)。机上の比較用で、'
            '実車の A/B/A の前は走行の標準はモデル B のまま',
    'A+FT': 'シミュレーター+実機の混合学習(案 C、2026-10-13)の対照 A+FT(10/12 の v0 から実機だけで同じ微調整)。'
            '机上の比較用で、走行の標準はモデル B のまま',
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


def fixed_protocol_sha():
    """logs/protocol.log の最後の「固定しました: ...(sha256 xxx)」の sha256"""
    sha = None
    for line in PROTOCOL_LOG.read_text(encoding='utf-8').splitlines():
        if line.startswith('固定しました') and 'sha256 ' in line:
            sha = line.split('sha256 ')[1].rstrip(')').strip()
    return sha


def load_protocol():
    sha = sha256(PROTOCOL)
    want = fixed_protocol_sha()
    if sha != want:
        raise SystemExit('protocol.json の sha256 %s が固定したもの %s と違います。止めます' % (sha, want))
    return json.load(open(PROTOCOL, encoding='utf-8')), sha


def command_of(proto, arm, seed, resume):
    """protocol.json のコマンド → (train.py の引数のリスト, 学習結果の名前)"""
    c = proto['arms'][arm]['commands'][str(seed)]
    text = c['resume' if resume else 'train']
    if not text.startswith(CMD_PREFIX):
        raise SystemExit('知らない形のコマンド: %s' % text)
    return shlex.split(text[len(CMD_PREFIX):]), c['run']


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
        if pid in me or ppid in me or 'cotrain_train.py' in args or 'grep' in args:
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


def write_run_json(run, args, arm, seed, created_at):
    """管理画面の学習タブ・走行タブの一覧に出すための run.json(管理画面の学習と同じ欄 + 微調整の元)"""
    sys.path.insert(0, str(REPO))
    from jetformer import policies      # torch を読まない
    meta = {'dataset': Path(arg_value(args, '--data')).name, 'stride': int(arg_value(args, '--stride')),
            'epochs': int(arg_value(args, '--epochs')), 'batch': int(arg_value(args, '--batch')),
            'lr': float(arg_value(args, '--lr')), 'pretrained': True, 'build_engine': True,
            'command_input': '--no-command-input' not in args, 'mask_impacts': True, 'policy': arg_value(args, '--policy'),
            'policy_cfg': policies.get(arg_value(args, '--policy')).resolve({}), 'impact_g': float(arg_value(args, '--impact-g')),
            'seed': seed, 'name': run.name, 'created_at': created_at, 'note': NOTE[arm],
            'experiment': 'sim_cotrain_20261013', 'arm': arm}
    if arg_value(args, '--init'):
        meta['init'] = arg_value(args, '--init')
    if arg_value(args, '--drop-mask'):
        meta['drop_mask'] = arg_value(args, '--drop-mask')
    write_json(run / 'run.json', meta)


def write_engine_check(run, src):
    """確認の json → <学習結果>/engine_check.json(走行タブ。10/12 の write_engine_check.py と同じ形)。
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
           'checked_at': c['checked_at'], 'note': '',
           'engine': {'file': eng.name, 'size': st.st_size, 'mtime_ns': st.st_mtime_ns},
           'source': str(src), 'method': '実フレーム 64 個で TensorRT(run_numpy)と PyTorch FP32(CPU・best.pth)の actions の差。'
                                         '速さは 100 回の空回しの後 600 回(sim_cotrain_20261013/train/engine_load_check.py)'}
    write_json(run / 'engine_check.json', out)
    return out


def save_state(state):
    state['updated'] = now()
    write_json(STATE_PATH, state)


def cmd_train(state):
    proto, sha = load_protocol()
    state['protocol_sha256'] = sha
    ENGINE_CHECK_DIR.mkdir(parents=True, exist_ok=True)
    runs_state = state.setdefault('runs', {})
    for arm, seed in ORDER:
        key = '%s/s%d' % (arm, seed)
        args0, name = command_of(proto, arm, seed, resume=False)
        run = REPO / arg_value(args0, '--out')
        if run.name != name:
            raise SystemExit('protocol.json の run と --out が違います: %s / %s' % (name, run.name))
        epochs = int(arg_value(args0, '--epochs'))
        rs = runs_state.setdefault(key, {'run': name, 'arm': arm, 'seed': seed, 'attempts': []})
        # ---- 学習
        if not run_done(run, epochs):
            init = arg_value(args0, '--init')
            if init and not (REPO / init).exists():
                log('%s: 初期値 %s がまだありません。止めます' % (key, init))
                return 1
            run.mkdir(parents=True, exist_ok=True)
            rs.setdefault('created_at', datetime.datetime.now().isoformat(timespec='seconds'))
            done_ep = last_epoch(run)
            resume = (run / 'last.pth').exists() and done_ep > 0
            args, _ = command_of(proto, arm, seed, resume=resume)
            cmd = [PY, '-u'] + args
            header = '==== %s %s(%s)====' % (now(), '再開: %d エポック目まで済み' % done_ep if resume else '開始', ' '.join(args))
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
        hist = res['history']
        rs.update({'train_wall_s': round(sum(a.get('wall_s') or 0 for a in rs['attempts']), 1),
                   'resumes': sum(1 for a in rs['attempts'] if a.get('resume_from_epoch')),
                   'best_val_us': res.get('best_val_us'), 'best_epoch': res.get('best_epoch'),
                   'last_val_us': hist[-1]['val_us'], 'peak_mem_mb': res.get('peak_mem_mb'),
                   'train_samples': res.get('train_samples'), 'val_samples': res.get('val_samples'),
                   'epoch1_s': hist[0]['seconds'], 'epoch2_s': hist[1]['seconds'] if len(hist) > 1 else None,
                   'epoch_s_median': sorted(h['seconds'] for h in hist)[len(hist) // 2]})
        save_state(state)
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
            write_run_json(run, args0, arm, seed, rs['created_at'])
        # ---- エンジンの確認(走行タブと同じ手順で読む。別のプロセス)
        out = ENGINE_CHECK_DIR / ('%s.json' % name)
        if not out.exists() or out.stat().st_mtime < engine.stat().st_mtime:
            code, sec, _ = run_step([PY, '-u', HERE / 'engine_load_check.py', run, out], LOGS / ('check_%s.log' % name))
            rs['check'] = {'exit': code, 'wall_s': sec, 'at': now()}
            save_state(state)
            if code != 0 or not out.exists():
                log('%s のエンジンの確認が失敗しました(終了コード %d)。止めます' % (key, code))
                return 1
        ec = write_engine_check(run, out)
        rs['engine_check'] = ec and {k: ec[k] for k in ('ok', 'accuracy_ok', 'speed_ok', 'actions_max_us',
                                                         'run_numpy_median_ms', 'run_numpy_p99_ms')}
        rs['done'] = True
        save_state(state)
        log('%s 済み: best %.1f µs(%d エポック目)・エンジンの確認 %s' % (
            key, res['best_val_us'], res['best_epoch'], 'OK' if ec and ec['ok'] else 'NG'))
    state['all_done'] = now()
    save_state(state)
    log('学習・エンジン・確認: 全部済みました')
    return 0


def cmd_status(state):
    print(json.dumps(state, ensure_ascii=False, indent=1))
    return 0


def main(argv):
    if len(argv) != 2 or argv[1] not in ('train', 'status'):
        raise SystemExit(__doc__)
    LOGS.mkdir(parents=True, exist_ok=True)
    os.chdir(REPO)
    state = load(STATE_PATH, {})
    return {'train': cmd_train, 'status': cmd_status}[argv[1]](state)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
