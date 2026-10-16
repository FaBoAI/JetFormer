#!/usr/bin/env python3
"""real_accuracy_20261015 の GPU の段の道具: protocol.json の gpu_order を上から 1 つずつ(GPU は 1 プロセスずつ・電源断のあとは
同じコマンドで続きから)。

    cd ~/jetformer && nohup setsid /home/jetson/miniconda/envs/jetracer/bin/python -u \\
        experiments/20261015_real_accuracy/train/realacc_driver.py all \\
        >> /home/jetson/jetformer_reports/real_accuracy_20261015/logs/driver.log 2>&1 < /dev/null &
    python experiments/20261015_real_accuracy/train/realacc_driver.py status     # 状態(train/state.json)
    python experiments/20261015_real_accuracy/train/realacc_driver.py plan       # 段の一覧とコマンドを出すだけ
    ... realacc_driver.py posthoc   # 固定の外の参考の腕 SIZE-EQ(deviations.md 21:55)。all が終わるのを待ってから

段(protocol.json の gpu_order と同じ順):
  1〜4  ENS の 4 本(msf-s2・v0-s2・msf-s3・v0-s3)
  5     crosscheck(A_s0 の best を新しい道具で予測し直す)→ 4 本の予測(best・last)
  6〜9  SIZE の H・Q(種 0・1)
  10    ROT の動作確認(2 エポック・2 ステップ → rot_log.jsonl に位相 0・1)→ rot-s0・rot-s1
  11    6 本の予測
  12    ENS のゲート(ens/ens_gate.py)
  13    (任意)sizeT-s0・s1 + 予測
  14    (任意)GAINL の動作確認・s0・s1 + 予測
各学習の後に、結果の確かめ(学習・検証のサンプル数・ROT の位相の表)。各予測の後に、npz の読み直し(key・有限)→ その学習結果の
last.pth を消す(protocol の disk.delete_last_pth)→ evaluate(CPU)→ progress.md。

学習のコマンドは protocol.json の arms.*.train の文字列そのまま(python = jetracer の環境の Python)。ただしメモリの扱いだけ
10/14〜10/15 と同じ包み(THP なし nothp.py・MALLOC_ARENA_MAX 2・expandable_segments・choom 1000・全体の空きメモリの見張り)を
足す(計算の結果は変わらない。deviations.md に 1 回書く)。メモリの上限で止まったら protocol の memory.on_cgroup_kill のとおり
上限を 800M 上げて続きから(deviations.md に自動で書く)。

時間と空き容量(protocol の time_plan_jst・cut_order_if_late・disk):
  - 任意の段は 2026-10-16 03:00 までに始められなければ削る。GPU の段は 06:00 を過ぎて終わる見込みなら始めない
  - ROT は腕ごと(2 本と予測が 06:00 までに収まらなければ 1 本目も始めない)
  - 各 GPU の段の前に空きを調べる: < 4.5 GB → 任意の段を削る。< 4.0 GB → 止めて deviations.md
protocol.json の sha256 が protocol.sha256 と違えば止める。状態は train/state.json(tmp → 置き換え)。車は動かさない・
PCA9685・I2C に触らない。jetformer-web を再起動しない。
"""

import datetime
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

REPO = Path('/home/jetson/jetformer')
HERE = Path(__file__).resolve().parent
EXP = HERE.parent
EXP_REL = 'experiments/20261015_real_accuracy'
REPORT = Path('/home/jetson/jetformer_reports/real_accuracy_20261015')
LOGS = REPORT / 'logs'
PROTOCOL = REPORT / 'protocol.json'
PROTOCOL_SHA = REPORT / 'protocol.sha256'
DEVIATIONS = REPORT / 'deviations.md'
PROGRESS = REPORT / 'progress.md'
STATE_PATH = HERE / 'state.json'
RUNS = REPO / 'data' / 'runs'
PRED_DIR = REPO / 'data' / 'zoo' / 'real_accuracy_20261015' / 'pred'
EVAL_TOOL = EXP / 'eval' / 'realacc_eval.py'
EVAL_SHA = REPORT / 'eval_sha256.txt'
GATE_TOOL = EXP / 'ens' / 'ens_gate.py'
GATE_JSON = EXP / 'ens' / 'gate.json'
GAINL_TOOL = EXP / 'train_gainl.py'
JF_HEAVY = '/home/jetson/bin/jf-heavy'
PY = '/home/jetson/miniconda/envs/jetracer/bin/python'
NOTHP = HERE / 'nothp.py'
PREFIX = 'realacc_20261015-'
GPU_ENV = {'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True', 'MALLOC_ARENA_MAX': '2'}
CPU_ENV = {'CUDA_VISIBLE_DEVICES': ''}
TZ_JST = datetime.timezone(datetime.timedelta(hours=9))
OPTIONAL_LATEST_START = datetime.datetime(2026, 10, 16, 3, 0, tzinfo=TZ_JST)
GPU_HARD_STOP = datetime.datetime(2026, 10, 16, 6, 0, tzinfo=TZ_JST)
POSTHOC_LATEST_START = datetime.datetime(2026, 10, 16, 4, 0, tzinfo=TZ_JST)   # 固定の外の参考の腕 SIZE-EQ(deviations.md 21:55)
DISK_CUT_OPTIONAL_GB = 4.5
DISK_STOP_GB = 4.0
MEM_RAISE_MB = 800
MAX_ATTEMPTS = 6
MAX_WATCHDOG_RETRIES = 30
WATCH_MIN_AVAIL_MB = 200        # 全体の空きメモリがこれを WATCH_SECONDS 秒続けて下回ったら、自分の学習だけを止める(10/14 と同じ)
WATCH_SECONDS = 8
NEED_WAIT_S = 3 * 60            # 空きメモリが protocol の need-mb に届くのを待つ長さ(過ぎたら見張りと choom 1000 に任せて始める。
                                # 18:21 に 20 分から短くした: 何もしていないときの空きが 3.8〜3.9 GB しかなく、上限を上げた後の need に届かない)
MIN_NEED_MB = 2600
EXPECTED_SAMPLES = {'ENS': 5912, 'SIZE_H': 3046, 'SIZE_Q': 1682, 'SIZE_T': 4682, 'ROT': 5912, 'GAINL': 5912}
EXPECTED_VAL = 1559
ROT_PHASE_SIZES = (5912, 5903, 5879)
GPU_PATTERNS = ('tools/train.py', 'train_rot.py', 'train_gainl.py', 'build_engine.py', 'policy_zoo.py', 'engine_check.py',
                'engine_load_check.py', 'rl_policies.py', 'rl_evaluate.py', 'compare_models.py', 'tools/evaluate.py',
                'offline_rl.py', 'zoo_train.py', 'sim_cotrain_eval', 'trtexec', 'realacc_eval.py predict',
                'realacc_eval.py crosscheck', 'ens_gate.py')
# 段の見積もり [分](protocol の gpu_budget_minutes と 10/12 の 1 エポックの秒数から)
EST_MIN = {'msfusion': 32, 'jetformer_v0': 22, 'SIZE_H': 13, 'SIZE_Q': 8, 'SIZE_T': 18, 'rot_smoke': 3, 'predict_per_run': 3.5,
           'crosscheck': 4, 'gate': 15, 'gainl_smoke': 3}
CODE_FILES = ['tools/train.py', 'jetformer/train_data.py', 'jetformer/policies/runtime.py', 'jetformer/policies/__init__.py',
              EXP_REL + '/train_rot.py']


def now():
    return time.strftime('%Y-%m-%d %H:%M:%S')


def jst_now():
    return datetime.datetime.now(TZ_JST)


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
        f.flush()
        os.fsync(f.fileno())
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
    """deviations.md の表の下に 1 行足す(時刻つき)"""
    with open(DEVIATIONS, 'a', encoding='utf-8') as f:
        f.write('- %s(realacc_driver.py が自動で記録)%s\n' % (now(), text))
    log('deviations.md: %s' % text)


def load_protocol():
    want = PROTOCOL_SHA.read_text().split()[0]
    sha = sha256(PROTOCOL)
    if sha != want:
        raise SystemExit('protocol.json の sha256 %s が固定したもの %s と違います。止めます' % (sha, want))
    return json.loads(PROTOCOL.read_text(encoding='utf-8')), sha


# ---------------------------------------------------------------- コマンド(テストする)

def parse_protocol_command(text):
    """protocol の 'jf-heavy --mem M --lock big --need-mb N -- nice -n 10 python -u <script> <引数>' →
    {'mem': M, 'need_mb': N, 'lock': L, 'args': [<script>, <引数> ...]}"""
    toks = shlex.split(text)
    if not toks or toks[0] != JF_HEAVY:
        raise ValueError('jf-heavy で始まらないコマンド: %s' % text)
    i, opt = 1, {'mem': '2G', 'lock': 'big', 'need_mb': 1500}
    while i < len(toks) and toks[i] != '--':
        k, v = toks[i], toks[i + 1]
        if k == '--mem':
            opt['mem'] = v
        elif k == '--lock':
            opt['lock'] = v
        elif k == '--need-mb':
            opt['need_mb'] = int(v)
        else:
            raise ValueError('知らない jf-heavy の引数 %s' % k)
        i += 2
    rest = toks[i + 1:]
    if rest[:3] != ['nice', '-n', '10'] or rest[3] != 'python' or rest[4] != '-u':
        raise ValueError('"nice -n 10 python -u" で始まらないコマンド: %s' % text)
    opt['args'] = rest[5:]
    return opt


def mem_to_mb(mem):
    s = str(mem).strip().upper()
    if s.endswith('G'):
        return int(round(float(s[:-1]) * 1024))
    if s.endswith('M'):
        return int(round(float(s[:-1])))
    raise ValueError('メモリの書き方がわかりません: %s' % mem)


def mb_to_mem(mb):
    return '%dM' % int(mb)


def arg_value(args, flag):
    return args[args.index(flag) + 1] if flag in args else None


def with_resume(args, resume):
    a = [x for x in args if x != '--resume']
    return a + ['--resume'] if resume else a


def wrap_gpu(args, mem, need_mb, lock='big'):
    """学習・予測のコマンドを jf-heavy + nice + choom 1000 + nothp で包む(計算の結果は変えない)"""
    inner = ['choom', '-n', '1000', '--', PY, '-I', str(NOTHP), PY, '-u'] + [str(a) for a in args]
    return [JF_HEAVY, '--mem', mem, '--lock', lock, '--need-mb', str(int(need_mb)), '--', 'nice', '-n', '10'] + inner


def wrap_cpu(args, mem='1500M', need_mb=700):
    return [JF_HEAVY, '--mem', mem, '--lock', 'none', '--need-mb', str(int(need_mb)), '--', 'nice', '-n', '10',
            PY, '-u'] + [str(a) for a in args]


def fits_before(start, minutes, deadline):
    """start から minutes 分で deadline までに終わるか"""
    return start + datetime.timedelta(minutes=float(minutes)) <= deadline


def raise_cap(mem, by_mb=MEM_RAISE_MB):
    return mb_to_mem(mem_to_mb(mem) + by_mb)


# ---------------------------------------------------------------- 段の一覧

def build_steps(proto):
    """protocol の gpu_order → 段のリスト(dict)。学習の段は protocol のコマンドの文字列を持つ"""
    arms = proto['arms']
    st = []

    def train(arm_key, name, text, kind, optional=False, group=None):
        c = parse_protocol_command(text)
        out = arg_value(c['args'], '--out')
        if Path(out).name != name:
            raise SystemExit('protocol の名前と --out が違います: %s / %s' % (name, out))
        st.append({'id': 'train:' + name, 'type': 'train', 'arm': arm_key, 'name': name, 'cmd': c, 'kind': kind,
                   'policy': arg_value(c['args'], '--policy'), 'optional': optional, 'group': group or arm_key})

    ens = arms['ENS']['train']
    for n in ('msf-s2', 'v0-s2', 'msf-s3', 'v0-s3'):
        name = PREFIX + n
        train('ENS', name, ens[name], 'ENS')
    ens_names = [PREFIX + n for n in ('msf-s2', 'v0-s2', 'msf-s3', 'v0-s3')]
    st.append({'id': 'crosscheck', 'type': 'crosscheck', 'optional': False, 'group': 'ENS'})
    st.append({'id': 'predict:ENS', 'type': 'predict', 'names': ens_names, 'optional': False, 'group': 'ENS'})
    sz = arms['SIZE']['train']
    for n in ('sizeH-s0', 'sizeH-s1', 'sizeQ-s0', 'sizeQ-s1'):
        name = PREFIX + n
        train('SIZE', name, sz[name], 'SIZE_' + n[4], group='SIZE_' + n[4])
    rot = arms['ROT']
    st.append({'id': 'rot_smoke', 'type': 'rot_smoke', 'cmd': parse_protocol_command(rot['smoke_test_gpu'].split(' → ')[0]),
               'optional': False, 'group': 'ROT'})
    for n in ('rot-s0', 'rot-s1'):
        name = PREFIX + n
        train('ROT', name, rot['train'][name], 'ROT')
    six = [PREFIX + n for n in ('sizeH-s0', 'sizeH-s1', 'sizeQ-s0', 'sizeQ-s1', 'rot-s0', 'rot-s1')]
    st.append({'id': 'predict:SIX', 'type': 'predict', 'names': six, 'optional': False, 'group': 'SIX'})
    st.append({'id': 'gate', 'type': 'gate', 'optional': False, 'group': 'GATE'})
    for n in ('sizeT-s0', 'sizeT-s1'):
        name = PREFIX + n
        train('SIZE', name, arms['SIZE']['optional_train'][name], 'SIZE_T', optional=True, group='SIZE_T')
    st.append({'id': 'predict:SIZE_T', 'type': 'predict', 'names': [PREFIX + 'sizeT-s0', PREFIX + 'sizeT-s1'],
               'optional': True, 'group': 'SIZE_T'})
    st.append({'id': 'gainl_smoke', 'type': 'gainl_smoke', 'optional': True, 'group': 'GAINL'})
    for n in ('gainl-s0', 'gainl-s1'):
        name = PREFIX + n
        train('GAINL', name, arms['GAINL_optional']['train'][name], 'GAINL', optional=True, group='GAINL')
    st.append({'id': 'predict:GAINL', 'type': 'predict', 'names': [PREFIX + 'gainl-s0', PREFIX + 'gainl-s1'],
               'optional': True, 'group': 'GAINL'})
    return st


def build_posthoc_steps(proto):
    """固定の外の参考の腕 SIZE-EQ(deviations.md 2026-10-15 21:55): sizeH・sizeQ と同じ引数で、エポック数だけ
    勾配のステップ数を全部(92 × 30 = 2,760)とそろえる(H 59 エポック・Q 106 エポック)"""
    sz = proto['arms']['SIZE']['train']
    st = []
    for sub, ep, est in (('H', 59, 30), ('Q', 106, 28)):
        names = []
        for seed in (0, 1):
            c = parse_protocol_command(sz[PREFIX + 'size%s-s%d' % (sub, seed)])
            name = PREFIX + 'size%seq-s%d' % (sub, seed)
            args = list(c['args'])
            args[args.index('--epochs') + 1] = str(ep)
            args[args.index('--out') + 1] = str(RUNS / name)
            c['args'] = args
            names.append(name)
            st.append({'id': 'train:' + name, 'type': 'train', 'arm': 'SIZE_EQ', 'name': name, 'cmd': c, 'kind': 'SIZE_' + sub,
                       'policy': 'jetformer_v0', 'optional': True, 'group': 'SIZEEQ_' + sub, 'est': est})
        st.append({'id': 'predict:SIZEEQ_' + sub, 'type': 'predict', 'names': names, 'optional': True,
                   'group': 'SIZEEQ_' + sub, 'est': 7})
    return st


def step_minutes(step):
    if 'est' in step:
        return step['est']
    t = step['type']
    if t == 'train':
        if step['kind'] in ('SIZE_H', 'SIZE_Q', 'SIZE_T'):
            return EST_MIN[step['kind']]
        return EST_MIN[step['policy']]
    if t == 'predict':
        return EST_MIN['predict_per_run'] * len(step['names'])
    if t == 'crosscheck':
        return EST_MIN['crosscheck']
    if t == 'rot_smoke':
        return EST_MIN['rot_smoke']
    if t == 'gate':
        return EST_MIN['gate']
    if t == 'gainl_smoke':
        return EST_MIN['gainl_smoke']
    return 10


def group_remaining_minutes(steps, i, done):
    """steps[i] から同じ group の残りの段(済みを除く)の見積もりの合計"""
    g = steps[i]['group']
    return sum(step_minutes(s) for s in steps[i:] if s['group'] == g and s['id'] not in done)


# ---------------------------------------------------------------- 状態・メモリ・GPU

def mem_available_mb():
    for line in open('/proc/meminfo'):
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) // 1024
    return 0


def disk_free_gb(path='/home/jetson'):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1e9


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
        if pid in me or ppid in me or 'realacc_driver.py' in args or 'grep' in args:
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
            return
        why = 'GPU: %s' % busy
        if why != said:
            log('待ちます(%s)' % why)
            said = why
        time.sleep(30)


def wait_need(need_mb):
    """空きメモリが need_mb に届くまで最大 NEED_WAIT_S 秒待つ → jf-heavy に渡す need-mb"""
    t0 = time.time()
    said = False
    while time.time() - t0 < NEED_WAIT_S:
        avail = mem_available_mb()
        if avail >= need_mb:
            return need_mb
        if not said:
            log('空きメモリ %d MB < %d MB。最大 %d 分待ちます' % (avail, need_mb, NEED_WAIT_S // 60))
            said = True
        time.sleep(20)
    avail = mem_available_mb()
    use = max(MIN_NEED_MB, min(need_mb, avail - 50))
    log('空きメモリが %d MB のままなので need-mb %d で始めます(見張りと choom 1000 があるので全体は守られる)' % (avail, use))
    return use


LAST_WATCHDOG = {'killed': False}


def own_pids(pattern):
    out = subprocess.run(['ps', '-eo', 'pid,args'], capture_output=True, text=True).stdout.splitlines()[1:]
    pids = []
    for l in out:
        sp = l.strip().split(None, 1)
        if len(sp) < 2 or 'realacc_driver.py' in sp[1]:
            continue
        if pattern in sp[1] and sp[1].startswith(PY):
            pids.append(int(sp[0]))
    return pids


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


def run_cmd(full, logfile, env_extra=None, append=True, header=None, watch=None):
    """1 つのコマンドを動かす → (終了コード, 秒)"""
    t = time.time()
    log('%s' % ' '.join(map(str, full)))
    LAST_WATCHDOG['killed'] = False
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with open(logfile, 'a' if append else 'w', encoding='utf-8') as f:
        if header:
            f.write(header + '\n')
            f.flush()
        proc = subprocess.Popen([str(x) for x in full], cwd=str(REPO), stdout=f, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, env=dict(os.environ, PYTHONUNBUFFERED='1', **(env_extra or {})))
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
    return code, sec


def killed_by_memory(code, logfile):
    if code in (-9, 137):
        return True
    try:
        tail = Path(logfile).read_text(encoding='utf-8', errors='replace')[-6000:]
    except OSError:
        return False
    return 'killed by signal: Killed' in tail or 'Out of memory' in tail or 'MemoryError' in tail or 'oom-kill' in tail


def run_done(run, epochs):
    res = load(run / 'result.json')
    return bool(res and res.get('state') == 'done' and res.get('epochs_done') == epochs
                and (run / 'model_config.json').exists() and (run / 'best.pth').exists())


def last_epoch(run):
    """last.pth のエポック(無ければ 0、読めなければ −1 = 電源断で壊れた)"""
    p = run / 'last.pth'
    if not p.exists():
        return 0
    if p.stat().st_size == 0:
        return -1
    code = ('import torch,sys; ck=torch.load(sys.argv[1], map_location="cpu", weights_only=False); '
            'print(int(ck.get("epoch", 0)))')
    r = subprocess.run([PY, '-c', code, str(p)], capture_output=True, text=True, env=dict(os.environ, **CPU_ENV))
    try:
        return int(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return -1


def save_state(state):
    state['updated'] = now()
    write_json(STATE_PATH, state)


# ---------------------------------------------------------------- 学習の段

def check_run_result(step, run):
    """学習の後の確かめ → (ok, 説明)"""
    res = load(run / 'result.json') or {}
    want = EXPECTED_SAMPLES[step['kind']]
    if step['kind'] == 'ROT':           # result.json の学習のサンプル数 = 最後のエポック(30)の位相の行数
        want = ROT_PHASE_SIZES[(30 - 1 + int(arg_value(step['cmd']['args'], '--seed'))) % 3]
    probs = []
    if res.get('train_samples') != want:
        probs.append('学習のサンプル数 %s ≠ %d' % (res.get('train_samples'), want))
    if res.get('val_samples') != EXPECTED_VAL:
        probs.append('検証のサンプル数 %s ≠ %d' % (res.get('val_samples'), EXPECTED_VAL))
    if step['kind'] == 'ROT':
        seed = int(arg_value(step['cmd']['args'], '--seed'))
        rows = []
        for line in (run / 'rot_log.jsonl').read_text().splitlines() if (run / 'rot_log.jsonl').exists() else []:
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
        last = {}
        for r in rows:              # 再開で同じエポックが 2 回あれば後のほう
            last[int(r['epoch'])] = r
        for ep in range(1, 31):
            r = last.get(ep)
            p = (ep - 1 + seed) % 3
            if r is None or int(r['phase']) != p or int(r['samples']) != ROT_PHASE_SIZES[p]:
                probs.append('rot_log.jsonl のエポック %d が違う(%s)' % (ep, r))
                break
    return (not probs), '; '.join(probs) or 'OK'


def train_one(state, step):
    name = step['name']
    run = RUNS / name
    rs = state.setdefault('runs', {}).setdefault(name, {'arm': step['arm'], 'kind': step['kind'], 'attempts': []})
    c = step['cmd']
    epochs = int(arg_value(c['args'], '--epochs'))
    logfile = run / 'train.log'
    mem = rs.get('mem_cap') or c['mem']
    fam = (state.get('family_cap') or {}).get(step['policy'])
    if not rs.get('mem_cap') and fam and mem_to_mb(fam) > mem_to_mb(mem) and not run_done(run, epochs):
        add_deviation('%s: 同じ種類(%s)の前の学習がメモリの上限 %s で止まって %s に上げたので、この学習は最初から上限 %s で始める'
                      '(止まるのを待って上げるのと同じ結果。計算の結果は変わらない)' % (name, step['policy'], mem, fam, fam))
        mem = fam
        rs['mem_cap'] = mem
        save_state(state)
    while True:
        wait_gpu_free()                 # 手で始めた学習などが動いていれば、終わるのを待ってから済みかを調べる
        if run_done(run, epochs):
            break
        own = [a for a in rs['attempts'] if not a.get('watchdog_killed')]
        if len(own) >= MAX_ATTEMPTS or len(rs['attempts']) >= MAX_ATTEMPTS + MAX_WATCHDOG_RETRIES:
            log('%s: 学習を %d 回試しても終わりませんでした。止めます' % (name, len(rs['attempts'])))
            save_state(state)
            return False
        run.mkdir(parents=True, exist_ok=True)
        rs.setdefault('created_at', now())
        done_ep = last_epoch(run)
        if done_ep < 0:
            log('%s: last.pth が読めません(電源断で壊れた?)。退避して最初から' % name)
            os.replace(run / 'last.pth', run / ('last.pth.broken-%d' % int(time.time())))
            add_deviation('%s: last.pth が読めなかった(電源断で壊れた)ので最初から学習し直した(protocol の power_loss のとおり)' % name)
            done_ep = 0
        resume = (run / 'last.pth').exists() and done_ep > 0
        args = with_resume(c['args'], resume)
        need = int(c['need_mb']) + max(0, mem_to_mb(mem) - mem_to_mb(c['mem']))
        need_use = wait_need(need)
        full = wrap_gpu(args, mem, need_use)
        header = '==== %s %s(上限 %s・%s)====' % (now(), '再開: %d エポック目まで済み' % done_ep if resume else '開始', mem,
                                               ' '.join(args))
        att = {'started': now(), 'resume_from_epoch': done_ep if resume else 0, 'mem_cap': mem, 'need_mb': need_use,
               'code': code_fingerprint(), 'env': GPU_ENV, 'wrapped': 'choom 1000 + nothp'}
        rs['attempts'].append(att)
        save_state(state)
        code, sec = run_cmd(full, logfile, env_extra=GPU_ENV, header=header, watch='--out %s ' % arg_value(args, '--out'))
        att.update({'exit': code, 'wall_s': sec, 'ended': now(), 'watchdog_killed': LAST_WATCHDOG['killed']})
        save_state(state)
        if run_done(run, epochs):
            continue
        if LAST_WATCHDOG['killed']:
            log('%s: 全体の空きメモリが足りなくなったので止めました(自分の上限ではない)。空くのを待って続きから' % name)
            time.sleep(60)
        elif killed_by_memory(code, logfile):
            new = raise_cap(mem)
            add_deviation('%s: メモリの上限 %s で止まったので、protocol の memory.on_cgroup_kill のとおり上限を %s にして続きから'
                          '(計算の結果は変わらない)' % (name, mem, new))
            mem = new
            rs['mem_cap'] = mem
            fc = state.setdefault('family_cap', {})
            if mem_to_mb(new) > mem_to_mb(fc.get(step['policy'], '0M')):
                fc[step['policy']] = new
            att['memory_cap_hit'] = True
            save_state(state)
        elif code == 0:
            log('%s: 終了コード 0 なのに終わっていません(停止の指示?)。止めます' % name)
            return False
        else:
            log('%s の学習が終わっていません(終了コード %d)。もう一度続きから試します' % (name, code))
            time.sleep(30)
    res = load(run / 'result.json')
    hist = res['history']
    ok, why = check_run_result(step, run)
    rs.update({'train_wall_s': round(sum(a.get('wall_s') or 0 for a in rs['attempts']), 1),
               'resumes': sum(1 for a in rs['attempts'] if a.get('resume_from_epoch')),
               'best_val_us': res.get('best_val_us'), 'best_epoch': res.get('best_epoch'),
               'last_val_us': hist[-1]['val_us'], 'peak_mem_mb': res.get('peak_mem_mb'),
               'train_samples': res.get('train_samples'), 'val_samples': res.get('val_samples'),
               'epoch_s_median': sorted(h['seconds'] for h in hist)[len(hist) // 2],
               'duration_s': res.get('duration_s'), 'check': why, 'check_ok': ok, 'done_at': rs.get('done_at') or now()})
    save_state(state)
    if not ok:
        log('%s: 学習の結果の確かめが通りません: %s。止めます' % (name, why))
        return False
    write_run_json(run, step, rs)
    log('%s 済み: 学習の検証 best %.1f µs(%d エポック目)・last %.1f µs' % (name, res['best_val_us'], res['best_epoch'],
                                                                   hist[-1]['val_us']))
    return True


NOTE = {
    'ENS': '実機の精度の試し(real_accuracy_20261015)の ENS: v0 と msfusion の予測の平均の確かめのための新しい種。机上の比較用で、'
           '走行の標準はモデル B のまま',
    'SIZE': '実機の精度の試し(real_accuracy_20261015)の SIZE: 実機のきれいな走行のデータの量の曲線(学習シーンを減らした)。'
            '机上の測定用で、走行には使わない',
    'ROT': '実機の精度の試し(real_accuracy_20261015)の ROT: 間引き 3 の位相をエポックごとに回した学習。机上の比較用で、'
           '走行の標準はモデル B のまま',
    'SIZE_EQ': '実機の精度の試し(real_accuracy_20261015)の SIZE-EQ(固定の外の参考の腕): SIZE と同じ部分集合で、勾配のステップ数を'
               '全部のデータとそろえた(エポックを増やした)。机上の測定用で、走行には使わない',
    'GAINL': '実機の精度の試し(real_accuracy_20261015)の GAINL: 電池の効きをそろえたラベルでの学習(任意の腕)。机上の比較用で、'
             '走行には使わない(後処理と組で使う前提)',
}


def write_run_json(run, step, rs):
    """管理画面の一覧に出すための run.json(既にあれば書かない)"""
    if (run / 'run.json').exists():
        return
    sys.path.insert(0, str(REPO))
    from jetformer import policies      # torch を読まない
    a = step['cmd']['args']
    meta = {'dataset': Path(arg_value(a, '--data')).name, 'stride': int(arg_value(a, '--stride')),
            'epochs': int(arg_value(a, '--epochs')), 'batch': int(arg_value(a, '--batch')), 'lr': float(arg_value(a, '--lr')),
            'pretrained': True, 'build_engine': False, 'command_input': '--no-command-input' not in a, 'mask_impacts': True,
            'policy': arg_value(a, '--policy'), 'policy_cfg': policies.get(arg_value(a, '--policy')).resolve({}),
            'impact_g': float(arg_value(a, '--impact-g')), 'seed': int(arg_value(a, '--seed')), 'name': run.name,
            'created_at': rs.get('created_at'), 'note': NOTE[step['arm']], 'experiment': 'real_accuracy_20261015',
            'arm': step['arm'], 'train_env': GPU_ENV}
    if arg_value(a, '--drop-mask'):
        meta['drop_mask'] = arg_value(a, '--drop-mask')
    if a and a[0] != 'tools/train.py':
        meta['wrapper'] = a[0]
    write_json(run / 'run.json', meta)


# ---------------------------------------------------------------- そのほかの段

def eval_tool_ready():
    """評価の道具が書けていて、eval_sha256.txt の sha256 と一致するか"""
    if not EVAL_TOOL.exists() or not EVAL_SHA.exists():
        return False, '評価の道具か eval_sha256.txt がまだありません'
    want = {}
    for line in EVAL_SHA.read_text().splitlines():
        sp = line.split()
        if len(sp) == 2 and len(sp[0]) == 64:
            want[sp[1]] = sp[0]
    rel = str(EVAL_TOOL.relative_to(REPO))
    for k in (rel, str(EVAL_TOOL)):
        if k in want:
            if want[k] != sha256(EVAL_TOOL):
                return False, '評価の道具の sha256 が eval_sha256.txt と違います'
            return True, 'OK'
    return False, 'eval_sha256.txt に評価の道具がありません'


def wait_eval_tool():
    said = None
    while True:
        ok, why = eval_tool_ready()
        if ok:
            return
        if why != said:
            log('待ちます(%s)' % why)
            said = why
        time.sleep(60)


def do_crosscheck(state):
    out = REPO / 'data' / 'zoo' / 'real_accuracy_20261015' / 'crosscheck.json'
    cc = load(out)
    if not cc:
        wait_eval_tool()
        wait_gpu_free()
        need = wait_need(3200)
        code, sec = run_cmd(wrap_gpu([EVAL_TOOL, 'crosscheck'], '3G', need), LOGS / 'crosscheck.log', env_extra=GPU_ENV)
        cc = load(out)
        state.setdefault('steps', {})['crosscheck'] = {'exit': code, 'wall_s': sec, 'at': now()}
    if not cc or not cc.get('pass'):
        log('crosscheck が通りません(%s)。予測を止めます(調べて deviations.md へ)' % cc)
        return False
    log('crosscheck: 最大の差 %s µs(≤ 0.01)' % cc.get('max_abs_us'))
    return True


def verify_pred(name):
    """best・last の予測の npz を読み直す(key・有限・行数)→ (ok, 説明)。torch を読まない"""
    sys.path.insert(0, str(REPO / 'tools'))
    import numpy as np
    want = load(PROTOCOL)['data']
    probs = []
    for suf in ('', '_last'):
        p = PRED_DIR / ('%s%s.npz' % (name, suf))
        try:
            with np.load(p) as z:
                pred = z['pred']
                if str(z['key']) != want['pred_table_key']:
                    probs.append('%s の key が違う' % p.name)
                if pred.shape != (want['pred_table_rows'], 10, 2):
                    probs.append('%s の形 %s' % (p.name, pred.shape))
                if not np.isfinite(pred).all():
                    probs.append('%s に有限でない値' % p.name)
                if str(z['ckpt']) != ('best' if not suf else 'last'):
                    probs.append('%s の ckpt %s' % (p.name, z['ckpt']))
        except Exception as e:     # noqa: BLE001
            probs.append('%s が読めない(%s)' % (p.name, e))
    return (not probs), '; '.join(probs) or 'OK'


def do_predict(state, step):
    names = [n for n in step['names'] if (load(RUNS / n / 'result.json') or {}).get('state') == 'done'
             and (RUNS / n / 'best.pth').exists()]
    missing = [n for n in step['names'] if n not in names]
    if missing:
        log('%s: 学習が済んでいない %s は予測しません' % (step['id'], missing))
    todo = [n for n in names if not verify_pred(n)[0]]
    if todo:
        wait_eval_tool()
        wait_gpu_free()
        need = wait_need(3200)
        code, sec = run_cmd(wrap_gpu([EVAL_TOOL, 'predict', '--names'] + todo, '3G', need), LOGS / 'predict.log',
                            env_extra=GPU_ENV)
        state.setdefault('steps', {})[step['id']] = {'exit': code, 'wall_s': sec, 'at': now(), 'names': todo}
        save_state(state)
    ok_all = True
    for n in names:
        ok, why = verify_pred(n)
        rs = state.setdefault('runs', {}).setdefault(n, {})
        rs['pred_check'] = why
        if not ok:
            ok_all = False
            log('%s の予測の確かめが通りません: %s' % (n, why))
            continue
        lp = RUNS / n / 'last.pth'
        if lp.exists():
            sz = lp.stat().st_size
            lp.unlink()
            rs['last_pth_deleted'] = {'at': now(), 'bytes': sz}
            log('%s: best・last の予測を確かめたので last.pth(%.0f MB)を消しました' % (n, sz / 1e6))
    save_state(state)
    run_evaluate(state)
    return ok_all


def run_evaluate(state):
    """評価(CPU)→ evaluation.json・evaluation.md・progress.md。失敗しても GPU の段は続ける"""
    ok, why = eval_tool_ready()
    if not ok:
        log('評価はまだしません(%s)' % why)
        write_progress(state)
        return
    code, sec = run_cmd(wrap_cpu([EVAL_TOOL, 'evaluate']), LOGS / 'evaluate.log', env_extra=CPU_ENV)
    state.setdefault('steps', {})['evaluate_last'] = {'exit': code, 'wall_s': sec, 'at': now()}
    save_state(state)
    write_progress(state)


def do_rot_smoke(state, step):
    st = state.setdefault('steps', {})
    if (st.get('rot_smoke') or {}).get('ok'):
        return True
    c = step['cmd']
    out = Path(arg_value(c['args'], '--out'))
    if out.exists():
        shutil.rmtree(out)
    wait_gpu_free()
    need = wait_need(int(c['need_mb']))
    code, sec = run_cmd(wrap_gpu(c['args'], c['mem'], need), LOGS / 'rot_smoke.log', env_extra=GPU_ENV,
                        header='==== %s ROT の動作確認 ====' % now())
    phases = []
    if (out / 'rot_log.jsonl').exists():
        phases = [json.loads(l)['phase'] for l in (out / 'rot_log.jsonl').read_text().splitlines() if l.strip()]
    res = load(out / 'result.json') or {}
    ok = code == 0 and phases == [0, 1] and res.get('state') == 'done'
    st['rot_smoke'] = {'exit': code, 'wall_s': sec, 'phases': phases, 'state': res.get('state'), 'ok': ok, 'at': now()}
    if out.exists():
        shutil.rmtree(out)
    save_state(state)
    log('ROT の動作確認: %s(位相 %s・終了コード %d)。フォルダーを消しました' % ('OK' if ok else 'NG', phases, code))
    return ok


def do_gainl_smoke(state, step):
    st = state.setdefault('steps', {})
    if (st.get('gainl_smoke') or {}).get('ok'):
        return True
    if not GAINL_TOOL.exists():
        log('GAINL の道具 %s がありません。GAINL を削ります(任意の腕)' % GAINL_TOOL)
        return False
    out = RUNS / (PREFIX + 'gainl-smoke')
    if out.exists():
        shutil.rmtree(out)
    args = [str(GAINL_TOOL.relative_to(REPO)), '--policy', 'jetformer_v0', '--data', 'data/datasets/clean_driving_20261011',
            '--epochs', '2', '--stride', '3', '--batch', '64', '--lr', '3e-4', '--workers', '4', '--no-command-input',
            '--impact-g', '2.5', '--max-steps', '2', '--out', str(out), '--seed', '0']
    wait_gpu_free()
    cap = (state.get('family_cap') or {}).get('jetformer_v0') or '3G'      # 22:43 に 3G で止まったので同じ種類の上限を使う
    need = wait_need(3200 + max(0, mem_to_mb(cap) - mem_to_mb('3G')))
    code, sec = run_cmd(wrap_gpu(args, cap, need), LOGS / 'gainl_smoke.log', env_extra=GPU_ENV,
                        header='==== %s GAINL の動作確認 ====' % now())
    res = load(out / 'result.json') or {}
    info = load(out / 'gainl.json') or {}
    ok = code == 0 and res.get('state') == 'done' and bool(info.get('label_check_ok'))
    st['gainl_smoke'] = {'exit': code, 'wall_s': sec, 'state': res.get('state'), 'label_check_ok': info.get('label_check_ok'),
                         'ok': ok, 'at': now()}
    if out.exists():
        shutil.rmtree(out)
    save_state(state)
    log('GAINL の動作確認: %s' % ('OK' if ok else 'NG'))
    return ok


def do_gate(state):
    st = state.setdefault('steps', {})
    if GATE_JSON.exists():
        return True
    if not GATE_TOOL.exists():
        log('ゲートの道具 %s がありません。待ちます(最大 60 分)' % GATE_TOOL)
        t0 = time.time()
        while not GATE_TOOL.exists() and time.time() - t0 < 3600:
            time.sleep(60)
        if not GATE_TOOL.exists():
            log('ゲートの道具がまだありません。ゲートを後回しにします')
            return False
    wait_gpu_free()
    need = wait_need(3200)
    code, sec = run_cmd(wrap_gpu([GATE_TOOL, 'all'], '3G', need), LOGS / 'ens_gate.log', env_extra=GPU_ENV)
    st['gate'] = {'exit': code, 'wall_s': sec, 'at': now(), 'gate_json': GATE_JSON.exists()}
    save_state(state)
    return GATE_JSON.exists()


# ---------------------------------------------------------------- progress.md

def run_status_ja(rs):
    """progress.md の「状態」: 確かめが通った → 済み、削った(電源断などで未完のまま打ち切った)→ 削った、ほか → 学習中・未完"""
    if rs.get('check_ok'):
        return '済み'
    if rs.get('cut'):
        return '削った(%s)' % rs['cut'] if isinstance(rs.get('cut'), str) else '削った'
    return '学習中・未完'


def write_progress(state):
    """progress.md(腕・種・学習の検証・保留の L1 best/last・今の判定)"""
    ev = load(REPORT / 'evaluation.json') or {}
    per = (ev.get('per_run') or {})
    lines = ['# 進み具合(real_accuracy_20261015)', '',
             '- 更新: %s(realacc_driver.py が自動で書く)' % now(),
             '- 学習の検証 = 学習の道具の val(間引き 3 の 1,559 サンプル)。保留の L1 = 主指標(人のきれいな 4,670 フレーム)',
             '- 基準 A = 49.146 / 49.457 µs(best)・49.235 / 49.752 µs(last)。閾値 best 0.626・last 1.322 µs', '',
             '| 学習結果 | 腕 | 状態 | 学習の検証 best / last [µs] | 保留の L1 best / last [µs] | 試行・再開 | 学習の時間 |',
             '|---|---|---|---|---|---|---|']
    for name, rs in (state.get('runs') or {}).items():
        if not name.startswith(PREFIX):
            continue
        pr = per.get(name) or {}
        lb, ll = pr.get('L_best'), pr.get('L_last')
        lines.append('| %s | %s | %s | %s / %s | %s / %s | %d・%s | %s |' % (
            name.replace(PREFIX, ''), rs.get('arm', ''), run_status_ja(rs),
            rs.get('best_val_us', '—'), rs.get('last_val_us', '—'), '—' if lb is None else '%.3f' % lb,
            '—' if ll is None else '%.3f' % ll, len(rs.get('attempts') or []), rs.get('resumes', 0),
            '%.0f 分' % (rs['train_wall_s'] / 60) if rs.get('train_wall_s') else '—'))
    vd = ev.get('verdicts_table') or []
    if vd:
        lines += ['', '## 今の判定(evaluation.json から)', '', '| 問い | best | last | 最終 | 平均の差 best [µs] |', '|---|---|---|---|---|']
        for v in vd:
            lines.append('| %s | %s | %s | **%s** | %s |' % (v.get('question'), v.get('best'), v.get('last'), v.get('final'),
                                                         v.get('gain_best_us')))
    steps = state.get('steps') or {}
    if steps:
        lines += ['', '## 段', '']
        for k, v in steps.items():
            lines.append('- %s: %s' % (k, json.dumps(v, ensure_ascii=False)))
    cuts = state.get('cuts') or []
    if cuts:
        lines += ['', '## 削った段', ''] + ['- %s' % c for c in cuts]
    lines += ['', '空き容量: %.1f GB' % disk_free_gb(), '']
    tmp = str(PROGRESS) + '.tmp'
    Path(tmp).write_text('\n'.join(lines), encoding='utf-8')
    os.replace(tmp, PROGRESS)


# ---------------------------------------------------------------- 本体

def ens_better(state):
    ev = load(REPORT / 'evaluation.json') or {}
    return ((ev.get('verdicts') or {}).get('ENS_vs_A') or {}).get('final_class') == 'better'


def cmd_all(state, posthoc=False):
    proto, sha = load_protocol()
    state['protocol_sha256'] = sha
    steps = build_posthoc_steps(proto) if posthoc else build_steps(proto)
    latest = POSTHOC_LATEST_START if posthoc else OPTIONAL_LATEST_START
    done = set(state.setdefault('done_steps', []))
    cuts = state.setdefault('cuts', [])
    if not state.get('wrap_deviation_logged'):
        add_deviation('学習・予測のコマンドに、メモリの扱いだけの包み(THP なし nothp.py・MALLOC_ARENA_MAX 2・expandable_segments・'
                      'choom 1000・全体の空きメモリの見張り)を足した。17:44 に msf-s2 が protocol のままの包み(上限 3800M)で 1 エポック目の'
                      '前にメモリの上限で止まったため(10/14〜10/15 の学習と同じ包み。計算の結果は変わらない)。python は jetracer の環境の '
                      '%s' % PY)
        state['wrap_deviation_logged'] = True
    save_state(state)
    for i, step in enumerate(steps):
        sid = step['id']
        if sid in done:
            continue
        if any(c.startswith(sid + ':') or c.startswith('group ' + step['group'] + ':') for c in cuts):
            continue
        t = jst_now()
        free = disk_free_gb()
        if free < DISK_STOP_GB:
            add_deviation('空き容量 %.2f GB < %.1f GB なので GPU の段を止めた(%s の前)' % (free, DISK_STOP_GB, sid))
            save_state(state)
            return 1
        if step['optional']:
            why = None
            if free < DISK_CUT_OPTIONAL_GB:
                why = '空き容量 %.2f GB < %.1f GB' % (free, DISK_CUT_OPTIONAL_GB)
            elif t > latest and step['group'] not in state.setdefault('groups_started', []):
                why = '%s を過ぎた' % latest.strftime('%H:%M')
            if why:
                cuts.append('group %s: 任意の段を削った(%s・%s)' % (step['group'], why, now()))
                log('任意の段 %s を削ります(%s)' % (step['group'], why))
                save_state(state)
                continue
        rem = group_remaining_minutes(steps, i, done)
        if not fits_before(t, rem, GPU_HARD_STOP):
            if step['group'] == 'GATE' and ens_better(state):
                log('ENS が「良い」なのでゲートは削りません(06:00 を過ぎる見込みでも始める)')
            else:
                cuts.append('group %s: 06:00 までに終わらない見込み(残り %.0f 分・%s)' % (step['group'], rem, now()))
                log('段 %s(%s)を削ります: 06:00 までに終わらない見込み' % (sid, step['group']))
                save_state(state)
                continue
        log('==== 段 %d/%d: %s(見積もり %.0f 分・空き %.1f GB)====' % (i + 1, len(steps), sid, step_minutes(step), free))
        if step['group'] not in state.setdefault('groups_started', []):
            state['groups_started'].append(step['group'])
            save_state(state)
        typ = step['type']
        if typ == 'train':
            ok = train_one(state, step)
            if ok:
                write_progress(state)
        elif typ == 'crosscheck':
            ok = do_crosscheck(state)
        elif typ == 'predict':
            ok = do_predict(state, step)
        elif typ == 'rot_smoke':
            ok = do_rot_smoke(state, step)
            if not ok:
                cuts.append('group ROT: 動作確認が通らなかった(%s)' % now())
                add_deviation('ROT の動作確認が通らなかったので ROT の腕を学習しなかった(state.json の steps.rot_smoke)')
                save_state(state)
                continue
        elif typ == 'gainl_smoke':
            ok = do_gainl_smoke(state, step)
            if not ok:
                cuts.append('group GAINL: 道具が無い・動作確認が通らなかった(%s)' % now())
                save_state(state)
                continue
        elif typ == 'gate':
            ok = do_gate(state)
            if not ok:
                cuts.append('gate: ゲートの道具が間に合わない・失敗(%s)' % now())
                save_state(state)
                continue
        else:
            raise SystemExit('知らない段 %s' % typ)
        if not ok:
            log('段 %s が済みませんでした。止めます(調べてから同じコマンドで続きから)' % sid)
            save_state(state)
            write_progress(state)
            return 1
        done.add(sid)
        order = [x['id'] for x in build_steps(proto) + build_posthoc_steps(proto)]
        state['done_steps'] = sorted(done, key=lambda s: order.index(s) if s in order else len(order))
        save_state(state)
    log('全部の段が済みました(削った段: %s)' % cuts)
    run_evaluate(state)
    return 0


def cmd_status(state):
    print(json.dumps({k: state.get(k) for k in ('updated', 'done_steps', 'cuts', 'steps')}, ensure_ascii=False, indent=1))
    for n, rs in (state.get('runs') or {}).items():
        print(n, {k: rs.get(k) for k in ('best_val_us', 'last_val_us', 'check', 'pred_check', 'mem_cap')},
              'attempts', len(rs.get('attempts') or []))
    print('空き容量 %.2f GB・空きメモリ %d MB' % (disk_free_gb(), mem_available_mb()))


def cmd_plan(_state):
    proto, _ = load_protocol()
    for s in build_steps(proto):
        extra = ' '.join(wrap_gpu(s['cmd']['args'], s['cmd']['mem'], s['cmd']['need_mb'])) if 'cmd' in s else s.get('names', '')
        print('%-34s %-10s 任意=%s 見積もり %4.0f 分  %s' % (s['id'], s['group'], s['optional'], step_minutes(s), extra))


def main(argv):
    if len(argv) < 2 or argv[1] not in ('all', 'posthoc', 'status', 'plan', 'progress'):
        raise SystemExit(__doc__)
    LOGS.mkdir(parents=True, exist_ok=True)
    if argv[1] in ('all', 'posthoc'):
        lock = open('/tmp/realacc_driver.lock', 'w')
        import fcntl
        if argv[1] == 'all':
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                raise SystemExit('realacc_driver.py all はもう動いています')
        else:                   # posthoc は all が終わるのを待ってから(GPU を重ねない)
            log('posthoc: all が終わるのを待ちます')
            fcntl.flock(lock, fcntl.LOCK_EX)
        state = load(STATE_PATH, {}) or {}
        return cmd_all(state, posthoc=(argv[1] == 'posthoc'))
    state = load(STATE_PATH, {}) or {}
    if argv[1] == 'progress':
        write_progress(state)
        return 0
    return {'status': cmd_status, 'plan': cmd_plan}[argv[1]](state) or 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
