#!/usr/bin/env python3
"""キュー v2(C4、2026-10-25): gpu_queue_v2.sh の手伝い。torch を読まない・GPU を使わない・車と装置に触れない。
experiments/20261021_bench_prep/gpu_queue.py(10/21 段 G2)と 20261022_prep/gpu_queue2.py の写し。足したこと(plan.md 4.5 の C4):
  - 表(gpu_queue_v2.tsv)は protocol_v2.json の runs から作る(tsv)。順番は B-FAST の 3 本が先、そのあと v2 の 19 本(protocol の順)
  - 名前・コマンドの {x}・{E}・{A}・{X_DATA}・{AUG_ARGS} は、その行に来たときに decisions.json(D1・D2・D3 の答え)で埋める。
    段 2 は答えが無ければ先回りで x = v2h(protocol の runs の note)。段 3・4 は答えが無ければそこで止まる
  - 始める前の確かめ(preflight): protocol の固定・封・表とデータセットの sha256・B の重み・data/modes.json のジャンル 9・
    表の行が protocol と同じか・データの確かめ(data_checks.json)で外れたものを main session が了承したか・ディスク
  - 学習ごと: 終わりの見込み(plan.md 4.2 の式 × 1.2)が締め切り(既定 2026-10-31 06:00)を超えるなら始めない。空きディスク 5 GB 未満なら止める
  - 読めない last.pth(zip の CRC が合わない)は脇へ動かして最初から。再開・メモリの上限の引き上げは deviations.md に書き足す
  - 走行中(GET /api/drive/status の running・recording)と管理画面の学習・収集中(GET /api/status)は待つ(GET だけ)

    python -I gpu_queue_v2.py tsv                      # protocol から gpu_queue_v2.tsv を作り直す(表を手で直したときは使わない)
    python -I gpu_queue_v2.py preflight                # 始める前の確かめ(合格で 0)
    python -I gpu_queue_v2.py templates                # 表の行(# で消していない行)の名前を順に
    python -I gpu_queue_v2.py resolve <行の名前>        # RUN<TAB>名前<TAB>コマンド / SKIP<TAB>理由 / STOP<TAB>理由
    python -I gpu_queue_v2.py trained <名前> <コマンド>   # 学習が済んでいれば 0
    python -I gpu_queue_v2.py fits <名前> <コマンド>      # 締め切りとディスクで始めてよければ 0(だめなら理由を出して 1)
    python -I gpu_queue_v2.py predicted <名前>          # eval_v2.py predict の予測(best・last)があれば 0
    python -I gpu_queue_v2.py prune-last [--yes <名前>...] # 予測を保存した学習の last.pth(消してよい一覧 / 消す。既定は消さない)
    python -I gpu_queue_v2.py lastpth <名前>            # resume / fresh / broken(読めない last.pth を脇へ動かした)
    python -I gpu_queue_v2.py need-engine <名前> / need-check <名前>
    python -I gpu_queue_v2.py begin <名前> <段> [メモ] / end <名前> <段> <rc> / fails <名前> <段> / last-rc <名前> <段>
    python -I gpu_queue_v2.py runjson <名前> <コマンド>   # run.json(無いときだけ。管理画面の一覧用)
    python -I gpu_queue_v2.py enginecheck <名前>        # logs/engine_check_<名前>.json → data/runs/<名前>/engine_check.json
    python -I gpu_queue_v2.py drivable <名前>           # drive.model_info で drivable(合格で 0)
    python -I gpu_queue_v2.py webbusy                  # 走行・収集・管理画面の学習中なら 0
    python -I gpu_queue_v2.py deviation <見出し> <本文>  # deviations.md に書き足す(本番だけ。試しは書かない)
    python -I gpu_queue_v2.py estimate                 # 時間の見込み(行ごと・合計。x = v2h と v2m)
    python -I gpu_queue_v2.py status / summary

試し(GPU_QUEUE_SMOKE=1): 先頭の 1 行だけ(GPU_QUEUE_SMOKE_ROW=<行の名前の型> で別の行)。名前は _smoke_<名前>、--epochs 1 と --max-steps(GPU_QUEUE_SMOKE_STEPS、既定 5)。
状態とログは logs/smoke/。deviations.md には書かない。
状態は logs/gpu_queue_v2_state.json(tmp → fsync → 置き換え。電源断でも半端に残らない)。
"""

import datetime
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

REPO = Path('/home/jetson/jetformer')
sys.path.insert(0, str(REPO))
HERE = REPO / 'experiments' / '20261025_train_v2'
REP = Path('/home/jetson/jetformer_reports/train_v2_20261025')
PROTOCOL = REP / 'protocol_v2.json'
PROTOCOL_SHA = REP / 'protocol_v2.sha256'
DECISIONS = Path(os.environ.get('GPU_QUEUE_DECISIONS') or REP / 'decisions.json')   # 試しのときだけ別のファイル
DEVIATIONS = REP / 'deviations.md'
DATA_CHECKS = REP / 'data_checks.json'
LOCK2 = REP / 'protocol_v2_lock2.json'
QUEUE = HERE / 'gpu_queue_v2.tsv'
SMOKE = os.environ.get('GPU_QUEUE_SMOKE') == '1'
SMOKE_STEPS = int(os.environ.get('GPU_QUEUE_SMOKE_STEPS') or 5)
LOGS = HERE / 'logs' / ('smoke' if SMOKE else '')
STATE = LOGS / 'gpu_queue_v2_state.json'
SUMMARY = LOGS / 'gpu_queue_v2_summary.json'
RUNS = REPO / 'data' / 'runs'
TABLES = REPO / 'data' / 'derived' / 'dataset_v2_20261025'
DATASET = REPO / 'data' / 'datasets' / 'dataset_v2_20261025'
MODES = REPO / 'data' / 'modes.json'
B_RUN = RUNS / 'clean_driving_20261011-nocmd-s3-20261011-024626'
ENGINE = 'jetformer_fp16.engine'
GATE_US = 2.5
DEADLINE = os.environ.get('GPU_QUEUE_DEADLINE') or '2026-10-31 06:00'      # protocol start_conditions.deadline
DISK_MIN_GB = float(os.environ.get('GPU_QUEUE_DISK_MIN_GB') or 5.0)         # protocol start_conditions.disk
WEB = (os.environ.get('GPU_QUEUE_WEB') or 'http://127.0.0.1:8080').rstrip('/')
PREFIX = 'cd ~/jetformer && '
TABLE_ARGS = ('--drop-mask', '--sample-weights-npz', '--mode-remap-npz', '--label-override-npz', '--init')
CODE_FILES = ('tools/train.py', 'jetformer/train_data.py', 'jetformer/dataset.py', 'tools/build_engine.py',
              'jetformer/trt_runtime.py', 'jetformer/policies/jetformer_v0.py', 'jetformer/policies/msfusion.py')
# 外れても学習を止めないデータの確かめ(protocol stage0_checks 7: 11/3 に出す前にユーザーに聞く)
DATA_FAILS_NOT_BLOCKING = ('action_reference_within_2us_of_B',)

# plan.md 4.2 の時間の式(protocol_numbers.py と同じ)+ 1 本ごとのエンジン・確認・あける時間
MS_PER_TRAIN = {'jetformer_v0': 4.96e-3, 'msfusion': 7.81e-3}
VAL_SAMPLES = 1559
MS_PER_VAL = 1.54e-3
EPOCH_FIXED_S = 1.9
PREP_S = 4.5 * 60
AUG2_FACTOR = 0.71 / 0.69
SAFETY = 1.2                     # protocol start_conditions.deadline の × 1.2
POST_S = 90 + 30 + 15 + 3 * 60   # エンジン 約 90 秒・確認 約 30 秒・走行の一覧 約 15 秒・あける時間 60 秒 × 3(10/23 の実績)
# 学習が読むサンプル数(data_checks.json。plan.md の数ではない)。base は B と同じ 5,912
SAMPLES_FALLBACK = {'v2h': 6506, 'v2m': 7638, 'bfast': 6697, 'base': 5912}


def now():
    return time.strftime('%Y-%m-%d %H:%M:%S')


def load(path, default=None):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
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


def protocol():
    return load(PROTOCOL)


def arg_value(args, key):
    if key in args:
        i = args.index(key)
        return args[i + 1] if i + 1 < len(args) else None
    return None


def train_args(cmd):
    """コマンド → tools/train.py の引数(train.py より後ろ)"""
    a = shlex.split(cmd)
    return a[a.index('tools/train.py') + 1:]


# ---------------------------------------------------------------- 表(gpu_queue_v2.tsv)

def protocol_rows():
    """protocol の runs → [(段, 名前の型, コマンドの型, run)]。B-FAST が先、そのあと v2 の 19 本(protocol の順)"""
    runs = protocol()['runs']
    bf = [r for r in runs if r['family'] in ('bfast', 'B-FAST') or str(r['stage']) == 'B-FAST']
    v2 = [r for r in runs if r not in bf]
    out = []
    for r in bf + v2:
        cmd = r['command']
        if not cmd.startswith(PREFIX):
            raise SystemExit('protocol の %s のコマンドが %r で始まっていません' % (r['name'], PREFIX))
        out.append((str(r['stage']), r['name'], cmd[len(PREFIX):], r))
    return out


def cmd_tsv():
    rows = protocol_rows()
    if len(rows) != 22 or len({r[1] for r in rows}) != 22:
        raise SystemExit('protocol の runs が 22 本(名前が別々)ではありません')
    lines = [
        '# キュー v2(C4、2026-10-25)の表。1 行 1 本: 段<TAB>名前<TAB>コマンド(~/jetformer で実行)。gpu_queue_v2.py tsv が '
        'protocol_v2.json の runs から作った(コマンドは protocol の command から "cd ~/jetformer && " を除いたもの)',
        '# 順番: B-FAST の 3 本(種 0・1・2)が先、そのあと v2 の 19 本(protocol の順。段 1 → 2 → 3 → 4)。'
        '削るときは行の頭に # を付ける(protocol cut_order の順。削ったことは deviations.md に書く)',
        '# {x}・{E}・{A}・{X_DATA}・{AUG_ARGS} はその行に来たときに %s の D1.x・D2.E・D3.aug で埋める。'
        '段 2 は D1 が無ければ先回りで x = v2h。段 3・4 は答えが無ければそこで止まる' % DECISIONS,
    ]
    for stage, name, cmd, _ in rows:
        lines.append('%s\t%s\t%s' % (stage, name, cmd))
    QUEUE.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('書きました: %s(%d 行)' % (QUEUE, len(rows)))
    return 0


def queue_rows(include_commented=False):
    """gpu_queue_v2.tsv → [(段, 名前の型, コマンドの型, 消してある)]"""
    out = []
    for line in QUEUE.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        commented = line.startswith('#')
        body = line.lstrip('#').strip() if commented else line
        parts = body.split('\t')
        if len(parts) != 3:
            continue                        # 注の行
        stage, name, cmd = (p.strip() for p in parts)
        if commented and not include_commented:
            continue
        out.append((stage, name, cmd, commented))
    return out


def check_tsv():
    """表の行(消した行も)が protocol の runs と同じか → 違いの一覧"""
    want = {name: (stage, cmd) for stage, name, cmd, _ in protocol_rows()}
    bad = []
    seen = []
    for stage, name, cmd, _ in queue_rows(include_commented=True):
        seen.append(name)
        if name not in want:
            bad.append('protocol に無い行: %s' % name)
        elif want[name] != (stage, cmd):
            bad.append('protocol とコマンドか段が違う: %s' % name)
    if sorted(seen) != sorted(want):
        bad.append('行の数・名前が protocol と違う(%d 行)' % len(seen))
    order = [n for _, n, _, _ in protocol_rows()]
    if seen != order:
        bad.append('行の順番が B-FAST → v2(protocol の順)と違う')
    return bad


# ---------------------------------------------------------------- 型を埋める(decisions.json)

def decisions():
    d = load(DECISIONS, {}) or {}
    x = ((d.get('D1') or {}).get('x'))
    e = ((d.get('D2') or {}).get('E'))
    aug = ((d.get('D3') or {}).get('aug'))
    return {'x': x, 'E': e, 'aug': aug}


def fill(stage, name, cmd, dec, proto):
    """→ ('RUN', 名前, コマンド) / ('STOP', 理由)"""
    ph = proto['choice_placeholders']
    need = set(re.findall(r'\{(\w+)\}', name + ' ' + cmd))
    if not need:
        return ('RUN', name, cmd)
    vals = {}
    if need & {'x', 'X_DATA'}:
        x = dec['x']
        if x is None:
            if stage == '2':
                x = 'v2h'              # protocol: 段 2 は先回りで x = v2h
            else:
                return ('STOP', 'D1(データ x)の答えが %s にありません。10/26 朝の D1 のあとに書いて、同じコマンドをもう一度' % DECISIONS)
        if x not in ph['{X_DATA}']:
            return ('STOP', 'D1.x = %r は v2h・v2m・base のどれでもありません' % x)
        vals['x'], vals['X_DATA'] = x, ph['{X_DATA}'][x]
    if 'E' in need:
        e = dec['E']
        if e is None:
            return ('STOP', 'D2(エポック E)の答えが %s にありません。10/26 朝の D2 のあとに書いて、同じコマンドをもう一度' % DECISIONS)
        if int(e) not in (30, 60):
            return ('STOP', 'D2.E = %r は 30 か 60 ではありません' % e)
        vals['E'] = str(int(e))
    if need & {'A', 'AUG_ARGS'}:
        aug = dec['aug']
        if aug is None:
            return ('STOP', 'D3(拡張)の答えが %s にありません。10/26 夕の D3 のあとに書いて、同じコマンドをもう一度' % DECISIONS)
        if aug not in ph['{AUG_ARGS}']:
            return ('STOP', 'D3.aug = %r は AUG2 か なし ではありません' % aug)
        vals['AUG_ARGS'] = ph['{AUG_ARGS}'][aug]
        vals['A'] = '-aug2' if aug == 'AUG2' else ''
    left = need - set(vals)
    if left:
        return ('STOP', '埋められない型: %s' % ' '.join(sorted(left)))
    for k, v in vals.items():
        name = name.replace('{%s}' % k, v)
        cmd = cmd.replace('{%s}' % k, v)
    cmd = ' '.join(cmd.split())             # {AUG_ARGS} が空のときの二重の空白
    return ('RUN', name, cmd)


def smoke_row():
    """試しの行(GPU_QUEUE_SMOKE_ROW の名前の型。既定は先頭の行)"""
    return os.environ.get('GPU_QUEUE_SMOKE_ROW') or queue_rows()[0][1]


def smoke(name, cmd):
    """試し: 名前 _smoke_<名前>・--epochs 1・--max-steps"""
    sname = '_smoke_' + name
    out = '--out data/runs/%s' % name
    if out not in cmd:
        raise SystemExit('試し: コマンドに %r がありません' % out)
    c = cmd.replace(out, '--out data/runs/%s' % sname)
    c = re.sub(r'--epochs \d+', '--epochs 1', c)
    return sname, c + ' --max-steps %d' % SMOKE_STEPS


def with_workers(cmd):
    """GPU_QUEUE_WORKERS=N なら --workers 4 を N に(protocol common_recipe.workers: 段 0 のメモリの最大が 3.5 GB を超えたら 2)。
    train.py の --resume は workers を比べないので、途中で変えても続きから学習できる"""
    w = (os.environ.get('GPU_QUEUE_WORKERS') or '').strip()
    if not w:
        return cmd
    if w not in ('1', '2', '3', '4'):
        raise SystemExit('GPU_QUEUE_WORKERS は 1〜4: %r' % w)
    return re.sub(r'--workers \d+', '--workers %s' % w, cmd)


def train_py_flags():
    src = (REPO / 'tools' / 'train.py').read_text(encoding='utf-8')
    return set(re.findall(r"add_argument\('(--[\w-]+)'", src))


def resolve(template):
    rows = queue_rows(include_commented=True)
    hit = [r for r in rows if r[1] == template]
    if not hit:
        return ('SKIP', '表に無い行になった')
    stage, name, cmd, commented = hit[0]
    if commented:
        return ('SKIP', '表で # が付いた(削った)')
    if SMOKE and template != smoke_row():
        return ('SKIP', '試しは 1 行だけ')
    r = fill(stage, name, cmd, decisions(), protocol())
    if r[0] != 'RUN':
        return r
    _, name, cmd = r
    cmd = with_workers(cmd)
    if SMOKE:
        name, cmd = smoke(name, cmd)
    missing = sorted({a for a in train_args(cmd) if a.startswith('--')} - train_py_flags())
    if missing:
        return ('STOP', 'tools/train.py にまだ無い引数: %s(C1 など。足してテストしたあとに同じコマンドをもう一度)' % ' '.join(missing))
    for key in TABLE_ARGS:
        v = arg_value(train_args(cmd), key)
        if v and not (REPO / v).exists():
            return ('STOP', '%s のファイルがありません: %s' % (key, v))
    return ('RUN', name, cmd)


def cmd_resolve(template):
    r = resolve(template)
    print('\t'.join(r))
    return 0


def cmd_templates():
    for _, name, _, _ in queue_rows():
        print(name)
    return 0


# ---------------------------------------------------------------- 済み・再開

def epochs_in(cmd):
    return int(arg_value(train_args(cmd), '--epochs') or 30)


def is_trained(name, cmd):
    run = RUNS / name
    res = load(run / 'result.json')
    return bool(res and res.get('state') == 'done' and res.get('epochs_done') == epochs_in(cmd)
                and all((run / f).exists() for f in ('best.pth', 'last.pth', 'jetformer.onnx', 'model_config.json')))


def lastpth_state(name):
    """last.pth: 無い → fresh、zip として全部読めて CRC が合う → resume、読めない → 脇へ動かして broken"""
    p = RUNS / name / 'last.pth'
    if not p.exists():
        return 'fresh', None
    try:
        with zipfile.ZipFile(p) as z:
            bad = z.testzip()
            names = z.namelist()
        if bad is None and any(n.endswith('data.pkl') for n in names):
            return 'resume', None
        why = 'CRC が合わない: %s' % bad if bad else 'data.pkl が無い'
    except (zipfile.BadZipFile, OSError, EOFError) as e:
        why = '%s: %s' % (type(e).__name__, e)
    moved = p.with_name('last.pth.broken-%s' % time.strftime('%Y%m%d-%H%M%S'))
    os.replace(p, moved)
    return 'broken', '%s(%s へ動かした)' % (why, moved.name)


def cmd_lastpth(name):
    st, why = lastpth_state(name)
    print(st if why is None else '%s\t%s' % (st, why))
    return 0


def need_engine(name):
    run = RUNS / name
    eng, onnx = run / ENGINE, run / 'jetformer.onnx'
    return (not eng.exists()) or eng.stat().st_mtime_ns < onnx.stat().st_mtime_ns


def check_current(run):
    """engine_check.json が今のエンジン(大きさ・更新時刻)のものなら dict(drive.engine_check_of と同じ規則)"""
    c = load(run / 'engine_check.json')
    eng = run / ENGINE
    if not isinstance(c, dict) or not eng.exists():
        return None
    st = eng.stat()
    e = c.get('engine') or {}
    if int(e.get('size', -1)) != st.st_size or int(e.get('mtime_ns', -1)) != st.st_mtime_ns:
        return None
    return c


# ---------------------------------------------------------------- 時間の見込み

def samples_of(cmd):
    dc = load(DATA_CHECKS, {}) or {}
    arms = dc.get('arms') or {}
    a = train_args(cmd)
    data = arg_value(a, '--data') or ''
    drop = arg_value(a, '--drop-mask') or ''
    if data.endswith('clean_driving_20261011'):
        return SAMPLES_FALLBACK['base']
    for key in ('v2h', 'v2m', 'bfast'):
        if drop.endswith('drop_%s.npz' % key):
            return int((arms.get(key) or {}).get('samples') or SAMPLES_FALLBACK[key])
    return SAMPLES_FALLBACK['v2m']


def hours_of(cmd, epochs_left=None):
    """plan.md 4.2 の式(学習だけ)[時間]"""
    a = train_args(cmd)
    policy = arg_value(a, '--policy') or 'jetformer_v0'
    n = samples_of(cmd)
    epoch_s = n * MS_PER_TRAIN.get(policy, MS_PER_TRAIN['msfusion']) + VAL_SAMPLES * MS_PER_VAL + EPOCH_FIXED_S
    e = epochs_in(cmd) if epochs_left is None else epochs_left
    h = (e * epoch_s + PREP_S) / 3600.0
    if '--aug-shift-px' in a:
        h *= AUG2_FACTOR
    return h


def epochs_left(name, cmd):
    """残りのエポック。last.pth があれば train.log の最後の「epoch N」の行(電源断でも残る)と result.json から"""
    run = RUNS / name
    done = 0
    if (run / 'last.pth').exists():
        try:
            hits = re.findall(r'^epoch +(\d+) ', (run / 'train.log').read_text(encoding='utf-8', errors='replace'), re.M)
            done = int(hits[-1]) if hits else 0
        except OSError:
            done = 0
        res = load(run / 'result.json') or {}
        if res.get('epochs_done'):
            done = max(done, int(res['epochs_done']))
    return max(epochs_in(cmd) - done, 0)


def cmd_fits(name, cmd):
    left = epochs_left(name, cmd)
    h = hours_of(cmd, left) * SAFETY
    end = datetime.datetime.now() + datetime.timedelta(hours=h)
    dl = datetime.datetime.strptime(DEADLINE, '%Y-%m-%d %H:%M')
    free_gb = shutil.disk_usage(str(RUNS)).free / 1e9
    if free_gb < DISK_MIN_GB:
        print('空きディスク %.1f GB < %.1f GB(protocol start_conditions.disk)' % (free_gb, DISK_MIN_GB))
        return 1
    if end > dl:
        print('見込み %.2f 時間(× %.1f)で %s に終わり、締め切り %s を超える(protocol start_conditions.deadline)' % (
            h, SAFETY, end.strftime('%m/%d %H:%M'), DEADLINE))
        return 1
    print('見込み %.2f 時間(× %.1f、残り %d エポック)・%s ごろまで・空きディスク %.1f GB' % (
        h, SAFETY, left, end.strftime('%m/%d %H:%M'), free_gb))
    return 0


def cmd_estimate():
    """行ごとの見込み(学習 + エンジン・確認・あける時間)。x = v2h(既定)と v2m、E = 60 と 30、拡張 = AUG2"""
    proto = protocol()
    print('| 順 | 段 | 名前 | サンプル | エポック | 学習 [時間] | エンジン・確認・あける [時間] | 合計 [時間] |')
    print('|---|---|---|---|---|---|---|---|')
    out = {}
    for x in ('v2h', 'v2m'):
        for e in (60, 30):
            dec = {'x': x, 'E': e, 'aug': 'AUG2'}
            tot = 0.0
            rows = []
            for i, (stage, name, cmd, _) in enumerate(queue_rows(include_commented=True), 1):
                r = fill(stage, name, cmd, dec, proto)
                h = hours_of(r[2])
                tot += h + POST_S / 3600.0
                rows.append((i, stage, r[1], samples_of(r[2]), epochs_in(r[2]), h))
            out['x=%s,E=%d' % (x, e)] = round(tot, 2)
            if x == 'v2h' and e == 60:
                for i, stage, n, s, ep, h in rows:
                    print('| %d | %s | %s | %d | %d | %.2f | %.2f | %.2f |' % (i, stage, n, s, ep, h, POST_S / 3600.0,
                                                                         h + POST_S / 3600.0))
                print('| | | **合計(x = v2h・E = 60)** | | | %.2f | %.2f | **%.2f** |' % (
                    sum(r[5] for r in rows), len(rows) * POST_S / 3600.0, tot))
    pre = sum(hours_of(fill(s, n, c, {'x': 'v2h', 'E': 60, 'aug': 'AUG2'}, proto)[2]) + POST_S / 3600.0
              for s, n, c, _ in queue_rows(include_commented=True) if s == '2')
    out['x=v2m,E=60(先回りの v2h の段 2 を含む)'] = round(out['x=v2m,E=60'] + pre, 2)
    print()
    print('合計の見込み [時間]: %s' % json.dumps(out, ensure_ascii=False))
    print('x = v2m のときは、先回りの段 2(v2h の 5 本、%.2f 時間)も先に回るので、合計 %.2f 時間' % (pre, out['x=v2m,E=60'] + pre))
    print('式: plan.md 4.2(1 エポック = サンプル × 4.96 ms(多段型 7.81 ms)+ 1,559 × 1.54 ms + 1.9 秒、準備 4.5 分、AUG2 × %.3f)。'
          'サンプルは data_checks.json の学習が読む数。1 本ごとに +%.0f 分(エンジン・確認・60 秒 × 3)' % (AUG2_FACTOR, POST_S / 60))
    return 0


# ---------------------------------------------------------------- 試みの記録

def state():
    return load(STATE, {'runs': {}})


def cmd_begin(name, step, note=''):
    s = state()
    r = s['runs'].setdefault(name, {})
    a = {'started': now(), 't0': time.time()}
    if note:
        a['note'] = note
    r.setdefault(step, []).append(a)
    s['updated'] = now()
    write_json(STATE, s)


def cmd_end(name, step, rc, cmd=None):
    s = state()
    atts = s['runs'].setdefault(name, {}).setdefault(step, [])
    if not atts or 'ended' in atts[-1]:
        atts.append({'started': None, 't0': None})
    a = atts[-1]
    a.update({'ended': now(), 'rc': int(rc), 'wall_s': round(time.time() - a['t0'], 1) if a.get('t0') else None,
              'ok': step_ok(name, step, int(rc), cmd)})
    s['updated'] = now()
    write_json(STATE, s)
    return a


def step_ok(name, step, rc, cmd=None):
    if step == 'train':
        return bool(cmd) and is_trained(name, cmd)
    if step == 'engine':
        return rc == 0 and not need_engine(name)
    if step == 'check':
        c = load(LOGS / ('engine_check_%s.json' % name))
        acc = (c or {}).get('accuracy') or {}
        return rc == 0 and bool(c) and bool(acc.get('gate_actions_2p5us')) and bool(acc.get('finite'))
    return rc == 0


def fails(name, step):
    atts = state()['runs'].get(name, {}).get(step, [])
    return sum(1 for a in atts if 'ended' in a and not a.get('ok'))


def last_rc(name, step):
    atts = [a for a in state()['runs'].get(name, {}).get(step, []) if 'ended' in a]
    return atts[-1]['rc'] if atts else ''


def wall(name, step):
    atts = state()['runs'].get(name, {}).get(step, [])
    w = [a.get('wall_s') for a in atts if a.get('wall_s')]
    return round(sum(w), 1) if w else None


def cmd_deviation(title, body):
    if SMOKE:
        print('試しなので deviations.md には書きません: %s' % title)
        return 0
    text = '\n## (キュー v2 が自動で書いた)%s(%s)\n\n%s\n' % (title, now(), body.strip())
    with open(DEVIATIONS, 'a', encoding='utf-8') as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    print('deviations.md に書き足しました: %s' % title)
    return 0


# ---------------------------------------------------------------- 始める前の確かめ

def cmd_preflight():
    ok, notes = [], []
    bad = []
    # 1. protocol の固定(sha256sum -c と同じ)
    for line in PROTOCOL_SHA.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        want, path = line.split(None, 1)
        p = Path(path) if path.startswith('/') else REP / path
        if not p.exists() or sha256(p) != want:
            bad.append('protocol の固定のファイルが違う: %s' % path)
    if not bad:
        ok.append('protocol の固定(protocol_v2.sha256 の 6 つ)')
    # 2. 封(check_seal_v2.py)
    r = subprocess.run([sys.executable, '-I', str(HERE / 'check_seal_v2.py')], capture_output=True, text=True)
    (ok if r.returncode == 0 else bad).append('封の 14 本(check_seal_v2.py rc=%d)' % r.returncode)
    proto = protocol()
    # 3. 表とデータセット
    card = load(TABLES / 'tables_card.json') or {}
    for f, ent in (card.get('files') or {}).items():
        p = TABLES / f
        if not p.exists() or sha256(p) != ent.get('sha256'):
            bad.append('表の sha256 が tables_card.json と違う: %s' % f)
    for key, f in proto['tables']['files'].items():
        if not (REPO / f).exists():
            bad.append('protocol の表がありません: %s' % f)
    dcard = DATASET / 'dataset_card.json'
    if not dcard.exists() or sha256(dcard) != card.get('dataset_card_sha256'):
        bad.append('dataset_card.json の sha256 が tables_card.json と違う')
    if card.get('protocol_v2_sha256') != sha256(PROTOCOL):
        bad.append('tables_card.json の protocol_v2_sha256 が今の protocol と違う')
    if not [b for b in bad if '表' in b or 'dataset_card' in b]:
        ok.append('表 6 つ・dataset_card.json(tables_card.json の sha256)')
    # 4. B の重み(B-FAST の --init)
    if sha256(B_RUN / 'best.pth') == proto['model_b_sha256']['best.pth']:
        ok.append('B の best.pth(B-FAST の --init)の sha256')
    else:
        bad.append('B の best.pth の sha256 が protocol と違う')
    # 5. ジャンル 9
    modes = load(MODES) or []
    ms = modes.get('modes', modes) if isinstance(modes, dict) else modes
    if any(int(m.get('id', -1)) == 9 for m in ms if isinstance(m, dict)):
        ok.append('data/modes.json にジャンル 9')
    else:
        bad.append('data/modes.json にジャンル 9 がありません')
    # 6. 表の行が protocol と同じ
    tb = check_tsv()
    if tb:
        bad.extend('表(gpu_queue_v2.tsv): %s' % b for b in tb)
    else:
        ok.append('gpu_queue_v2.tsv の 22 行が protocol の runs と同じ(順番 B-FAST → v2)')
    # 7. データの確かめで外れたもの(protocol tables.expected_after_build.if_off: 外れたら学習を始めない。main session に聞く)
    dc = load(DATA_CHECKS, {}) or {}
    fl = [f for f in dc.get('fails', []) if f not in DATA_FAILS_NOT_BLOCKING]
    warn = [f for f in dc.get('fails', []) if f in DATA_FAILS_NOT_BLOCKING]
    acc = {a.strip() for a in (os.environ.get('GPU_QUEUE_ACCEPT_DATA_FAILS') or '').split(',') if a.strip()}
    left = [f for f in fl if f not in acc and 'all' not in acc]
    if left and not SMOKE:
        bad.append('データの確かめで外れたもの(deviations.md の 3)を main session がまだ了承していません: %s。'
                   '了承したら GPU_QUEUE_ACCEPT_DATA_FAILS=%s を付けて動かす' % (' '.join(left), ','.join(fl)))
    elif fl:
        notes.append('データの確かめで外れたもの %s は%s' % (' '.join(fl), '試しなので通す' if left else '了承済み(GPU_QUEUE_ACCEPT_DATA_FAILS)'))
    if warn:
        notes.append('注意: %s(deviations.md の 4。学習は止めない。11/3 に出す前にユーザーに聞く)' % ' '.join(warn))
    # 8. ディスク
    free_gb = shutil.disk_usage(str(RUNS)).free / 1e9
    (ok if free_gb >= DISK_MIN_GB else bad).append('空きディスク %.1f GB(%.0f GB 以上)' % (free_gb, DISK_MIN_GB))
    if os.environ.get('GPU_QUEUE_WORKERS'):
        notes.append('GPU_QUEUE_WORKERS=%s(コマンドの --workers 4 を置き換える。protocol common_recipe.workers)' % os.environ['GPU_QUEUE_WORKERS'])
    if not LOCK2.exists():
        notes.append('protocol_v2_lock2.json はまだ無い(段 0 のあと・最初の予測の前に作る。それまでキューは予測をしない)')
    for o in ok:
        print('OK  %s' % o)
    for n in notes:
        print('注  %s' % n)
    for b in bad:
        print('NG  %s' % b)
    return 1 if bad else 0


# ---------------------------------------------------------------- run.json・engine_check.json・走行の一覧

def proto_run(name):
    """名前 → protocol の runs の行(型の名前と照らす。試しの _smoke_ は外して照らす)"""
    base = name[len('_smoke_'):] if name.startswith('_smoke_') else name
    for r in protocol()['runs']:
        pat = '^' + re.escape(r['name']).replace(r'\{x\}', '(v2h|v2m|base)').replace(r'\{E\}', '(30|60)').replace(
            r'\{A\}', '(-aug2)?') + '$'
        if re.match(pat, base):
            return r
    return None


def cmd_runjson(name, cmd):
    run = RUNS / name
    if (run / 'run.json').exists():
        print('run.json はあります: %s' % (run / 'run.json'))
        return 0
    from jetformer import policies      # torch を読まない
    args = train_args(cmd)
    pr = proto_run(name) or {}
    atts = state()['runs'].get(name, {}).get('train', [])
    created = next((a['started'] for a in atts if a.get('started')), None) or now()
    meta = {'dataset': Path(arg_value(args, '--data')).name, 'stride': int(arg_value(args, '--stride')),
            'epochs': int(arg_value(args, '--epochs')), 'batch': int(arg_value(args, '--batch')),
            'lr': float(arg_value(args, '--lr')), 'pretrained': '--no-pretrained' not in args, 'build_engine': True,
            'command_input': '--no-command-input' not in args, 'mask_impacts': '--keep-impacts' not in args,
            'policy': arg_value(args, '--policy'), 'policy_cfg': policies.get(arg_value(args, '--policy')).resolve({}),
            'impact_g': float(arg_value(args, '--impact-g')), 'seed': int(arg_value(args, '--seed')),
            'name': name, 'created_at': created,
            'note': ('試し(キュー v2 の動作の確かめ・--max-steps。すぐ消す。走らせない)' if SMOKE else
                     'データセット v2 / B-FAST の学習(protocol_v2 の %s・段 %s)。%s机上の門とユーザーの了承の前は走らせない' % (
                         pr.get('arm'), pr.get('stage'), (pr.get('note') + '。') if pr.get('note') else '')),
            'experiment': 'train_v2_20261025', 'arm': pr.get('arm'), 'stage': pr.get('stage'),
            'protocol_sha256': sha256(PROTOCOL)}
    tables = {}
    for key in TABLE_ARGS:
        v = arg_value(args, key)
        if v:
            meta[key.lstrip('-').replace('-', '_')] = v
            tables[key.lstrip('-').replace('-', '_')] = sha256(REPO / v)
    if tables:
        meta['tables_sha256'] = tables
    aug = {}
    for a in ('brightness', 'contrast', 'shift_px', 'tof_noise', 'top_brightness'):
        v = arg_value(args, '--aug-' + a.replace('_', '-'))
        if v:
            aug[a] = float(v)
    if aug:
        meta['aug'] = aug
    meta['train_env'] = {'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True', 'MALLOC_ARENA_MAX': '2'}
    meta['code_sha256'] = {f: sha256(REPO / f) for f in CODE_FILES if (REPO / f).exists()}
    meta['queue_cmd'] = cmd
    write_json(run / 'run.json', meta)
    print('run.json を書きました: %s' % (run / 'run.json'))
    return 0


def cmd_enginecheck(name):
    """確認の json → <学習結果>/engine_check.json(10/21 gpu_queue.py と同じ形)。確認の後にエンジンが変わっていたら写さない"""
    run = RUNS / name
    src = LOGS / ('engine_check_%s.json' % name)
    c = load(src)
    eng = run / ENGINE
    if not c or not eng.exists():
        print('確認の json かエンジンがありません: %s' % src)
        return 1
    st = eng.stat()
    checked = datetime.datetime.strptime(c['checked_at'], '%Y-%m-%d %H:%M:%S').timestamp()
    if st.st_mtime > checked:
        print('確認の後にエンジンが変わっています(写しません)')
        return 1
    acc, lat = c.get('accuracy') or {}, c.get('latency_ms') or {}
    out = {'ok': bool(c.get('all_gates')), 'accuracy_ok': bool(acc.get('gate_actions_2p5us')) and bool(acc.get('finite')),
           'speed_ok': bool(lat.get('gate_run_numpy')), 'actions_max_us': acc.get('actions_max_us'),
           'actions_p99_us': acc.get('actions_p99_us'), 'gate_us': GATE_US,
           'run_numpy_median_ms': lat.get('run_numpy_median'), 'run_numpy_p99_ms': lat.get('run_numpy_p99'),
           'checked_at': c['checked_at'], 'note': '',
           'engine': {'file': eng.name, 'size': st.st_size, 'mtime_ns': st.st_mtime_ns},
           'source': str(src),
           'method': '実フレーム 64 個で TensorRT(run_numpy)と PyTorch FP32(CPU・best.pth)の actions の差。速さは 100 回の'
                     '空回しの後 600 回(experiments/20261013_sim_cotrain/train/engine_load_check.py、10/25 キュー v2)'}
    write_json(run / 'engine_check.json', out)
    print(json.dumps({k: out[k] for k in ('ok', 'accuracy_ok', 'speed_ok', 'actions_max_us', 'run_numpy_median_ms',
                                          'run_numpy_p99_ms')}, ensure_ascii=False))
    return 0 if out['accuracy_ok'] else 1


def cmd_drivable(name):
    from jetformer import drive     # torch を読まない(cv2・numpy だけ)
    info = drive.model_info(name)
    listed = [m['name'] for m in drive.list_models()]
    rec = drive.recommended()
    out = {'checked_at': now(), 'drivable': bool(info and info.get('drivable')),
           'why_not': info and info.get('why_not'), 'engine_check': info and info.get('engine_check'),
           'listed': name in listed, 'recommended_run': rec and rec.get('run'),
           'policy': info and info.get('policy'), 'trained_modes': info and info.get('trained_modes'),
           'steer_center_us': info and info.get('steer_center_us'),
           'throttle_neutral_us': info and info.get('throttle_neutral_us')}
    s = state()
    s['runs'].setdefault(name, {})['drivable'] = out
    s['updated'] = now()
    write_json(STATE, s)
    print(json.dumps(out, ensure_ascii=False))
    ok = out['drivable'] and out['listed'] and bool(out['engine_check']) and out['engine_check'].get('accuracy_ok') is True
    return 0 if ok else 1


def _get(path):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(WEB + path, timeout=4) as r:
        return json.load(r)


def cmd_webbusy():
    """走行・収集(GET /api/drive/status の running・recording)と管理画面の学習・収集・走行(GET /api/status)。
    管理画面に届かない・読めないときは「使っていない」(jf-heavy と同じ)"""
    why = []
    try:
        d = _get('/api/drive/status')
        if isinstance(d, dict):
            why += [k for k in ('running', 'recording') if d.get(k) is True]
    except Exception:          # noqa: BLE001
        pass
    try:
        st = _get('/api/status')
        if isinstance(st, dict):
            why += [k for k, v in (('training', st.get('training')), ('driving', st.get('driving')),
                                   ('recorder', (st.get('recorder') or {}).get('recording')),
                                   ('session', (st.get('session') or {}).get('active'))) if v is True]
    except Exception:          # noqa: BLE001
        pass
    if why:
        print('使用中: %s' % '・'.join(sorted(set(why))))
        return 0
    return 1


def pred_ok(path, ck):
    """eval_v2.py predict の予測が読めて、表の key・ckpt・形・有限が合うか(numpy だけ)"""
    try:
        import numpy as np
        with np.load(path, allow_pickle=False) as z:
            return (str(z['key']) == '3d339512828b17195d568f1b93ee4a28208f2e85' and str(z['ckpt']) == ck
                    and z['pred'].shape == (10985, 10, 2) and bool(np.isfinite(z['pred']).all()))
    except Exception:        # noqa: BLE001
        return False


def cmd_prune_last(args):
    """protocol start_conditions.disk:「自分で作った last.pth は、予測を保存して確かめたあと消す(候補に残るものは消さない)」。
    prune-last → 消してよい候補の一覧(学習が済み・best と last の予測が読める)。prune-last --yes <名前>... → その last.pth だけ消す。
    どれが候補に残るかは決めないので、名前は main session が選ぶ(既定では何も消さない)"""
    yes = '--yes' in args
    names = [a for a in args if a != '--yes']
    pred = REP / 'eval' / 'pred'
    rows = [(n, c) for _, n, c, _, _, _ in current_rows() if c]
    if not names:
        tot = 0
        for n, c in rows:
            lp = RUNS / n / 'last.pth'
            if lp.exists() and is_trained(n, c) and pred_ok(pred / ('%s.npz' % n), 'best') and pred_ok(pred / ('%s_last.npz' % n), 'last'):
                tot += lp.stat().st_size
                print('%s\t%.0f MB' % (n, lp.stat().st_size / 2 ** 20))
        print('合計 %.1f GB(消すのは prune-last --yes <名前>...。候補に残るものは消さない)' % (tot / 1e9))
        return 0
    if not yes:
        raise SystemExit('消すときは --yes を付けてください')
    known = dict(rows)
    for n in names:
        lp = RUNS / n / 'last.pth'
        if n not in known or not lp.exists():
            print('%s: 表の学習でないか last.pth がありません(飛ばす)' % n)
            continue
        if not (is_trained(n, known[n]) and pred_ok(pred / ('%s.npz' % n), 'best') and pred_ok(pred / ('%s_last.npz' % n), 'last')):
            print('%s: 学習が済んでいないか、予測(best・last)が読めません(消さない)' % n)
            continue
        size = lp.stat().st_size
        lp.unlink()
        print('%s: last.pth を消しました(%.0f MB)' % (n, size / 2 ** 20))
    return 0


# ---------------------------------------------------------------- 状態・一覧

def current_rows():
    """表の行を今の答えで埋めた [(段, 名前, コマンド or None, 型の名前, 消してある, STOP の理由)]"""
    proto, dec = protocol(), decisions()
    out = []
    first = smoke_row()
    for stage, name, cmd, commented in queue_rows(include_commented=True):
        if SMOKE and name != first:
            continue                        # 試しは先頭の 1 行だけ
        r = fill(stage, name, cmd, dec, proto)
        if r[0] == 'RUN':
            n, c = r[1], with_workers(r[2])
            if SMOKE:
                n, c = smoke(n, c)
            out.append((stage, n, c, name, commented, None))
        else:
            out.append((stage, name, None, name, commented, r[1]))
    return out


def row(stage, name, cmd, template, commented, why):
    run = RUNS / name
    res = load(run / 'result.json') or {}
    hist = res.get('history') or []
    c = check_current(run)
    lc = load(LOGS / ('engine_check_%s.json' % name)) or {}
    s = state()['runs'].get(name, {})
    d = s.get('drivable') or {}
    if cmd is None:
        tstate = '未定'
    elif is_trained(name, cmd):
        tstate = 'done'
    elif (run / 'last.pth').exists():
        tstate = 'resume'
    else:
        tstate = 'fresh'
    pr = proto_run(name) or {}
    return {
        'stage': stage, 'name': name, 'template': template, 'arm': pr.get('arm'), 'cut': commented,
        'undecided': why, 'train': tstate, 'epochs_planned': epochs_in(cmd) if cmd else None,
        'epochs_done': res.get('epochs_done'), 'best_val_us': res.get('best_val_us'), 'best_epoch': res.get('best_epoch'),
        'last_val_us': hist[-1]['val_us'] if hist else None,
        'best_val_per_mode_us': res.get('best_val_per_mode_us'),
        'train_samples': res.get('train_samples'), 'val_samples': res.get('val_samples'),
        'steer_center_us': res.get('steer_center_us'), 'throttle_neutral_us': res.get('throttle_neutral_us'),
        'peak_mem_mb': res.get('peak_mem_mb'), 'train_duration_s': res.get('duration_s'),
        'engine': (not need_engine(name)) if (run / 'jetformer.onnx').exists() else False,
        'engine_mb': round((run / ENGINE).stat().st_size / 2 ** 20, 1) if (run / ENGINE).exists() else None,
        'engine_check': c and {k: c.get(k) for k in ('ok', 'accuracy_ok', 'speed_ok', 'actions_max_us', 'actions_p99_us',
                                                      'run_numpy_median_ms', 'run_numpy_p99_ms', 'checked_at')},
        'engine_only_ms': lc.get('latency_ms') and {k: lc['latency_ms'].get(k) for k in ('engine_median', 'engine_p99')},
        'drivable': d.get('drivable'), 'listed': d.get('listed'),
        'wall_s': {st: wall(name, st) for st in ('train', 'engine', 'check')},
        'attempts': {st: len(s.get(st, [])) for st in ('train', 'engine', 'check')},
        'fails': {st: fails(name, st) for st in ('train', 'engine', 'check')},
        'estimate_h': round(hours_of(cmd), 2) if cmd else None,
    }


def cmd_status():
    for r in (row(*x) for x in current_rows()):
        ec = r['engine_check'] or {}
        print('%-7s %-34s 学習 %-6s ep %-4s best %-6s last %-6s エンジン %-5s 確認 %-5s 精度 %-6s 走行 %s%s' % (
            r['stage'], r['name'], r['train'], r['epochs_done'], r['best_val_us'], r['last_val_us'], r['engine'],
            ec.get('ok'), ec.get('actions_max_us'), r['drivable'], '(削った)' if r['cut'] else ''))
    return 0


def cmd_summary():
    rows = [row(*x) for x in current_rows()]
    write_json(SUMMARY, {'made_at': now(), 'decisions': decisions(), 'runs': rows,
                         'note': 'best_val_us・last_val_us は tools/train.py の検証(H1 の 11 シーン、stride 3・衝突の前後を除く・'
                                 '1,559 サンプル)。机上の判定は eval_v2.py(H1 4,670 フレーム・介入 1,570)'})
    print('| 段 | 名前 | 学習 | L1 best [µs](エポック) | L1 last [µs] | エンジン | 精度 最大差 [µs] | run_numpy 中央/p99 [ms] | 走行の一覧 | 学習/エンジン/確認 [秒] |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for r in rows:
        ec = r['engine_check'] or {}
        w = r['wall_s']
        print('| %s | %s | %s | %s(%s) | %s | %s | %s(%s) | %s / %s | %s | %s / %s / %s |' % (
            r['stage'], r['name'] + ('(削った)' if r['cut'] else ''), r['train'], r['best_val_us'], r['best_epoch'],
            r['last_val_us'], '%s MB' % r['engine_mb'] if r['engine'] else 'なし', ec.get('actions_max_us'),
            '合格' if ec.get('accuracy_ok') else ('不合格' if ec else '未'), ec.get('run_numpy_median_ms'),
            ec.get('run_numpy_p99_ms'), 'drivable' if r['drivable'] else r['drivable'], w['train'], w['engine'], w['check']))
    return 0


def main(argv):
    if len(argv) < 2:
        raise SystemExit(__doc__)
    c, a = argv[1], argv[2:]
    simple = {'tsv': cmd_tsv, 'preflight': cmd_preflight, 'templates': cmd_templates, 'webbusy': cmd_webbusy,
              'estimate': cmd_estimate, 'status': cmd_status, 'summary': cmd_summary}
    if c in simple:
        return simple[c]()
    if c == 'resolve':
        return cmd_resolve(a[0])
    if c == 'trained':
        return 0 if is_trained(a[0], a[1]) else 1
    if c == 'prune-last':
        return cmd_prune_last(a)
    if c == 'predicted':                 # eval_v2.py predict の best・last があれば 0(GPU を使わずに飛ばすため)
        pred = REP / 'eval' / 'pred'
        return 0 if all((pred / ('%s%s.npz' % (a[0], k))).exists() for k in ('', '_last')) else 1
    if c == 'fits':
        return cmd_fits(a[0], a[1])
    if c == 'lastpth':
        return cmd_lastpth(a[0])
    if c == 'need-engine':
        return 0 if need_engine(a[0]) else 1
    if c == 'need-check':
        return 0 if check_current(RUNS / a[0]) is None else 1
    if c == 'begin':
        cmd_begin(a[0], a[1], a[2] if len(a) > 2 else '')
        return 0
    if c == 'end':
        r = cmd_end(a[0], a[1], a[2], a[3] if len(a) > 3 else None)
        print('%s %s: rc=%s ok=%s %s 秒' % (a[0], a[1], r['rc'], r['ok'], r['wall_s']))
        return 0
    if c == 'fails':
        print(fails(a[0], a[1]))
        return 0
    if c == 'last-rc':
        print(last_rc(a[0], a[1]))
        return 0
    if c == 'runjson':
        return cmd_runjson(a[0], a[1])
    if c == 'enginecheck':
        return cmd_enginecheck(a[0])
    if c == 'drivable':
        return cmd_drivable(a[0])
    if c == 'deviation':
        return cmd_deviation(a[0], a[1])
    raise SystemExit(__doc__)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
