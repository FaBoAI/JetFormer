#!/usr/bin/env python3
"""キュー v3(段 data、2026-10-26): gpu_queue_v3.sh の手伝い。torch を読まない・GPU を使わない・車と装置に触れない。
experiments/20261025_train_v2/gpu_queue_v2.py の写しを protocol_v3(/home/jetson/jetformer_reports/fix_20261026/protocol_v3.json)
に合わせたもの。違うところ:
  - 表(gpu_queue_v3.tsv)は protocol_v3 の runs_bfast2_conditional(B-FAST2 の 6 本)→ 段 0 の短い学習 4 本
    (diag_data/ablation_short.sh と同じコマンド)→ 段 1 の 12 本(runs)から作る(tsv)。型({x} など)は無い
  - B-FAST2 の行: BF2-0 の答え(decisions_v3.json の "BF2-0".train。無ければ data/out/bf2_0.json)が「学習しない」なら飛ばす。
    「学習する」で表(drop_bfast2.npz など)が無ければ止まる
  - 段 1 の行: protocol_v3_lock2.json(lock2_v3)と decisions_v3.json の D0.order(腕の順 3 つ)が無ければ止まる
    (protocol_v3 1 章: lock2_v3 は段 1 の学習の前)。行の順は D0.order で並べ替える(種 0 の 3 本 → 種 1 → …)
  - 途中でやめる決まり(protocol futility): 同じ腕の種 0・1 の best.pth の物差し(probe_gates.py run)が両方とも
    st23 > 42.4 µs か stand_brake_side > 0.192 なら、その腕の種 2・3 を飛ばす(deviations_v3.md に書く)
  - 学習のあと: 段 1・B-FAST2 はエンジン・確認・走行の一覧・物差し(best・last)・机上の予測(desk_v3/eval_v3.py が
    あって lock2_v3 のあとだけ)。段 0 は物差し(best・last)だけ(エンジンは作らない。走らせない)
  - 時間の式は protocol_v3 timing_estimate_hours(1 本 = 0.768 時間 × サンプル ÷ 6,506 × エポック ÷ 60 + 物差し + 準備 0.075)。
    締め切り 2026-10-30 06:00(× 1.2)

    python -I gpu_queue_v3.py tsv | preflight | templates | estimate | status | summary | webbusy
    python -I gpu_queue_v3.py resolve <行の名前>        # RUN<TAB>名前<TAB>コマンド / SKIP<TAB>理由 / STOP<TAB>理由
    python -I gpu_queue_v3.py rundir <名前> / post <名前> / trained <名前> <コマンド> / fits <名前> <コマンド> / lastpth <名前>
    python -I gpu_queue_v3.py need-engine <名前> / need-check <名前> / need-probe <名前> <best|last> / predicted <名前>
    python -I gpu_queue_v3.py begin <名前> <段> [メモ] / end <名前> <段> <rc> [コマンド] / fails <名前> <段> / last-rc <名前> <段>
    python -I gpu_queue_v3.py runjson <名前> <コマンド> / enginecheck <名前> / drivable <名前> / deviation <見出し> <本文>

試し(GPU_QUEUE_SMOKE=1): 1 行だけ(既定は段 1 の最初の行。GPU_QUEUE_SMOKE_ROW=<行の名前> で別の行)。lock2_v3・D0・
データの確かめの待ちは飛ばす(試しなので)。名前は _smoke_<名前>、出力は data/runs/_smoke_<名前>、--epochs 1 と
--max-steps(GPU_QUEUE_SMOKE_STEPS、既定 5)。状態とログは logs/smoke/。deviations_v3.md には書かない。終わったら sh が出力を消す。
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
HERE = REPO / 'experiments' / '20261026_fix'
REP = Path('/home/jetson/jetformer_reports/fix_20261026')
PROTOCOL = REP / 'protocol_v3.json'
PROTOCOL_SHA = REP / 'protocol_v3.sha256'
REP_V2 = Path('/home/jetson/jetformer_reports/train_v2_20261025')
SEAL_CHECK = REPO / 'experiments' / '20261025_train_v2' / 'check_seal_v2.py'
DECISIONS = Path(os.environ.get('GPU_QUEUE_DECISIONS') or REP / 'decisions_v3.json')
DEVIATIONS = REP / 'deviations_v3.md'
LOCK2 = Path(os.environ.get('GPU_QUEUE_LOCK2') or REP / 'protocol_v3_lock2.json')     # 試験のときだけ別のファイル
QUEUE = HERE / 'gpu_queue_v3.tsv'
BF2_0 = HERE / 'data' / 'out' / 'bf2_0.json'
DATA_CHECK = HERE / 'data' / 'out' / 'check_v3_data.json'
ONOFF = HERE / 'data' / 'out' / 'probe_onoff.json'
PROBE_GATES = HERE / 'diag_data' / 'probe_gates.py'
PROBE_SET = HERE / 'diag_data' / 'out' / 'probe_set.npz'
ABL = 'experiments/20261026_fix/diag_data/out/ablation'
EVAL_V3 = HERE / 'desk_v3' / 'eval_v3.py'          # 机上の台本(担当はまだ。あれば予測を回す)
SMOKE = os.environ.get('GPU_QUEUE_SMOKE') == '1'
SMOKE_STEPS = int(os.environ.get('GPU_QUEUE_SMOKE_STEPS') or 5)
LOGS = HERE / 'logs' / ('smoke' if SMOKE else '')
STATE = LOGS / 'gpu_queue_v3_state.json'
SUMMARY = LOGS / 'gpu_queue_v3_summary.json'
PROBE_DIR = Path(os.environ.get('GPU_QUEUE_PROBE_DIR') or LOGS / 'probe')            # 試験のときだけ別の場所
RUNS = REPO / 'data' / 'runs'
TABLES = REPO / 'data' / 'derived' / 'dataset_v3_20261026'
B_RUN = RUNS / 'clean_driving_20261011-nocmd-s3-20261011-024626'
B_BEST_SHA = '8b0dc57f51b79a6c9059b6cf882fb7b53c9acfd0e70de8f03a0e9e6a99a02a04'     # protocol_v2 model_b_sha256
ENGINE = 'jetformer_fp16.engine'
GATE_US = 2.5
DEADLINE = os.environ.get('GPU_QUEUE_DEADLINE') or '2026-10-30 06:00'      # protocol_v3 start_conditions.deadline
DISK_MIN_GB = float(os.environ.get('GPU_QUEUE_DISK_MIN_GB') or 5.0)         # protocol_v3 start_conditions.disk
WEB = (os.environ.get('GPU_QUEUE_WEB') or 'http://127.0.0.1:8080').rstrip('/')
PREFIX = 'cd ~/jetformer && '
TABLE_ARGS = ('--drop-mask', '--sample-weights-npz', '--mode-remap-npz', '--label-override-npz', '--init', '--probe-npz')
CODE_FILES = ('tools/train.py', 'jetformer/train_data.py', 'jetformer/dataset.py', 'tools/build_engine.py',
              'jetformer/trt_runtime.py', 'jetformer/policies/jetformer_v0.py',
              'experiments/20261026_fix/diag_data/probe_gates.py')
GPU_PREFIX = ('/home/jetson/bin/jf-heavy --mem 3872M --lock big --need-mb 3200 -- nice -n 10 env '
              'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MALLOC_ARENA_MAX=2 choom -n 1000 -- '
              '/home/jetson/miniconda/envs/jetracer/bin/python -I experiments/20261015_real_accuracy/train/nothp.py '
              '/home/jetson/miniconda/envs/jetracer/bin/python -u tools/train.py')
# 段 0 の短い学習(diag_data/ablation_short.sh と同じ引数・同じ出力先)
ABL_ARMS = {'same': '--drop-mask %s/drop_v2h_same.npz' % ABL, 'noTK': '--drop-mask %s/drop_v2h_noTK.npz' % ABL,
            'noTK_noLB': '--drop-mask %s/drop_v2h_noTK_noLB.npz' % ABL,
            'tkB': '--drop-mask %s/drop_v2h_same.npz --label-override-npz %s/override_tk_thr_B.npz' % (ABL, ABL)}
SAMPLES = {'drop_v3n.npz': 6105, 'drop_v3d.npz': 6469, 'drop_v3b.npz': 5875, 'drop_v2h_same.npz': 6506,
           'drop_v2h_noTK.npz': 6142, 'drop_v2h_noTK_noLB.npz': 6105}
# protocol_v3 timing_estimate_hours
H_PER_60E_FULL = 0.768          # 6,506 サンプル・60 エポック・AUG2(v2h-aug2-e60 の実測 0.70〜0.80)
N_FULL = 6506
PROBE_H = 0.07                  # 物差し 14 回(60 エポック)。protocol は 0.04。10/26 の試しで 1 回 15〜23 秒 → 14 回で約 0.07
PREP_H = 0.075
NOAUG_FACTOR = 0.69 / 0.71      # 拡張なし(段 0)は AUG2 より少し速い(v2 plan.md 4.2)
BF2_H = 0.23
SAFETY = 1.2
POST_S = {'0': 2 * 45 + 2 * 60, 'other': 90 + 30 + 15 + 2 * 45 + 5 * 60}     # 物差し 2 回・エンジン・確認・一覧・あける時間
FUTILITY = {'st23_thr_vs_b_us': 42.4, 'stand_brake_side_frac': 0.192}       # protocol_v3 futility(> で外れ)


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
    a = shlex.split(cmd)
    return a[a.index('tools/train.py') + 1:]


# ---------------------------------------------------------------- 表(gpu_queue_v3.tsv)

def abl_cmd(arm):
    return ('%s --policy jetformer_v0 --data data/datasets/dataset_v2_20261025 %s --epochs 15 --lr 3e-4 --stride 3 --batch 64 '
            '--workers 2 --no-command-input --impact-g 2.5 --seed 0 --out %s/runs/abl-%s-e15-s0' % (GPU_PREFIX, ABL_ARMS[arm], ABL, arm))


def protocol_rows():
    """→ [(段, 名前, コマンド, メタ)]。B-FAST2(6・条件つき)→ 段 0(4)→ 段 1(12、protocol の runs の順)"""
    p = protocol()
    out = []
    for r in p['runs_bfast2_conditional']:
        out.append(('BF2', r['name'], r['command'][len(PREFIX):], {'arm': r['arm'], 'seed': r['seed'], 'epochs': r['epochs']}))
    for arm in ('same', 'noTK', 'noTK_noLB', 'tkB'):
        out.append(('0', 'abl-%s-e15-s0' % arm, abl_cmd(arm), {'arm': arm, 'seed': 0, 'epochs': 15}))
    for r in p['runs']:
        if not r['command'].startswith(PREFIX):
            raise SystemExit('protocol の %s のコマンドが %r で始まっていません' % (r['name'], PREFIX))
        out.append((str(r['stage']), r['name'], r['command'][len(PREFIX):], {'arm': r['arm'], 'seed': r['seed'],
                                                                            'epochs': r['epochs'], 'n_train': r['n_train']}))
    return out


def cmd_tsv():
    rows = protocol_rows()
    if len(rows) != 22 or len({r[1] for r in rows}) != 22:
        raise SystemExit('行が 22 本(名前が別々)ではありません')
    lines = [
        '# キュー v3(段 data、2026-10-26)の表。1 行 1 本: 段<TAB>名前<TAB>コマンド(~/jetformer で実行)。gpu_queue_v3.py tsv が '
        'protocol_v3.json の runs_bfast2_conditional・runs と diag_data/ablation_short.sh から作った',
        '# 順番: B-FAST2 の 6 本(BF2-0 が「学習する」のときだけ。今は不合格で飛ばす)→ 段 0 の短い学習 4 本 → 段 1 の 12 本'
        '(lock2_v3 と decisions_v3.json の D0.order のあと。D0.order の腕の順で 種 0 → 1 → 2 → 3)',
        '# 削るときは行の頭に # を付ける(protocol_v3 cut_order: ① 段 0 の noTK_noLB・tkB → ② 種 3 → ③ 種 2 → ④ B-FAST2。'
        '削ったことは deviations_v3.md に書く)',
    ]
    for stage, name, cmd, _ in rows:
        lines.append('%s\t%s\t%s' % (stage, name, cmd))
    tmp = QUEUE.with_suffix('.tsv.tmp')
    tmp.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    os.replace(tmp, QUEUE)
    print('書きました: %s(%d 行)' % (QUEUE, len(rows)))
    return 0


def queue_rows(include_commented=False):
    out = []
    for line in QUEUE.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        commented = line.startswith('#')
        body = line.lstrip('#').strip() if commented else line
        parts = body.split('\t')
        if len(parts) != 3:
            continue
        stage, name, cmd = (p.strip() for p in parts)
        if commented and not include_commented:
            continue
        out.append((stage, name, cmd, commented))
    return out


def meta_of(name):
    base = name[len('_smoke_'):] if name.startswith('_smoke_') else name
    for stage, n, cmd, m in protocol_rows():
        if n == base:
            return dict(m, stage=stage, command=cmd)
    return {}


def check_tsv():
    want = {name: (stage, cmd) for stage, name, cmd, _ in protocol_rows()}
    bad, seen = [], []
    for stage, name, cmd, _ in queue_rows(include_commented=True):
        seen.append(name)
        if name not in want:
            bad.append('protocol に無い行: %s' % name)
        elif want[name] != (stage, cmd):
            bad.append('protocol とコマンドか段が違う: %s' % name)
    if sorted(seen) != sorted(want):
        bad.append('行の数・名前が protocol と違う(%d 行)' % len(seen))
    if seen != [n for _, n, _, _ in protocol_rows()]:
        bad.append('行の順番が B-FAST2 → 段 0 → 段 1(protocol の順)と違う')
    return bad


# ---------------------------------------------------------------- 決まりの答え(decisions_v3.json)

def decisions():
    return load(DECISIONS, {}) or {}


def bf2_answer():
    """→ (学習する True/False/None, 出どころ)。decisions_v3.json の "BF2-0".train が先、無ければ bf2_0_check.py の答え"""
    d = decisions().get('BF2-0')
    if isinstance(d, dict) and isinstance(d.get('train'), bool):
        return d['train'], 'decisions_v3.json'
    b = load(BF2_0)
    if isinstance(b, dict) and isinstance(b.get('train_bfast2'), bool):
        return b['train_bfast2'], 'data/out/bf2_0.json(上限 A %s・B %s µs、要るのは ≥ %s)' % (
            b['bound_g090_us'].get('A'), b['bound_g090_us'].get('B'), b.get('need_us'))
    return None, '無い'


def d0_order():
    d = decisions().get('D0')
    order = d.get('order') if isinstance(d, dict) else None
    if isinstance(order, list) and sorted(order) == ['v3b', 'v3d', 'v3n']:
        return order
    return None


def run_order():
    """段 1 の行の順: D0.order があれば (種, 腕の順)、無ければ protocol の順"""
    rows = queue_rows()
    order = d0_order()
    if not order:
        return rows
    head = [r for r in rows if r[0] != '1']
    st1 = [r for r in rows if r[0] == '1']
    st1.sort(key=lambda r: (meta_of(r[1]).get('seed', 9), order.index(meta_of(r[1]).get('arm', 'v3n'))))
    return head + st1


def probe_file(name, ck):
    return PROBE_DIR / ('%s_%s.json' % (name, ck))


def probe_result(name, ck='best'):
    r = load(probe_file(name, ck))
    if isinstance(r, dict) and len(r) == 1:
        return list(r.values())[0]
    return None


def futile(arm):
    """protocol_v3 futility: 種 0・1 の best の物差しが両方とも外れ → (True, 理由)。分からなければ (False, 理由)"""
    vals = []
    for s in (0, 1):
        r = probe_result('v3_20261026-%s-aug2-e60-s%d' % (arm, s))
        if r is None:
            return False, '種 %d の best の物差しがまだ無い' % s
        vals.append(r)
    bad = [any(float(v[k]) > t for k, t in FUTILITY.items()) for v in vals]
    why = '・'.join('種 %d: st23 %+.1f µs・止まった場面のブレーキ側 %.3f' % (s, v['st23_thr_vs_b_us'], v['stand_brake_side_frac'])
                    for s, v in enumerate(vals))
    return all(bad), why


# ---------------------------------------------------------------- 行を決める

def smoke_row():
    if os.environ.get('GPU_QUEUE_SMOKE_ROW'):
        return os.environ['GPU_QUEUE_SMOKE_ROW']
    st1 = [r for r in run_order() if r[0] == '1']
    return (st1 or queue_rows())[0][1]


def smoke(name, cmd):
    sname = '_smoke_' + name
    out = arg_value(train_args(cmd), '--out')
    c = cmd.replace('--out %s' % out, '--out data/runs/%s' % sname)
    c = re.sub(r'--epochs \d+', '--epochs 1', c)
    return sname, c + ' --max-steps %d' % SMOKE_STEPS


def train_py_flags():
    src = (REPO / 'tools' / 'train.py').read_text(encoding='utf-8')
    return set(re.findall(r"add_argument\('(--[\w-]+)'", src))


def resolve(template):
    hit = [r for r in queue_rows(include_commented=True) if r[1] == template]
    if not hit:
        return ('SKIP', '表に無い行になった')
    stage, name, cmd, commented = hit[0]
    if commented:
        return ('SKIP', '表で # が付いた(削った)')
    if SMOKE and template != smoke_row():
        return ('SKIP', '試しは 1 行だけ')
    m = meta_of(name)
    if stage == 'BF2':
        ans, src = bf2_answer()
        if ans is None:
            return ('STOP', 'BF2-0 の答えがありません(data/bf2_0_check.py を流すか decisions_v3.json に "BF2-0": {"train": …})')
        if ans is False and not SMOKE:
            return ('SKIP', 'BF2-0 が「学習しない」(%s)' % src)
    elif stage == '1' and not SMOKE:
        if not LOCK2.exists():
            return ('STOP', 'protocol_v3_lock2.json(lock2_v3)がまだ無い。段 0 の答え(D0・BF2-0)のあとに書いて、同じコマンドをもう一度')
        if not d0_order():
            return ('STOP', '%s の D0.order(例 ["v3d", "v3n", "v3b"])がまだ無い' % DECISIONS)
        dc = load(DATA_CHECK, {}) or {}
        if not dc.get('all_ok'):
            return ('STOP', 'データの確かめ(data/out/check_v3_data.json)が合格していない(protocol_v3 data.if_off: 学習を始めない)')
        if m.get('seed') in (2, 3):
            f, why = futile(m['arm'])
            if f:
                return ('SKIP', '途中でやめる決まり(protocol_v3 futility)で %s の種 2・3 を始めない(%s)' % (m['arm'], why))
    if SMOKE:
        name, cmd = smoke(name, cmd)
    missing = sorted({a for a in train_args(cmd) if a.startswith('--')} - train_py_flags())
    if missing:
        return ('STOP', 'tools/train.py にまだ無い引数: %s' % ' '.join(missing))
    for key in TABLE_ARGS:
        v = arg_value(train_args(cmd), key)
        if v and not (REPO / v).exists():
            return ('STOP', '%s のファイルがありません: %s' % (key, v))
    return ('RUN', name, cmd)


def cmd_resolve(template):
    print('\t'.join(resolve(template)))
    return 0


def cmd_templates():
    first = smoke_row() if SMOKE else None
    for _, name, _, _ in run_order():
        if first is None or name == first:
            print(name)
    return 0


# ---------------------------------------------------------------- 学習の場所・済み・再開

def run_dir(name):
    if name.startswith('_smoke_'):
        return RUNS / name
    m = meta_of(name)
    if not m:
        return RUNS / name
    return REPO / arg_value(train_args(m['command']), '--out')


def stage_of(name):
    return meta_of(name).get('stage', '')


def epochs_in(cmd):
    return int(arg_value(train_args(cmd), '--epochs') or 30)


def is_trained(name, cmd):
    run = run_dir(name)
    res = load(run / 'result.json')
    return bool(res and res.get('state') == 'done' and res.get('epochs_done') == epochs_in(cmd)
                and all((run / f).exists() for f in ('best.pth', 'last.pth', 'jetformer.onnx', 'model_config.json')))


def lastpth_state(name):
    p = run_dir(name) / 'last.pth'
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


def need_engine(name):
    run = run_dir(name)
    eng, onnx = run / ENGINE, run / 'jetformer.onnx'
    return (not eng.exists()) or eng.stat().st_mtime_ns < onnx.stat().st_mtime_ns


def check_current(run):
    c = load(run / 'engine_check.json')
    eng = run / ENGINE
    if not isinstance(c, dict) or not eng.exists():
        return None
    st = eng.stat()
    e = c.get('engine') or {}
    if int(e.get('size', -1)) != st.st_size or int(e.get('mtime_ns', -1)) != st.st_mtime_ns:
        return None
    return c


def need_probe(name, ck):
    """物差しの json が無いか、その重み(best.pth・last.pth)より古ければ要る"""
    f, w = probe_file(name, ck), run_dir(name) / ('%s.pth' % ck)
    return (not f.exists()) or (w.exists() and f.stat().st_mtime_ns < w.stat().st_mtime_ns) or probe_result(name, ck) is None


def post_steps(name):
    """学習のあとの手順: 段 0 は物差しだけ。ほかはエンジン・確認・一覧・物差し・予測"""
    return 'probe' if stage_of(name) == '0' else 'engine probe predict'


# ---------------------------------------------------------------- 時間の見込み

def samples_of(cmd):
    a = train_args(cmd)
    drop = Path(arg_value(a, '--drop-mask') or '').name
    return SAMPLES.get(drop, N_FULL)


def hours_of(cmd, epochs_left=None):
    a = train_args(cmd)
    e_all = epochs_in(cmd)
    e = e_all if epochs_left is None else epochs_left
    if '--init' in a and 'bfast2' in (arg_value(a, '--out') or '') + (arg_value(a, '--drop-mask') or ''):
        return BF2_H * e / max(e_all, 1)
    h = H_PER_60E_FULL * samples_of(cmd) / N_FULL * e / 60.0
    if '--aug-shift-px' not in a:
        h *= NOAUG_FACTOR
    if '--probe-npz' in a:
        h += PROBE_H * e / 60.0
    return h + PREP_H


def epochs_left(name, cmd):
    run = run_dir(name)
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
        print('空きディスク %.1f GB < %.1f GB(protocol_v3 start_conditions.disk)' % (free_gb, DISK_MIN_GB))
        return 1
    if end > dl and not SMOKE:
        print('見込み %.2f 時間(× %.1f)で %s に終わり、締め切り %s を超える(protocol_v3 start_conditions.deadline)' % (
            h, SAFETY, end.strftime('%m/%d %H:%M'), DEADLINE))
        return 1
    print('見込み %.2f 時間(× %.1f、残り %d エポック)・%s ごろまで・空きディスク %.1f GB' % (
        h, SAFETY, left, end.strftime('%m/%d %H:%M'), free_gb))
    return 0


def cmd_estimate():
    ans, _ = bf2_answer()
    print('| 順 | 段 | 名前 | サンプル | エポック | 学習 [時間] | 後の手順・あける [時間] | 合計 [時間] |')
    print('|---|---|---|---|---|---|---|---|')
    tot = {'BF2': 0.0, '0': 0.0, '1': 0.0}
    for i, (stage, name, cmd, commented) in enumerate(queue_rows(include_commented=True), 1):
        h = hours_of(cmd)
        post = POST_S['0' if stage == '0' else 'other'] / 3600.0
        if stage == 'BF2' and ans is False:
            print('| %d | %s | %s | — | %d | (BF2-0 不合格で飛ばす) | | 0 |' % (i, stage, name, epochs_in(cmd)))
            continue
        tot[stage] += h + post
        print('| %d | %s | %s%s | %s | %d | %.2f | %.2f | %.2f |' % (i, stage, name, '(削った)' if commented else '',
                                                                 samples_of(cmd) if stage != 'BF2' else '—', epochs_in(cmd),
                                                                 h, post, h + post))
    print()
    print('合計の見込み [時間]: B-FAST2 %.2f・段 0 %.2f・段 1 %.2f・計 %.2f(× %.1f で %.2f)。机上の予測・判定は入れていない'
          '(protocol_v3 は約 2 時間)' % (tot['BF2'], tot['0'], tot['1'], sum(tot.values()), SAFETY, sum(tot.values()) * SAFETY))
    print('式: protocol_v3 timing_estimate_hours(1 本 = 0.768 時間 × サンプル ÷ 6,506 × エポック ÷ 60 + 物差し × エポック ÷ 60 '
          '+ 準備 0.075。物差しは 10/26 の実測 0.07(protocol は 0.04)。拡張なしは × %.3f)。後の手順 = 物差し 2 回・エンジン・確認・一覧・60 秒 × 5' % NOAUG_FACTOR)
    return 0


# ---------------------------------------------------------------- 試みの記録

def state():
    return load(STATE, {'runs': {}})


def cmd_begin(name, step, note=''):
    s = state()
    a = {'started': now(), 't0': time.time()}
    if note:
        a['note'] = note
    s['runs'].setdefault(name, {}).setdefault(step, []).append(a)
    s['updated'] = now()
    write_json(STATE, s)


def step_ok(name, step, rc, cmd=None):
    if step == 'train':
        return bool(cmd) and is_trained(name, cmd)
    if step == 'engine':
        return rc == 0 and not need_engine(name)
    if step == 'check':
        c = load(LOGS / ('engine_check_%s.json' % name))
        acc = (c or {}).get('accuracy') or {}
        return rc == 0 and bool(c) and bool(acc.get('gate_actions_2p5us')) and bool(acc.get('finite'))
    if step in ('probe_best', 'probe_last'):
        return rc == 0 and not need_probe(name, step.split('_')[1])
    return rc == 0


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


def fails(name, step):
    return sum(1 for a in state()['runs'].get(name, {}).get(step, []) if 'ended' in a and not a.get('ok'))


def last_rc(name, step):
    atts = [a for a in state()['runs'].get(name, {}).get(step, []) if 'ended' in a]
    return atts[-1]['rc'] if atts else ''


def wall(name, step):
    w = [a.get('wall_s') for a in state()['runs'].get(name, {}).get(step, []) if a.get('wall_s')]
    return round(sum(w), 1) if w else None


def cmd_deviation(title, body):
    if SMOKE:
        print('試しなので deviations_v3.md には書きません: %s' % title)
        return 0
    s = state()
    done = s.setdefault('deviations', [])
    if title in done:
        print('もう書いてあります: %s' % title)
        return 0
    with open(DEVIATIONS, 'a', encoding='utf-8') as f:
        f.write('\n## (キュー v3 が自動で書いた)%s(%s)\n\n%s\n' % (title, now(), body.strip()))
        f.flush()
        os.fsync(f.fileno())
    done.append(title)
    write_json(STATE, s)
    print('deviations_v3.md に書き足しました: %s' % title)
    return 0


# ---------------------------------------------------------------- 始める前の確かめ

def sha_lines_ok(sha_file, base):
    bad = []
    for line in Path(sha_file).read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        want, path = line.split(None, 1)
        p = Path(path) if path.startswith('/') else Path(base) / path
        if not p.exists() or sha256(p) != want:
            bad.append(path)
    return bad


def cmd_preflight():
    ok, notes, bad = [], [], []
    proto = protocol()
    # 1. protocol_v3・protocol_v2・lock2 の固定(段 0 の 1)
    for f, base in ((PROTOCOL_SHA, REP), (REP_V2 / 'protocol_v2.sha256', REP_V2), (REP_V2 / 'protocol_v2_lock2.sha256', REP_V2)):
        b = sha_lines_ok(f, base)
        (bad.extend('固定のファイルが違う: %s' % x for x in b) if b else ok.append('固定(%s)' % f.name))
    # 2. 封
    r = subprocess.run([sys.executable, '-I', str(SEAL_CHECK)], capture_output=True, text=True)
    (ok if r.returncode == 0 else bad).append('封の 14 本(check_seal_v2.py rc=%d)' % r.returncode)
    # 3. v3 の表(tables_v3.json の sha256・protocol の sha256)とデータの確かめ(段 0 の 2・3)
    card = load(TABLES / 'tables_v3.json') or {}
    for f, ent in (card.get('files') or {}).items():
        if not (TABLES / f).exists() or sha256(TABLES / f) != ent.get('sha256'):
            bad.append('表の sha256 が tables_v3.json と違う: %s' % f)
    if not card:
        bad.append('data/derived/dataset_v3_20261026/tables_v3.json がない(data/make_v3_tables.py)')
    elif card.get('protocol_v3_sha256') != sha256(PROTOCOL):
        bad.append('tables_v3.json の protocol_v3_sha256 が今の protocol と違う')
    else:
        ok.append('v3 の表 4 つ(tables_v3.json の sha256)')
    for k, want in proto['stage0_tables_sha256'].items():
        p = REPO / ABL / k
        if not p.exists() or sha256(p) != want:
            bad.append('段 0 の表の sha256 が protocol と違う: %s' % k)
    dc = load(DATA_CHECK, {}) or {}
    if dc.get('all_ok'):
        ok.append('データの確かめ(check_v3_data.json: 数・封・作用の基準・差し替え・物差し)')
    else:
        (notes if SMOKE else bad).append('データの確かめ(data/check_v3_data.py)が無いか不合格')
    # 4. 物差し(probe_gates.py・probe_set.npz)の sha256
    if sha256(PROBE_GATES) == proto['probe']['script_sha256'] and sha256(PROBE_SET) == proto['probe']['set_sha256']:
        ok.append('物差し probe_gates.py・probe_set.npz の sha256')
    else:
        bad.append('物差しの sha256 が protocol_v3 と違う')
    # 5. 段 0 の 4(物差しの ON・OFF)
    oo = load(ONOFF, {}) or {}
    (ok if oo.get('ok') else notes).append('段 0 の 4(物差しの ON・OFF: %s)' % (oo.get('verdict') or '未'))
    # 6. B-FAST2
    ans, src = bf2_answer()
    if ans:
        if sha256(B_RUN / 'best.pth') != B_BEST_SHA:
            bad.append('B の best.pth の sha256 が protocol_v2 と違う(B-FAST2 の --init)')
        notes.append('BF2-0 が「学習する」(%s)' % src)
    else:
        notes.append('BF2-0: %s(%s)→ B-FAST2 の 6 行は飛ばす' % ('学習しない' if ans is False else '答えが無い', src))
    # 7. 表の行
    tb = check_tsv()
    if tb:
        bad.extend('表(gpu_queue_v3.tsv): %s' % b for b in tb)
    else:
        ok.append('gpu_queue_v3.tsv の 22 行が protocol と同じ(順番 B-FAST2 → 段 0 → 段 1)')
    # 8. train.py の引数
    need = {'--probe-npz', '--action-ref-from-samples', '--label-override-npz', '--drop-mask'}
    miss = sorted(need - train_py_flags())
    (bad.append('tools/train.py に無い引数: %s' % ' '.join(miss)) if miss else ok.append('train.py の引数(--probe-npz・--action-ref-from-samples)'))
    # 9. ディスク・lock2・D0
    free_gb = shutil.disk_usage(str(RUNS)).free / 1e9
    (ok if free_gb >= DISK_MIN_GB else bad).append('空きディスク %.1f GB(%.0f GB 以上)' % (free_gb, DISK_MIN_GB))
    if not LOCK2.exists():
        notes.append('protocol_v3_lock2.json はまだ無い(段 0 のあとに書く。それまで段 1 の行で止まる)')
    if not d0_order():
        notes.append('decisions_v3.json の D0.order はまだ無い(段 0 の短い学習のあと。それまで段 1 の行で止まる)')
    notes.append('電源: 夜の学習の前にユーザーが AC か満充電を確かめること(protocol_v3 段 0 の 9。キューは確かめられない)')
    for o in ok:
        print('OK  %s' % o)
    for n in notes:
        print('注  %s' % n)
    for b in bad:
        print('NG  %s' % b)
    return 1 if bad else 0


# ---------------------------------------------------------------- run.json・engine_check.json・走行の一覧

def cmd_runjson(name, cmd):
    run = run_dir(name)
    if (run / 'run.json').exists():
        print('run.json はあります: %s' % (run / 'run.json'))
        return 0
    from jetformer import policies
    args = train_args(cmd)
    m = meta_of(name)
    atts = state()['runs'].get(name, {}).get('train', [])
    created = next((a['started'] for a in atts if a.get('started')), None) or now()
    meta = {'dataset': Path(arg_value(args, '--data')).name, 'stride': int(arg_value(args, '--stride')),
            'epochs': int(arg_value(args, '--epochs')), 'batch': int(arg_value(args, '--batch')),
            'lr': float(arg_value(args, '--lr')), 'pretrained': '--no-pretrained' not in args, 'build_engine': stage_of(name) != '0',
            'command_input': '--no-command-input' not in args, 'mask_impacts': '--keep-impacts' not in args,
            'policy': arg_value(args, '--policy'), 'policy_cfg': policies.get(arg_value(args, '--policy')).resolve({}),
            'impact_g': float(arg_value(args, '--impact-g')), 'seed': int(arg_value(args, '--seed')),
            'name': name, 'created_at': created,
            'note': ('試し(キュー v3 の動作の確かめ・--max-steps。すぐ消す。走らせない)' if SMOKE else
                     '学習 v3(protocol_v3 の %s・段 %s)。机上の門・封(11/1)・ユーザーの了承の前は走らせない' % (m.get('arm'), m.get('stage'))),
            'experiment': 'fix_20261026', 'arm': m.get('arm'), 'stage': m.get('stage'), 'protocol_sha256': sha256(PROTOCOL)}
    tables = {}
    for key in TABLE_ARGS:
        v = arg_value(args, key)
        if v:
            meta[key.lstrip('-').replace('-', '_')] = v
            tables[key.lstrip('-').replace('-', '_')] = sha256(REPO / v)
    if tables:
        meta['tables_sha256'] = tables
    meta['action_ref_from_samples'] = '--action-ref-from-samples' in args
    meta['train_env'] = {'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True', 'MALLOC_ARENA_MAX': '2'}
    meta['code_sha256'] = {f: sha256(REPO / f) for f in CODE_FILES if (REPO / f).exists()}
    meta['queue_cmd'] = cmd
    write_json(run / 'run.json', meta)
    print('run.json を書きました: %s' % (run / 'run.json'))
    return 0


def cmd_enginecheck(name):
    run = run_dir(name)
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
           'engine': {'file': eng.name, 'size': st.st_size, 'mtime_ns': st.st_mtime_ns}, 'source': str(src),
           'method': '実フレーム 64 個で TensorRT(run_numpy)と PyTorch FP32(CPU・best.pth)の actions の差。速さは 100 回の'
                     '空回しの後 600 回(experiments/20261013_sim_cotrain/train/engine_load_check.py、10/26 キュー v3)'}
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


# ---------------------------------------------------------------- 状態・一覧

def current_rows():
    out = []
    first = smoke_row()
    for stage, name, cmd, commented in queue_rows(include_commented=True):
        if SMOKE and name != first:
            continue
        r = resolve(name) if not commented else ('SKIP', '削った')
        if r[0] == 'RUN':
            out.append((stage, r[1], r[2], name, commented, None))
        else:
            n, c = (smoke(name, cmd) if SMOKE else (name, cmd))
            out.append((stage, n, c, name, commented, '%s: %s' % r))
    return out


def row(stage, name, cmd, template, commented, why):
    run = run_dir(name)
    res = load(run / 'result.json') or {}
    hist = res.get('history') or []
    c = check_current(run)
    s = state()['runs'].get(name, {})
    d = s.get('drivable') or {}
    tstate = 'done' if is_trained(name, cmd) else ('resume' if (run / 'last.pth').exists() else 'fresh')
    m = meta_of(name)
    pb, pl = probe_result(name, 'best'), probe_result(name, 'last')
    return {
        'stage': stage, 'name': name, 'template': template, 'arm': m.get('arm'), 'seed': m.get('seed'), 'cut': commented,
        'resolve': why, 'train': tstate, 'run_dir': str(run), 'epochs_planned': epochs_in(cmd),
        'epochs_done': res.get('epochs_done'), 'best_val_us': res.get('best_val_us'), 'best_epoch': res.get('best_epoch'),
        'last_val_us': hist[-1]['val_us'] if hist else None,
        'train_samples': res.get('train_samples'), 'val_samples': res.get('val_samples'),
        'steer_center_us': res.get('steer_center_us'), 'throttle_neutral_us': res.get('throttle_neutral_us'),
        'probe_in_train_best': res.get('probe_best'), 'probe_best': pb, 'probe_last': pl,
        'peak_mem_mb': res.get('peak_mem_mb'), 'train_duration_s': res.get('duration_s'),
        'engine': (not need_engine(name)) if (run / 'jetformer.onnx').exists() else False,
        'engine_check': c and {k: c.get(k) for k in ('ok', 'accuracy_ok', 'speed_ok', 'actions_max_us', 'actions_p99_us',
                                                      'run_numpy_median_ms', 'run_numpy_p99_ms', 'checked_at')},
        'drivable': d.get('drivable'), 'listed': d.get('listed'),
        'wall_s': {st: wall(name, st) for st in ('train', 'engine', 'check', 'probe_best', 'probe_last')},
        'fails': {st: fails(name, st) for st in ('train', 'engine', 'check', 'probe_best', 'probe_last')},
        'estimate_h': round(hours_of(cmd), 2),
    }


def fmt_probe(p):
    if not p:
        return '—'
    return 'st10 %+.1f・st23 %+.1f・止 %.3f・低 %.3f' % (p['st10_thr_vs_b_us'], p['st23_thr_vs_b_us'], p['stand_brake_side_frac'],
                                                      p['low10_brake_frac'])


def cmd_status():
    for r in (row(*x) for x in current_rows()):
        ec = r['engine_check'] or {}
        print('%-4s %-34s 学習 %-6s ep %-4s best %-6s 中立 %-6s エンジン %-5s 確認 %-5s 物差し(best) %s%s' % (
            r['stage'], r['name'], r['train'], r['epochs_done'], r['best_val_us'], r['throttle_neutral_us'], r['engine'],
            ec.get('ok'), fmt_probe(r['probe_best']), '(%s)' % r['resolve'] if r['resolve'] else ''))
    return 0


def cmd_summary():
    rows = [row(*x) for x in current_rows()]
    write_json(SUMMARY, {'made_at': now(), 'd0_order': d0_order(), 'bf2_0': bf2_answer()[0], 'runs': rows})
    print('| 段 | 名前 | 学習 | L1 best [µs](エポック) | 中立 [µs] | 物差し best(probe_gates) | 物差し last | エンジンの差 [µs] | 走行の一覧 | 行の扱い |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for r in rows:
        ec = r['engine_check'] or {}
        print('| %s | %s | %s | %s(%s) | %s | %s | %s | %s | %s | %s |' % (
            r['stage'], r['name'] + ('(削った)' if r['cut'] else ''), r['train'], r['best_val_us'], r['best_epoch'],
            r['throttle_neutral_us'], fmt_probe(r['probe_best']), fmt_probe(r['probe_last']), ec.get('actions_max_us'),
            'drivable' if r['drivable'] else r['drivable'], r['resolve'] or 'RUN'))
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
    if c == 'rundir':
        print(run_dir(a[0]))
        return 0
    if c == 'post':
        print(post_steps(a[0]))
        return 0
    if c == 'trained':
        return 0 if is_trained(a[0], a[1]) else 1
    if c == 'fits':
        return cmd_fits(a[0], a[1])
    if c == 'lastpth':
        st, why = lastpth_state(a[0])
        print(st if why is None else '%s\t%s' % (st, why))
        return 0
    if c == 'need-engine':
        return 0 if need_engine(a[0]) else 1
    if c == 'need-check':
        return 0 if check_current(run_dir(a[0])) is None else 1
    if c == 'need-probe':
        return 0 if need_probe(a[0], a[1]) else 1
    if c == 'probe-out':
        print(probe_file(a[0], a[1]))
        return 0
    if c == 'predicted':                 # 机上の予測は eval_v3.py が決める(まだ無いので、あれば eval_v3.py predicted に任せる)
        if not EVAL_V3.exists():
            return 0
        r = subprocess.run([sys.executable, '-I', str(EVAL_V3), 'predicted', a[0]], capture_output=True, text=True)
        return r.returncode
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
