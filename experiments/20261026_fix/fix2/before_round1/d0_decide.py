#!/usr/bin/env python3
"""段 0 のあとの決まり D0 と BF2-0 の答えを decisions_v3.json の形で出す(2026-10-26、係 d0)。
numpy も torch も使わない・GPU を使わない・車と装置に触れない。読むだけ(--write のときだけ decisions_v3.json を 1 回書く)。

    python3 -I d0_decide.py                 # 標準出力に decisions_v3.json の中身(書かない)
    python3 -I d0_decide.py --write         # /home/jetson/jetformer_reports/fix_20261026/decisions_v3.json に書く
                                            # (もうあれば止まる。上書きしない)
    python3 -I d0_decide.py --same-noTK-only "<理由>"
                                            # protocol_v3 stage0 6 の if_off「止まったら same・noTK だけで D0」。
                                            # noTK_noLB・tkB が止まったときだけ使う(案 D・案 2 は「未」)

読むもの(どれも読むだけ):
  段 0 の 4 本: experiments/20261026_fix/diag_data/out/ablation/runs/abl-{same,noTK,noTK_noLB,tkB}-e15-s0/
    result.json(state done・15 エポック・種 0・サンプルの数・表)・best.pth
  best.pth の物差し(probe_gates.py run):
    (a) キュー v3 の logs/probe/abl-*-e15-s0_best.json(gpu_queue_v3.sh の probe_one が書く)
    (b) ablation_short.sh の diag_data/out/ablation/probe_best.json(あれば)
    (c) result.json の probe_best(段 0 のコマンドに --probe-npz は無いので、ふつうは無い)
    あるものは全部、best.pth より新しく、同じ値(±0.01 µs・±0.0001)でないと止まる。1 つも無ければ止まる
  B の物差し: diag_data/out/probe_check.json の B(B 自身なので st23 = 0.0)。「st23 − B」は この値を引く
  BF2-0: experiments/20261026_fix/data/out/bf2_0.json(data/bf2_0_check.py。GPU なし)。決まりの式で数え直して同じか確かめる

決まり(protocol_v3.json decisions.D0。protocol_v3.md 10 章。この台本は文が同じかも確かめる):
  - same の st23 − B が +20 µs(生)未満なら「確かめにならない」
  - noTK の st23 − B ≤ same の 1/3、かつ noTK の stand_brake_side ≤ 0.10 → 案 1 を確かめた。tkB で同じ式 → 案 D
  - noTK_noLB の low10 ≤ 0.01 → 案 2 を確かめた
  - 腕・門・閾値は変えない。順だけ: 確かめた案の腕が先(両方なら v3d → v3n → v3b)。
    どちらも確かめられない・確かめにならない → v3b → v3d → v3n
  読み方(protocol に書いていないところ。報告 d0_lock2.md に書いた):
  - 案 1 = v3n(交代の直後を外す)、案 D = v3d(交代の直後のスロットルを B に)。案 2(37 を外す)は 3 つの腕に全部入るので順に効かない
  - 片方だけ確かめたときは、その腕を先にして、残りは protocol の runs の順(v3n → v3d → v3b)
    → 案 1 だけ: v3n → v3d → v3b、案 D だけ: v3d → v3n → v3b
  - 「確かめにならない」(same が +20 未満)は st23 を使う案 1・案 D に掛ける。案 2 は low10 だけで読む(順に効かない)
決まりの計算は別の書き方(recount_d0)でも数え、答えが違えば止まる(protocol_v3 16 章「決まりの計算は別のコードで数え直す」)。
"""
import argparse
import datetime
import hashlib
import json
import math
import os
import sys
from pathlib import Path

REPO = Path('/home/jetson/jetformer')
HERE = REPO / 'experiments' / '20261026_fix'
REP = Path('/home/jetson/jetformer_reports/fix_20261026')
PROTOCOL = REP / 'protocol_v3.json'
PROTOCOL_SHA = REP / 'protocol_v3.sha256'
ABL_RUNS = HERE / 'diag_data' / 'out' / 'ablation' / 'runs'
ABL_PROBE = HERE / 'diag_data' / 'out' / 'ablation' / 'probe_best.json'
QUEUE_PROBE = HERE / 'logs' / 'probe'
PROBE_CHECK = HERE / 'diag_data' / 'out' / 'probe_check.json'
BF2_0 = HERE / 'data' / 'out' / 'bf2_0.json'
BF2_0_SCRIPT = HERE / 'data' / 'bf2_0_check.py'
DIAG_BFAST = HERE / 'diag_bfast' / 'diag_bfast.py'
DECISIONS = REP / 'decisions_v3.json'
B_RUN = 'clean_driving_20261011-nocmd-s3-20261011-024626'

EPOCHS = 15
VAL = 1559
FRAMES = {'st10': 441, 'st23': 581, 'stand': 372, 'low10': 582}
FIELDS = ('st10_thr_vs_b_us', 'st23_thr_vs_b_us', 'stand_brake_side_frac', 'stand_go_frac', 'low10_brake_frac')
TOL = {'us': 0.01, 'frac': 0.0001}           # protocol_v3 stage0 5 と同じ物差しの許し
# 段 0 の 4 本(diag_data/ablation_short.sh・gpu_queue_v3.py ABL_ARMS・protocol_v3 5 章の表)
ARMS = {
    'same': {'n_train': 6506, 'drop': 'drop_v2h_same.npz', 'override': None, 'override_used': 0},
    'noTK': {'n_train': 6142, 'drop': 'drop_v2h_noTK.npz', 'override': None, 'override_used': 0},
    'noTK_noLB': {'n_train': 6105, 'drop': 'drop_v2h_noTK_noLB.npz', 'override': None, 'override_used': 0},
    'tkB': {'n_train': 6506, 'drop': 'drop_v2h_same.npz', 'override': 'override_tk_thr_B.npz', 'override_used': 3640},
}
# 決まりの数(protocol_v3.json decisions.D0.rule の文から。文が変わっていれば止まる)
VALID_MIN_US = 20.0
RATIO = 1.0 / 3.0
STAND_MAX = 0.10
LOW10_MAX = 0.01
RULE_D0 = [
    'same の st23 − B が +20 µs(生)未満なら「確かめにならない」',
    'noTK の st23 − B ≤ same の 1/3、かつ noTK の stand_brake_side ≤ 0.10 → 案 1 を確かめた。tkB で同じ式 → 案 D を確かめた',
    'noTK_noLB の low10 ≤ 0.01 → 案 2 を確かめた',
]
EFFECT_D0_HEAD = '腕・門・閾値は変えない。順: 確かめた案の腕を先に(両方なら v3d → v3n → v3b)。どちらも確かめられない・確かめにならない → v3b → v3d → v3n'
PROTOCOL_ARM_ORDER = ['v3n', 'v3d', 'v3b']   # protocol_v3.json runs の腕の順
PLAN_ARM = {'plan1': 'v3n', 'planD': 'v3d'}
# BF2-0(protocol_v3.json bfast2.BF2-0: 上限 ≥ +10 かつ 上限 × 0.35 ≥ +10)
BF2_MIN_US = 10.0
BF2_LEARN = 0.35


class Stop(Exception):
    """決まりどおりに出せない(4 本がそろっていない・壊れている・食い違う)。止まる"""


def now_iso():
    return datetime.datetime.now().astimezone().isoformat(timespec='seconds')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def entry(path):
    p = Path(path)
    if not p.exists():
        return {'path': str(p), 'missing': True}
    st = p.stat()
    return {'path': str(p), 'bytes': st.st_size, 'sha256': sha256(p),
            'mtime': datetime.datetime.fromtimestamp(st.st_mtime).strftime('%Y-%m-%d %H:%M:%S')}


def load_json(path, what):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise Stop('%s がありません: %s' % (what, path))
    except (OSError, ValueError) as e:
        raise Stop('%s が読めません(壊れている): %s(%s: %s)' % (what, path, type(e).__name__, e))


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


# ---------------------------------------------------------------- protocol

def check_protocol(path):
    """protocol_v3.json の sha256 が protocol_v3.sha256 と同じか・D0 の文がこの台本の決まりと同じか → (protocol, sha256)"""
    proto = load_json(path, 'protocol_v3.json')
    got = sha256(path)
    want = None
    if Path(path) == PROTOCOL and PROTOCOL_SHA.exists():
        for line in PROTOCOL_SHA.read_text(encoding='utf-8').splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1] == 'protocol_v3.json':
                want = parts[0]
        if want != got:
            raise Stop('protocol_v3.json の sha256 が固定と違います(%s ≠ %s)' % (got[:16], (want or '?')[:16]))
    d0 = proto.get('decisions', {}).get('D0', {})
    if d0.get('rule') != RULE_D0:
        raise Stop('protocol_v3.json decisions.D0.rule の文が、この台本の決まりと違います: %r' % d0.get('rule'))
    if not str(d0.get('effect', '')).startswith(EFFECT_D0_HEAD):
        raise Stop('protocol_v3.json decisions.D0.effect の文が、この台本の決まりと違います: %r' % d0.get('effect'))
    arms = [r['arm'] for r in proto.get('runs', [])]
    seen = []
    for a in arms:
        if a not in seen:
            seen.append(a)
    if seen != PROTOCOL_ARM_ORDER:
        raise Stop('protocol_v3.json runs の腕の順が %r(この台本は %r を見込む)' % (seen, PROTOCOL_ARM_ORDER))
    return proto, got


# ---------------------------------------------------------------- 段 0 の 4 本

def run_name(arm):
    return 'abl-%s-e%d-s0' % (arm, EPOCHS)


def check_probe_values(v, where):
    if not isinstance(v, dict):
        raise Stop('%s: 物差しの形が違います' % where)
    for k in FIELDS:
        if not finite(v.get(k)):
            raise Stop('%s: 物差しの %s が数でありません(%r)' % (where, k, v.get(k)))
    for k in ('stand_brake_side_frac', 'stand_go_frac', 'low10_brake_frac'):
        if not 0.0 <= float(v[k]) <= 1.0:
            raise Stop('%s: %s が 0〜1 の外(%r)' % (where, k, v[k]))
    n = v.get('n')
    if n is not None and n != FRAMES:
        raise Stop('%s: 物差しのフレームの数が %r(protocol は %r)' % (where, n, FRAMES))
    return {k: float(v[k]) for k in FIELDS}


def pick_by_basename(obj, name, where):
    if not isinstance(obj, dict):
        raise Stop('%s: 物差しの json の形が違います' % where)
    hit = [k for k in obj if Path(str(k)).name == name]
    if len(hit) != 1:
        raise Stop('%s: %s の物差しが %d 個(1 個のはず)' % (where, name, len(hit)))
    return obj[hit[0]]


def read_arm(arm, runs_dir, probe_dir, abl_probe):
    """→ 1 本の結果(物差し・確かめ)。そろっていない・壊れている・食い違うときは Stop"""
    spec = ARMS[arm]
    name = run_name(arm)
    run = Path(runs_dir) / name
    res = load_json(run / 'result.json', '%s の result.json' % name)
    bad = []
    if res.get('state') != 'done':
        bad.append('state が %r(done でない)' % res.get('state'))
    if res.get('epochs_done') != EPOCHS or res.get('epochs_planned', EPOCHS) != EPOCHS:
        bad.append('エポックが %r / %r(%d のはず)' % (res.get('epochs_done'), res.get('epochs_planned'), EPOCHS))
    if res.get('seed') != 0:
        bad.append('種が %r(0 のはず)' % res.get('seed'))
    if res.get('train_samples') != spec['n_train']:
        bad.append('学習サンプルが %r(%d のはず)' % (res.get('train_samples'), spec['n_train']))
    if res.get('val_samples') != VAL:
        bad.append('検証サンプルが %r(%d のはず)' % (res.get('val_samples'), VAL))
    drop = str((res.get('rl_options') or {}).get('drop_mask') or '')
    if Path(drop).name != spec['drop']:
        bad.append('drop_mask が %r(%s のはず)' % (drop, spec['drop']))
    bo = res.get('bench_options') or {}
    ov = bo.get('label_override_npz')
    if (Path(str(ov)).name if ov else None) != spec['override']:
        bad.append('label_override_npz が %r(%r のはず)' % (ov, spec['override']))
    rows = bo.get('label_overridden_rows')
    if isinstance(rows, dict) and 'used' in rows and rows['used'] != spec['override_used']:
        bad.append('差し替えに使った行が %r(%d のはず)' % (rows['used'], spec['override_used']))
    best = run / 'best.pth'
    if not best.exists() or best.stat().st_size == 0:
        bad.append('best.pth が無いか 0 バイト')
    if bad:
        raise Stop('%s が壊れているか、まだ終わっていません: %s' % (name, '・'.join(bad)))
    best_ns = best.stat().st_mtime_ns

    sources = {}
    qf = Path(probe_dir) / ('%s_best.json' % name)
    if qf.exists():
        if qf.stat().st_mtime_ns < best_ns:
            raise Stop('%s: キューの物差し %s が best.pth より古い(学習し直したあと流していない)' % (name, qf))
        sources['queue'] = (check_probe_values(pick_by_basename(load_json(qf, '物差し'), name, str(qf)), str(qf)), entry(qf))
    if abl_probe and Path(abl_probe).exists():
        ap = Path(abl_probe)
        obj = load_json(ap, '物差し')
        if any(Path(str(k)).name == name for k in (obj if isinstance(obj, dict) else {})):
            if ap.stat().st_mtime_ns < best_ns:
                raise Stop('%s: ablation_short.sh の物差し %s が best.pth より古い' % (name, ap))
            sources['ablation_short'] = (check_probe_values(pick_by_basename(obj, name, str(ap)), str(ap)), entry(ap))
    if isinstance(res.get('probe_best'), dict):
        sources['result_json'] = (check_probe_values(res['probe_best'], '%s の result.json probe_best' % name),
                                  {'path': str(run / 'result.json')})
    if not sources:
        raise Stop('%s: best.pth の物差しがありません(%s)。キューの probe_one が済んでいない' % (name, qf))
    vals = [v for v, _ in sources.values()]
    for v in vals[1:]:
        for k in FIELDS:
            tol = TOL['us'] if k.endswith('_us') else TOL['frac']
            if abs(v[k] - vals[0][k]) > tol + 1e-12:
                raise Stop('%s: 物差しの出どころで %s が違います(%s)' % (
                    name, k, '・'.join('%s %r' % (s, x[0][k]) for s, x in sources.items())))
    return {
        'name': name,
        'probe_best': vals[0],
        'probe_sources': {s: e for s, (_, e) in sources.items()},
        'result': {k: res.get(k) for k in ('train_samples', 'val_samples', 'epochs_done', 'best_epoch', 'best_val_us', 'seed',
                                           'steer_center_us', 'throttle_neutral_us', 'duration_s')},
        'result_json': entry(run / 'result.json'),
        'best_pth': {'path': str(best), 'bytes': best.stat().st_size,
                     'mtime': datetime.datetime.fromtimestamp(best.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')},
    }


def b_reference(probe_check):
    """B の物差し(B 自身)。st23 は 0.0 のはず"""
    pc = load_json(probe_check, 'probe_check.json')
    v = check_probe_values(pc.get(B_RUN), 'probe_check.json の B')
    if abs(v['st23_thr_vs_b_us']) > TOL['us'] or abs(v['st10_thr_vs_b_us']) > TOL['us']:
        raise Stop('probe_check.json の B の st10・st23 が 0 でありません(%r・%r)' % (v['st10_thr_vs_b_us'], v['st23_thr_vs_b_us']))
    return v


# ---------------------------------------------------------------- 決まり

def order_from(plan1, planD, valid):
    if not valid or not (plan1 or planD):
        return ['v3b', 'v3d', 'v3n']
    if plan1 and planD:
        return ['v3d', 'v3n', 'v3b']
    first = PLAN_ARM['plan1'] if plan1 else PLAN_ARM['planD']
    return [first] + [a for a in PROTOCOL_ARM_ORDER if a != first]


def judge_d0(v, b_st23, partial=False):
    """v = {腕: probe_best}。→ D0 の答え(順・確かめた案・数)"""
    d = {arm: v[arm]['st23_thr_vs_b_us'] - b_st23 for arm in v}
    same_d = d['same']
    valid = same_d >= VALID_MIN_US
    lim = same_d * RATIO

    def plan_ok(arm):
        return valid and d[arm] <= lim and v[arm]['stand_brake_side_frac'] <= STAND_MAX

    plan1 = plan_ok('noTK')
    planD = None if partial else plan_ok('tkB')
    plan2 = None if partial else (v['noTK_noLB']['low10_brake_frac'] <= LOW10_MAX)
    order = order_from(plan1, bool(planD), valid)
    return {'valid': valid, 'plan1': plan1, 'planD': planD, 'plan2': plan2, 'order': order,
            'st23_minus_b_us': {a: round(x, 4) for a, x in d.items()}, 'same_third_us': round(lim, 4)}


def recount_d0(v, b_st23, partial=False):
    """judge_d0 と別の書き方(割り算をしない・表で引く)。答えが同じか確かめるため"""
    s = v['same']['st23_thr_vs_b_us'] - b_st23
    ok_base = not (s < 20.0)

    def confirmed(arm):
        x = v[arm]['st23_thr_vs_b_us'] - b_st23
        return bool(ok_base and 3.0 * x <= s and not (v[arm]['stand_brake_side_frac'] > 0.10))

    p1 = confirmed('noTK')
    pD = False if partial else confirmed('tkB')
    p2 = None if partial else not (v['noTK_noLB']['low10_brake_frac'] > 0.01)
    table = {(False, False): ('v3b', 'v3d', 'v3n'), (True, False): ('v3n', 'v3d', 'v3b'),
             (False, True): ('v3d', 'v3n', 'v3b'), (True, True): ('v3d', 'v3n', 'v3b')}
    order = list(table[(p1, pD)]) if ok_base else ['v3b', 'v3d', 'v3n']
    return {'valid': ok_base, 'plan1': p1, 'planD': None if partial else pD, 'plan2': p2, 'order': order}


def why_d0(j, v, partial):
    s = j['st23_minus_b_us']
    parts = ['same の st23 − B = %+.2f µs(%s +20)' % (s['same'], '≥' if j['valid'] else '<')]
    if not j['valid']:
        parts.append('→ 確かめにならない(案 1・案 D)')
    else:
        parts.append('same の 1/3 = %+.2f µs' % j['same_third_us'])
        parts.append('noTK %+.2f µs・stand_brake_side %.4f → 案 1 %s' % (
            s['noTK'], v['noTK']['stand_brake_side_frac'], '確かめた' if j['plan1'] else '確かめられない'))
        if partial:
            parts.append('tkB は止まったので見ない(案 D は未)')
        else:
            parts.append('tkB %+.2f µs・stand_brake_side %.4f → 案 D %s' % (
                s['tkB'], v['tkB']['stand_brake_side_frac'], '確かめた' if j['planD'] else '確かめられない'))
    if partial:
        parts.append('noTK_noLB は止まったので見ない(案 2 は未)')
    else:
        parts.append('noTK_noLB の low10 %.4f → 案 2 %s(順には効かない)' % (
            v['noTK_noLB']['low10_brake_frac'], '確かめた' if j['plan2'] else '確かめられない'))
    parts.append('順 %s' % ' → '.join(j['order']))
    return '。'.join(parts)


# ---------------------------------------------------------------- BF2-0

def judge_bf2(path, proto):
    b = load_json(path, 'bf2_0.json')
    rule = proto.get('bfast2', {}).get('BF2-0', {})
    bad = []
    if Path(path) == BF2_0:
        if b.get('script_sha256') != sha256(BF2_0_SCRIPT):
            bad.append('bf2_0_check.py の sha256 が答えを出したときと違う')
        if b.get('diag_bfast_sha256') != rule.get('script_sha256') or sha256(DIAG_BFAST) != rule.get('script_sha256'):
            bad.append('diag_bfast.py の sha256 が protocol_v3 と違う')
    bounds = b.get('bound_g090_us') or {}
    passed = {}
    for c in ('A', 'B'):
        x = bounds.get(c)
        if not finite(x):
            bad.append('上限 %s が数でない(%r)' % (c, x))
            continue
        passed[c] = bool(x >= BF2_MIN_US and x * BF2_LEARN >= BF2_MIN_US)
    if bad:
        raise Stop('BF2-0 の答え(%s)が使えません: %s' % (path, '・'.join(bad)))
    train = any(passed.values())
    if b.get('passed') != passed or b.get('train_bfast2') is not train:
        raise Stop('BF2-0: 決まりの式で数え直した答え(合格 %r・学習 %r)が bf2_0.json(%r・%r)と違います' % (
            passed, train, b.get('passed'), b.get('train_bfast2')))
    need = BF2_MIN_US / BF2_LEARN
    return {
        'train': train,
        'status': '決まった',
        'rule': rule.get('rule'),
        'values': {'bound_g090_us': {c: bounds[c] for c in ('A', 'B')}, 'bound_current_g090_us': bounds.get('current'),
                   'need_us': round(need, 3), 'passed': passed, 'frames': b.get('frames')},
        'candidate': b.get('candidate'),
        'why': '上限 A %+.3f・B %+.3f µs。要るのは ≥ +%.2f µs(上限 ≥ +10 かつ 上限 × 0.35 ≥ +10)→ %s' % (
            bounds['A'], bounds['B'], need, '学習する(候補 %s)' % b.get('candidate') if train else 'どちらも通らない → B-FAST2 は学習しない(GPU 0)'),
        'answered_at': b.get('made_at'),
        'src': entry(path),
        'scripts': {'bf2_0_check.py': entry(BF2_0_SCRIPT), 'diag_bfast.py': entry(DIAG_BFAST)} if Path(path) == BF2_0 else {},
    }


# ---------------------------------------------------------------- まとめ

def build(args):
    proto, proto_sha = check_protocol(args.protocol)
    arms = ['same', 'noTK'] if args.same_noTK_only else list(ARMS)
    runs, probe = {}, {}
    for arm in arms:
        r = read_arm(arm, args.runs_dir, args.probe_dir, args.abl_probe)
        runs[arm] = r
        probe[arm] = r['probe_best']
    partial = bool(args.same_noTK_only)
    skipped = {}
    if partial:
        for arm in ('noTK_noLB', 'tkB'):
            try:
                read_arm(arm, args.runs_dir, args.probe_dir, args.abl_probe)
            except Stop as e:
                skipped[arm] = str(e)
            else:
                raise Stop('--same-noTK-only ですが %s はそろっています。4 本で決めてください(protocol_v3 stage0 6)' % run_name(arm))
    b = b_reference(args.probe_check)
    j = judge_d0(probe, b['st23_thr_vs_b_us'], partial)
    r = recount_d0(probe, b['st23_thr_vs_b_us'], partial)
    for k in ('valid', 'plan1', 'planD', 'plan2', 'order'):
        if j[k] != r[k]:
            raise Stop('D0 の数え直しが合いません(%s: %r ≠ %r)' % (k, j[k], r[k]))
    d0rule = proto['decisions']['D0']
    d0 = {
        'order': j['order'],
        'status': '決まった' + ('(same・noTK だけ。protocol_v3 stage0 6 の if_off)' if partial else ''),
        'rule': d0rule['rule'],
        'effect': d0rule['effect'],
        'note': d0rule.get('note'),
        'why': why_d0(j, probe, partial),
        'confirmed': {
            'valid': j['valid'],
            '案 1(noTK → v3n)': j['plan1'],
            '案 D(tkB → v3d)': j['planD'],
            '案 2(noTK_noLB。順には効かない)': j['plan2'],
        },
        'values': {
            'b_probe_st23_us': b['st23_thr_vs_b_us'],
            'st23_minus_b_us': j['st23_minus_b_us'],
            'same_third_us': j['same_third_us'],
            'thresholds': {'valid_min_us': VALID_MIN_US, 'ratio': '1/3', 'stand_brake_side_max': STAND_MAX,
                           'low10_max': LOW10_MAX},
            'probe_best': probe,
        },
        'interpretation': [
            '案 1 = v3n、案 D = v3d(protocol_v3 5 章の腕の理由)。案 2(37 を外す)は 3 つの腕に全部入るので順に効かない',
            '片方だけ確かめたときは、その腕を先にして残りは protocol の runs の順(v3n → v3d → v3b)。案 1 だけ v3n → v3d → v3b・案 D だけ v3d → v3n → v3b',
            '「確かめにならない」(same が +20 µs 未満)は st23 を使う案 1・案 D に掛ける。案 2 は low10 だけで読む',
            '1 本・15 エポックの読み。腕・門・閾値は変えない。変えるのはキューの段 1 の順だけ',
        ],
        'recount': {'by': 'recount_d0(割り算なし・表で引く別の書き方)', 'same': True},
        'runs': runs,
        'written_at': now_iso(),
    }
    if partial:
        d0['partial'] = {'why': args.same_noTK_only, 'not_used': skipped}
    out = {
        'schema': 'jf-train-v3-decisions/v1',
        'written_at': now_iso(),
        'written_by': 'workflow のサブエージェント(係 d0)。experiments/20261026_fix/fix2/d0_decide.py(読むだけ・GPU なし)',
        'protocol_v3_sha256': proto_sha,
        'note': 'キュー v3 が読むのは D0.order(腕 3 つ)と "BF2-0".train(true/false)だけ(gpu_queue_v3.py d0_order・bf2_answer)。'
                'E1〜E5 はあとで足す(腕の種がそろったとき・11/1)',
        'D0': d0,
        'BF2-0': judge_bf2(args.bf2, proto),
        'pending': {
            'E1': '腕の種がそろったとき(10/27)',
            'E2': '10/27(遅れても 10/28)',
            'E3': 'E2 のあと',
            'E4': 'B-FAST2 を学習したときだけ(BF2-0 が「学習しない」なら無し)',
            'E5': '11/1 朝(封を開けたあと)',
        },
        'script': entry(Path(__file__).resolve()),
    }
    return out


def queue_view(out):
    """gpu_queue_v3.py の d0_order・bf2_answer と同じ読み方で、キューに読めるか"""
    order = out.get('D0', {}).get('order')
    bf2 = out.get('BF2-0', {}).get('train')
    return isinstance(order, list) and sorted(order) == ['v3b', 'v3d', 'v3n'] and isinstance(bf2, bool)


def main(argv=None):
    ap = argparse.ArgumentParser(description='段 0 のあとの D0・BF2-0 → decisions_v3.json')
    ap.add_argument('--write', action='store_true', help='decisions_v3.json に書く(もうあれば止まる)')
    ap.add_argument('--out', default=str(DECISIONS), help='書く場所(試験のときだけ変える)')
    ap.add_argument('--same-noTK-only', default=None, metavar='理由',
                    help='protocol_v3 stage0 6 の if_off: noTK_noLB・tkB が止まったときだけ。same・noTK で D0')
    ap.add_argument('--protocol', default=str(PROTOCOL))
    ap.add_argument('--runs-dir', default=str(ABL_RUNS))
    ap.add_argument('--probe-dir', default=str(QUEUE_PROBE))
    ap.add_argument('--abl-probe', default=str(ABL_PROBE))
    ap.add_argument('--probe-check', default=str(PROBE_CHECK))
    ap.add_argument('--bf2', default=str(BF2_0))
    args = ap.parse_args(argv)
    try:
        out = build(args)
    except Stop as e:
        print('止まりました: %s' % e, file=sys.stderr)
        return 2
    if not queue_view(out):
        print('止まりました: キュー v3 が読めない形になりました', file=sys.stderr)
        return 2
    text = json.dumps(out, ensure_ascii=False, indent=1) + '\n'
    dest = Path(args.out)
    if dest.exists():
        try:
            old = json.loads(dest.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            old = None
        same = isinstance(old, dict) and old.get('D0', {}).get('order') == out['D0']['order'] and \
            old.get('BF2-0', {}).get('train') == out['BF2-0']['train']
        print('注意: %s はもうあります(D0.order・BF2-0.train は %s)。上書きしません' % (
            dest, '同じ' if same else '違う: もとは %r・%r' % ((old or {}).get('D0', {}).get('order'),
                                                              (old or {}).get('BF2-0', {}).get('train'))), file=sys.stderr)
        if args.write:
            sys.stdout.write(text)
            return 3
    if args.write and not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + '.tmp-d0')
        with open(tmp, 'x', encoding='utf-8') as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(tmp, dest)        # もうあれば失敗する(上書きしない)
        except FileExistsError:
            os.unlink(tmp)
            print('止まりました: 書く間に %s ができました。上書きしません' % dest, file=sys.stderr)
            return 3
        os.unlink(tmp)
        print('書きました: %s(D0.order %s・BF2-0.train %s)' % (dest, ' → '.join(out['D0']['order']), out['BF2-0']['train']),
              file=sys.stderr)
    sys.stdout.write(text)
    return 0


if __name__ == '__main__':
    sys.exit(main())
