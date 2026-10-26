"""段 Decide(2026-10-23): A10 の「C1 の枠の決め方」を机上の JSON にそのまま当てる(標準ライブラリだけ・読むだけ)

    python3 -I decide_c1.py [--out logs/decide_c1.json]

- 門は protocol.json gates の式 + A1(家族 label_r1: 門 5 を l1_steer_best_us の差 < 10.0、brake_lowspeed_frac ≤ 0.01)で計算し直す。
  desk_lib・desk_eval・bench_report は import しない(独立に計算する)。
- SLOW の値は 10/21 の desk_bench.json。nonzone_thr_diff_us は 10/21 に無い欄なので 10/22 の desk_bench_20261022.json の値
  (A10 の新しい欄)。SLOW のほかの欄が 2 つのファイルで同じかも確かめる。
- 書くのは --out の JSON だけ。bench_candidates.json は変えない(変えるのは apply_c1.py)。
"""
import argparse
import hashlib
import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path

PLAN = Path('/home/jetson/jetformer_reports/bench_prep_20261021/plan')
PROTOCOL = PLAN / 'protocol.json'
PROTOCOL_SHA = PLAN / 'protocol.sha256'
AMEND = PLAN / 'protocol_amendments.md'
DESK21 = Path('/home/jetson/jetformer_reports/bench_prep_20261021/desk/desk_bench.json')
DESK22 = Path('/home/jetson/jetformer_reports/prep_20261022/desk/desk_bench_20261022.json')
TRAIN_LOGS = Path('/home/jetson/jetformer/experiments/20261022_prep/logs')
RUNS = Path('/home/jetson/jetformer/data/runs')
LABELS_SEC = Path('/home/jetson/jetformer/data/derived/bench_20261021/slow_labels_sections.npz')
LABELS_ALL = Path('/home/jetson/jetformer/data/derived/bench_20261021/slow_labels.npz')
BENCH = Path('/home/jetson/jetformer/data/bench_candidates.json')

# A10 の対象(id → run)。A10 の文の名前そのまま
A10_IDS = {'SLOW': 'bench_20261021-slow-s0', 'SLOW1': 'bench_20261021-slow-s1',
           'SLOW-SEC0': 'bench_20261022-slow-sec-s0', 'SLOW-SEC1': 'bench_20261022-slow-sec-s1'}
R1_APP_MIN = 20.0
L1_GATE_US = 10.0
BRAKE_MAX = 0.01
GATE_FIELDS = ('engine_ok', 'sanity_ok', 'trt_p99_ms', 'steer_amp_ratio', 'l1_steer_best_us', 'brake_lowspeed_frac')


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def ts(p):
    st = os.stat(p)
    return datetime.fromtimestamp(st.st_mtime).isoformat(timespec='seconds')


def rows(path):
    d = json.load(open(path, encoding='utf-8'))
    return d, {c['id']: c for c in d['candidates']}


def missing(v):
    return v is None or (isinstance(v, float) and not math.isfinite(v))


def gates(r, B):
    """protocol gates 1〜4 + 5(A1: l1_steer_best_us)+ A1 の brake_lowspeed_frac。式と値を返す"""
    fail, expr = [], []
    for k in GATE_FIELDS:
        if missing(r.get(k)):
            fail.append('missing:' + k)
    if missing(B.get('l1_steer_best_us')):
        fail.append('missing:B.l1_steer_best_us')
    if fail:
        return {'pass': False, 'fail': fail, 'expr': expr}
    def chk(name, ok, text):
        expr.append('%s %s' % (text, '○' if ok else '×'))
        if not ok:
            fail.append(name)
    chk('engine_ok', r['engine_ok'] is True, 'engine_ok = %s' % r['engine_ok'])
    chk('sanity_ok', r['sanity_ok'] is True, 'sanity_ok = %s' % r['sanity_ok'])
    chk('trt_p99_ms', r['trt_p99_ms'] <= 10.0, 'trt_p99_ms %.3f ≤ 10.0' % r['trt_p99_ms'])
    chk('steer_amp_ratio', 0.80 <= r['steer_amp_ratio'] <= 1.25, '0.80 ≤ steer_amp_ratio %.4f ≤ 1.25' % r['steer_amp_ratio'])
    d = r['l1_steer_best_us'] - B['l1_steer_best_us']
    chk('l1_steer_best_us', d < L1_GATE_US, 'l1_steer_best_us − B = %.3f − %.3f = %+.3f < 10.0'
        % (r['l1_steer_best_us'], B['l1_steer_best_us'], d))
    chk('brake_lowspeed_frac', r['brake_lowspeed_frac'] <= BRAKE_MAX, 'brake_lowspeed_frac %.5f ≤ 0.01' % r['brake_lowspeed_frac'])
    return {'pass': not fail, 'fail': fail, 'expr': expr, 'l1_steer_diff_us': round(d, 3)}


def brake_from_counts(r):
    """A1 の定義: 保留と 10/10 の AI 走行の 1.0 m/s 未満のフレームのうち、ブレーキ側(> 1598 µs)の割合(数から計算し直す)"""
    b = r.get('brake_lowspeed') or {}
    try:
        (hk, hn), (ak, an) = b['heldout'], b['ai_1010']
    except (KeyError, TypeError, ValueError):
        return None
    return {'k': hk + ak, 'n': hn + an, 'frac': (hk + ak) / (hn + an), 'heldout': [hk, hn], 'ai_1010': [ak, an]}


def first_train_start():
    """10/22 の新しい学習の最初の始まり: train_*.log の 1 行目の「==== 時刻 開始」の最も早いもの"""
    best = None
    for p in sorted(TRAIN_LOGS.glob('train_*.log')):
        line = open(p, encoding='utf-8').readline()
        m = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) 開始', line)
        if m:
            t = datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')
            if best is None or t < best[0]:
                best = (t, p, line.strip())
    return best


def train_args(run):
    """train ログの 1 行目から --label-override-npz と --seed を読む(A10 の作り方と合うか)"""
    p = TRAIN_LOGS / ('train_%s.log' % run)
    if not p.exists():
        return None
    line = open(p, encoding='utf-8').readline()
    lab = re.search(r'--label-override-npz (\S+)', line)
    seed = re.search(r'--seed (\d+)', line)
    data = re.search(r'--data (\S+)', line)
    return {'log': str(p), 'labels': lab and lab.group(1), 'seed': seed and int(seed.group(1)), 'data': data and data.group(1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(Path(__file__).with_name('logs') / 'decide_c1.json'))
    a = ap.parse_args()
    out = {'stage': 'Decide(prep_20261022)', 'created_at': datetime.now().isoformat(timespec='seconds'),
           'tool': 'experiments/20261022_prep/decide/decide_c1.py', 'checks': {}}
    ck = out['checks']

    # 1. protocol.json の sha256
    want = open(PROTOCOL_SHA).read().split()[0]
    got = sha256(PROTOCOL)
    ck['protocol_sha256'] = {'expected': want, 'actual': got, 'ok': want == got}

    # 2. A10 の文がファイルにあり、ファイルの時刻が新しい学習の始まりより前か
    text = open(AMEND, encoding='utf-8').read()
    a10 = re.search(r'^## A10.*?(?=^## |\Z)', text, flags=re.S | re.M)
    t0 = first_train_start()
    amend_mtime = datetime.fromtimestamp(os.stat(AMEND).st_mtime)
    d22, R22 = rows(DESK22)
    ck['a10'] = {
        'present': bool(a10), 'heading': a10.group(0).splitlines()[0] if a10 else None,
        'has_rule_lines': bool(a10 and all(s in a10.group(0) for s in ('A1 の門に合格', 'r1_app_thr_diff_us ≥ +20.0',
                                                                       'nonzone_thr_diff_us が小さい順', '小数第 1 位に四捨五入'))),
        'amend_file': str(AMEND), 'amend_mtime': amend_mtime.isoformat(timespec='seconds'),
        'amend_sha256': sha256(AMEND), 'amend_sha256_at_desk': d22.get('amendments_sha256'),
        'first_train_log': str(t0[1]) if t0 else None, 'first_train_start': t0[0].isoformat(timespec='seconds') if t0 else None,
        'amend_before_train': bool(t0 and amend_mtime < t0[0]),
    }
    ck['a10']['amend_same_as_desk'] = ck['a10']['amend_sha256'] == ck['a10']['amend_sha256_at_desk']
    ck['a10']['lead_s'] = round((t0[0] - amend_mtime).total_seconds(), 1) if t0 else None

    # 3. 値: B・SLOW は 10/21、SLOW1・SLOW-SEC0・SLOW-SEC1 は 10/22(SLOW の nonzone も 10/22)
    d21, R21 = rows(DESK21)
    B = R21['B']
    vals = {'SLOW': dict(R21['SLOW'], nonzone_thr_diff_us=R22['SLOW']['nonzone_thr_diff_us'])}
    for cid in ('SLOW1', 'SLOW-SEC0', 'SLOW-SEC1'):
        vals[cid] = R22[cid]
    same = {}
    for cid in ('B', 'SLOW'):
        want_keys = ('l1_best_us', 'l1_steer_best_us', 'l1_thr_best_us', 'l1_interv_us', 'r1_app_thr_diff_us',
                     'steer_amp_ratio', 'trt_p99_ms', 'engine_ok', 'sanity_ok', 'brake_lowspeed_frac',
                     'gates_pass', 'engine_sha256', 'run')
        keys = [k for k in want_keys if k in R21[cid] and k in R22[cid]]
        diff = [k for k in keys if R21[cid][k] != R22[cid][k]]
        same[cid] = {'compared_fields': len(keys), 'missing': [k for k in want_keys if k not in keys], 'diff': diff}
    ck['desk_21_vs_22'] = same
    ck['baseline_B'] = {'engine_ok': B['engine_ok'], 'sanity_ok': B['sanity_ok'], 'ok': B['engine_ok'] is True and B['sanity_ok'] is True}

    # 4. 候補ごと: 門・brake の数からの計算し直し・runs のエンジンが机上で見たものと同じか・作り方
    table = []
    for cid, run in A10_IDS.items():
        r = vals[cid]
        assert r['run'] == run, (cid, r['run'], run)
        g = gates(r, B)
        br = brake_from_counts(r)
        eng = RUNS / run / 'jetformer_fp16.engine'
        esha = sha256(eng) if eng.exists() else None
        ec = json.load(open(RUNS / run / 'engine_check.json'))
        st = os.stat(eng)
        ec_current = (ec.get('engine') or {}).get('size') == st.st_size and (ec.get('engine') or {}).get('mtime_ns') == st.st_mtime_ns
        row = {'id': cid, 'run': run, 'family': r.get('family'), 'source': str(DESK21 if cid == 'SLOW' else DESK22),
               'gate': g, 'gates_pass_in_desk': r.get('gates_pass'), 'gate_agrees_with_desk': bool(r.get('gates_pass')) == g['pass'],
               'r1_app_thr_diff_us': r['r1_app_thr_diff_us'], 'nonzone_thr_diff_us': r['nonzone_thr_diff_us'],
               'r1_app_ok': (not missing(r['r1_app_thr_diff_us'])) and r['r1_app_thr_diff_us'] >= R1_APP_MIN,
               'l1_steer_best_us': r['l1_steer_best_us'], 'brake_lowspeed_frac': r['brake_lowspeed_frac'],
               'brake_recount': br, 'brake_recount_agrees': br is not None and round(br['frac'], 5) == round(r['brake_lowspeed_frac'], 5),
               'steer_amp_ratio': r['steer_amp_ratio'], 'trt_p99_ms': r['trt_p99_ms'], 'engine_ok': r['engine_ok'], 'sanity_ok': r['sanity_ok'],
               'engine_sha256_now': esha, 'engine_sha256_desk': r.get('engine_sha256'), 'engine_same_as_desk': esha == r.get('engine_sha256'),
               'engine_check_current': ec_current, 'engine_check_ok': ec.get('ok'), 'engine_check_accuracy_ok': ec.get('accuracy_ok'),
               'train': train_args(run)}
        row['target'] = g['pass'] and row['r1_app_ok']
        table.append(row)

    # 5. A10 の 3: 小さい順(nonzone 小数第 1 位)→ r1_app 大きい順 → id
    targets = [t for t in table if t['target']]
    targets.sort(key=lambda t: (round(t['nonzone_thr_diff_us'], 1), -t['r1_app_thr_diff_us'], t['id']))
    for i, t in enumerate(targets, 1):
        t['rank'] = i
    for t in table:
        t.setdefault('rank', None)
        t['why_out'] = (None if t['target'] else
                        ('門 ' + ','.join(t['gate']['fail']) if not t['gate']['pass'] else 'r1_app_thr_diff_us < +20.0'))
    ties = [(a['id'], b['id']) for a, b in zip(targets, targets[1:]) if round(a['nonzone_thr_diff_us'], 1) == round(b['nonzone_thr_diff_us'], 1)]
    pick = targets[0] if targets else None
    bench = json.load(open(BENCH, encoding='utf-8'))
    cur = next((c for c in bench['candidates'] if c.get('slot') == 'C1'), None)
    out['table'] = table
    out['order'] = [t['id'] for t in targets]
    out['ties_on_nonzone'] = ties
    out['empty_fallback'] = not targets
    out['c1_id'] = pick['id'] if pick else 'SLOW'
    out['c1_run'] = pick['run'] if pick else A10_IDS['SLOW']
    out['current_c1'] = {'id': cur and cur['id'], 'run': cur and cur['run']}
    out['changed'] = out['c1_run'] != (cur and cur['run'])
    out['agrees_with_desk_a10'] = (d22.get('a10_c1') or {}).get('pick') == out['c1_id']
    labels = {'sections': {'path': str(LABELS_SEC), 'sha256': sha256(LABELS_SEC)}, 'all': {'path': str(LABELS_ALL), 'sha256': sha256(LABELS_ALL)}}
    labels['sections']['prefix_matches_a10'] = labels['sections']['sha256'].startswith('f090567f')
    ck['labels'] = labels
    ok_all = (ck['protocol_sha256']['ok'] and ck['a10']['present'] and ck['a10']['has_rule_lines'] and ck['a10']['amend_before_train']
              and ck['a10']['amend_same_as_desk'] and ck['baseline_B']['ok'] and all(not v['diff'] for v in ck['desk_21_vs_22'].values())
              and all(t['gate_agrees_with_desk'] and t['brake_recount_agrees'] and t['engine_same_as_desk'] and t['engine_check_current']
                      for t in table))
    out['all_checks_ok'] = ok_all
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    # 画面
    print('protocol sha256 ok:', ck['protocol_sha256']['ok'])
    print('A10:', ck['a10']['heading'], '| amend mtime', ck['a10']['amend_mtime'], '< first train', ck['a10']['first_train_start'],
          '=', ck['a10']['amend_before_train'], '| same as desk:', ck['a10']['amend_same_as_desk'])
    print('B baseline ok:', ck['baseline_B']['ok'], '| 10/21 vs 10/22 diff:', ck['desk_21_vs_22'])
    print('labels sections sha256 prefix f090567f:', labels['sections']['prefix_matches_a10'])
    for t in table:
        print('%-10s %-28s gate=%-5s %-28s r1_app=%+7.3f nonzone=%+7.3f target=%-5s rank=%s eng_same=%s ec_cur=%s brake_recount=%s train=%s'
              % (t['id'], t['run'], t['gate']['pass'], ','.join(t['gate']['fail']) or '-', t['r1_app_thr_diff_us'], t['nonzone_thr_diff_us'],
                 t['target'], t['rank'], t['engine_same_as_desk'], t['engine_check_current'], t['brake_recount_agrees'],
                 t['train'] and (Path(t['train']['labels'] or '').name, t['train']['seed'])))
        for e in t['gate']['expr']:
            print('    ', e)
    print('order:', out['order'], 'ties:', ties)
    print('C1 =', out['c1_id'], out['c1_run'], '| current', out['current_c1'], '| changed', out['changed'],
          '| agrees with desk a10:', out['agrees_with_desk_a10'], '| all checks ok:', ok_all)
    return 0 if ok_all else 1


if __name__ == '__main__':
    sys.exit(main())
