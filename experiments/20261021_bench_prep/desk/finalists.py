#!/usr/bin/env python3
"""段 Finalists(bench_prep_20261021): protocol.json の決まり(+ 事前登録の直し A1)を desk_bench.json に当てて、
10/23 の候補 5 本・順番・ブロックの並びを決め、data/bench_candidates.json を書く。標準ライブラリだけ(GPU・torch なし)。

    python3 -I finalists.py            # 決めるだけ(logs/finalists_select.json に結果)
    python3 -I finalists.py --write    # data/bench_candidates.json も書く

- 決まりの元は plan/protocol.json(sha256 が plan/protocol.sha256 と一致しなければ止める)と plan/protocol_amendments.md の A1。
- 決まりの外の判断はしない。門は desk_bench.json の値から式で計算し直す(gates.recompute)。
- 照合: desk_bench.json の gates_pass・tools/bench_report.py の recompute_gates(A1 入り)・plan/src/select_ref.py
  (A1 なし。SLOW の門だけ食い違うのが正しい)と比べ、結果を logs に残す。
- 書かないもの: drive_recommended.json・drive_settings.json・DEFAULT_PARAMS・protocol.json。
"""
import argparse
import hashlib
import importlib.util
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
JF = Path('/home/jetson/jetformer')
PREP = Path('/home/jetson/jetformer_reports/bench_prep_20261021')
PROTOCOL = PREP / 'plan' / 'protocol.json'
PROTOCOL_SHA = PREP / 'plan' / 'protocol.sha256'
DESK = PREP / 'desk' / 'desk_bench.json'
OUT_JSON = JF / 'data' / 'bench_candidates.json'
LOG = HERE / 'logs' / 'finalists_select.json'
JST = timezone(timedelta(hours=9))

# A1(plan/protocol_amendments.md、2026-10-21 11:30): 家族 label_r1 だけ門 5 を l1_steer_best_us で比べ、brake_lowspeed_frac ≤ 0.01 を足す
A1_FAMILIES = ('label_r1',)
A1_BRAKE_MAX = 0.01
L1_GATE_US = 10.0
NEW_POLICY_FILES = ('ens_average', 'ens_config')      # 2026-10-21 に jetformer/policies に足したファイル(risk_rule の low の条件)
RISK_RANK = {'low': 0, 'mid': 1, 'high': 2}
STEP3_FAMILIES = ('arch', 'robust', 'ens', 'weight')
RUN_FAMILY_RULE = (('-slow-', 'label_r1'), ('-wgt-', 'weight'), ('-aug-', 'robust'), ('-ens-', 'ens'))


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def num(x):
    if isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if not math.isfinite(v) else v


def rnd(x, n):
    """四捨五入(0.5 は 0 から遠い方へ)。select_ref.py と同じ"""
    if x is None:
        return None
    q = 10 ** n
    return math.copysign(math.floor(abs(x) * q + 0.5), x) / q


def family(row, pool_family):
    """protocol pool.family_rule: run の名前が先、無ければ pool の表"""
    run = str(row.get('run') or '')
    for key, fam in RUN_FAMILY_RULE:
        if key in run:
            return fam
    return pool_family.get(row['id'])


def gates(row, B, fam):
    """protocol gates.rules(5 つ)+ A1。返り値 (落ちた門の名前のリスト, 門 5 の値の説明)"""
    fails = []
    for k in ('engine_ok', 'sanity_ok'):
        v = row.get(k)
        if v is None:
            fails.append('missing:' + k)
        elif v is not True:
            fails.append(k)
    p99 = num(row.get('trt_p99_ms'))
    if p99 is None:
        fails.append('missing:trt_p99_ms')
    elif not p99 <= 10.0:
        fails.append('trt_p99_ms')
    amp = num(row.get('steer_amp_ratio'))
    if amp is None:
        fails.append('missing:steer_amp_ratio')
    elif not 0.80 <= amp <= 1.25:
        fails.append('steer_amp_ratio')
    a1 = fam in A1_FAMILIES
    key5 = 'l1_steer_best_us' if a1 else 'l1_best_us'
    v5, b5 = num(row.get(key5)), num(B.get(key5))
    if v5 is None:
        fails.append('missing:' + key5)
    elif b5 is None:
        fails.append('missing:B.' + key5)
    elif not (v5 - b5) < L1_GATE_US:
        fails.append(key5)
    detail = {'gate5_key': key5, 'gate5_diff_us': None if v5 is None or b5 is None else round(v5 - b5, 3), 'A1': a1}
    if a1:
        bl = num(row.get('brake_lowspeed_frac'))
        if bl is None:
            fails.append('missing:brake_lowspeed_frac')
        elif not bl <= A1_BRAKE_MAX:
            fails.append('brake_lowspeed_frac')
        detail['brake_lowspeed_frac'] = bl
    return fails, detail


def uses_new_policy_code(run):
    cfg = json.loads((JF / 'data' / 'runs' / run / 'model_config.json').read_text(encoding='utf-8'))
    return str(cfg.get('policy') or '') in NEW_POLICY_FILES, cfg.get('policy')


def risk(row, new_code):
    """protocol risk_rule"""
    amp, r1, p99 = num(row.get('steer_amp_ratio')), num(row.get('r1_app_thr_diff_us')), num(row.get('trt_p99_ms'))
    if (amp is not None and (amp < 0.85 or amp > 1.20)) or (r1 is not None and r1 < -20):
        return 'high'
    if r1 is None:
        return 'mid'
    if amp is not None and 0.90 <= amp <= 1.10 and r1 >= -10 and p99 is not None and p99 <= 8.0 and not new_code:
        return 'low'
    return 'mid'


def desk_key(row):
    """protocol selection.desk_key(6 つ。四捨五入してから比べる。null は最後)"""
    def asc(v, n):
        v = rnd(num(v), n)
        return (1, 0.0) if v is None else (0, v)

    def desc(v, n):
        v = rnd(num(v), n)
        return (1, 0.0) if v is None else (0, -v)
    amp = num(row.get('steer_amp_ratio'))
    lamp = rnd(abs(math.log(amp)), 3) if amp and amp > 0 else None
    return (asc(row.get('l1_interv_us'), 1), desc(row.get('r1_app_thr_diff_us'), 1), asc(row.get('l1_best_us'), 1),
            (1, 0.0) if lamp is None else (0, lamp), asc(row.get('trt_p99_ms'), 1), row['id'])


def engine_current(run):
    """engine_check.json が今のエンジンのものか(大きさ・mtime)と精度の合格(読み取りだけ)"""
    rd = JF / 'data' / 'runs' / run
    eng = rd / 'jetformer_fp16.engine'
    ck = json.loads((rd / 'engine_check.json').read_text(encoding='utf-8'))
    ce = ck.get('engine') if isinstance(ck.get('engine'), dict) else {}
    st = eng.stat()
    cur = (ce.get('mtime_ns') is not None and int(ce['mtime_ns']) == st.st_mtime_ns and ce.get('size') in (None, st.st_size))
    return {'ok': ck.get('ok'), 'accuracy_ok': ck.get('accuracy_ok'), 'actions_max_us': ck.get('actions_max_us'),
            'current': bool(cur), 'engine_sha256_prefix': sha256(eng)[:16]}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--write', action='store_true', help='data/bench_candidates.json を書く')
    ap.add_argument('--no-crosscheck', action='store_true', help='bench_report.py との照合を省く(jetformer を読まない)')
    a = ap.parse_args()

    # 0. 決まりが変わっていないか
    sha = sha256(PROTOCOL)
    want = PROTOCOL_SHA.read_text(encoding='utf-8').split()[0]
    if sha != want:
        sys.exit('protocol.json の sha256 が protocol.sha256 と違う(%s ≠ %s)。止める' % (sha, want))
    proto = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    desk = json.loads(DESK.read_text(encoding='utf-8'))
    if desk.get('protocol_sha256') != sha:
        sys.exit('desk_bench.json の protocol_sha256 が今の protocol.json と違う。止める')
    rows = {r['id']: r for r in desk['candidates']}
    pool = proto['pool']
    pool_family = {c['id']: c['family'] for c in pool['candidates']}
    pool_by_id = {c['id']: c for c in pool['candidates']}
    B = rows['B']
    out = {'protocol_sha256': sha, 'desk_bench_sha256': sha256(DESK), 'desk_created_at': desk.get('created_at'),
           'amendments': 'A1(plan/protocol_amendments.md)', 'baseline_check': {}, 'gates': {}, 'risk': {}, 'family': {},
           'desk_key': {}, 'how': {}, 'not_selected': {}, 'crosscheck': {}}

    # B は門に掛けない。engine_ok・sanity_ok が false なら 10/23 は走らずに知らせる(gates.baseline)
    out['baseline_check'] = {'engine_ok': B.get('engine_ok'), 'sanity_ok': B.get('sanity_ok'),
                             'engine_check_now': engine_current(B['run']),
                             'run_10_23': bool(B.get('engine_ok') is True and B.get('sanity_ok') is True)}
    if not out['baseline_check']['run_10_23']:
        sys.exit('B の engine_ok か sanity_ok が false。10/23 は走らずにユーザーに知らせる(protocol gates.baseline)')

    # step1: 門
    passed = []
    for cid in [c['id'] for c in pool['candidates']]:
        r = rows.get(cid)
        if r is None:
            out['gates'][cid] = {'gates_pass': False, 'gate_fail_reasons': ['missing:row']}
            continue
        fam = family(r, pool_family)
        f, det = gates(r, B, fam)
        new_code, pol = uses_new_policy_code(r['run'])
        out['family'][cid] = fam
        out['gates'][cid] = dict(gates_pass=not f, gate_fail_reasons=f, desk_gates_pass=r.get('gates_pass'),
                                 agrees_with_desk=(r.get('gates_pass') is (not f)), engine_check_now=engine_current(r['run']), **det)
        out['risk'][cid] = risk(r, new_code)
        out['risk_inputs'] = out.get('risk_inputs', {})
        out['risk_inputs'][cid] = {'steer_amp_ratio': r.get('steer_amp_ratio'), 'r1_app_thr_diff_us': r.get('r1_app_thr_diff_us'),
                                   'trt_p99_ms': r.get('trt_p99_ms'), 'policy': pol, 'new_policy_code': new_code}
        out['desk_key'][cid] = [list(x) if isinstance(x, tuple) else x for x in desk_key(r)]
        if not f:
            passed.append(r)
    b_new, b_pol = uses_new_policy_code(B['run'])
    out['risk']['B'] = risk(B, b_new)
    passed.sort(key=desk_key)
    out['passed_in_desk_key_order'] = [r['id'] for r in passed]

    # step2: SLOW(slow-s0)が通れば必ず。落ちて label_r1 の別の候補が通ればそれ
    chosen = []
    slow = [r for r in passed if str(r.get('run')) == 'bench_20261021-slow-s0']
    slow_other = [r for r in passed if out['family'][r['id']] == 'label_r1' and r not in slow]
    if slow:
        chosen.append(slow[0]); out['how'][slow[0]['id']] = 'step2_slow'
    elif slow_other:
        chosen.append(slow_other[0]); out['how'][slow_other[0]['id']] = 'step2_slow(slow-s1)'
    # step3: 残りの家族から desk_key のいちばん良い 1 本ずつ
    for fam in STEP3_FAMILIES:
        cand = [r for r in passed if out['family'][r['id']] == fam and r not in chosen]
        if cand and len(chosen) < proto['selection']['n_slots']:
            chosen.append(cand[0]); out['how'][cand[0]['id']] = 'step3_' + fam
    # step4: 足りなければ desk_key の順に足す
    for r in passed:
        if len(chosen) >= proto['selection']['n_slots']:
            break
        if r not in chosen:
            chosen.append(r); out['how'][r['id']] = 'step4_fill'
    out['chosen'] = [r['id'] for r in chosen]

    # 落ちた・入らなかった候補と理由
    for cid in pool_by_id:
        if cid in out['chosen']:
            continue
        g = out['gates'][cid]
        if not g['gates_pass']:
            out['not_selected'][cid] = 'step1_gate: ' + ','.join(g['gate_fail_reasons'])
            continue
        fam = out['family'][cid]
        winner = next((x for x in out['chosen'] if out['family'].get(x) == fam), None)
        out['not_selected'][cid] = ('step3_one_per_family: 家族 %s は %s(desk_key で上)。step4_fill は 5 本そろって使わない' % (fam, winner)
                                    if winner else 'step4_fill: 5 本がそろった')

    # 順番(order.rule)
    order = []
    first = [r for r in chosen if out['how'][r['id']].startswith('step2')]
    ens = [r for r in chosen if out['family'][r['id']] == 'ens' and r not in first]
    order += first[:1] + ens[:1]
    rest = sorted([r for r in chosen if r not in order], key=lambda r: (RISK_RANK[out['risk'][r['id']]], desk_key(r)))
    order += rest
    ids = [r['id'] for r in order]
    C = {'C%d' % (i + 1): x for i, x in enumerate(ids)}
    layout = proto['block_layout']['n%d' % len(ids)]
    blocks = [{'block': i + 1, 'id': C.get(x, x)} for i, x in enumerate(layout)]
    out['order'] = C
    out['blocks_2026_10_23'] = blocks

    # 照合 1: plan/src/select_ref.py(A1 なし)。SLOW の門だけ違い、ほかは同じはず
    sr = load_module(PREP / 'plan' / 'src' / 'select_ref.py', 'select_ref')
    ref = sr.select([dict(r) for r in desk['candidates']])
    out['crosscheck']['select_ref_no_A1'] = {
        'gates_differ': sorted(k for k, v in ref['gates'].items() if v['gates_pass'] != out['gates'][k]['gates_pass']),
        'risk_differ': sorted(k for k, v in ref['risk'].items() if v != out['risk'][k]),
        'chosen': ref['chosen'], 'order': ref['order']}
    # 照合 2: select_ref.py に A1 の門だけ差し替えた版(門以外は select_ref のまま)→ 選び方・順番が同じはず
    orig_gates = sr.gates

    def gates_a1(row, b_l1):
        if sr.family(row) in A1_FAMILIES:
            f, _ = gates(row, B, 'label_r1')
            return f
        return orig_gates(row, b_l1)
    sr.gates = gates_a1
    ref_a1 = sr.select([dict(r) for r in desk['candidates']])
    sr.gates = orig_gates
    out['crosscheck']['select_ref_with_A1_gates'] = {
        'chosen': ref_a1['chosen'], 'order': ref_a1['order'], 'blocks': ref_a1['blocks'],
        'same_order': ref_a1['order'] == C, 'same_blocks': ref_a1['blocks'] == blocks,
        'same_chosen_set': sorted(ref_a1['chosen']) == sorted(out['chosen'])}
    # 照合 3: tools/bench_report.py の recompute_gates(A1 入り。S3 の段)
    if not a.no_crosscheck:
        sys.path.insert(0, str(JF))
        sys.path.insert(0, str(JF / 'tools'))
        try:
            br = load_module(JF / 'tools' / 'bench_report.py', 'bench_report_x')
            rg = br.recompute_gates(rows, out['family'])
            out['crosscheck']['bench_report_recompute_gates'] = {
                'agree': all(rg[k]['gates_pass'] == out['gates'][k]['gates_pass'] for k in rg),
                'by_id': {k: [v['gates_pass'], v['gate_fail_reasons'], v.get('amendment')] for k, v in rg.items()}}
        except Exception as e:      # 照合ができなくても決め方は変えない(結果に書く)
            out['crosscheck']['bench_report_recompute_gates'] = {'error': repr(e)}

    ok = (out['crosscheck']['select_ref_with_A1_gates']['same_order'] and out['crosscheck']['select_ref_with_A1_gates']['same_blocks']
          and out['crosscheck']['select_ref_no_A1']['gates_differ'] in ([], ['SLOW'])
          and all(g.get('agrees_with_desk') for g in out['gates'].values())
          and out['crosscheck'].get('bench_report_recompute_gates', {}).get('agree', a.no_crosscheck))
    out['crosscheck_ok'] = bool(ok)

    # bench_candidates.json
    rs = proto['run_settings']
    params = dict(rs['baseline_params'])
    # 「ほかは DEFAULT_PARAMS」: protocol の baseline_params に無いのは rev_us・rev_s だけ(drive.py DEFAULT_PARAMS、assist.py の既定)
    params.setdefault('rev_us', 300.0)
    params.setdefault('rev_s', 0.4)
    assert params['zone_caps'] == 0 and params['speed_cap'] == 0.0 and params['steer_gain'] == 1.0
    now = datetime.now(JST).replace(microsecond=0).isoformat()
    cands = [{'id': 'B', 'label': pool['baseline']['label'], 'run': pool['baseline']['run'], 'hypothesis': '基準',
              'risk': out['risk']['B'], 'family': None, 'slot': '基準', 'selected_by': 'pool.baseline(毎日走る。門に掛けない)'}]
    for slot, cid in C.items():
        p = pool_by_id[cid]
        cands.append({'id': cid, 'label': p['label'], 'run': rows[cid]['run'], 'hypothesis': p['hypothesis'],
                      'risk': out['risk'][cid], 'family': out['family'][cid], 'slot': slot, 'selected_by': out['how'][cid]})
    bench = {
        'version': 1, 'bench': proto['bench'], 'protocol_sha256': sha, 'created_at': now,
        'baseline': {'mode_id': rs['baseline_mode_id'], 'params': params},
        'candidates': cands,
        'days': [{'date': proto['day1']['date'], 'blocks': blocks},
                 {'date': proto['day2']['date'], 'blocks': [],
                  'note': '10/23 の夜に決める(protocol ranking の day1_top2 で T1・T2。並びは B・T1・T2・B・T2・T1・B)'}],
        'description_template': rs['description_template'],
        'selection': {
            'stage': 'Finalists(bench_prep_20261021)',
            'rule': 'protocol.json selection・order・block_layout + plan/protocol_amendments.md A1(label_r1 の門)',
            'desk_bench': str(DESK), 'desk_bench_sha256': out['desk_bench_sha256'],
            'report': str(PREP / 'plan' / 'finalists.md'),
            'not_selected': [{'id': k, 'run': rows[k]['run'], 'reason': v} for k, v in out['not_selected'].items()],
        },
    }
    out['bench_candidates'] = bench
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(json.dumps({k: out[k] for k in ('chosen', 'order', 'blocks_2026_10_23', 'risk', 'not_selected', 'crosscheck_ok')},
                     ensure_ascii=False, indent=1))
    if a.write:
        if not ok:
            sys.exit('照合が合わないので data/bench_candidates.json は書かない(logs/finalists_select.json を見る)')
        tmp = OUT_JSON.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(bench, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
        tmp.replace(OUT_JSON)
        print('書いた:', OUT_JSON)


if __name__ == '__main__':
    main()
