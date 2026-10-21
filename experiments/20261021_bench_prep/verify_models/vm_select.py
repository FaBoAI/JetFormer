#!/usr/bin/env python3
"""V-models: 選び方と順番を protocol.json + A1 から自分で計算し直し、data/bench_candidates.json と比べる(標準ライブラリだけ)。
select_ref.py・finalists.py・desk_lib.py・bench_report.py を import しない。"""
import hashlib, json, math, os, sys, re
from pathlib import Path
PLAN = Path('/home/jetson/jetformer_reports/bench_prep_20261021/plan')
DESKJ = Path('/home/jetson/jetformer_reports/bench_prep_20261021/desk/desk_bench.json')
BC = Path('/home/jetson/jetformer/data/bench_candidates.json')
OUT = Path(sys.argv[1])
P = json.load(open(PLAN / 'protocol.json'))
D = json.load(open(DESKJ))
B = json.load(open(BC))
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
res = {}
res['protocol_sha256'] = sha(PLAN / 'protocol.json')
res['protocol_sha256_file'] = open(PLAN / 'protocol.sha256').read().split()[0]
res['bc_protocol_sha256'] = B['protocol_sha256']
res['desk_protocol_sha256'] = D['protocol_sha256']
res['desk_sha256'] = sha(DESKJ); res['bc_desk_sha256'] = B['selection']['desk_bench_sha256']
mt = {n: os.stat(p).st_mtime for n, p in (('protocol.json', PLAN / 'protocol.json'), ('protocol_amendments.md', PLAN / 'protocol_amendments.md'),
                                           ('desk_bench.json', DESKJ), ('bench_candidates.json', BC))}
res['mtimes'] = mt
res['protocol_before_desk'] = mt['protocol.json'] < mt['desk_bench.json']
res['amend_before_desk'] = mt['protocol_amendments.md'] < mt['desk_bench.json']
res['protocol_written_at'] = P['written_at']; res['desk_created_at'] = D['created_at']; res['bc_created_at'] = B['created_at']
rows = {c['id']: c for c in D['candidates']}
pool = {c['id']: c for c in P['pool']['candidates']}

def fam(cid):
    run = pool[cid]['run']
    for k, f in (('-slow-', 'label_r1'), ('-wgt-', 'weight'), ('-aug-', 'robust'), ('-ens-', 'ens')):
        if k in run:
            return f
    return pool[cid]['family']

def num(x):
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None

base = rows['B']
def gate(cid):
    r, f = rows[cid], fam(cid)
    fails = []
    for k in ('engine_ok', 'sanity_ok'):
        if r.get(k) is not True:
            fails.append(k if r.get(k) is not None else 'missing:' + k)
    p99 = num(r.get('trt_p99_ms'))
    if p99 is None or p99 > 10.0: fails.append('trt_p99_ms')
    a = num(r.get('steer_amp_ratio'))
    if a is None or not (0.80 <= a <= 1.25): fails.append('steer_amp_ratio')
    key = 'l1_steer_best_us' if f == 'label_r1' else 'l1_best_us'
    x, y = num(r.get(key)), num(base.get(key))
    if x is None or y is None or not (x - y < 10.0): fails.append(key)
    if f == 'label_r1':
        bf = num(r.get('brake_lowspeed_frac'))
        if bf is None or bf > 0.01: fails.append('brake_lowspeed_frac')
    return not fails, fails

def gate_no_a1(cid):
    r = rows[cid]
    x, y = num(r.get('l1_best_us')), num(base.get('l1_best_us'))
    return x - y < 10.0

def r1(x, nd):
    return None if x is None else round(x, nd)
def dkey(cid):
    r = rows[cid]
    li, rd, lb = num(r.get('l1_interv_us')), num(r.get('r1_app_thr_diff_us')), num(r.get('l1_best_us'))
    amp, p99 = num(r.get('steer_amp_ratio')), num(r.get('trt_p99_ms'))
    return ((li is None, r1(li, 1) if li is not None else 0),
            (rd is None, -r1(rd, 1) if rd is not None else 0),
            (lb is None, r1(lb, 1) if lb is not None else 0),
            (amp is None, round(abs(math.log(amp)), 3) if amp else 0),
            (p99 is None, r1(p99, 1) if p99 is not None else 0), cid)

def risk(cid):
    r = rows[cid]
    amp, rd, p99 = num(r.get('steer_amp_ratio')), num(r.get('r1_app_thr_diff_us')), num(r.get('trt_p99_ms'))
    if (amp is not None and (amp < 0.85 or amp > 1.20)) or (rd is not None and rd < -20):
        return 'high'
    new_code = pool[cid]['run'] == 'bench_20261021-ens-p2'   # ens_average(10/21 に足した方策のファイル)
    if rd is None:
        return 'mid'
    if 0.90 <= amp <= 1.10 and rd >= -10 and p99 <= 8.0 and not new_code:
        return 'low'
    return 'mid'

ids = [c for c in pool]
G = {c: gate(c) for c in ids}
res['gates'] = {c: {'pass': G[c][0], 'fails': G[c][1], 'desk_gates_pass': rows[c].get('gates_pass'),
                    'pass_without_A1': gate_no_a1(c) if fam(c) == 'label_r1' else None, 'family': fam(c), 'risk': risk(c)} for c in ids}
passed = [c for c in ids if G[c][0]]
chosen = []
if 'SLOW' in passed:
    chosen.append(('SLOW', 'step2'))
else:
    alt = [c for c in passed if fam(c) == 'label_r1' and c != 'SLOW']
    if alt: chosen.append((sorted(alt, key=dkey)[0], 'step2'))
for f in ('arch', 'robust', 'ens', 'weight'):
    cs = sorted([c for c in passed if fam(c) == f], key=dkey)
    if cs: chosen.append((cs[0], 'step3_' + f))
rest = sorted([c for c in passed if c not in [x for x, _ in chosen]], key=dkey)
while len(chosen) < 5 and rest:
    chosen.append((rest.pop(0), 'step4'))
sel = [c for c, _ in chosen][:5]
order = []
if any(fam(c) == 'label_r1' for c in sel): order.append([c for c in sel if fam(c) == 'label_r1'][0])
if 'ENS' in sel: order.append('ENS')
rk = {'low': 0, 'mid': 1, 'high': 2}
order += sorted([c for c in sel if c not in order], key=lambda c: (rk[risk(c)], dkey(c)))
lay = P['block_layout']['n%d' % len(order)]
slots = {'C%d' % (i + 1): c for i, c in enumerate(order)}
blocks = [slots.get(x, x) for x in lay]
bc_blocks = [b['id'] for b in B['days'][0]['blocks']]
res['selected_mine'] = sel; res['order_mine'] = order; res['blocks_mine'] = blocks; res['blocks_bc'] = bc_blocks
res['blocks_same'] = blocks == bc_blocks
res['bc_block_numbers_ok'] = [b['block'] for b in B['days'][0]['blocks']] == list(range(1, len(bc_blocks) + 1))
bcc = {c['id']: c for c in B['candidates']}
res['bc_candidate_checks'] = {}
for cid, c in bcc.items():
    pc = pool.get(cid) or P['pool']['baseline']
    res['bc_candidate_checks'][cid] = {'run_same': c['run'] == pc['run'], 'label_same': c['label'] == pc['label'],
                                       'hyp_same': c['hypothesis'] == pc.get('hypothesis', '基準'),
                                       'risk_bc': c['risk'], 'risk_mine': risk(cid) if cid != 'B' else 'n/a',
                                       'family_bc': c.get('family'), 'family_mine': fam(cid) if cid != 'B' else None}
# 基準の値
bp = P['run_settings']['baseline_params']
res['baseline_mode_same'] = B['baseline']['mode_id'] == P['run_settings']['baseline_mode_id']
res['baseline_param_diffs'] = {k: [bp.get(k), B['baseline']['params'].get(k)] for k in set(bp) | set(B['baseline']['params'])
                               if bp.get(k) != B['baseline']['params'].get(k)}
res['description_template_same'] = B['description_template'] == P['run_settings']['description_template']
res['day2'] = B['days'][1]
# 決め手の近さ(感度): risk の境と desk_key の差
res['margins'] = {c: {'r1_app_thr_diff_us': rows[c]['r1_app_thr_diff_us'], 'to_low_boundary(-10)': round(rows[c]['r1_app_thr_diff_us'] + 10, 3),
                      'steer_amp_ratio': rows[c]['steer_amp_ratio'], 'trt_p99_ms': rows[c]['trt_p99_ms'], 'l1_interv_us': rows[c]['l1_interv_us']}
                  for c in ids}
# AUG が mid だったら(感度)
def order_with(override):
    o = [x for x in order if x in ('SLOW', 'ENS')]
    rr = lambda c: override.get(c, risk(c))
    o += sorted([c for c in sel if c not in o], key=lambda c: (rk[rr(c)], dkey(c)))
    return [({'C%d' % (i + 1): c for i, c in enumerate(o)}).get(x, x) for x in lay]
res['sensitivity_AUG_mid_blocks'] = order_with({'AUG': 'mid'})
OUT.mkdir(parents=True, exist_ok=True)
json.dump(res, open(OUT / 'select.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps({k: res[k] for k in ('protocol_sha256', 'protocol_sha256_file', 'bc_protocol_sha256', 'desk_sha256', 'bc_desk_sha256',
                                      'protocol_before_desk', 'amend_before_desk', 'selected_mine', 'order_mine', 'blocks_mine', 'blocks_bc',
                                      'blocks_same', 'baseline_mode_same', 'baseline_param_diffs', 'description_template_same',
                                      'sensitivity_AUG_mid_blocks')}, ensure_ascii=False, indent=0))
print(json.dumps(res['gates'], ensure_ascii=False))
print(json.dumps(res['bc_candidate_checks'], ensure_ascii=False))
