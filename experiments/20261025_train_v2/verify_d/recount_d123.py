#!/usr/bin/env python3
"""別の目の数え直し(2026-10-25 夜): decisions.json の D1・D2・D3・BF を、protocol_v2.json の決まりと
保存した予測(eval/pred/*.npz)から数え直す。eval_v2.py・recount_stage12_v2.py は読まない・使わない。
numpy だけ(torch を読まない)。車・装置に触れない。書くのは verify_d/recount_d123.json だけ。

    /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I \
        /home/jetson/jetformer/experiments/20261025_train_v2/verify_d/recount_d123.py
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

REP = Path('/home/jetson/jetformer_reports/train_v2_20261025')
PRED = REP / 'eval' / 'pred'
OUT = Path(__file__).resolve().parent / 'recount_d123.json'
BUNDLE = Path('/home/jetson/jetformer_reports/prep_20261022/desk/pred/desk_inputs_20261022.npz')
DATA = Path('/home/jetson/jetformer/data')
ALIGNED = DATA / 'datasets' / 'clean_driving_20261011' / 'aligned'
ZOO = DATA / 'zoo' / 'policy_zoo_20261012' / 'pred'
RACC = DATA / 'zoo' / 'real_accuracy_20261015' / 'pred'
KEY = '3d339512828b17195d568f1b93ee4a28208f2e85'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


proto = json.load(open(REP / 'protocol_v2.json'))
checks = {}
checks['protocol_sha256'] = sha(REP / 'protocol_v2.json')
checks['protocol_sha256_ok'] = checks['protocol_sha256'] == open(REP / 'protocol_v2.sha256').read().split('\n')[1].split()[0]

# ---- 集合(desk_inputs の bundle)
checks['bundle_sha256_ok'] = sha(BUNDLE) == proto['reference_predictions_sha256']['bundle']
bz = np.load(BUNDLE, allow_pickle=False)
assert str(bz['key']) == KEY
Y = bz['y'].astype(np.float64)
PRIM = bz['primary'].astype(bool)
INTV = bz['interv'].astype(bool)
scene = bz['scene']
frame = bz['frame']
checks['n_primary'] = int(PRIM.sum())
checks['n_interv'] = int(INTV.sum())
checks['n_primary_ok'] = checks['n_primary'] == proto['holdouts']['H1']['frames_desk']
checks['n_interv_ok'] = checks['n_interv'] == proto['holdouts']['interv']['frames_desk']

# ジャンルは H1 の aligned の mode から自分で引く(予測の npz の genre と照らす)。正解 y も aligned と照らす
G = np.full(len(Y), -1, np.int64)
y_max_diff = 0.0
for s in sorted(set(scene[PRIM])):
    with np.load(ALIGNED / ('%s.npz' % s), allow_pickle=False) as a:
        m = PRIM & (scene == s)
        G[m] = a['mode'][frame[m]]
        y_max_diff = max(y_max_diff, float(np.abs(a['future_actions'][frame[m]].astype(np.float64) - Y[m]).max()))
checks['y_vs_aligned_max_abs_us'] = y_max_diff
G4 = PRIM & (G == 4)
G5 = PRIM & (G == 5)
checks['genre_frames'] = {'4': int(G4.sum()), '5': int(G5.sum())}
checks['genre_frames_ok'] = checks['genre_frames'] == {k: int(v) for k, v in proto['holdouts']['H1']['genre_frames'].items()}
DAY = PRIM & np.isin(scene, proto['holdouts']['day_night']['day'])
NIGHT = PRIM & np.isin(scene, proto['holdouts']['day_night']['night'])


def load(path, expect_ck=None, expect_run=None):
    with np.load(path, allow_pickle=False) as z:
        assert str(z['key']) == KEY, path
        if expect_ck is not None:
            assert str(z['ckpt']) == expect_ck, (path, str(z['ckpt']))
        if expect_run is not None:
            assert str(z['run']) == expect_run, (path, str(z['run']))
        p = z['pred'].astype(np.float64)
        g = z['genre'].astype(np.int64)
        sub = z['substituted'].astype(bool) if 'substituted' in z.files else np.zeros(len(p), bool)
    assert p.shape == Y.shape
    assert np.isfinite(p[PRIM | INTV]).all()
    return p, g, sub


def fields(pb, pl):
    def l1(p, m):
        return float(np.abs(p[m] - Y[m]).mean())
    return {
        'l1_best': l1(pb, PRIM), 'l1_last': l1(pl, PRIM),
        'interv_best': l1(pb, INTV), 'interv_last': l1(pl, INTV),
        'steer_bias_best': float((pb[PRIM, 0, 0] - Y[PRIM, 0, 0]).mean()),
        'thr_bias_best': float((pb[PRIM, 0, 1] - Y[PRIM, 0, 1]).mean()),
        'steer_bias_last': float((pl[PRIM, 0, 0] - Y[PRIM, 0, 0]).mean()),
        'thr_bias_last': float((pl[PRIM, 0, 1] - Y[PRIM, 0, 1]).mean()),
        'g4_best': l1(pb, G4), 'g5_best': l1(pb, G5), 'g4_last': l1(pl, G4), 'g5_last': l1(pl, G5),
        'day_best': l1(pb, DAY), 'night_best': l1(pb, NIGHT),
    }


# ---- 5 本(sha256 を protocol と照らす)
FIVE = [('A_s0', ZOO / 'jetformer_v0_s0.npz', ZOO / 'jetformer_v0_s0_last.npz'),
        ('A_s1', ZOO / 'jetformer_v0_s1.npz', ZOO / 'jetformer_v0_s1_last.npz'),
        ('v0-s2', RACC / 'realacc_20261015-v0-s2.npz', RACC / 'realacc_20261015-v0-s2_last.npz'),
        ('v0-s3', RACC / 'realacc_20261015-v0-s3.npz', RACC / 'realacc_20261015-v0-s3_last.npz'),
        ('B', ZOO / 'B.npz', ZOO / 'B_last.npz')]
five = {}
checks['five_sha256_ok'] = {}
for fid, pb_path, pl_path in FIVE:
    ref = proto['reference_predictions_sha256'][fid]
    checks['five_sha256_ok'][fid] = (sha(pb_path) == ref['best'], sha(pl_path) == ref['last'])
    pb, gb, _ = load(pb_path)
    pl, gl, _ = load(pl_path)
    five[fid] = fields(pb, pl)
F5 = {k: float(np.mean([five[f][k] for f in five])) for k in five['B']}
F5sd = {k: float(np.std([five[f][k] for f in five], ddof=1)) for k in five['B']}

# ---- 16 本
ARMS = {
    'v2h-e30': ['v2_20261025-v2h-e30-s%d' % k for k in range(4)],
    'v2m-e30': ['v2_20261025-v2m-e30-s%d' % k for k in range(4)],
    'v2h-e60': ['v2_20261025-v2h-e60-s%d' % k for k in range(4)],
    'v2h-e100': ['v2_20261025-v2h-e100-s0'],
    'bfast-e15': ['bfast_20261025-e15-s%d' % k for k in range(3)],
}
rows = {}
checks['genre_input_equals_aligned_mode'] = {}
checks['run_done'] = {}
for arm, names in ARMS.items():
    for n in names:
        pb, gb, sb = load(PRED / ('%s.npz' % n), 'best', n)
        pl, gl, sl = load(PRED / ('%s_last.npz' % n), 'last', n)
        checks['genre_input_equals_aligned_mode'][n] = bool((gb[PRIM] == G[PRIM]).all() and (gl[PRIM] == G[PRIM]).all())
        r = json.load(open(DATA / 'runs' / n / 'result.json'))
        checks['run_done'][n] = (r.get('state') == 'done' and r.get('epochs_done') == r.get('epochs_planned'),
                                 int(sb[PRIM | INTV].sum() + sl[PRIM | INTV].sum()))
        rows[n] = fields(pb, pl)

T = proto['thresholds_us']['values']
NOTW = proto['thresholds_us']['not_worse_us']


def vs5(names, key):
    vals = [rows[n][key] for n in names]
    n = len(vals)
    thr = T['vs5']['n%d' % n][key]
    d = float(np.mean(vals)) - F5[key]
    return {'d': d, 'thr': thr, 'all_above': all(v > F5[key] for v in vals), 'all_below': all(v < F5[key] for v in vals),
            'seeds': vals}


def vs5_verdict(names):
    b, l = vs5(names, 'l1_best'), vs5(names, 'l1_last')
    good = (b['d'] <= -b['thr'] and b['all_below']) and (l['d'] <= -l['thr'] and l['all_below'])
    bad = (b['d'] >= b['thr'] and b['all_above']) or (l['d'] >= l['thr'] and l['all_above'])
    return {'best': b, 'last': l, 'verdict': '良い' if good else ('悪い' if bad else '同等')}


def veto(names, v2h_names=None):
    iv = vs5(names, 'interv_best')
    v_interv = iv['d'] > iv['thr'] and iv['all_above']
    n = len(names)
    if v2h_names is None:
        bias = float(np.mean([rows[x]['steer_bias_best'] for x in names]))
        lo, hi = F5['steer_bias_best'] - T['vs5']['n%d' % n]['steer_bias_best'], F5['steer_bias_best'] + T['vs5']['n%d' % n]['steer_bias_best']
        bias_info = {'mean': bias, 'lo': lo, 'hi': hi, 'thr': T['vs5']['n%d' % n]['steer_bias_best'], 'kind': '5 本の平均 ± thr_vs5'}
    else:
        bias = float(np.mean([rows[a]['steer_bias_best'] - rows[b]['steer_bias_best'] for a, b in zip(names, v2h_names)]))
        thr = T['pairs']['k%d' % n]['steer_bias_best']
        lo, hi = -thr, thr
        bias_info = {'mean_pair_diff': bias, 'lo': lo, 'hi': hi, 'thr': thr, 'kind': '同じ種の v2h との差の平均 ± thr_pairs'}
    v_bias = not (lo <= bias <= hi)
    return {'interv': iv, 'veto_interv': v_interv, 'bias': bias_info, 'veto_bias': v_bias, 'any': v_interv or v_bias}


def pair(a_names, b_names, key):
    d = [rows[a][key] - rows[b][key] for a, b in zip(a_names, b_names)]
    k = len(d)
    return {'d': d, 'mean': float(np.mean(d)), 'k': k, 'thr': T['pairs']['k%d' % k][key],
            'all_neg': all(x < 0 for x in d), 'all_pos': all(x > 0 for x in d)}


def pair_verdict(a_names, b_names, kb, kl):
    pb, pl = pair(a_names, b_names, kb), pair(a_names, b_names, kl)
    good = pb['mean'] <= -pb['thr'] and pl['mean'] <= -pl['thr'] and pb['all_neg'] and pl['all_neg']
    bad = (pb['mean'] >= pb['thr'] and pb['all_pos']) or (pl['mean'] >= pl['thr'] and pl['all_pos'])
    not_worse = pb['mean'] <= NOTW and pl['mean'] <= NOTW
    return {'best': pb, 'last': pl, 'good': good, 'bad': bad, 'not_worse': not_worse}


out = {'checks': checks, 'five_mean': F5, 'five_sd': F5sd, 'rows': rows}

# ---- D1
h30, m30 = ARMS['v2h-e30'], ARMS['v2m-e30']
d1_1 = {'vs5': vs5_verdict(h30), 'veto': veto(h30)}
step1 = 'v2h' if (d1_1['vs5']['verdict'] != '悪い' and not d1_1['veto']['any']) else 'base'
h1 = pair_verdict(m30, h30, 'l1_best', 'l1_last')
iv = pair_verdict(m30, h30, 'interv_best', 'interv_last')
sb = pair(m30, h30, 'steer_bias_best')
tb = pair(m30, h30, 'thr_bias_best')
m_veto = veto(m30, v2h_names=h30)
cond = {
    'H1 が悪くない(not_worse: d の平均 ≤ +0.5 best・last)': h1['not_worse'],
    'H1 が「悪い」でない(参考)': not h1['bad'],
    'steer_bias の差の平均が ±4.125 の中': abs(sb['mean']) <= T['pairs']['k4']['steer_bias_best'],
    'l1_interv の差の平均 ≤ +1.718(best)': iv['best']['mean'] <= T['pairs']['k4']['interv_best'],
    'thr_bias の差の平均 ≥ −2.887(best)': tb['mean'] >= -T['pairs']['k4']['thr_bias_best'],
    'H1 か介入で「良い」': h1['good'] or iv['good'],
    'v2-M に拒否権が無い': not m_veto['any'],
}
need = [c for c in cond if c != 'H1 が「悪い」でない(参考)']
x = step1
if step1 == 'v2h' and all(cond[c] for c in need):
    x = 'v2m'
out['D1'] = {'step1': d1_1, 'step1_answer': step1, 'v2m_vs_v2h': {'H1': h1, 'interv': iv, 'steer_bias': sb, 'thr_bias': tb,
             'veto_v2m': m_veto}, 'conditions': cond, 'x': x}

# ---- D2(x = v2h のときの 60 と 30。last だけ)
h60 = ARMS['v2h-e60']
p_last = pair(h60, h30, 'l1_last')
bad60 = p_last['mean'] >= p_last['thr'] and p_last['all_pos']
too_long = {n: rows[n]['l1_last'] - rows[n]['l1_best'] for n in h60}
TOO_LONG = 1.6  # protocol marks.too_long「l1_last_us − l1_best_us > 1.6 µs」
n_too_long = sum(v > TOO_LONG for v in too_long.values())
gth = proto['marks']['genre_skew_threshold_us']
skew = {}
for a, b in zip(h60, h30):
    d4 = rows[a]['g4_last'] - rows[b]['g4_last']
    d5 = rows[a]['g5_last'] - rows[b]['g5_last']
    skew[a] = {'d4': d4, 'd5': d5, 'mark': (d4 > gth['genre4_last'] and d5 <= gth['genre5_last']) or
               (d5 > gth['genre5_last'] and d4 <= gth['genre4_last'])}
v60 = veto(h60)
E = 30 if (bad60 or n_too_long >= 2 or sum(s['mark'] for s in skew.values()) >= 2 or v60['any']) else 60
out['D2'] = {'pair_last': p_last, 'bad60': bad60, 'too_long_us': too_long, 'n_too_long': n_too_long,
             'genre_skew': skew, 'n_skew': sum(s['mark'] for s in skew.values()), 'veto_60': v60, 'E': E,
             'vs5_60_ref': vs5_verdict(h60)}

# ---- D3: 段 3 の予測があるか
st3 = ['v2_20261025-%s-aug2-e%d-s%d' % (x, E, k) for k in range(4)]
have3 = [n for n in st3 if (PRED / ('%s.npz' % n)).exists()]
desk_c3 = REP / 'desk' / 'desk_v2.json'
out['D3'] = {'stage3_names': st3, 'stage3_preds_present': have3, 'desk_v2_json_present': desk_c3.exists(),
             'aug': None if (len(have3) < 4 or not desk_c3.exists()) else 'needs_full_rule',
             'why': 'D3 は段 3 の 4 本と C3 の robust_dl1_us が要る'}

# ---- BF(B-FAST の腕 n = 3 に拒否権)
bf = ARMS['bfast-e15']
out['BF'] = {'veto': veto(bf), 'vs5': vs5_verdict(bf)}
out['BF']['BF1_BF2'] = None if out['BF']['veto']['any'] else 'gates_needed'
out['ref_e100'] = {'vs5': vs5_verdict(ARMS['v2h-e100']), 'veto': veto(ARMS['v2h-e100'])}

# ---- decisions.json と照らす
dec = json.load(open(REP / 'decisions.json'))
cmp = {
    'D1.x': (dec['D1']['x'], x),
    'D2.E': (dec['D2']['E'], E),
    'D3.aug': (dec['D3']['aug'], out['D3']['aug']),
    'BF.BF1': (dec['BF']['BF1'], out['BF']['BF1_BF2']),
    'BF.BF2': (dec['BF']['BF2'], out['BF']['BF1_BF2']),
    'D1.values.steer_bias_diff': (dec['D1']['values']['steer_bias_diff'], sb['mean']),
    'D1.values.interv_diff': (dec['D1']['values']['interv_diff'], iv['best']['mean']),
    'D1.values.thr_bias_diff': (dec['D1']['values']['thr_bias_diff'], tb['mean']),
    'D2.mean_d_last': (dec['D2']['pair_last']['mean_d_last'], p_last['mean']),
    'BF.interv_diff': (dec['BF']['veto']['interv_diff'], out['BF']['veto']['interv']['d']),
    'BF.bias_value': (dec['BF']['veto']['bias_value'], float(np.mean([rows[n]['steer_bias_best'] for n in bf]))),
}
for i, v in enumerate(dec['D2']['pair_last']['d_last']):
    cmp['D2.d_last[%d]' % i] = (v, p_last['d'][i])
for n, v in dec['BF']['veto_interv_seeds_us'].items():
    cmp['BF.interv[%s]' % n] = (v, rows[n]['interv_best'])
cmp['BF.five_mean_interv'] = (dec['BF']['five_mean_interv_best_us'], F5['interv_best'])
res = {}
for k, (a, b) in cmp.items():
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        res[k] = {'decisions': a, 'mine': b, 'abs_diff': abs(a - b), 'same': abs(a - b) <= 1e-3}
    else:
        res[k] = {'decisions': a, 'mine': b, 'same': a == b}
out['compare_with_decisions'] = res
out['all_same'] = all(v['same'] for v in res.values())

# 5 本の平均と標準偏差を protocol の丸めと照らす
out['five_vs_protocol'] = {k: {'mine': F5[k], 'protocol': T['five_mean'][k], 'sd_mine': F5sd[k], 'sd_protocol': T['five_sd'][k]}
                           for k in T['five_mean']}


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    return o


tmp = OUT.with_suffix('.json.tmp')
tmp.write_text(json.dumps(clean(out), ensure_ascii=False, indent=1))
tmp.replace(OUT)
print(json.dumps(clean({'all_same': out['all_same'], 'D1.x': x, 'D2.E': E, 'D3': out['D3']['aug'],
                        'BF': out['BF']['BF1_BF2'], 'checks': checks}), ensure_ascii=False, indent=1))
for k, v in res.items():
    if not v['same']:
        print('DIFF', k, v)
