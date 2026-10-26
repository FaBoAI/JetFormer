"""verify/aggregate.py — recount.json(生の記録だけから数えたもの)を回・設定ごとにまとめる。
2 つの周の決まりを並べる:
  V = 車の記録(VEHICLE)が 0.2 秒を超えて途切れない周(この検証の決まり)
  S = さらに AI_CMD(モデルの出力のフレーム)も 0.2 秒を超えて途切れない周(scorecard と同じ決まり)
"""
import json, os
import numpy as np
H = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(H, 'recount.json')))
NAME = {'B': 'B', 'SLOW': '遅く入る型', 'ENS': '2本の平均', 'AUG': 'ずれに強い型', 'ACT': 'ACT型', 'WGT': '苦手区間重み型'}


def session(lg, no):
    if lg.startswith('20261023'):
        return '10/23 公式'
    no = int(no)
    return '10/24 1回目' if no <= 4 else ('10/24 2回目' if no <= 10 else '10/24 3回目')


def setting(d):
    return '%s %.2f/%.2f' % (d['model'], round(d['gains'][1], 2), round(d['gains'][0], 2))


def laps_of(d, rule):
    out = []
    for l in d['laps']:
        if not l['ok']:
            continue
        if rule == 'S' and l.get('gap_cmd', 0) > 0.2:
            continue
        out.append(l)
    return out


def windows(laps):
    w = []
    for i in range(len(laps) - 2):
        a, b, c = laps[i:i + 3]
        if abs(a['t1'] - b['t0']) < 0.01 and abs(b['t1'] - c['t0']) < 0.01:
            g = max(a['gmax'], b['gmax'], c['gmax'])
            w.append(dict(t0=a['t0'], laps=[a['lap'], b['lap'], c['lap']], total=round(a['lap'] + b['lap'] + c['lap'], 3),
                          kind='clean' if g == 0 else ('light' if g < 4 else 'strong'), gmax=g))
    return w


res = {'rules': {}}
for rule in ('V', 'S'):
    per_run, by_sess, by_set = {}, {}, {}
    for lg, d in R.items():
        if 'error' in d:
            continue
        L = laps_of(d, rule)
        W = windows(L)
        s = session(lg, d['ai_no']); st = setting(d)
        per_run[lg] = dict(no=d['ai_no'], session=s, setting=st, laps=[(l['lap'], l['clean']) for l in L],
                           windows=[(x['total'], x['kind']) for x in W])
        for key, dic in ((s, by_sess), ((s if s == '10/23 公式' else s[:5]) + '|' + st, by_set)):
            e = dic.setdefault(key, dict(laps=[], clean=[], win=[], win_clean=[], nos=set()))
            e['laps'] += [l['lap'] for l in L]
            e['clean'] += [l['lap'] for l in L if l['clean']]
            e['win'] += [x['total'] for x in W]
            e['win_clean'] += [x['total'] for x in W if x['kind'] == 'clean']
            e['nos'].add(int(d['ai_no']))
    def summ(e):
        c = sorted(e['clean']); w = sorted(e['win'])
        return dict(nos=sorted(e['nos']), n_laps=len(e['laps']), n_clean=len(c), best=c[0] if c else None,
                    med_clean=float(np.median(c)) if c else None, sub9=sum(x < 9 for x in c),
                    n_win=len(w), win_min=w[0] if w else None, win_med=float(np.median(w)) if w else None,
                    win_clean_min=min(e['win_clean']) if e['win_clean'] else None, n_win_clean=len(e['win_clean']),
                    le25=sum(x <= 25 for x in w), le255=sum(x <= 25.5 for x in w), le26=sum(x <= 26 for x in w),
                    le265=sum(x <= 26.5 for x in w), le27=sum(x <= 27 for x in w))
    res['rules'][rule] = dict(per_run=per_run, by_session={k: summ(v) for k, v in by_sess.items()},
                              by_setting={k: summ(v) for k, v in by_set.items()})

# 衝突(AI 側・立て直し中を除く)
col = {}
for lg, d in R.items():
    if 'error' in d or not lg.startswith('20261024'):
        continue
    s = session(lg, d['ai_no'])
    e = col.setdefault(s, dict(n=0, strong=0, restart=0, loc={}, loc_strong={}, weak_lt35=0))
    for c in d['collisions']:
        if c['restart']:
            e['restart'] += 1; continue
        e['n'] += 1; e['strong'] += c['strong']
        e['weak_lt35'] += c['g'] < 3.5
        e['loc'][c.get('loc')] = e['loc'].get(c.get('loc'), 0) + 1
        if c['strong']:
            e['loc_strong'][c.get('loc')] = e['loc_strong'].get(c.get('loc'), 0) + 1
res['collisions_by_session'] = col

# R1 の進入の帯(本物の起点・AI 側)と、その通過の R1 の強い衝突(起点 −0.5〜+2.0 秒・場所 R1)
bands = {'<5500': [0, 0], '5500-6200': [0, 0], '>=6200': [0, 0]}
bands_with_false = {'<5500': [0, 0], '5500-6200': [0, 0], '>=6200': [0, 0]}
passes = {}
for lg, d in R.items():
    if 'error' in d or not lg.startswith('20261024'):
        continue
    s = session(lg, d['ai_no'])
    for a in d['anchors']:
        if not a['ai'] or a['entry_erpm'] is None:
            continue
        b = '<5500' if a['entry_erpm'] < 5500 else ('5500-6200' if a['entry_erpm'] < 6200 else '>=6200')
        hit = any(c['strong'] and c.get('loc') in ('R1',) and a['t'] - 0.5 <= c['t_first'] <= a['t'] + 2.0 for c in d['collisions'])
        hit_any = any(c['strong'] and a['t'] - 0.5 <= c['t_first'] <= a['t'] + 0.6 for c in d['collisions'])
        bands_with_false[b][0] += 1; bands_with_false[b][1] += hit_any
        if a['false']:
            continue
        bands[b][0] += 1; bands[b][1] += hit
        passes[s] = passes.get(s, 0) + 1
res['r1_bands_real_anchors'] = bands
res['r1_bands_incl_false_strong_within_0.6s'] = bands_with_false
res['passes_by_session'] = passes
res['false_anchors'] = [(lg, d['ai_no'], round(a['t'], 2), round(a['dpsi_prev'], 0)) for lg, d in R.items() if 'error' not in d
                        for a in d['anchors'] if a['false'] and a['ai']]
json.dump(res, open(os.path.join(H, 'aggregate.json'), 'w'), ensure_ascii=False, indent=1, default=lambda o: sorted(o) if isinstance(o, set) else float(o))

for rule in ('V', 'S'):
    print('=== rule', rule)
    for k, v in sorted(res['rules'][rule]['by_session'].items()):
        print(' ', k, v)
    for k, v in sorted(res['rules'][rule]['by_setting'].items()):
        print(' ', k, v)
print('No.9 laps', res['rules']['V']['per_run']['20261024-162033'])
print('No.18 laps V', res['rules']['V']['per_run']['20261024-185703'], 'S', res['rules']['S']['per_run']['20261024-185703'])
print('No.7', res['rules']['V']['per_run']['20261024-160654'])
print('collisions', json.dumps(col, ensure_ascii=False))
print('bands', bands, 'incl false', bands_with_false)
print('passes', passes, sum(passes.values()))
print('false anchors', res['false_anchors'])
