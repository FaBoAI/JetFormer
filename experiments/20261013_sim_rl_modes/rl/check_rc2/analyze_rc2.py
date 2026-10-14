"""離脱復活の確認 r2: 新しい開始(F2-P・F2-KY・F2-KL)の判定の値・ゲートの外し・ラベル・ずるを自前で数える(2026-10-14)。

乱しなしの線(nominal line)は確認 r1 の NOM-recover(同じ方策・種 990000〜)から作る。数え方は確認 r1 の analyze_rc.py と同じ(protocol definitions の文から)。
  python -I rl/check_rc2/analyze_rc2.py <r2 の runs> <r1 の runs> <out.json>
"""
import glob
import json
import math
import sys

import numpy as np

RUNS, RUNS1, OUT = sys.argv[1:4]
REF = json.load(open('/home/jetson/jetformer/experiments/20261013_sim_rl_modes/web/rl/reference_loop.json'))
L = REF['length_m']
NB = int(math.ceil(L / 0.05))
GS = np.array(sorted(REF['gate_s']))
MAXGAP = max(((GS[(i + 1) % len(GS)] - GS[i]) % L) for i in range(len(GS)))
THR = MAXGAP + 0.5


def load(d):
    by = {}
    for p in sorted(glob.glob(d + '/rows_*.jsonl')):
        for line in open(p):
            r = json.loads(line)
            if r.get('invalid_spawn'):
                by.setdefault('_invalid', []).append(r)
                continue
            by.setdefault(r['set'], []).append(r)
    for k in by:
        by[k].sort(key=lambda r: (len(r['id']), r['id']))
    return by


by = load(RUNS)
by1 = load(RUNS1)


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def q(a, p):
    return float(np.quantile(np.asarray(sorted(a), float), p)) if a else None


def nominal_line(runs):
    ey = [[] for _ in range(NB)]
    ps = [[] for _ in range(NB)]
    for r in runs:
        s, e, h = r['pf']['s'], r['pf']['ey'], r['pf']['epsi']
        seen = set()
        for i in range(len(s)):
            if s[i] - s[0] < 3:
                continue
            b = int((s[i] % L) / 0.05) % NB
            if b in seen:
                continue
            seen.add(b)
            ey[b].append(e[i])
            ps[b].append(h[i])
    eyM, psM = np.full(NB, np.nan), np.full(NB, np.nan)
    for b in range(NB):
        if ey[b]:
            eyM[b] = np.mean(ey[b])
            psM[b] = math.atan2(np.mean(np.sin(ps[b])), np.mean(np.cos(ps[b])))
    have = np.nonzero(~np.isnan(eyM))[0]
    for b in range(NB):
        if np.isnan(eyM[b]):
            d = np.minimum(np.abs(have - b), NB - np.abs(have - b))
            j = have[np.argmin(d)]
            eyM[b], psM[b] = eyM[j], psM[j]
    return {'ey': eyM, 'psi': psM}


NL = nominal_line(by1['NOM-recover'])


def stall_wrong(r):
    v, s = r['pf']['v'], r['pf']['s']
    stall, wrong, run, smax = [], [], 0, -1e9
    for f in range(len(v)):
        run = run + 1 if (f / 30 >= 1 and v[f] < 0.2) else 0
        stall.append(run >= 60)
        smax = max(smax, s[f])
        wrong.append(s[f] < smax - 1)
    return stall, wrong


def recoveries(r):
    pf = r['pf']
    T = len(pf['s'])
    stall, wrong = stall_wrong(r)
    evs = r['events']
    collF = T if r['collisions'] else math.inf
    out = []
    for k, ev in enumerate(evs):
        pe = ev['pe'] if ev['pe'] is not None else T
        nxt = evs[k + 1]['ps'] if k + 1 < len(evs) else T
        until = min(nxt, T)
        runlen, recA = 0, None
        for f in range(pe, T):
            b = int((pf['s'][f] % L) / 0.05) % NB
            ok = abs(pf['ey'][f] - NL['ey'][b]) <= 0.10 and abs(wrap(pf['epsi'][f] - NL['psi'][b])) <= 0.15
            runlen = runlen + 1 if ok else 0
            if runlen >= 15:
                recA = f - 14
                break
        bad = collF <= until
        why = 'collision' if bad else None
        if not bad:
            for f in range(pe, until):
                if stall[f]:
                    bad, why = True, 'stall'
                    break
                if wrong[f]:
                    bad, why = True, 'wrong_way'
                    break
        tA = None if recA is None else (recA - pe) / 30
        smod = (pf['s'][ev['ps']] % L) if ev['ps'] < T else None
        out.append({'type': ev['type'], 'cls': ev['cls'], 'sign': ev.get('sign'), 'tA': tA, 'bad': bad, 'why': why,
                    'ok': tA is not None and tA <= 3.0 and not bad, 'smod': smod, 'v_ps': pf['v'][ev['ps']] if ev['ps'] < T else None})
    return out


def mask_frac(r):
    l = r['pf']['l']
    m = np.zeros(len(l), bool)
    for j in range(len(l)):
        if l[j] == 0:
            m[max(0, j - 10):j] = True
    return float(m.sum()) / 600


def gate_stuck(r):
    """sim の passed_gates が変わらないまま s が THR を越えた回数(rl/gate_miss.mjs と同じ決まり)と、最後の止まりの長さ"""
    s, g = r['pf']['s'], r['pf']['g']
    last, n, counted, smax_end = s[0], 0, False, 0.0
    for f in range(1, len(s)):
        if g[f] != g[f - 1]:
            last, counted = s[f], False
        st = s[f] - last
        if st > THR and not counted:
            counted, n = True, n + 1
    end_stuck = s[-1] - last
    # 最後の止まり: 止まった所の次のゲートまでの距離 + 0.5 m を越えたのに THR に届かない(= 回の終わりの近くで外して数えられない候補)
    sm = last % L
    nxt = GS[GS > sm + 0.15]
    gap = (nxt[0] - sm) if len(nxt) else (GS[0] + L - sm)
    end_suspect = (n == 0 or end_stuck <= THR) and end_stuck > gap + 0.5 and end_stuck <= THR
    return n, end_stuck, bool(end_suspect)


res = {'thr_m': THR, 'max_gap_m': MAXGAP}
P = by['F2-P']
ev, epm, mfs = [], [], []
for r in P:
    rec = recoveries(r)
    r['_rec'] = rec
    ev += rec
    epm.append(len(r['events']))
    mfs.append(mask_frac(r))
valid = [r for r in P if r['valid_episode']]
okt = [e['tA'] for e in ev if e['ok']]
fail = {}
for e in ev:
    if not e['ok']:
        w = e['why'] or ('not_recovered' if e['tA'] is None else 'late')
        fail[w] = fail.get(w, 0) + 1
bytype = {}
for e in ev:
    d = bytype.setdefault(e['type'], {'n': 0, 'ok': 0})
    d['n'] += 1
    d['ok'] += int(e['ok'])
exitR = [e for e in ev if e['type'] == 'steer_pulse' and e['smod'] is not None and 8.0 <= e['smod'] < 8.25]
gm = [(r['id'],) + gate_stuck(r) for r in valid]
res['F2_P'] = {
    'episodes': len(P), 'valid': len(valid), 'end_reasons': {}, 'events': len(ev),
    'C2_success': sum(e['ok'] for e in ev) / max(1, len(ev)), 'C3': {'median': q(okt, .5), 'p90': q(okt, .9)},
    'C4_events_per_episode': float(np.mean(epm)), 'C4_masked_fraction': float(np.mean(mfs)),
    'valid_only_events_per_episode': float(np.mean([len(r['events']) for r in valid])) if valid else None,
    'valid_only_masked_fraction': float(np.mean([mask_frac(r) for r in valid])) if valid else None,
    'fail_why': fail, 'by_type': bytype,
    'exit_steer': {'left_n': sum(1 for e in exitR if e['sign'] == 1), 'left_coll': sum(1 for e in exitR if e['sign'] == 1 and e['why'] == 'collision'),
                   'right_n': sum(1 for e in exitR if e['sign'] == -1), 'right_coll': sum(1 for e in exitR if e['sign'] == -1 and e['why'] == 'collision')},
    'event_start_share_8_00_8_25': sum(1 for e in ev if e['smod'] is not None and 8.0 <= e['smod'] < 8.25) / max(1, len(ev)),
    'event_start_share_19_25_19_50': sum(1 for e in ev if e['smod'] is not None and 19.25 <= e['smod'] < 19.5) / max(1, len(ev)),
    'min_clearance_valid': min([r['min_clearance'] for r in valid], default=None),
    'valid_clearance_lt_3cm': sum(1 for r in valid if r['min_clearance'] < 0.03),
    'gate_miss_valid': [x[0] for x in gm if x[1] > 0],
    'gate_miss_end_suspect_valid': [(x[0], round(x[2], 3)) for x in gm if x[3]],
    'overlap_frames_total': sum(r['overlap_frames'] for r in P), 'audit_pos_err_max': max(r['audit_pos_err'] for r in P),
}
for r in P:
    res['F2_P']['end_reasons'][r['reason']] = res['F2_P']['end_reasons'].get(r['reason'], 0) + 1
# 黒シート: 有効な回の横断が全部開口の中か
bad_cross = [r['id'] for r in valid if any(not c['inside'] for c in r['own_crossings'])]
res['F2_P']['black_sheet_outside_crossings_valid'] = bad_cross
# ラベル
lab = {'changed_but_l1': sum(r['labels']['changed_but_l1'] for r in P), 'info_bad_keys': sum(r['labels']['info_bad_keys'] for r in P),
       'p0_missing': sum(1 for r in P if not r['p0']), 'l0_outside_event': 0, 'event_l0_mismatch': 0}
for r in P:
    l = r['pf']['l']
    T = len(l)
    win = np.zeros(T, bool)
    for e in r['events']:
        pe = e['pe'] if e['pe'] is not None else T
        win[e['ps']:pe] = True
        if sum(1 for f in range(e['ps'], min(pe, T)) if l[f] == 0) != min(pe, T) - e['ps']:
            lab['event_l0_mismatch'] += 1
    lab['l0_outside_event'] += int(sum(1 for f in range(T) if l[f] == 0 and not win[f]))
res['F2_P']['labels'] = lab
# キック(判定に使う = キックの位置が乱してよい区間)
for name in ('F2-KY', 'F2-KL'):
    rs = by[name]
    pre = [r for r in rs if not r['kick']]
    j = [r for r in rs if r['kick'] and r['kick']['allowed']]
    a = [r for r in rs if r['kick']]
    gmk = [r['id'] for r in j if r['valid_episode'] and gate_stuck(r)[0] > 0]
    res[name] = {'judged': len(j), 'survived': sum(r['valid_episode'] for r in j), 'rate': sum(r['valid_episode'] for r in j) / max(1, len(j)),
                 'all': len(a), 'all_survived': sum(r['valid_episode'] for r in a), 'ended_before_kick': len(pre),
                 'judged_survived_with_gate_miss': len(gmk), 'overlap_frames_total': sum(r['overlap_frames'] for r in rs)}
res['invalid_spawn'] = len(by.get('_invalid', []))


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - w, c + w]


res['F2_P']['valid_ci95'] = wilson(len(valid), len(P))
res['F2_P']['C2_ci95'] = wilson(sum(e['ok'] for e in ev), len(ev))
for name in ('F2-KY', 'F2-KL'):
    d = res[name]
    d['ci95'] = wilson(d['survived'], d['judged']) if d['judged'] else None
json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
