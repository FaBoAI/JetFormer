"""離脱復活 r3 の確認: 自前の走り(ck_batch.mjs の rows_*.jsonl)から判定と報告の値を出す(2026-10-14)。
学習・評価のコード(eval_rc3.mjs・variation_metrics_rc3.mjs など)を使わず、protocol.json definitions と
protocol_recover_r3.json definitions_added・acceptance の文から書いた。
  python3 -I ck_analyze.py <runs のフォルダ> <out.json>
"""
import glob
import json
import math
import sys
from collections import Counter, defaultdict

import numpy as np

RUNS, OUT = sys.argv[1:3]
L = 26.119648724030924
NB = int(math.ceil(L / 0.05))
rows = []
for p in sorted(glob.glob(RUNS + '/rows_*.jsonl')):
    for line in open(p):
        if line.strip():
            rows.append(json.loads(line))
by = defaultdict(list)
_seen = set()
DUP = 0
for r in rows:
    if (r['set'], r['id']) in _seen:   # 同じ回が 2 つの出力にあれば最初の 1 つだけ(20:40 の分け直しの用心)
        DUP += 1
        continue
    _seen.add((r['set'], r['id']))
    by[r['set']].append(r)
for k in by:
    by[k].sort(key=lambda r: r['id'])


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def q(a, p):
    return float(np.quantile(np.asarray(a, float), p)) if len(a) else None


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


# ---- 乱しなしの線(definitions.nominal_line) ----
def nominal_line(runs):
    ey = [[] for _ in range(NB)]
    ps = [[] for _ in range(NB)]
    for r in runs:
        s, e, h = r['pf']['s'], r['pf']['ey'], r['pf']['epsi']
        s0, seen = s[0], set()
        for i in range(len(s)):
            if s[i] - s0 < 3:
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
    filled = 0
    for b in range(NB):
        if np.isnan(eyM[b]):
            d = np.minimum(np.abs(have - b), NB - np.abs(have - b))
            j = have[np.argmin(d)]
            eyM[b], psM[b] = eyM[j], psM[j]
            filled += 1
    return {'ey': eyM, 'psi': psM, 'filled': filled, 'runs': len(runs), 'valid': sum(1 for r in runs if r['valid_episode'])}


NL = {'r3': nominal_line(by['NOM-r3']), 'fastest': nominal_line(by['NOM-fastest'])}
if by.get('NOMF'):
    NL['fastest_mode'] = nominal_line(by['NOMF'])   # 最速のモードそのもの(mode 'fastest')の乱しなしの線(20:50 に足した)


def brake_windows(r, T):
    ex = np.zeros(T, bool)
    for e in r['events']:
        if e['type'] == 'speed_pulse' and e.get('kind') == 'brake':
            pe = e['pe'] if e['pe'] is not None else T
            ex[e['ps']:min(T, pe + 30)] = True
    return ex


def stall_r3(r, reset_in_window=False):
    """definitions_added.stall: フレーム 30 以降に |v| < 0.2 が 30 フレーム以上。ブレーキの乱しの区間とそのあと 30 フレームは数えない。
    既定は「数えず、続きも切らない」(きびしい側)。reset_in_window=True は区間で切る読み"""
    v = r['pf']['v']
    T = len(v)
    ex = brake_windows(r, T)
    run = mx = 0
    at = None
    for f in range(30, T):
        if ex[f]:
            if reset_in_window:
                run = 0
            continue
        if abs(v[f]) < 0.2:
            run += 1
            if run > mx:
                mx = run
            if run >= 30 and at is None:
                at = f - 29
        else:
            run = 0
    return {'stall': mx >= 30, 'max_run': mx, 'at': at}


def progress(r):
    return r['final']['s'] - r['pf']['s'][0]


def gate_miss(r):
    """自前のゲートの外し: sim の passed_gates が増えないまま、自前の s が最後に増えたところから 2.02 m(ゲートの間の最大 1.52 m + 0.5 m)を越えて進んだ"""
    s, g = r['pf']['s'], r['pf']['g']
    last_s, last_g, misses, at = s[0], g[0], 0, []
    for f in range(len(s)):
        if g[f] != last_g:
            last_g, last_s = g[f], s[f]
        elif s[f] - last_s > 2.02:
            misses += 1
            at.append(f)
            last_s = s[f]
    return misses, at


def valid_r3(r):
    if r.get('invalid_spawn'):
        return False
    st = stall_r3(r)
    return bool(r['valid_episode'] and not st['stall'] and progress(r) >= L - 1e-9)


def recoveries(r, nl):
    pf = r['pf']
    T = len(pf['s'])
    v = pf['v']
    s = pf['s']
    # 停止(確認 r1 と同じ 60 フレームの読み・r3 の止まりの 30 フレームの読み)と逆走
    st60, st30, wrong = [False] * T, [False] * T, [False] * T
    ex = brake_windows(r, T)
    run60 = run30 = 0
    smax = -1e9
    for f in range(T):
        run60 = run60 + 1 if (f >= 30 and v[f] < 0.2) else 0
        st60[f] = run60 >= 60
        if f >= 30 and not ex[f]:
            run30 = run30 + 1 if abs(v[f]) < 0.2 else 0
        st30[f] = run30 >= 30
        smax = max(smax, s[f])
        wrong[f] = s[f] < smax - 1
    evs = r['events']
    collF = r['frames'] if r['collisions'] else math.inf
    out = []
    for k, ev in enumerate(evs):
        pe = ev['pe'] if ev['pe'] is not None else T
        nxt = evs[k + 1]['ps'] if k + 1 < len(evs) else T
        until = min(nxt, T)
        runlen, rec = 0, None
        for f in range(pe, T):
            b = int((s[f] % L) / 0.05) % NB
            ok = abs(pf['ey'][f] - nl['ey'][b]) <= 0.10 and abs(wrap(pf['epsi'][f] - nl['psi'][b])) <= 0.15
            runlen = runlen + 1 if ok else 0
            if runlen >= 15:
                rec = f - 14
                break
        bad, why = (collF <= until), None
        if bad:
            why = 'collision'
        bad30 = bad
        for f in range(pe, until):
            if not bad and (st60[f] or wrong[f]):
                bad, why = True, 'stall60' if st60[f] else 'wrong_way'
            if st30[f] or wrong[f]:
                bad30 = True
        t = None if rec is None else (rec - pe) / 30
        ps = ev['ps']
        out.append({'n': ev['n'], 'type': ev['type'], 'kind': ev.get('kind'), 'cls': ev['cls'], 'sign': ev['sign'], 'ps': ps, 'pe': ev['pe'],
                    's': ev['s'] % L, 'v': ev.get('v'), 'req': ev.get('req'), 'fallback': ev.get('fallback'), 'wait': ev.get('wait'),
                    'frames': ev.get('frames'), 'k': ev.get('k'), 'd': ev.get('d'), 'T': ev.get('T'), 'aborted': ev.get('aborted'),
                    't': t, 'bad': bad, 'why': why, 'ok': t is not None and t <= 3.0 and not bad,
                    'ok_strict30': t is not None and t <= 3.0 and not bad30,
                    'gate_miss_after': None})
    return out


def mask_frac(r):
    lab = r['pf']['l']
    n = len(lab)
    m = np.zeros(n, bool)
    for j in range(n):
        if lab[j] == 0:
            m[max(0, j - 10):j] = True
    return float(m.sum()) / 600


def label_audit(r):
    """l = 0 のフレーム = 出来事の区間 [ps, pe) のフレーム、ほかは l = 1。替えた操作で l = 1 はシールドだけ"""
    pf = r['pf']
    T = len(pf['l'])
    inwin = np.zeros(T, bool)
    for e in r['events']:
        pe = e['pe'] if e['pe'] is not None else T
        inwin[e['ps']:pe] = True
    l0 = np.array(pf['l']) == 0
    ph = np.array(pf['ph'])
    sh = np.array(pf['sh'])
    ch = (np.abs(np.array(pf['st']) - np.array(pf['pst'])) > 1e-3) | (np.abs(np.array(pf['sp']) - np.array(pf['psp'])) > 1e-2)
    return {'l0_outside_events': int((l0 & ~inwin).sum()), 'l1_inside_events': int((~l0 & inwin).sum()),
            'changed_l1_not_shield': int((ch & ~l0 & (sh == 0)).sum()), 'shield_in_perturb': int(((sh == 1) & (ph == 1)).sum()),
            'runner': r['labels']}


def episode_summary(rs, nl, name):
    ev = []
    out = {'n': len(rs)}
    rs = [r for r in rs if not r.get('invalid_spawn')]
    vr3 = [valid_r3(r) for r in rs]
    out['invalid_spawn'] = out['n'] - len(rs)
    out['valid_episode'] = sum(r['valid_episode'] for r in rs)
    out['valid_r3'] = sum(vr3)
    out['valid_r3_ci95'] = wilson(sum(vr3), len(rs))
    out['stall'] = sum(stall_r3(r)['stall'] for r in rs)
    out['stall_reset_reading'] = sum(stall_r3(r, True)['stall'] for r in rs)
    out['progress_short'] = sum(1 for r in rs if r['valid_episode'] and progress(r) < L)
    out['collisions'] = sum(1 for r in rs if r['collisions'])
    inval = []
    for r, ok in zip(rs, vr3):
        if not ok:
            st = stall_r3(r)
            inval.append({'id': r['id'], 'reason': r['reason'], 'frames': r['frames'], 'collision': r['collision'], 'stall': st, 'progress': round(progress(r), 2),
                          'start': r['spec']['start'], 'start_clearance': round(r['start_clearance'], 4)})
    out['invalid_runs'] = inval
    evcount, mfs, gm, gm_ids = [], [], 0, []
    sh_after = sh_other = n_after = n_other = 0
    labs = Counter()
    for r in rs:
        rec = recoveries(r, nl)
        T = len(r['pf']['s'])
        misses, at = gate_miss(r)
        for e in rec:
            pe = e['pe'] if e['pe'] is not None else T
            e['gate_miss_after'] = any(pe - 30 <= f <= pe + 90 for f in at)
            e['run'] = r['id']
            e['run_valid_r3'] = valid_r3(r)
        ev += rec
        evcount.append(len(r['events']))
        mfs.append(mask_frac(r))
        if valid_r3(r) and misses:
            gm += 1
            gm_ids.append(r['id'])
        after = np.zeros(T, bool)
        for e in r['events']:
            pe = e['pe'] if e['pe'] is not None else T
            after[pe:min(T, pe + 90)] = True
        ph = np.array(r['pf']['ph'])
        sh = np.array(r['pf']['sh'])
        npert = ph != 1
        n_after += int((after & npert).sum())
        sh_after += int((after & npert & (sh == 1)).sum())
        n_other += int((~after & npert).sum())
        sh_other += int((~after & npert & (sh == 1)).sum())
        la = label_audit(r)
        for k2 in ('l0_outside_events', 'l1_inside_events', 'changed_l1_not_shield', 'shield_in_perturb'):
            labs[k2] += la[k2]
        for k2, v2 in la['runner'].items():
            labs['runner_' + k2] += v2
    out['events'] = len(ev)
    out['events_per_run'] = round(float(np.mean(evcount)), 4) if evcount else None
    out['masked_fraction_mean'] = round(float(np.mean(mfs)), 4) if mfs else None
    if ev:
        ok = [e for e in ev if e['ok']]
        out['C2_success'] = round(len(ok) / len(ev), 4)
        out['C2_ci95'] = wilson(len(ok), len(ev))
        out['C2_success_strict30'] = round(sum(e['ok_strict30'] for e in ev) / len(ev), 4)
        ts = [e['t'] for e in ok]
        out['C3_median_s'] = round(q(ts, .5), 3) if ts else None
        out['C3_p90_s'] = round(q(ts, .9), 3) if ts else None
        cls = Counter(e['cls'] for e in ev)
        typ = Counter(e['type'] for e in ev)
        out['C7_cls_frac'] = {k: round(v / len(ev), 3) for k, v in cls.items()}
        out['C7_type_frac'] = {k: round(v / len(ev), 3) for k, v in typ.items()}
        out['C7_pass'] = all(cls.get(c, 0) / len(ev) >= 0.15 for c in ('弱', '中', '強')) and all(typ.get(t, 0) / len(ev) >= 0.10 for t in ('steer_pulse', 'excursion', 'speed_pulse'))
        out['by_cls'] = {c: f"{sum(e['ok'] for e in ev if e['cls'] == c)}/{sum(1 for e in ev if e['cls'] == c)}" for c in ('弱', '中', '強')}
        out['by_type'] = {t: f"{sum(e['ok'] for e in ev if e['type'] == t)}/{sum(1 for e in ev if e['type'] == t)}" for t in ('steer_pulse', 'excursion', 'speed_pulse')}
        out['by_kind'] = dict(Counter(f"{e['type']}:{e['kind']}" for e in ev))
        out['fallback'] = dict(Counter(e['fallback'] for e in ev))
        out['req_type'] = dict(Counter((e['req'] or {}).get('type') for e in ev))
        out['wait_median'] = q([e['wait'] for e in ev if e['wait'] is not None], .5)
        out['wait_p90'] = q([e['wait'] for e in ev if e['wait'] is not None], .9)
        out['start_black_sheet_exit'] = round(sum(1 for e in ev if 8.0 <= e['s'] <= 8.25) / len(ev), 3)
        out['start_back_straight_exit'] = round(sum(1 for e in ev if 19.3 <= e['s'] <= 19.6) / len(ev), 3)
        out['failed_events'] = [{k: e[k] for k in ('run', 'n', 'type', 'kind', 'cls', 'sign', 'ps', 'pe', 's', 'v', 't', 'why', 'fallback', 'frames', 'k', 'd', 'T', 'aborted', 'run_valid_r3')}
                                for e in ev if not e['ok']]
        out['events_gate_miss_after'] = sum(1 for e in ev if e['gate_miss_after'])
    out['gate_miss_valid_runs'] = f"{gm}/{out['valid_r3']}"
    out['gate_miss_ids'] = gm_ids[:40]
    out['shield_after3s'] = round(sh_after / max(1, n_after), 5)
    out['shield_other'] = round(sh_other / max(1, n_other), 5)
    out['labels'] = dict(labs)
    vals = [r for r, ok in zip(rs, vr3) if ok]
    out['min_clearance_valid'] = round(min(r['min_clearance'] for r in vals), 5) if vals else None
    out['clear_lt3cm_valid'] = sum(1 for r in vals if r['min_clearance'] < 0.03)
    out['audit_pos_err_max'] = max((r['audit_pos_err'] or 0) for r in rs) if rs else None
    out['overlap_without_collision'] = sum(1 for r in rs if r['overlap_frames'] and not r['collisions'])
    out['proj_vs_runtime_ds_max'] = max(r['proj_vs_runtime']['ds'] for r in rs) if rs else None
    out['_events'] = ev
    return out


def kick_summary(rs):
    res = {}
    for kind in ('yaw', 'lateral'):
        sub = [r for r in rs if r['spec'].get('kind') == kind]
        judged = [r for r in sub if r['kick'] and r['kick']['allowed_rt']]
        surv = [r for r in judged if r['reason'] == 'collection_duration_complete' and r['collisions'] == 0 and not stall_r3(r)['stall']]
        fails = []
        for r in judged:
            if r in surv:
                continue
            kf = r['kick']['frame']
            fails.append({'id': r['id'], 'reason': r['reason'], 'frames': r['frames'], 'kick_frame': kf, 'same_step': r['collisions'] > 0 and r['frames'] <= kf + 1,
                          'clr_after_kick': round(r['kick']['clr_after'], 4), 's': round(r['kick']['rt_s'] % L, 2), 'v': round(r['kick']['v'], 2),
                          'ey_pre': round(r['kick']['ey_pre'], 3), 'stall': stall_r3(r)['stall'], 'start': r['spec']['start']})
        allsurv = [r for r in sub if r['reason'] == 'collection_duration_complete' and r['collisions'] == 0 and not stall_r3(r)['stall']]
        res[kind] = {'judged': len(judged), 'survived': len(surv), 'rate': round(len(surv) / len(judged), 4) if judged else None,
                     'ci95': wilson(len(surv), len(judged)), 'all_positions': f'{len(allsurv)}/{len(sub)}', 'fails': fails,
                     'own_vs_rt_allowed_mismatch': sum(1 for r in sub if r['kick'] and r['kick']['allowed_rt'] != r['kick']['allowed_own'])}
    return res


def unperturbed_summary(rs):
    rs = [r for r in rs if not r.get('invalid_spawn')]
    out = {'n': len(rs), 'valid_r3': sum(valid_r3(r) for r in rs), 'stall': sum(stall_r3(r)['stall'] for r in rs),
           'collisions': sum(1 for r in rs if r['collisions']), 'progress_short': sum(1 for r in rs if r['valid_episode'] and progress(r) < L)}
    mins = [r['min_clearance'] for r in rs]
    out['min_clearance'] = round(min(mins), 5) if mins else None
    out['lt3cm'] = [{'id': r['id'], 'min': round(r['min_clearance'], 4), 'start_clearance': round(r['start_clearance'], 4), 'min_frame': r['min_frame'],
                     'min_after_frame30': round(min(r['pf']['cl'][30:]), 4) if r['pf']['cl'] and len(r['pf']['cl']) > 30 else None,
                     'start': r['spec']['start']} for r in rs if r['min_clearance'] < 0.03]
    good = [r for r in rs if r['start_clearance'] >= 0.03]
    out['start_ge3cm'] = {'n': len(good), 'valid_r3': sum(valid_r3(r) for r in good),
                          'min_clearance': round(min(r['min_clearance'] for r in good), 5) if good else None,
                          'lt3cm': sum(1 for r in good if r['min_clearance'] < 0.03)}
    out['min_after_frame30_all'] = round(min(min(r['pf']['cl'][30:]) for r in rs if r['pf']['cl'] and len(r['pf']['cl']) > 30), 5) if rs else None
    out['invalid'] = [{'id': r['id'], 'reason': r['reason'], 'frames': r['frames'], 'collision': r['collision'], 'stall': stall_r3(r), 'progress': round(progress(r), 2),
                       'start': r['spec']['start'], 'start_clearance': round(r['start_clearance'], 4)} for r in rs if not valid_r3(r)]
    out['gate_miss_valid'] = sum(1 for r in rs if valid_r3(r) and gate_miss(r)[0])
    return out


res = {'duplicates_dropped': DUP, 'counts': {k: len(v) for k, v in by.items()}, 'nominal_line': {k: {'runs': v['runs'], 'valid': v['valid'], 'filled_bins': v['filled']} for k, v in NL.items()}}
# 最速と離脱復活 r3 の乱しなしの線の違い
d = np.abs(NL['r3']['ey'] - NL['fastest']['ey'])
res['nominal_r3_vs_fastest'] = {'mean_abs_dey_m': round(float(d.mean()), 4), 'p90_abs_dey_m': round(float(np.quantile(d, .9)), 4), 'max_abs_dey_m': round(float(d.max()), 4),
                                'mean_abs_dpsi_rad': round(float(np.mean([abs(wrap(a - b)) for a, b in zip(NL['r3']['psi'], NL['fastest']['psi'])])), 4)}
for nm in ('NOM-r3', 'NOM-fastest'):
    rs = by[nm]
    run = [v for r in rs for v in r['pf']['v'] if v > 0.3]
    res[nm] = {'valid': sum(r['valid_episode'] for r in rs), 'mean_speed_running': round(float(np.mean(run)), 3), 'laps': [l['elapsed'] for r in rs for l in r['laps']][:20],
               'min_clearance': round(min(r['min_clearance'] for r in rs), 4)}
res['STD1_lap_from_stop'] = by['STD1'][0]['lap_from_stop'] if by['STD1'] else None
res['STD1_min_clearance'] = by['STD1'][0]['min_clearance'] if by['STD1'] else None
res['STD1F_lap_from_stop'] = by['STD1F'][0]['lap_from_stop'] if by['STD1F'] else None
for nm, pol in (('FP', 'r3'), ('FC', 'r3'), ('FF', 'fastest'), ('FV1', 'r3'), ('FSTRONG', 'r3'), ('FMX', 'r3'),
                ('RP', 'r3'), ('FWEAK', 'r3'), ('FMED', 'r3'), ('FNS', 'r3')):
    if by.get(nm):
        s = episode_summary(by[nm], NL[pol], nm)
        evs = s.pop('_events')
        res[nm] = s
        if nm in ('FP', 'FC', 'FMX', 'FF', 'RP'):
            # 速さのマスごとの成功(報告)
            vb = defaultdict(lambda: [0, 0])
            for e in evs:
                b = min(9, int((e['v'] or 0) / 0.5))
                vb[b][1] += 1
                vb[b][0] += e['ok']
            res[nm]['by_speed_bin'] = {f'{b * 0.5:.1f}-{b * 0.5 + 0.5:.1f}': f'{a}/{n}' for b, (a, n) in sorted(vb.items())}
        if nm == 'FMX':
            res[nm]['magnitude_at_upper'] = sum(1 for e in evs if (e['type'] != 'steer_pulse' or (e['k'] or 0) >= 0.79) and True)
# FP と同じ開始で強さ(フレーム数・k・d・T)の値ごとの成功(範囲の中の値で偏りがないか)
if by.get('FP'):
    evs = []
    for r in by['FP']:
        evs += recoveries(r, NL['r3'])
    mags = defaultdict(lambda: [0, 0])
    for e in evs:
        if e['type'] == 'steer_pulse':
            key = f"steer:{e['cls']}:frames{e['frames']}"
        elif e['type'] == 'excursion':
            key = f"exc:{e['cls']}:d{'hi' if e['d'] is not None and e['d'] > {'弱': .16, '中': .25, '強': .35}[e['cls']] else 'lo'}"
        else:
            key = f"{e['kind']}:{e['cls']}:frames{e['frames']}"
        mags[key][1] += 1
        mags[key][0] += e['ok']
    res['FP_by_magnitude'] = {k: f'{a}/{n}' for k, (a, n) in sorted(mags.items())}
res['FK'] = kick_summary(by.get('FK', []))
res['FKR'] = kick_summary(by.get('FKR', []))
res['REC60'] = unperturbed_summary(by.get('REC60', []))
res['FU'] = unperturbed_summary(by.get('FU', []))
res['FUC'] = unperturbed_summary(by.get('FUC', []))
res['GRID'] = unperturbed_summary(by.get('GRID', []))
# 判定(この確認の組で)
FP = res.get('FP', {})
C5y, C5l = res['FK']['yaw'], res['FK']['lateral']
c6_un = [r for k in ('REC60', 'FU') for r in by.get(k, []) if not r.get('invalid_spawn')]
res['judged_fresh'] = {
    'C1_r3': {'value': f"{FP.get('valid_r3')}/{FP.get('n')}", 'pass': FP.get('valid_r3', 0) >= 0.9 * FP.get('n', 1)},
    'C2': {'value': FP.get('C2_success'), 'pass': (FP.get('C2_success') or 0) >= 0.95},
    'C3': {'value': [FP.get('C3_median_s'), FP.get('C3_p90_s')], 'pass': FP.get('C3_median_s') is not None and FP['C3_median_s'] <= 1.2 and FP['C3_p90_s'] <= 2.5},
    'C4': {'value': [FP.get('events_per_run'), FP.get('masked_fraction_mean')], 'pass': (FP.get('events_per_run') or 0) >= 3.5 and 0.10 <= (FP.get('masked_fraction_mean') or 0) <= 0.40},
    'C5_r3': {'value': [f"{C5y['survived']}/{C5y['judged']}", f"{C5l['survived']}/{C5l['judged']}"], 'pass': (C5y['rate'] or 0) >= 0.90 and (C5l['rate'] or 0) >= 0.90},
    'C6_r3': {'value': {'lap_from_stop': res['STD1_lap_from_stop'], 'REC60_valid': f"{res['REC60']['valid_r3']}/{res['REC60']['n']}", 'REC60_min': res['REC60']['min_clearance'],
                        'FU_valid': f"{res['FU']['valid_r3']}/{res['FU']['n']}", 'FU_min': res['FU']['min_clearance']},
              'pass': (res['STD1_lap_from_stop'] or 99) <= 9.0 and all(valid_r3(r) and r['min_clearance'] >= 0.03 for r in c6_un)},
    'C7': {'value': [FP.get('C7_cls_frac'), FP.get('C7_type_frac')], 'pass': FP.get('C7_pass')},
}
# ======== 20:50 に足した: 決めた試験の組の作り直し・学習の段の行との 1 回ずつの比べ・乱しの大きさ・表と予定の守り・最速との違い ========
EVT = json.load(open('/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/rc3/eval_test.json'))
TBL = json.load(open('/home/jetson/jetformer/experiments/20261013_sim_rl_modes/r3/candidates/recover_r3/recoverable_table_v1.json'))
ZONES = [(5.9, 8.0), (14.6, 19.3)]   # modes.json common.zones の黒シート・奥の直線(乱しを始めない区間)
CLS = {'弱': {'steer': ((3, 5), (0.6, 0.8)), 'exc': ((0.12, 0.2), (0.4, 0.7)), 'brake': (10, 15), 'burst': (6, 10)},
       '中': {'steer': ((6, 9), (0.8, 1.0)), 'exc': ((0.2, 0.3), (0.6, 1.0)), 'brake': (15, 25), 'burst': (10, 15)},
       '強': {'steer': ((10, 12), (1.0, 1.0)), 'exc': ((0.3, 0.4), (0.8, 1.2)), 'brake': (25, 30), 'burst': (15, 20)}}


def combo_key(e):
    if e['type'] == 'speed_pulse':
        return '%s:%s' % (e['kind'], e['cls'])
    return '%s:%s:%s' % (e['type'], e['cls'], 'L' if (e['sign'] or 0) > 0 else 'R')


def table_cell(e):
    sm = (e['s'] % L + L) % L
    return TBL['cells'].get('%d|%d|%s' % (int(sm // TBL['s_bin_m']), int(max(0.0, e['v'] or 0) // TBL['v_bin_mps']), combo_key(e)))


def event_checks(rs):
    """出来事ごと: 表の守り(当てた組のマスが n ≥ 3・割合 ≥ 0.95)・予定(最初 ≥ 2 秒・あいだ ≥ 2 秒・18.5 秒より前・区間の外)・強さの範囲"""
    out = Counter()
    bad = []
    for r in rs:
        evs = r['events']
        for i, e in enumerate(evs):
            c = table_cell(e)
            ok_t = bool(c) and c[0] >= 3 and c[1] / c[0] >= 0.95 - 1e-12
            out['events'] += 1
            out['table_ok'] += ok_t
            if not ok_t:
                bad.append({'run': r['set'] + '/' + r['id'], 'n': e['n'], 'key': combo_key(e), 's': round(e['s'] % L, 3), 'v': e['v'], 'cell': c})
            if i == 0 and e['ps'] < 60:
                out['first_before_2s'] += 1
            if i > 0 and evs[i - 1]['pe'] is not None and e['ps'] - evs[i - 1]['pe'] < 60:
                out['gap_lt_2s'] += 1
            if e['ps'] > 555:
                out['start_after_18_5s'] += 1
            sm = e['s'] % L
            if any(a - 2.0 - 1e-9 <= sm <= b + 1e-9 or a - 2.0 - 1e-9 <= sm - L <= b + 1e-9 for a, b in ZONES):
                out['start_in_or_before_zone'] += 1
            # 自前の射影の s でも(ps のフレーム)
            if e['ps'] < len(r['pf']['s']):
                so = r['pf']['s'][e['ps']] % L
                if any(a - 2.0 - 1e-9 <= so <= b + 1e-9 for a, b in ZONES):
                    out['start_in_or_before_zone_own_s'] += 1
            cl = CLS[e['cls']]
            pe = e['pe'] if e['pe'] is not None else 600
            if e['type'] == 'steer_pulse':
                (f0, f1), (k0, k1) = cl['steer']
                if not (f0 <= (e['frames'] or 0) <= f1 and k0 - 1e-9 <= (e['k'] or 0) <= k1 + 1e-9):
                    out['magnitude_out_of_class'] += 1
            elif e['type'] == 'excursion':
                (d0, d1), (t0, t1) = cl['exc']
                if not (d0 - 1e-9 <= (e['d'] or 0) <= d1 + 1e-9 and t0 - 1e-9 <= (e['T'] or 0) <= t1 + 1e-9):
                    out['magnitude_out_of_class'] += 1
            else:
                f0, f1 = cl[e['kind']]
                if not (f0 <= (e['frames'] or 0) <= f1):
                    out['magnitude_out_of_class'] += 1
            if e.get('aborted'):
                out['aborted'] += 1
            # 乱しの区間の指令: 舵のパルスは ±0.52·k、ブレーキは速さ 0、踏み込みは V_cap(4.5)
            st, sp = r['pf']['st'], r['pf']['sp']
            for f in range(e['ps'], min(pe, len(st))):
                if e['type'] == 'steer_pulse' and abs(abs(st[f]) - 0.52 * (e['k'] or 0)) > 2e-4:
                    out['steer_value_mismatch_frames'] += 1
                if e['type'] == 'speed_pulse' and e['kind'] == 'brake' and abs(sp[f]) > 1e-3:
                    out['brake_value_mismatch_frames'] += 1
                if e['type'] == 'speed_pulse' and e['kind'] == 'burst' and abs(sp[f] - 4.5) > 1e-3:
                    out['burst_value_mismatch_frames'] += 1
    return {'counts': dict(out), 'table_violations': bad[:20]}


def event_dev(rs, nl):
    """出来事ごとの乱しの大きさ: [ps, min(pe + 90, 次の ps, 回の終わり)) で、乱しなしの線からの |Δe_y|・|Δψ| の最大。
    「帯から出なかった」= その区間で一度も |Δe_y| > 0.10 m・|Δψ| > 0.15 rad にならない(protocol の recovered の帯)"""
    rows = []
    for r in rs:
        pf = r['pf']
        T = len(pf['s'])
        evs = r['events']
        recs = recoveries(r, nl)
        for i, e in enumerate(evs):
            pe = e['pe'] if e['pe'] is not None else T
            end = min(pe + 90, evs[i + 1]['ps'] if i + 1 < len(evs) else T, T)
            dey = dps = 0.0
            out_band = out_band_after = False
            for f in range(e['ps'], end):
                b = int((pf['s'][f] % L) / 0.05) % NB
                a = abs(pf['ey'][f] - nl['ey'][b])
                h = abs(wrap(pf['epsi'][f] - nl['psi'][b]))
                dey, dps = max(dey, a), max(dps, h)
                if a > 0.10 or h > 0.15:
                    out_band = True
                    if f >= pe:
                        out_band_after = True
            rows.append({'type': e['type'], 'kind': e.get('kind'), 'cls': e['cls'], 'dey': dey, 'dpsi': dps, 'out_band': out_band, 'out_band_after_pe': out_band_after,
                         'ok': recs[i]['ok'], 't': recs[i]['t']})
    if not rows:
        return None
    res_ = {'events': len(rows), 'never_left_band': round(sum(not x['out_band'] for x in rows) / len(rows), 3),
            'no_dev_after_pe': round(sum(not x['out_band_after_pe'] for x in rows) / len(rows), 3),
            'dey_median_m': round(q([x['dey'] for x in rows], .5), 4), 'dey_p90_m': round(q([x['dey'] for x in rows], .9), 4),
            'dpsi_median_rad': round(q([x['dpsi'] for x in rows], .5), 4), 'dpsi_p90_rad': round(q([x['dpsi'] for x in rows], .9), 4)}
    lb = [x for x in rows if x['out_band_after_pe']]
    res_['left_band_after_pe'] = {'n': len(lb), 'success': f"{sum(x['ok'] for x in lb)}/{len(lb)}", 'rate': round(sum(x['ok'] for x in lb) / len(lb), 4) if lb else None,
                                  'ci95': wilson(sum(x['ok'] for x in lb), len(lb)),
                                  'C3_median_s': round(q([x['t'] for x in lb if x['ok']], .5), 3) if lb else None,
                                  'C3_p90_s': round(q([x['t'] for x in lb if x['ok']], .9), 3) if lb else None}
    lb2 = [x for x in rows if x['out_band']]
    res_['left_band_any'] = {'n': len(lb2), 'success': f"{sum(x['ok'] for x in lb2)}/{len(lb2)}", 'rate': round(sum(x['ok'] for x in lb2) / len(lb2), 4) if lb2 else None}
    by_t = {}
    for t in ('steer_pulse', 'excursion', 'brake', 'burst'):
        sub = [x for x in rows if (x['type'] == t) or (x['type'] == 'speed_pulse' and x['kind'] == t)]
        if sub:
            by_t[t] = {'n': len(sub), 'never_left_band': round(sum(not x['out_band'] for x in sub) / len(sub), 3),
                       'dey_median_m': round(q([x['dey'] for x in sub], .5), 4), 'dpsi_median_rad': round(q([x['dpsi'] for x in sub], .5), 4)}
    res_['by_type'] = by_t
    return res_


def vm7(rs, N=9000, seed=12345):
    pts = []
    for r in rs:
        pf = r['pf']
        s0 = pf['s'][0]
        for i in range(len(pf['s'])):
            if pf['s'][i] - s0 < 3:
                continue
            pts.append((pf['s'][i], pf['ey'][i], pf['epsi'][i], pf['v'][i]))
    if len(pts) > N:
        idx = np.random.default_rng(seed).permutation(len(pts))[:N]
        pts = [pts[i] for i in idx]
    return len({(int((s_ % L) // 0.25), math.floor(e / 0.05), math.floor(h / 0.1), math.floor(v / 0.5)) for s_, e, h, v in pts}), len(pts)


# ---- 決めた試験の組の作り直しと、学習の段の行との比べ ----
T_P = {r['id']: r for r in EVT['tests']['RC-P100-R3']}
T_K = {r['id']: r for r in EVT['tests']['RC-KICK-R3']}
T_U = {r['id']: r for r in EVT['tests']['RC-U60-R3']}
T_R = {r['id']: r for r in EVT['tests']['RC-REC60']}
cmpP = Counter()
diffs = []
for r in by.get('RP', []):
    t = T_P.get(r['id'])
    if t is None:
        cmpP['missing'] += 1
        continue
    cmpP['n'] += 1
    sig = lambda evs: [(e['type'], e['cls'], e['sign'], e.get('kind'), e['ps'], e['pe'], e.get('fallback')) for e in evs]
    same = {'reason': r['reason'] == t['reason'], 'frames': r['frames'] == t['frames'], 'collisions': r['collisions'] == t['collisions'],
            'events': sig(r['events']) == sig(t['events']), 'valid_r3': valid_r3(r) == t['valid_episode_r3']}
    for k_, v_ in same.items():
        cmpP['same_' + k_] += v_
    if not all(same.values()):
        diffs.append({'id': r['id'], **{k_: v_ for k_, v_ in same.items() if not v_}})
    cmpP['max_abs_dmin_m'] = max(cmpP.get('max_abs_dmin_m', 0), abs((r['min_clearance'] or 0) - (t['min_clearance'] or 0)))
    mine = recoveries(r, NL['r3'])
    if len(mine) == len(t['recovery']):
        for a, b in zip(mine, t['recovery']):
            cmpP['events_compared'] += 1
            cmpP['success_same'] += (a['ok'] == b['success'])
            if a['t'] is not None and b['recovered_s'] is not None:
                cmpP['t_max_abs_diff_s'] = max(cmpP.get('t_max_abs_diff_s', 0), abs(a['t'] - b['recovered_s']))
    gm_mine = gate_miss(r)[0] > 0
    cmpP['gate_miss_same'] += (gm_mine == bool(t.get('gate_miss')))
res['replay_vs_training'] = {'RP': dict(cmpP), 'RP_diffs': diffs[:20]}
cmpK = Counter()
kd = []
for r in by.get('RK', []):
    t = T_K.get(r['id'])
    if t is None:
        continue
    cmpK['n'] += 1
    surv = r['reason'] == 'collection_duration_complete' and r['collisions'] == 0 and not stall_r3(r)['stall']
    same = {'reason': r['reason'] == t['reason'], 'frames': r['frames'] == t['frames'], 'allowed': bool(r['kick']['allowed_rt']) == bool(t['kick_allowed']),
            'survival': surv == bool(t['kick_survival_r3'])}
    for k_, v_ in same.items():
        cmpK['same_' + k_] += v_
    if not all(same.values()):
        kd.append({'id': r['id'], 'mine': [r['reason'], r['frames']], 'train': [t['reason'], t['frames']], **{k_: v_ for k_, v_ in same.items() if not v_}})
res['replay_vs_training']['RK'] = dict(cmpK)
res['replay_vs_training']['RK_diffs'] = kd[:20]
cmpU = Counter()
for nm, TT in (('RU', T_U), ('REC60', T_R)):
    for r in by.get(nm, []):
        t = TT.get(r['id'])
        if t is None:
            continue
        cmpU[nm + '_n'] += 1
        cmpU[nm + '_same_valid'] += (valid_r3(r) == t['valid_episode_r3'])
        cmpU[nm + '_max_abs_dmin_m'] = max(cmpU.get(nm + '_max_abs_dmin_m', 0), abs(r['min_clearance'] - t['min_clearance']))
res['replay_vs_training']['RU_REC60'] = dict(cmpU)
res['RK'] = kick_summary(by.get('RK', []))
res['RU'] = unperturbed_summary(by.get('RU', []))
RP = res.get('RP', {})
c6_rep = [r for k_ in ('REC60', 'RU') for r in by.get(k_, []) if not r.get('invalid_spawn')]
res['judged_replay'] = {
    'C1_r3': {'value': f"{RP.get('valid_r3')}/{RP.get('n')}", 'pass': RP.get('valid_r3', 0) >= 90},
    'C2': {'value': RP.get('C2_success'), 'n': RP.get('events'), 'pass': (RP.get('C2_success') or 0) >= 0.95},
    'C3': {'value': [RP.get('C3_median_s'), RP.get('C3_p90_s')], 'pass': RP.get('C3_median_s') is not None and RP['C3_median_s'] <= 1.2 and RP['C3_p90_s'] <= 2.5},
    'C4': {'value': [RP.get('events_per_run'), RP.get('masked_fraction_mean')], 'pass': (RP.get('events_per_run') or 0) >= 3.5 and 0.10 <= (RP.get('masked_fraction_mean') or 0) <= 0.40},
    'C5_r3': {'value': [f"{res['RK']['yaw']['survived']}/{res['RK']['yaw']['judged']}", f"{res['RK']['lateral']['survived']}/{res['RK']['lateral']['judged']}"],
              'pass': (res['RK']['yaw']['rate'] or 0) >= 0.90 and (res['RK']['lateral']['rate'] or 0) >= 0.90},
    'C6_r3': {'value': {'lap_from_stop': res['STD1_lap_from_stop'], 'REC60': f"{res['REC60']['valid_r3']}/{res['REC60']['n']}", 'REC60_min': res['REC60']['min_clearance'],
                        'U60': f"{res['RU']['valid_r3']}/{res['RU']['n']}", 'U60_min': res['RU']['min_clearance'],
                        'min_excluding_spawn_frame': round(min(min(r['pf']['cl']) for r in c6_rep), 5) if c6_rep else None,
                        'starts_lt3cm': [(r['id'], round(r['start_clearance'], 4)) for r in c6_rep if r['start_clearance'] < 0.03]},
              'pass': (res['STD1_lap_from_stop'] or 99) <= 9.0 and all(valid_r3(r) and r['min_clearance'] >= 0.03 for r in c6_rep)},
    'C7': {'value': [RP.get('C7_cls_frac'), RP.get('C7_type_frac')], 'pass': RP.get('C7_pass')},
}
res['judged_replay_n_pass'] = sum(v['pass'] for v in res['judged_replay'].values())
# ---- 乱しの大きさ・表と予定の守り ----
res['event_dev'] = {}
for nm, pol in (('RP', 'r3'), ('FP', 'r3'), ('FV1', 'r3'), ('FF', 'fastest'), ('FSTRONG', 'r3'), ('FMX', 'r3'), ('FC', 'r3'), ('FNS', 'r3')):
    if by.get(nm):
        res['event_dev'][nm] = event_dev([r for r in by[nm] if not r.get('invalid_spawn')], NL[pol])
res['event_checks'] = {}
for nm in ('RP', 'FP', 'FC', 'FF', 'FSTRONG', 'FMX', 'FWEAK', 'FMED', 'FNS'):
    if by.get(nm):
        res['event_checks'][nm] = event_checks([r for r in by[nm] if not r.get('invalid_spawn')])
# ---- 最速との違い(乱しなし。最速のモードそのもの) ----
if by.get('FUF') and by.get('FU'):
    pairs = [(a, b) for a in by['FU'] for b in by['FUF'] if a['id'] == b['id'] and not a.get('invalid_spawn') and not b.get('invalid_spawn')]
    both = [(a, b) for a, b in pairs if valid_r3(a) and valid_r3(b)]
    run = lambda rs: [v for r in rs for v in r['pf']['v'] if v > 0.3]
    lap1 = lambda rs: [r['laps'][0]['elapsed'] for r in rs if r['laps']]
    res['vs_fastest_mode'] = {
        'pairs': len(pairs), 'both_valid': len(both),
        'valid_r3': [sum(valid_r3(a) for a, _ in pairs), sum(valid_r3(b) for _, b in pairs)],
        'mean_speed_running': [round(float(np.mean(run([a for a, _ in both]))), 3), round(float(np.mean(run([b for _, b in both]))), 3)],
        'lap1_median_s': [round(q(lap1([a for a, _ in both]), .5), 3), round(q(lap1([b for _, b in both]), .5), 3)],
        'min_clearance_median': [round(q([a['min_clearance'] for a, _ in both], .5), 4), round(q([b['min_clearance'] for _, b in both], .5), 4)],
        'min_clearance_min': [round(min(a['min_clearance'] for a, _ in both), 4), round(min(b['min_clearance'] for _, b in both), 4)],
        'lt3cm_runs': [sum(1 for a, _ in both if min(a['pf']['cl']) < 0.03), sum(1 for _, b in both if min(b['pf']['cl']) < 0.03)],
        'shield_frames_total': [sum(sum(a['pf']['sh']) for a, _ in pairs), sum(sum(b['pf']['sh']) for _, b in pairs)],
    }
    if 'fastest_mode' in NL:
        d2 = np.abs(NL['r3']['ey'] - NL['fastest_mode']['ey'])
        res['vs_fastest_mode']['nominal_line_mean_abs_dey_m'] = round(float(d2.mean()), 4)
        res['vs_fastest_mode']['nominal_line_p90_abs_dey_m'] = round(float(np.quantile(d2, .9)), 4)
    # 同じ開始の線の差(開始から 3 m 以降、5 cm のマスで |Δe_y| の平均)
    dl = []
    for a, b in both:
        ma, mb = {}, {}
        for rr, mm in ((a, ma), (b, mb)):
            s0 = rr['pf']['s'][0]
            for i in range(len(rr['pf']['s'])):
                if rr['pf']['s'][i] - s0 < 3:
                    continue
                k_ = int((rr['pf']['s'][i] - s0) / 0.05)
                mm.setdefault(k_, rr['pf']['ey'][i])
        ks = set(ma) & set(mb)
        if ks:
            dl.append(float(np.mean([abs(ma[k_] - mb[k_]) for k_ in ks])))
    res['vs_fastest_mode']['same_start_line_diff_mean_m'] = round(float(np.mean(dl)), 4) if dl else None
# ばらつき(VM7、有効な回の最初の 15 シーン、9,000 フレームに間引く)
first15 = lambda rs: [r for r in rs if not r.get('invalid_spawn') and valid_r3(r)][:15]
res['VM7_first15'] = {nm: vm7(first15(by[nm]))[0] for nm in ('FP', 'RP', 'FF', 'FV1', 'FU', 'FUF', 'FNS', 'FSTRONG') if by.get(nm)}
# 止まり: 有効に見えて(valid_episode)止まりで無効になる回
stalls = []
for nm in ('FP', 'FC', 'FUC', 'FU', 'GRID', 'RP', 'RU', 'REC60', 'FF', 'FV1', 'FSTRONG', 'FMX', 'FWEAK', 'FMED', 'FNS', 'FUF'):
    for r in by.get(nm, []):
        if r.get('invalid_spawn'):
            continue
        st_ = stall_r3(r)
        if r['valid_episode'] and (st_['stall'] or progress(r) < L):
            stalls.append({'set': nm, 'id': r['id'], 'stall': st_, 'progress': round(progress(r), 2), 'shield_frames': sum(r['pf']['sh']),
                           'events': len(r['events']), 'start': r['spec']['start'], 'start_clearance': round(r['start_clearance'], 4)})
res['valid_but_stalled'] = {'count': len(stalls), 'by_set': dict(Counter(x['set'] for x in stalls)), 'list': stalls[:60]}
json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
print(json.dumps({'judged_replay': res['judged_replay'], 'judged_fresh': res['judged_fresh']}, ensure_ascii=False, indent=1))

