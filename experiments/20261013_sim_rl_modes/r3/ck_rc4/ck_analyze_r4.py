"""離脱復活 r4 の確認: 自前の走り(ck_batch.mjs の rows_*.jsonl)から判定と報告の値を出す(2026-10-15)。
確認 r3 の ck_analyze.py(sha256 81573123…)を元に、protocol_recover_r4.json(C5_r4・C6_r4・C8・gate_miss_real・start_band)に合わせて書き直した。
学習・評価のコード(eval_rc4.mjs・variation_metrics_rc4.mjs・gate_miss_rc4.mjs など)を使わず、protocol.json definitions・
protocol_recover_r3.json definitions_added・protocol_recover_r4.json definitions_added / acceptance の文から書いた。
  python3 -I ck_analyze_r4.py <runs のフォルダ> <out.json>
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
GATE_MISS_M = 1.519 + 0.5          # gate_miss_real: ゲートの間の最大 1.519 m + 0.5 m
BAND = (0.30, 3.70)                # start_band
rows = []
for p in sorted(glob.glob(RUNS + '/rows_*.jsonl')):
    for line in open(p):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass                # 停電で切れた最後の行は飛ばす(ck_batch.mjs が走らせ直す)
by = defaultdict(list)
_seen = set()
DUP = 0
for r in rows:
    if (r['set'], r['id']) in _seen:
        DUP += 1
        continue
    _seen.add((r['set'], r['id']))
    by[r['set']].append(r)
for k in by:
    by[k].sort(key=lambda r: (len(r['id']), r['id']))


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


def frac(k, n):
    return f'{k}/{n}'


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


NL = {k: nominal_line(by[n]) for k, n in (('r4', 'NOM-r4'), ('r3', 'NOM-r3'), ('fastest', 'NOM-fastest'), ('fastest_mode', 'NOMF')) if by.get(n)}
POL = {'FP': 'r4', 'FC': 'r4', 'FCS': 'r4', 'FF': 'fastest', 'FV1': 'r4', 'FSTRONG': 'r4', 'FMX': 'r4', 'RP': 'r4', 'FWEAK': 'r4', 'FMED': 'r4',
       'FNS': 'r4', 'FR3': 'r3'}


def brake_windows(r, T):
    ex = np.zeros(T, bool)
    for e in r['events']:
        if e['type'] == 'speed_pulse' and e.get('kind') == 'brake':
            pe = e['pe'] if e['pe'] is not None else T
            ex[e['ps']:min(T, pe + 30)] = True
    return ex


def stall_r3(r, reset_in_window=False):
    """definitions_added.stall: フレーム 30 以降に |v| < 0.2 が 30 フレーム以上。ブレーキの乱しの区間とそのあと 30 フレームは数えない
    (既定は「数えず、続きも切らない」= きびしい側。reset_in_window=True は区間で切る読み)"""
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
    """gate_miss_real(自前): sim の passed_gates と completed_laps が変わらないまま、自前の R 上の s が最後に変わったところから
    2.019 m より先へ進んだ。戻り値 (回数, フレームの列)"""
    s, g = r['pf']['s'], r['pf']['g']
    lap = r['pf'].get('lap') or [0] * len(s)
    last_s, last_k, misses, at = s[0], (g[0], lap[0]), 0, []
    for f in range(len(s)):
        k = (g[f], lap[f])
        if k != last_k:
            last_k, last_s = k, s[f]
        elif f and abs(s[f] - s[f - 1]) > 0.5:
            # 02:30 足した: 自前の射影が 1 フレームで 0.5 m より跳んだ(曲がりの内側から大きく外れた開始で、近い別の区間へ移る)ところは
            # 進んだ距離に数えない(角のゲート 17・右 0.2 m・+15° で s が 2 フレームで 2.2 m 跳び、ゲートの外しと誤って数えた。sim はゲートを普通に数えている)
            last_s = s[f] - (s[f - 1] - last_s)
        elif s[f] - last_s > GATE_MISS_M:
            misses += 1
            at.append(f)
            last_s = s[f]
    return misses, at


def valid_r3(r):
    if r.get('invalid_spawn'):
        return False
    return bool(r['valid_episode'] and not stall_r3(r)['stall'] and progress(r) >= L - 1e-9)


def unstick_frames(r):
    return sum(1 for x in (r['pf'].get('fb') or []) if x == 'unstick')


def recoveries(r, nl):
    pf = r['pf']
    T = len(pf['s'])
    v = pf['v']
    s = pf['s']
    st60, wrong = [False] * T, [False] * T
    run60 = 0
    smax = -1e9
    for f in range(T):
        run60 = run60 + 1 if (f >= 30 and v[f] < 0.2) else 0
        st60[f] = run60 >= 60
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
        for f in range(pe, until):
            if not bad and (st60[f] or wrong[f]):
                bad, why = True, 'stall60' if st60[f] else 'wrong_way'
        t = None if rec is None else (rec - pe) / 30
        out.append({'n': ev['n'], 'type': ev['type'], 'kind': ev.get('kind'), 'cls': ev['cls'], 'sign': ev['sign'], 'ps': ev['ps'], 'pe': ev['pe'],
                    's': ev['s'] % L, 'v': ev.get('v'), 'req': ev.get('req'), 'fallback': ev.get('fallback'), 'wait': ev.get('wait'),
                    'frames': ev.get('frames'), 'k': ev.get('k'), 'd': ev.get('d'), 'T': ev.get('T'), 'aborted': ev.get('aborted'),
                    't': t, 'bad': bad, 'why': why, 'ok': t is not None and t <= 3.0 and not bad, 'gate_miss_after': None})
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
            'shield_l0': int(((sh == 1) & l0).sum()), 'runner': r['labels']}


def episode_summary(rs, nl):
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
    out['invalid_runs'] = [{'id': r['id'], 'reason': r['reason'], 'frames': r['frames'], 'collision': r['collision'], 'stall': stall_r3(r),
                            'progress': round(progress(r), 2), 'start': r['spec']['start'], 'start_clearance': round(r['start_clearance'], 4),
                            'shield_frames': sum(r['pf']['sh']), 'unstick_frames': unstick_frames(r), 'events': len(r['events'])}
                           for r, ok in zip(rs, vr3) if not ok]
    evcount, mfs, gm, gm_ids = [], [], 0, []
    sh_after = sh_other = n_after = n_other = 0
    labs = Counter()
    for r in rs:
        rec = recoveries(r, nl)
        T = len(r['pf']['s'])
        misses, at = gate_miss(r)
        for e in rec:
            pe = e['pe'] if e['pe'] is not None else T
            e['gate_miss_after'] = any(e['ps'] <= f <= pe + 90 for f in at)
            e['run'] = r['id']
            e['run_valid_r3'] = valid_r3(r)
        ev += rec
        evcount.append(len(r['events']))
        mfs.append(mask_frac(r))
        if valid_r3(r) and misses:
            gm += 1
            gm_ids.append({'id': r['id'], 'frames': at[:4], 's': [round(r['pf']['s'][f] % L, 2) for f in at[:4]]})
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
        for k2 in ('l0_outside_events', 'l1_inside_events', 'changed_l1_not_shield', 'shield_in_perturb', 'shield_l0'):
            labs[k2] += la[k2]
        for k2, v2 in la['runner'].items():
            labs['runner_' + k2] += v2
    out['events'] = len(ev)
    out['events_per_run'] = round(float(np.mean(evcount)), 4) if evcount else None
    out['masked_fraction_mean'] = round(float(np.mean(mfs)), 4) if mfs else None
    if ev:
        ok = [e for e in ev if e['ok']]
        out['C2_success'] = round(len(ok) / len(ev), 4)
        out['C2_k_n'] = frac(len(ok), len(ev))
        out['C2_ci95'] = wilson(len(ok), len(ev))
        ts = [e['t'] for e in ok]
        out['C3_median_s'] = round(q(ts, .5), 3) if ts else None
        out['C3_p90_s'] = round(q(ts, .9), 3) if ts else None
        cls = Counter(e['cls'] for e in ev)
        typ = Counter(e['type'] for e in ev)
        out['C7_cls_frac'] = {k: round(v / len(ev), 3) for k, v in cls.items()}
        out['C7_type_frac'] = {k: round(v / len(ev), 3) for k, v in typ.items()}
        out['C7_pass'] = all(cls.get(c, 0) / len(ev) >= 0.15 for c in ('弱', '中', '強')) and all(typ.get(t, 0) / len(ev) >= 0.10 for t in ('steer_pulse', 'excursion', 'speed_pulse'))
        out['by_cls'] = {c: frac(sum(e['ok'] for e in ev if e['cls'] == c), sum(1 for e in ev if e['cls'] == c)) for c in ('弱', '中', '強')}
        out['by_type'] = {t: frac(sum(e['ok'] for e in ev if e['type'] == t), sum(1 for e in ev if e['type'] == t)) for t in ('steer_pulse', 'excursion', 'speed_pulse')}
        out['by_kind'] = dict(Counter(f"{e['type']}:{e['kind']}" for e in ev))
        out['fallback'] = dict(Counter(e['fallback'] for e in ev))
        out['req_type'] = dict(Counter((e['req'] or {}).get('type') for e in ev))
        out['start_black_sheet_exit'] = round(sum(1 for e in ev if 8.0 <= e['s'] <= 8.25) / len(ev), 3)
        out['start_in_band'] = round(sum(1 for e in ev if BAND[0] <= e['s'] <= BAND[1]) / len(ev), 3)
        out['failed_events'] = [{k: e[k] for k in ('run', 'n', 'type', 'kind', 'cls', 'sign', 'ps', 'pe', 's', 'v', 't', 'why', 'fallback', 'frames', 'k', 'd', 'T', 'aborted', 'run_valid_r3')}
                                for e in ev if not e['ok']]
        out['events_gate_miss_after'] = sum(1 for e in ev if e['gate_miss_after'])
    out['C8_gate_miss_valid_runs'] = frac(gm, out['valid_r3'])
    out['C8_rate'] = round(gm / out['valid_r3'], 4) if out['valid_r3'] else None
    out['gate_miss_ids'] = gm_ids[:40]
    out['shield_after3s'] = round(sh_after / max(1, n_after), 5)
    out['shield_other'] = round(sh_other / max(1, n_other), 5)
    out['unstick_frames'] = sum(unstick_frames(r) for r in rs)
    out['runs_with_unstick'] = sum(1 for r in rs if unstick_frames(r))
    out['labels'] = dict(labs)
    vals = [r for r, ok in zip(rs, vr3) if ok]
    out['min_clearance_valid'] = round(min(r['min_clearance'] for r in vals), 5) if vals else None
    out['clear_lt3cm_valid'] = sum(1 for r in vals if r['min_clearance'] < 0.03)
    out['audit_pos_err_max'] = max((r['audit_pos_err'] or 0) for r in rs) if rs else None
    out['overlap_without_collision'] = sum(1 for r in rs if r['overlap_frames'] and not r['collisions'])
    out['proj_vs_runtime_ds_max'] = max(r['proj_vs_runtime']['ds'] for r in rs) if rs else None
    out['_events'] = ev
    return out


def in_band(s):
    m = s % L
    return BAND[0] <= m <= BAND[1]


def kick_survived(r):
    return r['reason'] == 'collection_duration_complete' and r['collisions'] == 0 and not stall_r3(r)['stall']


def kick_summary(rs):
    res = {}
    rs = [r for r in rs if not r.get('invalid_spawn')]
    res['not_fired'] = sum(1 for r in rs if not r['kick'])
    for kind in ('yaw', 'lateral'):
        sub = [r for r in rs if r['spec'].get('kind') == kind and r['kick']]
        judged = [r for r in sub if r['kick']['allowed_rt']]
        surv = [r for r in judged if kick_survived(r)]
        fails = []
        for r in judged:
            if kick_survived(r):
                continue
            kf = r['kick']['frame']
            fails.append({'id': r['id'], 'reason': r['reason'], 'frames': r['frames'], 'kick_frame': kf, 'same_step': r['collisions'] > 0 and r['frames'] <= kf + 1,
                          'clr_after_kick': round(r['kick']['clr_after'], 4), 's': round(r['kick']['rt_s'] % L, 2), 'v': round(r['kick']['v'], 2),
                          'ey_pre': round(r['kick']['ey_pre'], 3), 'sign': 1 if (r['spec']['kick'].get(kind) or 0) > 0 else -1, 'stall': stall_r3(r)['stall'],
                          'band': in_band(r['kick']['rt_s']), 'start': r['spec']['start']})
        allsurv = [r for r in sub if kick_survived(r)]
        res[kind] = {'judged': len(judged), 'survived': len(surv), 'rate': round(len(surv) / len(judged), 4) if judged else None,
                     'ci95': wilson(len(surv), len(judged)), 'all_positions': frac(len(allsurv), len(sub)), 'fails': fails,
                     'own_vs_rt_allowed_mismatch': sum(1 for r in sub if r['kick']['allowed_rt'] != r['kick']['allowed_own'])}
    j = [r for r in rs if r['kick'] and r['kick']['allowed_rt']]
    side = lambda r: 1 if (r['spec']['kick'].get('yaw') or r['spec']['kick'].get('lateral') or 0) > 0 else -1
    bR = [r for r in j if in_band(r['kick']['rt_s']) and side(r) < 0]
    bL = [r for r in j if in_band(r['kick']['rt_s']) and side(r) > 0]
    out_ = [r for r in j if not in_band(r['kick']['rt_s'])]
    res['band_right'] = frac(sum(kick_survived(r) for r in bR), len(bR))
    res['band_left'] = frac(sum(kick_survived(r) for r in bL), len(bL))
    res['outside'] = frac(sum(kick_survived(r) for r in out_), len(out_))
    res['band_right_clr_before'] = [round(r['pf']['cl'][r['kick']['frame'] - 1], 3) for r in bR if r['pf'].get('cl') and r['kick']['frame'] >= 1][:60]
    res['band_right_v'] = round(float(np.mean([r['kick']['v'] for r in bR])), 3) if bR else None
    return res


def unperturbed_summary(rs):
    rs = [r for r in rs if not r.get('invalid_spawn')]
    out = {'n': len(rs), 'valid_r3': sum(valid_r3(r) for r in rs), 'stall': sum(stall_r3(r)['stall'] for r in rs),
           'collisions': sum(1 for r in rs if r['collisions']), 'progress_short': sum(1 for r in rs if r['valid_episode'] and progress(r) < L)}
    mins = [r['min_clearance'] for r in rs]
    out['min_clearance'] = round(min(mins), 5) if mins else None
    out['lt3cm'] = [{'id': r['id'], 'min': round(r['min_clearance'], 4), 'start_clearance': round(r['start_clearance'], 4), 'min_frame': r['min_frame'],
                     'min_after_frame30': round(min(r['pf']['cl'][30:]), 4) if r['pf']['cl'] and len(r['pf']['cl']) > 30 else None,
                     'unstick_frames': unstick_frames(r), 'start': r['spec']['start']} for r in rs if r['min_clearance'] < 0.03]
    good = [r for r in rs if r['start_clearance'] >= 0.03]
    out['start_ge3cm'] = {'n': len(good), 'valid_r3': sum(valid_r3(r) for r in good),
                          'min_clearance': round(min(r['min_clearance'] for r in good), 5) if good else None,
                          'lt3cm': sum(1 for r in good if r['min_clearance'] < 0.03),
                          'invalid': [r['id'] for r in good if not valid_r3(r)],
                          'gate_miss_valid': sum(1 for r in good if valid_r3(r) and gate_miss(r)[0])}
    out['invalid'] = [{'id': r['id'], 'reason': r['reason'], 'frames': r['frames'], 'collision': r['collision'], 'stall': stall_r3(r), 'progress': round(progress(r), 2),
                       'start': r['spec']['start'], 'start_clearance': round(r['start_clearance'], 4), 'shield_frames': sum(r['pf']['sh']),
                       'unstick_frames': unstick_frames(r)} for r in rs if not valid_r3(r)]
    out['gate_miss_valid'] = sum(1 for r in rs if valid_r3(r) and gate_miss(r)[0])
    out['gate_miss_ids'] = [r['id'] for r in rs if valid_r3(r) and gate_miss(r)[0]][:30]
    out['unstick_frames'] = sum(unstick_frames(r) for r in rs)
    out['runs_with_unstick'] = sum(1 for r in rs if unstick_frames(r))
    out['shield_frames'] = sum(sum(r['pf']['sh']) for r in rs)
    return out


res = {'duplicates_dropped': DUP, 'counts': {k: len(v) for k, v in by.items()},
       'policy_sha_by_set': {k: sorted({str(r.get('policy_sha')) for r in v}) for k, v in by.items()},
       'nominal_line': {k: {'runs': v['runs'], 'valid': v['valid'], 'filled_bins': v['filled']} for k, v in NL.items()}}
for a, b in (('r4', 'fastest_mode'), ('r4', 'r3'), ('r3', 'fastest_mode'), ('r4', 'fastest')):
    if a in NL and b in NL:
        d = np.abs(NL[a]['ey'] - NL[b]['ey'])
        bandb = [i for i in range(NB) if BAND[0] <= i * 0.05 <= BAND[1]]
        res[f'nominal_{a}_vs_{b}'] = {'mean_abs_dey_m': round(float(d.mean()), 4), 'p90_abs_dey_m': round(float(np.quantile(d, .9)), 4),
                                     'max_abs_dey_m': round(float(d.max()), 4), 'band_mean_abs_dey_m': round(float(d[bandb].mean()), 4)}
for nm in ('NOM-r4', 'NOM-r3', 'NOM-fastest', 'NOMF'):
    rs = by.get(nm, [])
    if not rs:
        continue
    run = [v for r in rs for v in r['pf']['v'] if v > 0.3]
    bandv = [r['pf']['v'][i] for r in rs for i in range(len(r['pf']['s'])) if in_band(r['pf']['s'][i]) and r['pf']['s'][i] - r['pf']['s'][0] > 3]
    bandcl = [r['pf']['cl'][i] for r in rs for i in range(len(r['pf']['s'])) if in_band(r['pf']['s'][i]) and r['pf']['s'][i] - r['pf']['s'][0] > 3]
    res[nm] = {'valid': sum(r['valid_episode'] for r in rs), 'mean_speed_running': round(float(np.mean(run)), 3), 'laps': [round(l['elapsed'], 3) for r in rs for l in r['laps']][:20],
               'min_clearance': round(min(r['min_clearance'] for r in rs), 4), 'band_speed_mean': round(float(np.mean(bandv)), 3) if bandv else None,
               'band_min_clearance': round(float(min(bandcl)), 4) if bandcl else None}
for k in ('STD1', 'STD1F'):
    if by.get(k):
        res[k] = {'lap_from_stop': by[k][0]['lap_from_stop'], 'min_clearance': by[k][0]['min_clearance'], 'reason': by[k][0]['reason']}
EVS = {}
for nm, pol in POL.items():
    if by.get(nm) and pol in NL:
        s_ = episode_summary(by[nm], NL[pol])
        EVS[nm] = s_.pop('_events')
        res[nm] = s_
        vb = defaultdict(lambda: [0, 0])
        for e in EVS[nm]:
            b = min(9, int((e['v'] or 0) / 0.5))
            vb[b][1] += 1
            vb[b][0] += e['ok']
        res[nm]['by_speed_bin'] = {f'{b * 0.5:.1f}-{b * 0.5 + 0.5:.1f}': f'{a}/{n}' for b, (a, n) in sorted(vb.items())}
for nm in ('RK', 'RKR', 'FK', 'FKR', 'FKB', 'FKBR', 'FKZ'):
    if by.get(nm):
        res[nm] = kick_summary(by[nm])
for nm in ('REC60', 'RU', 'FU', 'FUC', 'GRID', 'FUF', 'FUL'):
    if by.get(nm):
        res[nm] = unperturbed_summary(by[nm])


def judged(P, K1, K2, U, RECk='REC60', STD='STD1', name=''):
    """protocol_recover_r4.json acceptance の 8 項目を、P(乱しあり 100 回の組)・K1(時刻の格子のキック)・K2(新しい開始のキック)・U(乱しなし)で"""
    Pp = res.get(P, {})
    k1, k2 = res.get(K1, {}), res.get(K2, {})
    rates = [x['rate'] for kk in (k1, k2) for x in (kk.get('yaw', {}), kk.get('lateral', {})) if x]
    un = [r for k_ in (RECk, U) for r in by.get(k_, []) if not r.get('invalid_spawn')]
    if U in ('FU',):   # 新しい乱しなしの開始は、C6_r4 と同じく開始のすき間 ≥ 3 cm の開始だけで判定する(そうでない開始は報告だけ)
        un = [r for r in un if r['set'] == RECk or r['start_clearance'] >= 0.03]
    lap = res.get(STD, {}).get('lap_from_stop')
    pert_valid = Pp.get('valid_r3', 0)
    gmP = int(Pp.get('C8_gate_miss_valid_runs', '0/0').split('/')[0])
    gmU = sum(1 for r in un if valid_r3(r) and gate_miss(r)[0])
    out = {
        'C1_r3': {'value': f"{Pp.get('valid_r3')}/{Pp.get('n')}", 'pass': Pp.get('valid_r3', 0) >= 0.9 * max(1, Pp.get('n', 1) - Pp.get('invalid_spawn', 0))},
        'C2': {'value': Pp.get('C2_success'), 'k_n': Pp.get('C2_k_n'), 'pass': (Pp.get('C2_success') or 0) >= 0.95},
        'C3': {'value': [Pp.get('C3_median_s'), Pp.get('C3_p90_s')], 'pass': Pp.get('C3_median_s') is not None and Pp['C3_median_s'] <= 1.2 and Pp['C3_p90_s'] <= 2.5},
        'C4': {'value': [Pp.get('events_per_run'), Pp.get('masked_fraction_mean')], 'pass': (Pp.get('events_per_run') or 0) >= 3.5 and 0.10 <= (Pp.get('masked_fraction_mean') or 0) <= 0.40},
        'C5_r4': {'value': [f"{x['survived']}/{x['judged']}" for kk in (k1, k2) for x in (kk.get('yaw', {}), kk.get('lateral', {})) if x],
                  'pass': len(rates) == 4 and all((x or 0) >= 0.90 for x in rates)},
        'C6_r4': {'value': {'lap_from_stop': lap, 'unperturbed_valid': f"{sum(valid_r3(r) for r in un)}/{len(un)}",
                            'min_clearance': round(min(r['min_clearance'] for r in un), 5) if un else None,
                            'fails': [r['set'] + '/' + r['id'] for r in un if not (valid_r3(r) and r['min_clearance'] >= 0.03)][:20]},
                  'pass': (lap or 99) <= 9.0 and all(valid_r3(r) and r['min_clearance'] >= 0.03 for r in un)},
        'C7': {'value': [Pp.get('C7_cls_frac'), Pp.get('C7_type_frac')], 'pass': bool(Pp.get('C7_pass'))},
        'C8': {'value': {'perturbed': f'{gmP}/{pert_valid}', 'unperturbed': f'{gmU}/{len(un)}'}, 'pass': pert_valid > 0 and gmP <= 0.10 * pert_valid and gmU == 0},
    }
    out['n_pass'] = sum(v['pass'] for v in out.values() if isinstance(v, dict))
    return out


res['judged_replay'] = judged('RP', 'RK', 'RKR', 'RU')
res['judged_fresh'] = judged('FP', 'FK', 'FKR', 'FU')
# ======== 学習の段の試験の行(eval_test.json)との 1 回ずつの比べ ========
EVT = json.load(open('/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/rc4/eval_test.json'))
TBL = json.load(open('/home/jetson/jetformer/experiments/20261013_sim_rl_modes/r3/candidates/recover_r4/recoverable_table_v2.json'))
T_P = {r['id']: r for r in EVT['tests']['RC-P100-R4']}
T_K = {r['id']: r for r in EVT['tests']['RC-KICK-R4']}
T_KR = {r['id']: r for r in EVT['tests']['RC-KICKR-R4']}
T_U = {r['id']: r for r in EVT['tests']['RC-U60-R4']}
T_R = {r['id']: r for r in EVT['tests']['RC-REC60']}
cmpP = Counter()
diffs = []
sig = lambda evs: [(e['type'], e['cls'], e['sign'], e.get('kind'), e['ps'], e['pe'], e.get('fallback')) for e in evs]
for r in by.get('RP', []):
    t = T_P.get(r['id'])
    if t is None:
        cmpP['missing'] += 1
        continue
    cmpP['n'] += 1
    same = {'reason': r['reason'] == t['reason'], 'frames': r['frames'] == t['frames'], 'collisions': r['collisions'] == t['collisions'],
            'events': sig(r['events']) == sig(t['events']), 'valid_r3': valid_r3(r) == t['valid_episode_r3'], 'gate_miss': (gate_miss(r)[0] > 0) == bool(t.get('gate_miss'))}
    for k_, v_ in same.items():
        cmpP['same_' + k_] += v_
    if not all(same.values()):
        diffs.append({'id': r['id'], **{k_: v_ for k_, v_ in same.items() if not v_}})
    cmpP['max_abs_dmin_m'] = max(cmpP.get('max_abs_dmin_m', 0), abs((r['min_clearance'] or 0) - (t['min_clearance'] or 0)))
    mine = recoveries(r, NL['r4']) if 'r4' in NL else []
    if mine and len(mine) == len(t['recovery']):
        for a, b in zip(mine, t['recovery']):
            cmpP['events_compared'] += 1
            cmpP['success_same'] += (a['ok'] == b['success'])
            if a['t'] is not None and b['recovered_s'] is not None:
                cmpP['t_max_abs_diff_s'] = max(cmpP.get('t_max_abs_diff_s', 0), abs(a['t'] - b['recovered_s']))
res['replay_vs_training'] = {'RP': dict(cmpP), 'RP_diffs': diffs[:20]}
for nm, TT in (('RK', T_K), ('RKR', T_KR)):
    cmpK = Counter()
    kd = []
    for r in by.get(nm, []):
        t = TT.get(r['id'])
        if t is None:
            continue
        cmpK['n'] += 1
        same = {'reason': r['reason'] == t['reason'], 'frames': r['frames'] == t['frames'], 'allowed': bool(r['kick'] and r['kick']['allowed_rt']) == bool(t['kick_allowed']),
                'survival': kick_survived(r) == bool(t['kick_survival_r3'])}
        for k_, v_ in same.items():
            cmpK['same_' + k_] += v_
        if not all(same.values()):
            kd.append({'id': r['id'], 'mine': [r['reason'], r['frames']], 'train': [t['reason'], t['frames']], **{k_: v_ for k_, v_ in same.items() if not v_}})
    res['replay_vs_training'][nm] = dict(cmpK)
    res['replay_vs_training'][nm + '_diffs'] = kd[:20]
cmpU = Counter()
for nm, TT in (('RU', T_U), ('REC60', T_R)):
    for r in by.get(nm, []):
        t = TT.get(r['id'])
        if t is None:
            continue
        cmpU[nm + '_n'] += 1
        cmpU[nm + '_same_valid'] += (valid_r3(r) == t['valid_episode_r3'])
        cmpU[nm + '_max_abs_dmin_m'] = max(cmpU.get(nm + '_max_abs_dmin_m', 0), abs(r['min_clearance'] - t['min_clearance']))
        cmpU[nm + '_same_gate_miss'] += ((gate_miss(r)[0] > 0) == bool(t.get('gate_miss')))
res['replay_vs_training']['RU_REC60'] = dict(cmpU)
if by.get('STD1'):
    res['replay_vs_training']['STD_lap_diff_s'] = abs(by['STD1'][0]['lap_from_stop'] - EVT['tests']['RC-STD'][0]['lap_from_stop'])
# ======== 乱しの大きさ・表と予定の守り ========
ZONES = [(5.9, 8.0), (14.6, 19.3)]
CLS = {'弱': {'steer': ((3, 5), (0.6, 0.8)), 'exc': ((0.12, 0.2), (0.4, 0.7)), 'brake': (10, 15), 'burst': (6, 10)},
       '中': {'steer': ((6, 9), (0.8, 1.0)), 'exc': ((0.2, 0.3), (0.6, 1.0)), 'brake': (15, 25), 'burst': (10, 15)},
       '強': {'steer': ((10, 12), (1.0, 1.0)), 'exc': ((0.3, 0.4), (0.8, 1.2)), 'brake': (25, 30), 'burst': (15, 20)}}


def combo_key(e):
    if e['type'] == 'speed_pulse':
        return '%s:%s' % (e['kind'], e['cls'])
    return '%s:%s:%s' % (e['type'], e['cls'], 'L' if (e['sign'] or 0) > 0 else 'R')


def table_cell(e):
    sm = ((e.get('s') or 0) % L + L) % L
    return TBL['cells'].get('%d|%d|%s' % (int(sm // TBL['s_bin_m']), int(max(0.0, e.get('v') or 0) // TBL['v_bin_mps']), combo_key(e)))


def event_checks(rs, v1=False):
    out = Counter()
    bad = []
    for r in rs:
        evs = r['events']
        for i, e in enumerate(evs):
            c = table_cell(e)
            ok_t = bool(c) and c[0] >= 3 and c[1] / c[0] >= 0.95 - 1e-12
            out['events'] += 1
            out['table_ok'] += ok_t
            if not ok_t and not v1:
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
    rows_ = []
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
            rows_.append({'type': e['type'], 'kind': e.get('kind'), 'cls': e['cls'], 'dey': dey, 'dpsi': dps, 'out_band': out_band, 'out_band_after_pe': out_band_after,
                          'ok': recs[i]['ok'], 't': recs[i]['t']})
    if not rows_:
        return None
    r_ = {'events': len(rows_), 'never_left_band': round(sum(not x['out_band'] for x in rows_) / len(rows_), 3),
          'dey_median_m': round(q([x['dey'] for x in rows_], .5), 4), 'dey_p90_m': round(q([x['dey'] for x in rows_], .9), 4)}
    lb = [x for x in rows_ if x['out_band']]
    r_['left_band'] = {'n': len(lb), 'success': frac(sum(x['ok'] for x in lb), len(lb)), 'rate': round(sum(x['ok'] for x in lb) / len(lb), 4) if lb else None,
                       'ci95': wilson(sum(x['ok'] for x in lb), len(lb)),
                       'C3_median_s': round(q([x['t'] for x in lb if x['ok']], .5), 3) if lb else None,
                       'C3_p90_s': round(q([x['t'] for x in lb if x['ok']], .9), 3) if lb else None}
    r_['by_type'] = {t: {'n': len(sub), 'never_left_band': round(sum(not x['out_band'] for x in sub) / len(sub), 3)}
                     for t in ('steer_pulse', 'excursion', 'brake', 'burst')
                     for sub in [[x for x in rows_ if x['type'] == t or (x['type'] == 'speed_pulse' and x['kind'] == t)]] if sub}
    return r_


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
    return len({(int((s_ % L) // 0.25), math.floor(e / 0.05), math.floor(h / 0.1), math.floor(v / 0.5)) for s_, e, h, v in pts})


res['event_dev'] = {nm: event_dev([r for r in by[nm] if not r.get('invalid_spawn')], NL[POL[nm]]) for nm in POL if by.get(nm) and POL[nm] in NL}
res['event_checks'] = {nm: event_checks([r for r in by[nm] if not r.get('invalid_spawn')], v1=(nm == 'FV1')) for nm in POL if by.get(nm)}
# ======== 最速との違い(乱しなし。最速のモードそのもの) ========
if by.get('FUF') and by.get('FU'):
    fu = {r['id']: r for r in by['FU']}
    pairs = [(fu[b['id']], b) for b in by['FUF'] if b['id'] in fu and not b.get('invalid_spawn') and not fu[b['id']].get('invalid_spawn')]
    both = [(a, b) for a, b in pairs if valid_r3(a) and valid_r3(b)]
    run = lambda rs: [v for r in rs for v in r['pf']['v'] if v > 0.3]
    lap1 = lambda rs: [r['laps'][0]['elapsed'] for r in rs if r['laps']]
    res['vs_fastest_mode'] = {
        'pairs': len(pairs), 'both_valid': len(both),
        'valid_r3': [sum(valid_r3(a) for a, _ in pairs), sum(valid_r3(b) for _, b in pairs)],
        'collisions': [sum(1 for a, _ in pairs if a['collisions']), sum(1 for _, b in pairs if b['collisions'])],
        'mean_speed_running': [round(float(np.mean(run([a for a, _ in both]))), 3), round(float(np.mean(run([b for _, b in both]))), 3)],
        'lap1_median_s': [round(q(lap1([a for a, _ in both]), .5), 3), round(q(lap1([b for _, b in both]), .5), 3)],
        'min_clearance_median': [round(q([a['min_clearance'] for a, _ in both], .5), 4), round(q([b['min_clearance'] for _, b in both], .5), 4)],
        'min_clearance_min': [round(min(a['min_clearance'] for a, _ in both), 4), round(min(b['min_clearance'] for _, b in both), 4)],
        'lt3cm_runs': [sum(1 for a, _ in both if a['min_clearance'] < 0.03), sum(1 for _, b in both if b['min_clearance'] < 0.03)],
        'gate_miss_valid': [sum(1 for a, _ in pairs if valid_r3(a) and gate_miss(a)[0]), sum(1 for _, b in pairs if valid_r3(b) and gate_miss(b)[0])],
    }
    dl, dlb = [], []
    for a, b in both:
        ma, mb = {}, {}
        for rr, mm in ((a, ma), (b, mb)):
            s0 = rr['pf']['s'][0]
            for i in range(len(rr['pf']['s'])):
                if rr['pf']['s'][i] - s0 < 3:
                    continue
                mm.setdefault(int((rr['pf']['s'][i] - s0) / 0.05), (rr['pf']['ey'][i], rr['pf']['s'][i]))
        ks = set(ma) & set(mb)
        if ks:
            dl.append(float(np.mean([abs(ma[k_][0] - mb[k_][0]) for k_ in ks])))
            kb = [k_ for k_ in ks if in_band(ma[k_][1])]
            if kb:
                dlb.append(float(np.mean([abs(ma[k_][0] - mb[k_][0]) for k_ in kb])))
    res['vs_fastest_mode']['same_start_line_diff_mean_m'] = round(float(np.mean(dl)), 4) if dl else None
    res['vs_fastest_mode']['same_start_line_diff_band_mean_m'] = round(float(np.mean(dlb)), 4) if dlb else None
first15 = lambda rs: [r for r in rs if not r.get('invalid_spawn') and valid_r3(r)][:15]
res['VM7_first15'] = {nm: vm7(first15(by[nm])) for nm in ('FP', 'RP', 'FF', 'FV1', 'FR3', 'FU', 'FUF', 'FNS', 'FSTRONG') if by.get(nm)}
# ======== 止まり: 有効に見えて(valid_episode)止まり・進み不足で無効になる回 ========
stalls = []
for nm in by:
    if nm.startswith('NOM') or nm in ('STD1', 'STD1F'):
        continue
    for r in by[nm]:
        if r.get('invalid_spawn'):
            continue
        st_ = stall_r3(r)
        if r['valid_episode'] and (st_['stall'] or progress(r) < L):
            stalls.append({'set': nm, 'id': r['id'], 'stall': st_, 'progress': round(progress(r), 2), 'shield_frames': sum(r['pf']['sh']),
                           'unstick_frames': unstick_frames(r), 'events': len(r['events']), 'start': r['spec']['start'],
                           'start_clearance': round(r['start_clearance'], 4), 'policy': r['spec'].get('policy')})
res['valid_but_stalled'] = {'count': len(stalls), 'by_set': dict(Counter(x['set'] for x in stalls)), 'list': stalls[:80]}
# 乱しありの有効な回のすき間の最小(報告)
res['perturbed_valid_min_clearance'] = {nm: res[nm]['min_clearance_valid'] for nm in POL if nm in res}
json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
print(json.dumps({'counts': res['counts'], 'judged_replay': res['judged_replay'], 'judged_fresh': res['judged_fresh'],
                  'replay_vs_training': {k: v for k, v in res['replay_vs_training'].items() if not k.endswith('_diffs')}}, ensure_ascii=False, indent=1))
