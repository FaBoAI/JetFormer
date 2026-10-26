#!/usr/bin/env python3
"""速さの天井・速さのゲインの自動合わせの記録の再生(2026-10-25、spec.md 7 章)。読むだけ。生の記録・ラベル・設定は変えない。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \
        /home/jetson/miniconda/envs/jetracer/bin/python -I experiments/20261025_speed_control/replay_speed_ctrl.py

実装した純粋な関数(jetformer.safety_caps.ceil_step・jetformer.speed_auto)をそのまま使う。版は 2 つ:
  impl: 実装の既定(仕様からの 2 つの直し start_from_hold・release_bumpless あり)
  spec: 仕様の文のまま(直しなし)
1. 開ループの影(7.2): 記録の入力(v・a・舵・gz5・車に出したスロットル)を通す。記録の車は動きを変えない
2. 縦の仮想の閉ループ(7.3): 周ごとに、天井の出した前進の量で仮想の車の速さ v' を作る(惰性 c0・こすれ k_s·v²·|κ|・
   駆動 τ_up・最大 a_cap は P_cal、指令 → 速さは 10/24 の当てはめ eRPM = 26.314·f + 1343.5 に走行ごとの直線の効き b)。
   v' は記録を超えない。天井が通したフレームは記録へ戻る(駆動か記録の加速の大きい方)。むだ時間は入れていない。
   モデルの反応(減速を見て戻すなど)は入れていない
3. 出っ張り 9 件・LH 3 件(7.4)、No.9 の失う時間と 9 組(7.5)、自動合わせの開ループの影(7.6)、机の上の門(7.7)
"""
import json
import math
import os
import sys
import time

import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(H, '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'experiments', '20261024_report', 'verify'))
import rawload  # noqa: E402
from jetformer import safety_caps as sc  # noqa: E402
from jetformer import speed_auto as sa  # noqa: E402

REP = '/home/jetson/jetformer_reports/bench_20261024_report'
OUT_DIR = '/home/jetson/jetformer_reports/speed_control_20261025'
COL = os.path.join(ROOT, 'experiments', '20261024_report', 'col')
PCAL = '/home/jetson/jetformer_reports/sim_calibration_20261016/profiles/P_cal.json'
KV = sc.KV
G0 = 9.80665
lo = json.load(open(PCAL))['longitudinal']
C0, KS, TAU_UP, A_CAP = lo['c0'], lo['k_s'], lo['tau_up'], lo['a_cap']
VARIANTS = {'impl': {}, 'spec': {'start_from_hold': False, 'release_bumpless': False}}

tt = json.load(open(os.path.join(REP, 'tt_data.json')))
RUNS = {r['log']: r for r in tt['runs']}
LAPS = [l for l in tt['laps'] if not l.get('false_end')]
col = json.load(open(os.path.join(COL, 'col_data.json')))
cam = json.load(open(os.path.join(COL, 'cam_check.json')))['events']
EVENTS = []
for e in col['events']:
    k = '%s@%s' % (e['log'], e['t_s'])
    if cam.get(k) in ('SWX', 'LHC'):
        EVENTS.append(dict(kind=cam[k], log=e['log'], no=e['ai_no'], t_us=e['t_us'], g=e['g'], setting=e['setting']))


def T_lin(f_us):
    """前進の量 f [µs] → 落ち着いた直線の速さ [m/s](10/24 の 20 本の当てはめ、b = 1)"""
    return KV * (sc.CeilParams().hold_slope * max(f_us, 0.0) + sc.CeilParams().hold_offset) if f_us > 0 else 0.0


def load_run(log):
    r = rawload.load(log)
    extra = rawload._cat(log, 'AI_CMD', ['hold', 'guard_applied'])
    meta = r['meta']
    drv = meta.get('drive') or {}
    center = float(drv.get('steer_center_us') or 1441.0)
    neutral = float(drv.get('throttle_neutral_us') or 1496.0)
    trim = float((drv.get('params') or {}).get('steer_trim_us', 0.0) or 0.0)
    at = r['a_t']
    hold = extra['hold'].astype(bool) if extra['hold'] is not None else np.zeros(len(at), bool)
    guard = extra['guard_applied'].astype(bool) if extra['guard_applied'] is not None else np.zeros(len(at), bool)
    o = np.argsort(at, kind='stable')
    at, hold, guard = at[o], hold[o], guard[o]
    out, sel, dry, assist = r['a_out'][o], r['a_sel'][o], r['a_dry'][o], r['a_assist'][o]
    vt, ve = r['v_t'], r['v_erpm']
    ov = np.argsort(vt, kind='stable')
    vt, ve = vt[ov], ve[ov]

    def med3(tq):
        j = np.searchsorted(vt, tq, side='right') - 1
        jj = np.clip(np.stack([j - 2, j - 1, j]), 0, len(vt) - 1)        # 3 つ無ければ古い方で埋める(drive と同じ)
        return np.median(-ve[jj] * KV, axis=0), j
    v, j = med3(at)
    v_old, j_old = med3(at - 0.1)
    a = np.where(j_old >= 0, (v - v_old) / 0.1, np.nan)
    fresh = (j >= 0) & (at - vt[np.clip(j, 0, None)] < 0.1)
    it, gz, acc = r['i_t'], r['i_gz'], r['i_acc']
    oi = np.argsort(it, kind='stable')
    it, gz, acc = it[oi], gz[oi], acc[oi]
    k = np.searchsorted(it, at, side='right') - 1
    kk = np.clip(np.stack([k - m for m in range(5)]), 0, len(it) - 1)
    gz5 = gz[kk].mean(axis=0)
    hg = np.hypot(acc[:, 0], acc[:, 1]) / G0
    a3 = np.abs(np.linalg.norm(acc, axis=1) / G0 - 1.0)
    # フレームごとの IMU の衝撃の最大(前のフレームより後・このフレームまで)
    hg_f, a3_f = np.zeros(len(at)), np.zeros(len(at))
    prev = 0
    for i in range(len(at)):
        e_ = k[i] + 1
        if e_ > prev:
            hg_f[i] = hg[prev:e_].max()
            a3_f[i] = a3[prev:e_].max()
            prev = e_
    ai = sel & ~dry & ~hold
    steer = out[:, 0] - (center + trim)
    thr = out[:, 1].astype(float)
    kap = np.where(v > 0.3, np.deg2rad(gz5) / np.maximum(v, 0.3), 0.0)
    f_in = neutral - thr
    straight = ai & (np.abs(steer) <= 60) & (np.abs(gz5) <= 20) & (v > 2.0) & (np.abs(np.nan_to_num(a)) < 0.5) & (f_in > 60)
    b = float(np.median(v[straight] / np.array([T_lin(x) for x in f_in[straight]]))) if straight.sum() > 30 else 1.0
    return dict(log=log, t0_us=int(r['t0']), at=at, ai=ai, active=ai & fresh, v=v, a=a, gz5=gz5, steer=steer, thr=thr,
                kap=kap, neutral=neutral, assist=assist, guard=guard, hg=hg_f, a3=a3_f, b=b,
                tg=float((drv.get('params') or {}).get('throttle_gain', 1.0) or 1.0))


def ceil_params(variant, cap_all=4.5, cap_left=4.0):
    return sc.ceil_params(cap_all, cap_left, **VARIANTS[variant])


def open_loop(d, p):
    st = sc.CeilState()
    n = len(d['at'])
    thr_out, mode, left, cap = np.zeros(n), np.zeros(n, int), np.zeros(n, bool), np.full(n, np.nan)
    for i in range(n):
        st, o = sc.ceil_step(st, float(d['at'][i]), float(d['v'][i]), float(d['a'][i]), float(d['steer'][i]),
                             float(d['gz5'][i]), float(d['thr'][i]), d['neutral'], bool(d['active'][i]), p)
        thr_out[i], mode[i], left[i], cap[i] = o.thr_us, o.mode, o.left, o.cap
    return dict(thr_out=thr_out, mode=mode, left=left, cap=cap)


def virtual(d, p, i0, i1):
    """[i0, i1) のフレームの縦の仮想の閉ループ。戻り値 (失う時間, 仮想の速さ, 天井の出し方, 変えたか)"""
    st = sc.CeilState()
    at, v, thr, neu = d['at'], d['v'], d['thr'], d['neutral']
    vp = float(v[i0])
    hist = []                                  # (t, v') 0.1 秒前の速さ用
    loss = 0.0
    vv, modes, changed = [], [], []
    for i in range(i0, i1):
        t = float(at[i])
        hist.append((t, vp))
        while len(hist) > 2 and hist[1][0] <= t - 0.1:
            hist.pop(0)
        touched = vp < v[i] - 1e-6
        if touched and hist[0][0] <= t - 0.1 + 1e-9:
            a_ = (vp - hist[0][1]) / max(t - hist[0][0], 1e-3)
        else:
            a_ = float(d['a'][i])
        st, o = sc.ceil_step(st, t, vp, a_, float(d['steer'][i]), float(d['gz5'][i]), float(thr[i]), neu,
                             bool(d['active'][i]), p)
        assert o.thr_us >= thr[i] and (o.thr_us <= neu or o.thr_us == thr[i]) and np.isfinite(o.thr_us)
        vv.append(vp)
        modes.append(o.mode)
        changed.append(o.thr_us != thr[i])
        if i + 1 >= i1:
            break
        dt = float(at[i + 1] - at[i])
        if dt <= 0 or dt > 0.2:
            vp = float(v[i + 1])
            continue
        vr, vr2 = float(v[i]), float(v[i + 1])
        f_in, f_out = neu - thr[i], neu - o.thr_us
        kap = abs(float(d['kap'][i]))
        if f_out >= f_in or f_in <= 0:
            if vp >= vr - 1e-6:
                vn = vr2
            else:
                a_rec = (vr2 - vr) / dt
                a_d = max(min(A_CAP, (vr2 - vp) / TAU_UP), a_rec)
                vn = min(vr2, vp + a_d * dt)
        else:
            if f_out <= 0:
                acc = -(C0 + KS * vp * vp * kap)
            else:
                a_d = min((T_lin(f_out) * d['b'] - vp) / TAU_UP, A_CAP)
                acc = max(a_d, -C0) - KS * vp * vp * kap
            vn = min(vr2, max(0.0, vp + acc * dt))
        ds = 0.5 * (vr + vr2) * dt
        if ds > 0:
            loss += ds / max(0.5 * (vp + vn), 0.3) - dt
        vp = vn
    return loss, np.array(vv), np.array(modes), np.array(changed)


def lap_idx(d, t0, t1):
    idx = np.nonzero((d['at'] >= t0) & (d['at'] < t1))[0]
    return (int(idx[0]), int(idx[-1]) + 1) if len(idx) >= 30 else None


def main():
    t_start = time.time()
    D = {}
    for log in sorted(RUNS):
        try:
            D[log] = load_run(log)
        except Exception as e:      # noqa: BLE001
            print('skip', log, e)
    res = {'note': __doc__, 'pcal': {'c0': C0, 'k_s': KS, 'tau_up': TAU_UP, 'a_cap': A_CAP}, 'runs': [], 'laps': [],
           'events': [], 'combos': [], 'auto': [], 'gates': {}}
    OL = {}
    g_c_bad = 0
    for log, d in D.items():
        r = RUNS[log]
        mv = d['ai'] & (d['v'] > 0.5)
        if mv.sum() < 60:
            continue
        row = {'log': log, 'no': r['no'], 'session': r['session'], 'setting': r['setting'], 'ai_frames': int(mv.sum()),
               'b': round(d['b'], 3)}
        for var in VARIANTS:
            ol = open_loop(d, ceil_params(var))
            OL[(log, var)] = ol
            ch = mv & (ol['thr_out'] != d['thr'])
            neu = ch & (ol['mode'] == sc.CEIL_NEUTRAL)
            bad = ~np.isfinite(ol['thr_out']) | (ol['thr_out'] < d['thr']) | ((ol['thr_out'] > d['neutral']) & (ol['thr_out'] != d['thr']))
            g_c_bad += int(bad.sum())
            row[var] = {'changed_frac': round(float(ch.sum() / mv.sum()), 4), 'neutral_frac': round(float(neu.sum() / mv.sum()), 4),
                        'left_frac': round(float((mv & ol['left']).sum() / mv.sum()), 4),
                        'over_frac': round(float((mv & (d['v'] > ol['cap'])).sum() / mv.sum()), 4),
                        'cut_us_mean': round(float(np.mean(ol['thr_out'][ch] - d['thr'][ch])), 1) if ch.any() else 0.0,
                        'cut_us_p90': round(float(np.percentile(ol['thr_out'][ch] - d['thr'][ch], 90)), 1) if ch.any() else 0.0}
        res['runs'].append(row)
    # ---- 周ごとの仮想の閉ループ
    for lp in LAPS:
        d = D.get(lp['log'])
        if d is None:
            continue
        ij = lap_idx(d, lp['t_start'], lp['t_end'])
        if ij is None:
            continue
        row = {'log': lp['log'], 'no': lp['no'], 'session': lp['session'], 'setting': lp['setting'],
               't_start': lp['t_start'], 'lap_s': lp['lap_s'], 'kind': lp['kind'], 'p90': lp['speed_mps']['p90']}
        for var in VARIANTS:
            loss, vv, modes, changed = virtual(d, ceil_params(var), *ij)
            row[var] = {'loss_s': round(loss, 4), 'changed_frac': round(float(changed.mean()), 4),
                        'vmax_virtual': round(float(vv.max()), 2)}
        res['laps'].append(row)
    # ---- 出っ張り・LH
    for e in EVENTS:
        d = D.get(e['log'])
        if d is None:
            continue
        th = (e['t_us'] - d['t0_us']) / 1e6
        ij = lap_idx(d, th - 4.0, th - 0.03)
        if ij is None:
            continue
        ev = dict(e, t_hit=round(th, 3))
        w = (d['at'][ij[0]:ij[1]] >= th - 2.0)
        ev['vmax_2s_rec'] = round(float(d['v'][ij[0]:ij[1]][w].max()), 2)
        for var in VARIANTS:
            loss, vv, modes, changed = virtual(d, ceil_params(var), *ij)
            tt_ = d['at'][ij[0]:ij[1]]
            first = np.nonzero(changed & w)[0]
            lead = round(float(th - tt_[first[0]]), 2) if len(first) else None
            vmax = round(float(vv[w].max()), 2)
            ev[var] = {'lead_s': lead, 'vmax_2s': vmax, 'v_hit': round(float(vv[-1]), 2),
                       'effective': bool(lead is not None and lead >= 0.3 and vmax <= 4.3)}
        res['events'].append(ev)
    # ---- 9 組(No.9 と 2 回目)
    for ca in (4.3, 4.5, 4.8):
        for cl in (3.8, 4.0, 4.2):
            for key, f in (('No.9', lambda lp: lp['no'] == '9' and lp['session'] == '10/24 2回目'),
                           ('2回目', lambda lp: lp['session'] == '10/24 2回目')):
                L = []
                for lp in LAPS:
                    d = D.get(lp['log'])
                    if d is None or not f(lp):
                        continue
                    ij = lap_idx(d, lp['t_start'], lp['t_end'])
                    if ij:
                        L.append(virtual(d, ceil_params('impl', ca, cl), *ij)[0])
                if L:
                    res['combos'].append({'cap_all': ca, 'cap_left': cl, 'set': key, 'laps': len(L),
                                          'loss_mean': round(float(np.mean(L)), 4), 'loss_max': round(float(np.max(L)), 4)})
    # ---- 自動合わせの開ループの影(周の区切りは tt_data の周。偽の周は除く。電圧は記録に無いので源 0)
    for log, d in D.items():
        laps = sorted([lp for lp in LAPS if lp['log'] == log], key=lambda x: x['t_start'])
        if not laps or (log, 'impl') not in OL:
            continue
        ol = OL[(log, 'impl')]
        frames = [sa.LapFrame(float(d['at'][i]), float(d['v'][i]), bool(d['ai'][i]), int(d['assist'][i]), bool(d['guard'][i]),
                              float(d['hg'][i]), float(d['a3'][i]), bool(ol['mode'][i] >= sc.CEIL_LIMIT))
                  for i in range(len(d['at']))]
        st = sa.FbState()
        rows = []
        for lp in laps:
            ls = sa.lap_stats(frames, lp['t_start'], lp['t_end'])
            st, ev, dl = sa.fb_lap(st, ls, 1.0)
            rows.append({'t_start': lp['t_start'], 'lap_s': round(ls.dur, 2), 'v90': round(ls.v90, 3) if np.isfinite(ls.v90) else None,
                         'ok': ls.ok, 'reason': ls.reason, 'event': sa.EVENT_NAMES[ev], 'm_fb': round(st.m_fb, 4),
                         'ceil_frac': round(ls.ceil_frac, 3), 'a3_max': round(ls.a3_max, 2),
                         'one_shot': round(4.2 / ls.v90, 3) if np.isfinite(ls.v90) and ls.v90 > 0 else None})
        res['auto'].append({'log': log, 'no': RUNS[log]['no'], 'session': RUNS[log]['session'], 'tg': d['tg'],
                            'laps': rows, 'end': sa.end_summary(st, False, d['tg'])})
    # ---- 門(7.7)
    no9 = [lp for lp in res['laps'] if lp['no'] == '9' and lp['session'] == '10/24 2回目']
    g = {}
    for var in VARIANTS:
        L = [lp[var]['loss_s'] for lp in no9]
        three = [sum(L[k:k + 3]) for k in range(max(len(L) - 2, 0))]
        eff = sum(1 for e in res['events'] if e['kind'] == 'SWX' and e[var]['effective'])
        n_swx = sum(1 for e in res['events'] if e['kind'] == 'SWX')
        r9 = next(r for r in res['runs'] if r['no'] == '9' and r['session'] == '10/24 2回目')
        no9_virtual = float(np.mean([lp[var]['changed_frac'] for lp in no9])) if no9 else None
        g[var] = {'G-a': {'loss_mean': round(float(np.mean(L)), 4) if L else None, 'loss_3laps_max': round(max(three), 4) if three else None,
                          'pass': bool(L and np.mean(L) <= 0.05 and (not three or max(three) <= 0.15))},
                  'G-b': {'effective': eff, 'of': n_swx, 'pass': eff >= 7},
                  'G-d': {'open_loop_changed_frac': r9[var]['changed_frac'], 'virtual_changed_frac_mean': round(no9_virtual, 4),
                          'pass_open_loop': r9[var]['changed_frac'] <= 0.05,
                          'pass_virtual': no9_virtual is not None and no9_virtual <= 0.05}}
    g['G-c'] = {'bad_frames': g_c_bad, 'pass': g_c_bad == 0}
    res['gates'] = g
    res['elapsed_s'] = round(time.time() - t_start, 1)
    json.dump(res, open(os.path.join(H, 'replay.json'), 'w'), ensure_ascii=False, indent=1)
    write_md(res)
    print(open(os.path.join(OUT_DIR, 'replay.md'), encoding='utf-8').read())


def write_md(res):
    L = ['# 速さの天井・自動合わせの記録の再生(2026-10-25)', '',
         '- 台本: `~/jetformer/experiments/20261025_speed_control/replay_speed_ctrl.py`(数は `replay.json`)。実装した純粋な関数で再生した。',
         '- 版: **impl** = 実装の既定(仕様からの 2 つの直しあり)、**spec** = 仕様の文のまま。',
         '- 開ループ = 記録の車は動きを変えない。仮想の閉ループ = 天井の出した前進で仮想の速さを作る(P_cal。むだ時間なし・'
         'モデルの反応なし)。どちらも粗い。', '']
    L += ['## 1. 走行ごと(開ループの影。AI・0.5 m/s 超のフレームに対する割合)', '',
          '| 回 | No. | 設定 | b | 天井を超えた | 出力を変えた impl | 中立 impl | 変えた spec | 弱めた量の平均 impl [µs] |',
          '|---|---|---|---|---|---|---|---|---|']
    for r in res['runs']:
        L.append('| %s | %s | %s | %.2f | %.1f%% | %.1f%% | %.1f%% | %.1f%% | %.0f |' % (
            r['session'], r['no'], r['setting'], r['b'], 100 * r['impl']['over_frac'], 100 * r['impl']['changed_frac'],
            100 * r['impl']['neutral_frac'], 100 * r['spec']['changed_frac'], r['impl']['cut_us_mean']))
    L += ['', '## 2. 周の失う時間(仮想の閉ループ)', '', '| 回 | 周 | impl 平均 | impl 最大 | spec 平均 | spec 最大 | impl 出力を変えた |',
          '|---|---|---|---|---|---|---|']
    groups = (('No.9', lambda r: r['no'] == '9' and r['session'] == '10/24 2回目'),
              ('10/24 2 回目', lambda r: r['session'] == '10/24 2回目'), ('10/24 3 回目', lambda r: r['session'] == '10/24 3回目'),
              ('10/23 公式', lambda r: r['session'] == '10/23 公式'), ('10/24 1 回目', lambda r: r['session'] == '10/24 1回目'))
    for name, f in groups:
        rows = [r for r in res['laps'] if f(r)]
        if rows:
            a = [r['impl']['loss_s'] for r in rows]
            b = [r['spec']['loss_s'] for r in rows]
            c = [r['impl']['changed_frac'] for r in rows]
            L.append('| %s | %d | %.3f 秒 | %.3f | %.3f | %.3f | %.1f%% |' % (name, len(rows), np.mean(a), np.max(a), np.mean(b),
                                                                             np.max(b), 100 * np.mean(c)))
    L += ['', '## 3. 出っ張り(SWX)9 件・LH(LHC)3 件(当たる前 2 秒、仮想の閉ループ)', '',
          '| 種類 | No. | 時刻 | 記録の最高 | impl 掛かる(何秒前) | impl 最高 | spec 掛かる | spec 最高 | 効く impl/spec |',
          '|---|---|---|---|---|---|---|---|---|']
    for e in res['events']:
        L.append('| %s | %s | %.1f | %.2f | %s | %.2f | %s | %.2f | %s/%s |' % (
            e['kind'], e['no'], e['t_hit'], e['vmax_2s_rec'], e['impl']['lead_s'], e['impl']['vmax_2s'], e['spec']['lead_s'],
            e['spec']['vmax_2s'], '○' if e['impl']['effective'] else '×', '○' if e['spec']['effective'] else '×'))
    L += ['', '- 効く = 当たる 0.3 秒以上前に掛かり、当たる前 2 秒の仮想の最高の速さが 4.3 m/s 以下(spec 7.4)。', '',
          '## 4. 9 組(impl・仮想の閉ループの失う時間 秒/周)', '', '| 全体 \\ 左 | 3.8 | 4.0 | 4.2 |', '|---|---|---|---|']
    for key in ('No.9', '2回目'):
        for ca in (4.3, 4.5, 4.8):
            cells = []
            for cl in (3.8, 4.0, 4.2):
                c = next((x for x in res['combos'] if x['set'] == key and x['cap_all'] == ca and x['cap_left'] == cl), None)
                cells.append('—' if c is None else '%.3f' % c['loss_mean'])
            L.append('| %s %.1f | %s |' % (key, ca, ' | '.join(cells)))
    L += ['', '## 5. 自動合わせ(開ループの影。電圧なし = 源 0。周は tt_data の周)', '',
          '| 回 | No. | ゲイン | 使った周 | m_fb の終わり | 次の throttle_gain の目安(使う/使わない) | 知らせ |', '|---|---|---|---|---|---|---|']
    for a in res['auto']:
        e = a['end']
        L.append('| %s | %s | %.2f | %d | %.3f | %s / %s | %s |' % (a['session'], a['no'], a['tg'], e['laps_used'], e['m_fb_end'],
                                                                 e['suggest_throttle_gain_auto'], e['suggest_throttle_gain_fixed'],
                                                                 ','.join(e['warn_seen']) or '—'))
    L += ['', '## 6. 机の上の門(spec 7.7)', '', '| 門 | impl | spec |', '|---|---|---|']
    g = res['gates']
    L.append('| G-a No.9 の失う時間 平均 ≤ 0.05・3 周 ≤ 0.15 | %s(%.3f・%.3f) | %s(%.3f・%.3f) |' % (
        '合格' if g['impl']['G-a']['pass'] else '不合格', g['impl']['G-a']['loss_mean'], g['impl']['G-a']['loss_3laps_max'],
        '合格' if g['spec']['G-a']['pass'] else '不合格', g['spec']['G-a']['loss_mean'], g['spec']['G-a']['loss_3laps_max']))
    L.append('| G-b 出っ張りの 9 件で 7 件以上「効く」 | %s(%d/%d) | %s(%d/%d) |' % (
        '合格' if g['impl']['G-b']['pass'] else '不合格', g['impl']['G-b']['effective'], g['impl']['G-b']['of'],
        '合格' if g['spec']['G-b']['pass'] else '不合格', g['spec']['G-b']['effective'], g['spec']['G-b']['of']))
    L.append('| G-c 全フレーム 出力 ≥ 入力・中立以下・NaN なし | %s(%d) | (同じ数に含む) |' % (
        '合格' if g['G-c']['pass'] else '不合格', g['G-c']['bad_frames']))
    L.append('| G-d No.9 で出力を変えるフレーム ≤ 5%%(開ループ / 仮想の閉ループ) | %.1f%% / %.1f%% | %.1f%% / %.1f%% |' % (
        100 * g['impl']['G-d']['open_loop_changed_frac'], 100 * g['impl']['G-d']['virtual_changed_frac_mean'],
        100 * g['spec']['G-d']['open_loop_changed_frac'], 100 * g['spec']['G-d']['virtual_changed_frac_mean']))
    L += ['', '- G-e(自動合わせの試験)・G-f(時間)は pytest(build_drive.md)。G-e の位相の起点の一致は段 tools。G-g は 11/2 の dry run。',
          '- 再生にかかった時間 %.0f 秒。' % res['elapsed_s'], '']
    with open(os.path.join(OUT_DIR, 'replay.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))


if __name__ == '__main__':
    main()
