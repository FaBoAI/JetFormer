#!/usr/bin/env python3
"""反証 V-replay: 速さの天井の効き目を記録の再生で数える(2026-10-25)。読むだけ。

    cd ~/jetformer/experiments/20261025_speed_control/replay
    /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I rr_cache.py
    /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I rr_ceiling.py

1. 開ループの影: 10/23・10/24 の全部の AI 走行(39 本)の記録の入力を、実装の ceil_step に通す。走行ごと・区間ごと
2. 仮想の閉ループ(段 drive とは別の作り): 記録の道のり(eRPM の積分)の刻みごとに、仮想の車の時間と速さを進める。
   - 舵・道すじ・モデルの指令は記録のまま(道のりの関数)。ヨーは道の曲がりが同じとして gz × v'/v
   - 前後は「記録との差」だけを 1 次の遅れで: a' = a_rec + [k·b·(f_eff − f_rec) + (v_rec − v')] / τ_up
     (k = 10/24 の当てはめ 26.314 eRPM/µs × 0.5e-3、b = 走行ごとの直線の効き、τ_up 0.318・上限 4.12 m/s² は P_cal)。
     天井で中立なら惰性 c0 0.576 + こすれ 0.404·v²·|κ|。前進を弱めても惰性より強くは減速しない
   - むだ時間 dead: 天井の出した量が効き始めるまでの遅れ(0 = 段 drive と同じ・0.05 秒 = 撮像 → 出力 30 ms + ESC)
   - モデルの反応(感度だけ): モデルの指令を f_rec − kv·(v_rec − v') − ka·(a_rec − a') にする(解析 B 6 章: モデル B は
     自分の速さ・前後の加速度で踏む。kv 25・50 µs/(m/s)、ka 15 µs/(m/s²) は入れ替えの試しからの粗い見積もり)
   - ゲイン倍 gm: モデルの前進の指令 × gm(自動合わせの効き目の見込み)。このときだけ v' > v_rec を許す
3. 出っ張り 9 件・LH 3 件・DE の衝突・10/23 速さ上げの衝突の前で、速さがどれだけ下がるか
4. SW の通過 98 回(10/24、col_data)の最高の速さ → 当たる率の帯(analysis_collisions 4 章)で、当たる数の見込み
5. No.9 の失う時間(周・3 周・区間・原因ごと)と、9 組・むだ時間・モデルの反応・ゲイン倍の感度
"""
import collections
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rr_cache  # noqa: E402
import rr_common as C  # noqa: E402
from rr_common import sc  # noqa: E402

PCAL = {'c0': 0.5762225785807149, 'k_s': 0.4044908485500322, 'tau_up': 0.31845547836796556, 'a_cap': 4.12321166697387}
SLOPE = 26.314 * C.KV              # 指令 1 µs あたりの落ち着いた速さ [m/s](b = 1)
OFFSET = 1343.5 * C.KV
SECS = ['R1', 'GM_in', 'GM_out', 'DE', 'SW', 'HS', 'LH', '?']
NO9 = '20261024-162033'
# SW の中の最高の速さの帯 → 出っ張りに当たった / 通過(analysis_collisions 4 章)
SW_BINS = [(0.0, 3.5, 0, 47), (3.5, 4.0, 2, 34), (4.0, 4.5, 2, 11), (4.5, 99.0, 4, 6)]


def sw_rate(v):
    for lo, hi, h, n in SW_BINS:
        if lo <= v < hi:
            return h / n
    return float('nan')


def run_b(d):
    """走行ごとの直線の効き b = 記録の速さ ÷ 当てはめの速さ(落ち着いた直線のフレームの中央値)"""
    f = d['neutral'] - d['thr']
    a = np.nan_to_num(d['a'])
    m = d['ai'] & (np.abs(d['steer']) <= 60) & (np.abs(d['gz5']) <= 20) & (d['v'] > 2.0) & (np.abs(a) < 0.5) & (f > 60)
    if m.sum() < 30:
        return 1.0
    return float(np.median(d['v'][m] / (SLOPE * f[m] + OFFSET)))


def smooth_v(d):
    """記録の速さを 5 フレームの中心の平均でならす(仮想の車の動きはこれに合わせる。記録の跳ねで仮想の車が
    下がっていく(上は駆動の上限で切られ、下は切られない)のを防ぐ)。跳ね(記録 − ならし)は天井の入力に足し戻す"""
    if 'vs' not in d:
        v = d['v']
        ok = np.isfinite(v)
        num = np.convolve(np.where(ok, v, 0.0), np.ones(5), 'same')
        den = np.convolve(ok.astype(float), np.ones(5), 'same')
        d['vs'] = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)
    return d['vs']


def virtual(d, p, i0, i1, b, dead=0.05, kv=0.0, ka=0.0, gm=1.0, ceil=True, keep=False):
    """[i0, i1) の仮想の閉ループ(1 本の続いた AI の区間)。戻り値 dict(失う時間・フレームごとの値)"""
    t, thr, neu = d['t'], d['thr'], d['neutral']
    vs = smooth_v(d)
    noise = np.nan_to_num(d['v'] - vs)
    ar = np.nan_to_num(d['a'])
    kap = np.abs(d['kap'])
    st = sc.CeilState()
    vp = float(vs[i0])
    tv = float(t[i0])
    hist = collections.deque()          # (tv, 測った速さ) 0.1 秒前の速さ
    cmd = collections.deque()           # (tv, f_rec, f_eff) むだ時間のため
    n = i1 - i0
    out = {'loss': np.zeros(n), 'vp': np.zeros(n), 'vm': np.zeros(n), 'mode': np.full(n, -1, int),
           'changed': np.zeros(n, bool), 'tv': np.zeros(n), 'f_out': np.zeros(n), 'f_in': np.zeros(n)}
    bad = 0
    for k, i in enumerate(range(i0, i1)):
        vm = vp + float(noise[i])
        out['vm'][k] = vm        # 天井が測る速さ(仮想の速さ + 記録の跳ね)
        hist.append((tv, vm))
        while len(hist) > 2 and hist[1][0] <= tv - 0.1:
            hist.popleft()
        if hist[0][0] <= tv - 0.1 + 1e-9:
            a_v = (vm - hist[0][1]) / max(tv - hist[0][0], 1e-3)
        else:
            a_v = float(ar[i])
        f_rec = neu - float(thr[i])
        f_in = f_rec
        if f_rec > 0:
            f_in = f_rec * gm - kv * (float(vs[i]) - vp) - ka * (float(ar[i]) - a_v)
            f_in = max(f_in, 0.0)
        thr_in = neu - f_in
        gz_v = float(d['gz5'][i]) * (vp / float(vs[i]) if vs[i] > 0.3 else 1.0)
        if ceil:
            st, o = sc.ceil_step(st, tv, vm, a_v, float(d['steer'][i]), gz_v, thr_in, neu, bool(d['active'][i]), p)
            thr_out = o.thr_us
            mode = o.mode
            if not (thr_out >= thr_in - 1e-9 and (thr_out <= neu + 1e-9 or thr_out == thr_in) and math.isfinite(thr_out)):
                bad += 1
        else:
            thr_out, mode = thr_in, -1
        f_out = neu - thr_out
        out['vp'][k], out['mode'][k], out['tv'][k] = vp, mode, tv
        out['changed'][k] = thr_out != thr_in
        out['f_out'][k], out['f_in'][k] = f_out, f_in
        cmd.append((tv, f_rec, f_out))
        if k == n - 1:
            break
        dt = float(t[i + 1] - t[i])
        if dt <= 0 or dt > 0.2 or not d['active'][i] or not np.isfinite(vs[i + 1]):
            vp = float(np.nan_to_num(vs[i + 1]))
            tv += dt
            st = sc.CeilState()
            cmd.clear()
            hist.clear()
            continue
        # むだ時間: tv − dead の時点に出ていた量
        while len(cmd) > 1 and cmd[1][0] <= tv - dead + 1e-9:
            cmd.popleft()
        _, f_rec_d, f_eff_d = cmd[0] if cmd[0][0] <= tv - dead + 1e-9 else (tv, f_rec, f_rec * gm if f_rec > 0 else f_rec)
        v1, v2 = float(vs[i]), float(vs[i + 1])
        ds = 0.5 * (v1 + v2) * dt
        if gm <= 1.0 + 1e-12 and f_eff_d >= f_rec_d - 1e-9 and vp >= v1 - 1e-6:
            out['loss'][k] = 0.0             # 天井が効いていない・記録に追いついている → 記録のまま
            tv += dt
            vp = v2
            continue
        a_seg = (v2 - v1) / dt
        coast = -(PCAL['c0'] + PCAL['k_s'] * vp * vp * float(kap[i]))
        if f_rec_d > 0 and f_eff_d <= 0:
            acc = coast
        elif f_rec_d > 0:
            acc = a_seg + (SLOPE * b * (f_eff_d - f_rec_d) + (v1 - vp)) / PCAL['tau_up']
            if f_eff_d < f_rec_d:
                acc = max(acc, coast)
        else:
            acc = a_seg
        acc = min(acc, max(PCAL['a_cap'], a_seg))
        dtv = ds / max(vp, 0.3)
        for _ in range(4):
            vn = vp + acc * dtv
            dtv = ds / max(0.5 * (vp + max(vn, 0.0)), 0.3)
        vn = max(vp + acc * dtv, 0.0)
        if gm <= 1.0 + 1e-12:                      # ゲインを上げないときは記録より速くならない
            vn = min(vn, v2)
        if ds <= 1e-9:
            dtv = dt
        out['loss'][k] = dtv - dt
        tv += dtv
        vp = vn
    out['bad'] = bad
    if not keep:
        for key in ('f_out', 'f_in'):
            out.pop(key)
    return out


def lap_chains(d):
    """続いた周の鎖(前の周の終わり = 次の周の始まり)。[(t_start, [laps…])]"""
    laps = sorted(d['laps'], key=lambda x: x['t_start'])
    chains = []
    for lp in laps:
        lp = dict(lp, t_end=round(lp['t_start'] + lp['lap_s'], 3))
        if chains and abs(chains[-1][-1]['t_end'] - lp['t_start']) < 0.02:
            chains[-1].append(lp)
        else:
            chains.append([lp])
    return chains


def idx(d, t0, t1):
    ii = np.nonzero((d['t'] >= t0) & (d['t'] < t1))[0]
    return (int(ii[0]), int(ii[-1]) + 1) if len(ii) else None


def lap_losses(d, p, b, **kw):
    """周ごとの失う時間(鎖ごとに続けて回す)。戻り値 [{t_start, lap_s, loss, changed_frac, sec_loss, cause_loss}]"""
    rows = []
    for ch in lap_chains(d):
        ij = idx(d, ch[0]['t_start'], ch[-1]['t_end'])
        if ij is None:
            continue
        o = virtual(d, p, ij[0], ij[1], b, **kw)
        tt = d['t'][ij[0]:ij[1]]
        locd = d['locd'][ij[0]:ij[1]]
        for lp in ch:
            m = (tt >= lp['t_start']) & (tt < lp['t_end'])
            sec = {s: float(o['loss'][m & (locd == j)].sum()) for j, s in enumerate(C.LOC_DETAIL)}
            # 原因: 天井が出力を変えたフレーム(弱めた・中立)と、そのあとの戻り
            ch_ = o['changed'] & m
            neu_ = (o['mode'] == sc.CEIL_NEUTRAL) & m
            cause = {'limit': float(o['loss'][ch_ & ~neu_].sum()), 'neutral': float(o['loss'][neu_].sum()),
                     'recover': float(o['loss'][m & ~o['changed']].sum())}
            rows.append({'t_start': lp['t_start'], 'lap_s': lp['lap_s'], 'clean': lp.get('clean'),
                         'loss': float(o['loss'][m].sum()), 'changed_frac': float(ch_.sum() / max(m.sum(), 1)),
                         'neutral_frac': float(neu_.sum() / max(m.sum(), 1)), 'vmax_v': float(o['vp'][m].max()),
                         'v90_rec': float(np.percentile(d['v'][ij[0]:ij[1]][m], 90)),
                         'v90_v': float(np.percentile(o['vm'][m], 90)), 'sec_loss': sec, 'cause_loss': cause,
                         'bad': o['bad']})
    return rows


def windows3(rows):
    out = []
    for k in range(len(rows) - 2):
        a, b_, c = rows[k:k + 3]
        if abs(a['t_start'] + a['lap_s'] - b_['t_start']) < 0.02 and abs(b_['t_start'] + b_['lap_s'] - c['t_start']) < 0.02:
            out.append({'t_start': a['t_start'], 'rec_s': a['lap_s'] + b_['lap_s'] + c['lap_s'],
                        'loss': a['loss'] + b_['loss'] + c['loss']})
    return out


def open_loop_tables(D):
    runs, secs, sess = [], [], collections.defaultdict(lambda: collections.Counter())
    gc_bad = 0
    OL = {}
    for log, d in D.items():
        mv = d['ai'] & (d['v'] > 0.5)
        row = {'log': log, 'no': d['no'], 'session': d['session'], 'tg': round(d['tg'], 2), 'ai_frames': int(mv.sum()),
               'vmax': round(float(d['v'][mv].max()), 2), 'v90': round(float(np.percentile(d['v'][mv], 90)), 2)}
        for var in ('impl', 'spec'):
            ol = C.open_loop(d, C.P(var))
            OL[(log, var)] = ol
            ch = mv & (ol['thr_out'] != d['thr'])
            neu = ch & (ol['mode'] == sc.CEIL_NEUTRAL)
            over = mv & (d['v'] > ol['cap'])
            bad = (~np.isfinite(ol['thr_out']) | (ol['thr_out'] < d['thr'])
                   | ((ol['thr_out'] > d['neutral']) & (ol['thr_out'] != d['thr'])))
            gc_bad += int(bad.sum())
            row[var] = {'over': float(over.sum() / mv.sum()), 'changed': float(ch.sum() / mv.sum()),
                        'neutral': float(neu.sum() / mv.sum()), 'left': float((mv & ol['left']).sum() / mv.sum()),
                        'cut_us': float(np.mean(ol['thr_out'][ch] - d['thr'][ch])) if ch.any() else 0.0}
            if var == 'impl':
                for j, s in enumerate(C.LOC_DETAIL):
                    ms = mv & (d['locd'] == j)
                    if ms.sum() == 0:
                        continue
                    secs.append({'log': log, 'no': d['no'], 'session': d['session'], 'sec': s, 'frames': int(ms.sum()),
                                 'over': int((ms & over).sum()), 'changed': int((ms & ch).sum()),
                                 'neutral': int((ms & neu).sum()), 'left': int((ms & ol['left']).sum())})
                    c = sess[(d['session'], s)]
                    c['frames'] += int(ms.sum())
                    c['over'] += int((ms & over).sum())
                    c['changed'] += int((ms & ch).sum())
                    c['neutral'] += int((ms & neu).sum())
                    c['left'] += int((ms & ol['left']).sum())
        runs.append(row)
    sess_rows = [dict(session=k[0], sec=k[1], **v) for k, v in sorted(sess.items())]
    return runs, secs, sess_rows, gc_bad, OL


def event_list(D):
    """出っ張り・LH(10/24 カメラで分けたもの)・DE の衝突(全部)・10/23 速さ上げの衝突(SW・HS・LH)"""
    ev = []
    for key, kind in C.CAM.items():
        if kind not in ('SWX', 'LHC'):
            continue
        log, ts = key.split('@')
        if log in D:
            ev.append({'kind': kind, 'log': log, 't_hit': float(ts)})
    for log, d in D.items():
        for c in d['coll']:
            if c.get('class') != 'main':
                continue
            if c['loc'] == 'DE':
                ev.append({'kind': 'DE', 'log': log, 't_hit': c['t_s'], 'g': c['g'], 'strong': c['strong']})
            elif d['session'] == '10/23 速さ上げ' and c['loc'] in ('other', 'HS', 'LH'):
                ev.append({'kind': '23HI-' + (c.get('loc_detail') or c['loc']), 'log': log, 't_hit': c['t_s'], 'g': c['g'],
                           'strong': c['strong']})
    return ev


def event_effect(D, OL, e, B, p, dead=0.05):
    d = D[e['log']]
    th = e['t_hit']
    ij = idx(d, th - 4.0, th - 0.03)
    if ij is None or ij[1] - ij[0] < 20:
        return None
    tt = d['t'][ij[0]:ij[1]]
    w = tt >= th - 2.0
    act = d['active'][ij[0]:ij[1]]
    vrec = d['v'][ij[0]:ij[1]]
    ol = OL[(e['log'], 'impl')]
    chg = (ol['thr_out'][ij[0]:ij[1]] != d['thr'][ij[0]:ij[1]]) & w
    first_ol = np.nonzero(chg)[0]
    o = virtual(d, p, ij[0], ij[1], B[e['log']], dead=dead)
    ch_v = o['changed'] & w
    first_v = np.nonzero(ch_v)[0]
    r = dict(e, no=d['no'], session=d['session'], tg=round(d['tg'], 2), ai_2s=bool(act[w].all()),
             vmax_rec=float(vrec[w].max()), v_hit_rec=float(vrec[-1]),
             lead_ol=float(th - tt[first_ol[0]]) if len(first_ol) else None,
             lead_v=float(th - tt[first_v[0]]) if len(first_v) else None,
             vmax_v=float(o['vm'][w].max()), v_hit_v=float(o['vm'][-1]), vmax_v_smooth=float(o['vp'][w].max()),
             vmax_rec_smooth=float(smooth_v(d)[ij[0]:ij[1]][w].max()),
             cap_at_hit=float(ol['cap'][ij[1] - 1]) if np.isfinite(ol['cap'][ij[1] - 1]) else None,
             left_at_hit=bool(ol['left'][ij[1] - 1]), steer_hit=float(d['steer'][ij[1] - 1]),
             gz_hit=float(d['gz5'][ij[1] - 1]))
    r['drop'] = r['vmax_rec'] - r['vmax_v']
    r['effective'] = bool(r['lead_v'] is not None and r['lead_v'] >= 0.3 and r['vmax_v'] <= 4.3)
    r['effective_real'] = bool(r['effective'] and r['drop'] >= 0.10)
    return r


def sw_pass_expectation(D, B, p, dead=0.05, gm=1.0):
    """10/24 の SW の通過(col_data.sw_passes)ごとに、SW の中の最高の速さ 記録 → 仮想(天井)。帯の率で当たる数の見込み"""
    rows = []
    for sp in C.COL['sw_passes']:
        d = D.get(sp['log'])
        if d is None:
            continue
        fa, fe = int(sp['frame_a']), int(sp['frame_exit'])
        i0 = int(np.searchsorted(d['t'], d['t'][fa] - 3.0))
        i1 = min(fe + 1, len(d['t']))
        if not d['active'][fa:i1].all():
            continue
        o = virtual(d, p, i0, i1, B[sp['log']], dead=dead, gm=gm)
        o0 = virtual(d, p, i0, i1, B[sp['log']], dead=dead, gm=gm, ceil=False) if gm != 1.0 else None
        sl = slice(fa - i0, i1 - i0)
        vmax_rec = float(d['v'][fa:i1].max())
        vmax_v = float(o['vm'][sl].max())
        rows.append({'log': sp['log'], 'no': d['no'], 'session': d['session'], 'hit': bool(sp['hit']),
                     'vmax_col': sp['v_max_mps'], 'vmax_rec': vmax_rec, 'vmax_v': vmax_v,
                     'vmax_nocap': float(o0['vm'][sl].max()) if o0 is not None else vmax_rec})
    return rows


def main():
    t0 = time.time()
    cache = rr_cache.load()
    D = cache['runs']
    B = {log: run_b(d) for log, d in D.items()}
    res = {'note': __doc__, 'pcal': PCAL, 'skipped': cache['skipped'], 'b': B}
    runs, secs, sess_rows, gc_bad, OL = open_loop_tables(D)
    res['open_loop'] = {'runs': runs, 'secs': secs, 'sessions': sess_rows, 'gc_bad': gc_bad}
    print('open loop done %.1f s' % (time.time() - t0))
    p_impl = C.P('impl')
    # ---- 出来事
    evs = []
    for e in event_list(D):
        r = event_effect(D, OL, e, B, p_impl, dead=0.05)
        if r is not None:
            r0 = event_effect(D, OL, e, B, p_impl, dead=0.0)
            r['vmax_v_dead0'] = r0['vmax_v'] if r0 else None
            evs.append(r)
    res['events'] = evs
    print('events done %.1f s' % (time.time() - t0))
    # ---- SW の通過の見込み
    res['sw_passes'] = {'dead0.05': sw_pass_expectation(D, B, p_impl, 0.05), 'dead0': sw_pass_expectation(D, B, p_impl, 0.0)}
    print('sw done %.1f s' % (time.time() - t0))
    # ---- 周の失う時間(全部の走行・既定 = むだ時間 0.05・モデルの反応なし)
    laps = {}
    for log, d in D.items():
        rows = lap_losses(d, p_impl, B[log], dead=0.05)
        if rows:
            laps[log] = rows
    res['laps'] = laps
    print('laps done %.1f s' % (time.time() - t0))
    # ---- No.9 の感度
    d9 = D[NO9]
    sens = []
    for name, var, kw, pk in [
            ('impl・むだ時間 0(段 drive と同じ前提)', 'impl', dict(dead=0.0), {}),
            ('impl・むだ時間 0.05', 'impl', dict(dead=0.05), {}),
            ('impl・むだ時間 0.10', 'impl', dict(dead=0.10), {}),
            ('spec・むだ時間 0.05', 'spec', dict(dead=0.05), {}),
            ('impl・反応 kv 25', 'impl', dict(dead=0.05, kv=25.0), {}),
            ('impl・反応 kv 50', 'impl', dict(dead=0.05, kv=50.0), {}),
            ('impl・反応 kv 50 + ka 15', 'impl', dict(dead=0.05, kv=50.0, ka=15.0), {}),
            ('掛け始め 0・先読みなし', 'impl', dict(dead=0.05), dict(engage_e=0.0, pred_s=0.0)),
            ('左 4.2', 'impl', dict(dead=0.05), dict(cap_left=4.2)),
    ]:
        pk = dict(pk)
        cl = pk.pop('cap_left', 4.0)
        p = C.P(var, 4.5, cl, **pk)
        rows = lap_losses(d9, p, B[NO9], **kw)
        w3 = windows3(rows)
        sens.append({'name': name, 'laps': len(rows), 'loss_mean': float(np.mean([r['loss'] for r in rows])),
                     'loss_max': float(np.max([r['loss'] for r in rows])),
                     'changed_frac': float(np.mean([r['changed_frac'] for r in rows])),
                     'w3_max': float(max(w['loss'] for w in w3)) if w3 else None,
                     'w3_best_rec': float(min(w['rec_s'] for w in w3)) if w3 else None,
                     'w3_best_loss': float(min(w3, key=lambda w: w['rec_s'])['loss']) if w3 else None})
    res['no9_sens'] = sens
    # 9 組(No.9・むだ時間 0.05)
    combos = []
    for ca in (4.3, 4.5, 4.8):
        for cl in (3.8, 4.0, 4.2):
            rows = lap_losses(d9, C.P('impl', ca, cl), B[NO9], dead=0.05)
            combos.append({'cap_all': ca, 'cap_left': cl, 'loss_mean': float(np.mean([r['loss'] for r in rows])),
                           'changed_frac': float(np.mean([r['changed_frac'] for r in rows]))})
    res['no9_combos'] = combos
    print('no9 sens done %.1f s' % (time.time() - t0))
    # ---- ゲイン倍(自動合わせが上げたら): No.9 ×1.07(目標 4.20 の 1 回合わせ)・3 回目の B 0.90 ×1.10(上限)
    gains = []
    for log, gm_list in [(NO9, (1.0, 1.03, 1.07, 1.10)), ('20261024-183123', (1.0, 1.10)), ('20261024-183437', (1.0, 1.10)),
                         ('20261024-184549', (1.0, 1.10))]:
        d = D[log]
        for gm in gm_list:
            for ceil in (False, True):
                rows = []
                for ch in lap_chains(d):
                    ij = idx(d, ch[0]['t_start'], ch[-1]['t_end'])
                    if ij is None:
                        continue
                    o = virtual(d, p_impl, ij[0], ij[1], B[log], dead=0.05, gm=gm, ceil=ceil)
                    tt = d['t'][ij[0]:ij[1]]
                    locd = d['locd'][ij[0]:ij[1]]
                    for lp in ch:
                        m = (tt >= lp['t_start']) & (tt < lp['t_end'])
                        sw = m & (locd == C.LOC_DETAIL.index('SW'))
                        de = m & (locd == C.LOC_DETAIL.index('DE'))
                        r1 = m & (locd == C.LOC_DETAIL.index('R1'))
                        rows.append({'lap_s': lp['lap_s'], 'loss': float(o['loss'][m].sum()),
                                     'v90': float(np.percentile(o['vm'][m], 90)),
                                     'sw_vmax': float(o['vm'][sw].max()) if sw.any() else None,
                                     'de_vmax': float(o['vm'][de].max()) if de.any() else None,
                                     'r1_entry': float(o['vm'][r1][:12].max()) if r1.any() else None,
                                     'changed_frac': float((o['changed'] & m).sum() / m.sum())})
                gains.append({'log': log, 'no': d['no'], 'session': d['session'], 'tg': d['tg'], 'gm': gm, 'ceil': ceil,
                              'laps': len(rows), 'lap_mean_rec': float(np.mean([r['lap_s'] for r in rows])),
                              'lap_mean_v': float(np.mean([r['lap_s'] + r['loss'] for r in rows])),
                              'v90_mean': float(np.mean([r['v90'] for r in rows])),
                              'sw_vmax_med': float(np.median([r['sw_vmax'] for r in rows if r['sw_vmax'] is not None])),
                              'sw_rate': float(np.mean([sw_rate(r['sw_vmax']) for r in rows if r['sw_vmax'] is not None])),
                              'de_vmax_med': float(np.median([r['de_vmax'] for r in rows if r['de_vmax'] is not None])),
                              'changed_frac': float(np.mean([r['changed_frac'] for r in rows]))})
    res['gain_up'] = gains
    # SW の通過で、No.9 のゲイン ×1.07 の見込み
    res['sw_passes_gm107'] = [r for r in sw_pass_expectation(D, B, p_impl, 0.05, gm=1.07) if r['log'] == NO9]
    res['elapsed_s'] = round(time.time() - t0, 1)
    C.jdump(res, os.path.join(C.HERE, 'rr_ceiling.json'))
    print('done %.1f s' % (time.time() - t0))


if __name__ == '__main__':
    main()
