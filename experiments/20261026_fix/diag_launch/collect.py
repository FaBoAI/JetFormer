"""原因 C の診断(1): 止まってから AI で走り出した場面を全部集める(10/23・10/24 の AI 走行 50 本)。2026-10-26

読むだけ。data/raw・data/labels は変えない。

発進 = AI の区間(labels の ai が続くところ)の最初のフレームで |eRPM| < 400(0.2 m/s 未満)。
  補助(assist = 4)が出た発進・出なかった発進の両方。
最初の R1 の起点 = label_events.checked_anchors(偽の起点を外したもの。label_log と同じ作り方)。
走りながらの同じ区間 = 同じ走行の「続いた周」(起点 → 次の起点がずっと AI・5〜16 秒・向きの変化 300〜420°)の、
  起点の d m 手前 → 起点(d = 発進から最初の起点までの車輪の距離)。同じ走行に周が無ければ、同じモデル・同じ速さのゲイン・
  同じ回の周を使う(ref = 'same_setting')。

出すもの(out/launches.json):
  発進ごと: 起点までの時間・失う時間・動き出し(0.2 / 0.4 m/s)・補助の長さ・モデルが出した前進・舵・加速(車輪と IMU)・
  空転の目安(車輪の速さの増え ÷ IMU の加速の積分)・距離の格子(0.1 m)の上の発進と周の比べ(時間・速さ・指令・舵)。
  周の区間(参考): 同じ走行の周の、同じ場所の値。

    /home/jetson/bin/jf-heavy --lock none --mem 1500M -- python3 -I collect.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '/home/jetson/jetformer/tools')
sys.path.insert(0, '/home/jetson/jetformer')
import label_events as le  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / 'out'
RAW = Path('/home/jetson/jetformer/data/raw')
LAB = Path('/home/jetson/jetformer/data/labels')
KV = 0.5e-3
STOP_ERPM = 400.0
MOVE1_ERPM, MOVE2_ERPM = 400.0, 800.0     # 動き出し: 0.2 m/s・0.4 m/s(StartKick.MOVE_ERPM と同じ 800)
DX = 0.1                                  # 距離の格子 [m]
MAX_TO_R1_S = 15.0
MODEL = {
    'clean_driving_20261011-nocmd-s3-20261011-024626': 'B',
    'bench_20261022-slow-sec-s1': 'SLOW-SEC1',
    'bench_20261021-ens-p2': 'ENS',
    'bench_20261021-aug-s0': 'AUG',
    'clean_driving_20261011-act-s0-nocmd-stride3-e30': 'ACT',
    'bench_20261021-wgt-s0': 'WGT',
}


def logs():
    return sorted(p.name for p in RAW.iterdir() if p.is_dir() and p.name[:8] in ('20261023', '20261024')
                  and (LAB / (p.name + '.npz')).exists())


def session_of(log, no):
    if log.startswith('20261023'):
        if no is None:
            return '10/23 前'
        return '10/23 前' if no <= 7 else ('10/23 公式' if no <= 15 else '10/23 後')
    if no is None:
        return '10/24 ?'
    return '10/24 1回目' if no <= 4 else ('10/24 2回目' if no <= 10 else '10/24 3回目')


def ai_no(desc):
    for x in (desc or '').split():
        if x.startswith('ai_no='):
            try:
                return int(x[6:])
            except ValueError:
                return None
    return None


def first_time(t, mask):
    i = np.nonzero(mask)[0]
    return float(t[i[0]]) if len(i) else None


def smooth(x, n):
    if len(x) < n:
        return x
    k = np.ones(n) / n
    return np.convolve(x, k, 'same')


def load(log):
    lab = json.loads((LAB / (log + '.json')).read_text())
    z = dict(np.load(LAB / (log + '.npz')))
    L = le.load_log(log)
    T0 = int(lab['t0_us'])
    V, I = L['VEHICLE'], L['IMU']
    vt = (V['timestamps_us'].astype(np.int64) - T0) / 1e6
    verpm = V['erpm'].astype(np.float64)
    it = (I['timestamps_us'].astype(np.int64) - T0) / 1e6
    ie = np.interp(it, vt, verpm)
    gz = I['gyro'][:, 2].astype(np.float64)
    still = (np.abs(ie) < 50) & (np.abs(gz) < 5)
    hg, bias = le.horizontal_g(I['accel'], still)
    hd, gzu = le.heading_deg(it, gz, still)
    impacts = le.detect_collisions(it, hg)
    hits = [(round(float(it[e['i']]), 3), e['g']) for e in impacts if not e['suspect']]
    anchors, gm, gzs, rej = le.checked_anchors(it, hd, gzu, vt, verpm, z['t_s'], z['ai'].astype(bool), hits)
    D, Df = le.cumulative_distance(vt, verpm)
    ay = I['accel'][:, 1].astype(np.float64) - bias[1]      # 前 + [m/s²](止まっているときのずれを引く)
    return {'log': log, 'lab': lab, 'z': z, 'meta': L['meta'], 'vt': vt, 'erpm': verpm, 'v': -verpm * KV, 'Df': Df,
            'it': it, 'gz': gzu, 'hd': hd, 'ay': ay, 'hg': hg, 'hits': hits, 'anchors': np.asarray(anchors, float),
            'ref': (float(lab['ref_us']['steer_center']), float(lab['ref_us']['throttle_neutral']))}


def ai_continuous(z, a, b):
    t, ai = z['t_s'], z['ai'].astype(bool)
    m = (t >= a - 0.05) & (t <= b + 0.05)
    if not m.any():
        return False
    tt = t[m]
    return bool(ai[m].all() and (np.diff(tt).max(initial=0) <= 0.2) and tt[0] <= a + 0.1 and tt[-1] >= b - 0.1)


def laps_of(d):
    """続いた周: 起点 a → 次の起点 b(ずっと AI・5〜16 秒・向きの変化 300〜420°)"""
    A, out = d['anchors'], []
    for a, b in zip(A[:-1], A[1:]):
        if not (5.0 <= b - a <= 16.0):
            continue
        dpsi = float(np.interp(b, d['it'], d['hd']) - np.interp(a, d['it'], d['hd']))
        if not (300 <= dpsi <= 420):
            continue
        if not ai_continuous(d['z'], a, b):
            continue
        out.append((float(a), float(b)))
    return out


def frame_vals(d, tq):
    """フレーム(30 Hz)の値を時刻 tq に(直前のフレーム)"""
    z = d['z']
    t = z['t_s']
    k = np.clip(np.searchsorted(t, tq, side='right') - 1, 0, len(t) - 1)
    c, n = d['ref']
    return {'f_out': n - z['out'][k, 1], 'f_raw': n - z['raw'][k, 1], 'f_model': n - z['model_out'][k, 1],
            's_out': z['out'][k, 0] - c, 'assist': z['assist'][k], 'loc': z['loc_code'][k], 'gain': z['gains'][k, 1]}


def lap_profile(d, a, b, dmax):
    """周の終わりの起点 b から手前 dmax m までを、起点までの距離 r の格子に: 起点までの時間・速さ・指令・舵・加速"""
    vt, Df = d['vt'], d['Df']
    m = (vt >= a) & (vt <= b)
    tt, DD = vt[m], Df[m]
    Db = float(np.interp(b, vt, Df))
    r_all = Db - DD                                # 起点までの距離(減っていく)
    r = np.arange(0.0, dmax + 1e-9, DX)
    ok = r <= r_all[0]
    # r → 時刻(r は単調に減る。止まったところは同じ値なので、最後に通った時刻)
    rr, ti = r_all[::-1], tt[::-1]
    t_at = np.interp(r, rr, ti)
    t_at[~ok] = np.nan
    fv = frame_vals(d, np.nan_to_num(t_at, nan=a))
    v = np.interp(np.nan_to_num(t_at, nan=a), vt, d['v'])
    ay = np.interp(np.nan_to_num(t_at, nan=a), d['it'], smooth(d['ay'], 9))
    nanify = lambda x: np.where(ok, x, np.nan)   # noqa: E731
    return {'r': r, 'tau': nanify(b - t_at), 'v': nanify(v), 'f_out': nanify(fv['f_out']), 'f_raw': nanify(fv['f_raw']),
            's_out': nanify(fv['s_out']), 'ay': nanify(ay), 'loc': np.where(ok, fv['loc'], -1)}


def slip_window(d, t_a, t_b):
    """車輪の速さの増え と IMU の前の加速の積分 の比(1 より大きい = 車輪が先に回った = 空転の疑い)"""
    vt, it = d['vt'], d['it']
    if t_b - t_a < 0.15:
        return None
    dv_w = float(np.interp(t_b, vt, d['v']) - np.interp(t_a, vt, d['v']))
    m = (it >= t_a) & (it <= t_b)
    if m.sum() < 5:
        return None
    dv_i = float(np.trapz(d['ay'][m], it[m]))
    return {'t_a': round(t_a, 3), 't_b': round(t_b, 3), 'dv_wheel': round(dv_w, 3), 'dv_imu': round(dv_i, 3),
            'ratio': round(dv_w / dv_i, 3) if dv_i > 0.2 else None}


def launch_rows(d, laps_ref, ref_kind):
    z, lab = d['z'], d['lab']
    t, ai = z['t_s'], z['ai'].astype(bool)
    vt, it = d['vt'], d['it']
    c, n = d['ref']
    no = ai_no(lab.get('description'))
    run = lab.get('run') or ''
    model = MODEL.get(run, run[:24])
    rows = []
    for i0, i1 in le.runs(ai):
        t0, t1 = float(t[i0]), float(t[i1])
        e0 = float(np.interp(t0, vt, d['erpm']))
        if abs(e0) >= STOP_ERPM or t1 - t0 < 0.1:
            continue
        seg = slice(i0, i1 + 1)
        a_seg = z['assist'][seg]
        kick = np.nonzero(a_seg == 4)[0]
        esc0 = np.nonzero(a_seg[:15] == 1)[0]             # 最初の 0.5 秒の「止まっている間のブレーキ → 中立」
        g = float(np.median(z['gains'][seg, 1]))
        mv = (vt >= t0) & (vt <= t1)
        t_m1 = first_time(vt[mv], -d['erpm'][mv] >= MOVE1_ERPM)
        t_m2 = first_time(vt[mv], -d['erpm'][mv] >= MOVE2_ERPM)
        # モデルが前へ出すまで(ゲイン後の model_out で 80 µs 以上・raw で 80 µs 以上)
        fm = n - z['model_out'][seg, 1]
        fr = n - z['raw'][seg, 1]
        ts = t[seg]
        t_model80 = first_time(ts, fm >= 80)
        t_model40 = first_time(ts, fm >= 40)
        row = {'log': d['log'], 'no': no, 'session': session_of(d['log'], no), 'model': model, 'gain': round(g, 3),
               'start_kick_param': (lab.get('params_at_start') or {}).get('start_kick'),
               't_ai': round(t0, 3), 'erpm_at_ai': round(e0, 0), 'ai_seg_s': round(t1 - t0, 2),
               'kick': bool(len(kick)), 'kick_frames': int(len(kick)), 'kick_s': round(len(kick) / 30.0, 3),
               'kick_end_s': round(float(ts[kick[-1]] - t0) + 1 / 30.0, 3) if len(kick) else None,
               'esc_neutral_frames_0_5s': int(len(esc0)),
               'move_0p2_s': round(t_m1 - t0, 3) if t_m1 is not None else None,
               'move_0p4_s': round(t_m2 - t0, 3) if t_m2 is not None else None,
               'model80_s': round(t_model80 - t0, 3) if t_model80 is not None else None,
               'model40_s': round(t_model40 - t0, 3) if t_model40 is not None else None,
               'f_model_first5': [round(float(x), 0) for x in fm[:5]],
               'f_raw_first5': [round(float(x), 0) for x in fr[:5]]}
        # 最初の 1.0 秒・1.5 秒の指令(車に出した前進・モデルの前進(ゲイン後)・ゲイン前)
        for w in (0.5, 1.0, 1.5):
            mm = (ts - t0) < w
            row['f_out_mean_%.1fs' % w] = round(float(np.mean(n - z['out'][seg, 1][mm])), 1)
            row['f_model_mean_%.1fs' % w] = round(float(np.nanmean(fm[mm])), 1)
            row['s_out_absmean_%.1fs' % w] = round(float(np.nanmean(np.abs(z['out'][seg, 0][mm] - c))), 1)
        nxt = [a for a in d['anchors'] if t0 < a <= t1]
        if not nxt or nxt[0] - t0 > MAX_TO_R1_S:
            row['to_r1_s'] = None
            row['why_no_r1'] = 'AI の区間が短い' if t1 - t0 < 3 else '15 秒以内に起点なし'
            rows.append(row)
            continue
        a1 = float(nxt[0])
        D0 = float(np.interp(t0, vt, d['Df']))
        dist = float(np.interp(a1, vt, d['Df']) - D0)
        col = [h for h in d['hits'] if t0 <= h[0] < a1]
        rec = bool(np.any(z['assist'][seg][(ts < a1)] == 2))
        row.update({'to_r1_s': round(a1 - t0, 3), 'dist_m': round(dist, 2), 'collisions': col, 'recovery': rec,
                    'dpsi_deg': round(float(np.interp(a1, it, d['hd']) - np.interp(t0, it, d['hd'])), 1)})
        # 発進の距離の格子(発進から x m)
        m = (vt >= t0) & (vt <= a1)
        tt, DD = vt[m], d['Df'][m] - D0
        x = np.arange(0.0, dist + 1e-9, DX)
        # x に初めて届いた時刻(DD は単調に増える)
        idx = np.searchsorted(DD, x, side='left')
        idx = np.clip(idx, 1, len(DD) - 1)
        x0, x1 = DD[idx - 1], DD[idx]
        frac = np.where(x1 > x0, (x - x0) / np.maximum(x1 - x0, 1e-9), 0.0)
        tL = tt[idx - 1] + frac * (tt[idx] - tt[idx - 1])
        tL[0] = t0 if t_m1 is None else first_time(tt, DD > 0.005) or t0
        fvL = frame_vals(d, tL)
        prof = {'x': x, 't': tL - t0, 'v': np.interp(tL, vt, d['v']), 'f_out': fvL['f_out'], 'f_raw': fvL['f_raw'],
                'f_model': fvL['f_model'], 's_out': fvL['s_out'], 'assist': fvL['assist'],
                'ay': np.interp(tL, it, smooth(d['ay'], 9)), 'loc': fvL['loc']}
        # 走りながら(周の終わりの起点の dist m 手前 → 起点)
        P = []
        for (la, lb, dd) in laps_ref:
            p = lap_profile(dd, la, lb, dist)
            if np.isfinite(p['tau'][-1 if len(p['tau']) else 0]):
                P.append(p)
        row['fly_n'] = len(P)
        row['fly_ref'] = ref_kind
        if P:
            # 発進の x = 起点までの距離 r = dist − x
            r_of_x = dist - x
            def at(p, key):
                return np.interp(r_of_x, p['r'], p[key])
            tauF = np.nanmedian(np.stack([at(p, 'tau') for p in P]), axis=0)
            T_fly = float(np.nanmedian([p['tau'][-1] for p in P]))     # 起点の dist m 手前 → 起点
            tF = T_fly - tauF                                           # 周の、発進の場所から x m までの時間
            F = {k: np.nanmedian(np.stack([at(p, k) for p in P]), axis=0) for k in ('v', 'f_out', 'f_raw', 's_out', 'ay')}
            locF = np.stack([np.round(at(p, 'loc')) for p in P])
            loc_mode = np.array([np.bincount(col_.astype(int)[col_ >= 0], minlength=7).argmax() if (col_ >= 0).any() else -1
                                 for col_ in locF.T])
            loss_x = prof['t'] - tF
            row.update({'fly_s': round(T_fly, 3), 'loss_s': round(a1 - t0 - T_fly, 3),
                        'start_loc': le.LOC_NAMES[int(loc_mode[0])] if loc_mode[0] >= 0 else None})
            # LH の入口(周の場所で LH が始まる x)
            lh = np.nonzero(loc_mode == 6)[0]
            x_lh = float(x[lh[0]]) if len(lh) else None
            row['x_lh_m'] = round(x_lh, 2) if x_lh is not None else None
            # 失う時間の内訳: 動き出すまで(x = 0)・直線(x_lh まで)・LH〜起点
            k_lh = int(lh[0]) if len(lh) else len(x) - 1
            row['loss_parts'] = {'dead_s': round(float(loss_x[0]), 3),
                                 'straight_s': round(float(loss_x[k_lh] - loss_x[0]), 3),
                                 'lh_to_r1_s': round(float((a1 - t0 - T_fly) - loss_x[k_lh]), 3)}
            # 周の速さに追いつくところ(発進の速さ ≥ 周の速さ − 0.1 m/s)
            catch = np.nonzero(prof['v'] >= F['v'] - 0.1)[0]
            catch = catch[catch > 2] if len(catch) else catch
            row['catch_x_m'] = round(float(x[catch[0]]), 2) if len(catch) else None
            row['v_fly_at_start'] = round(float(F['v'][0]), 2)
            row['v_fly_at_lh'] = round(float(F['v'][k_lh]), 2)
            row['v_launch_at_lh'] = round(float(prof['v'][k_lh]), 2)
            row['t_launch_to_lh'] = round(float(prof['t'][k_lh]), 3)
            row['t_fly_to_lh'] = round(float(tF[k_lh]), 3)
            # 直線の指令の比べ(発進 vs 周、同じ場所。x が 0.3 m〜x_lh)
            sm = (x >= 0.3) & (x <= (x_lh if x_lh is not None else dist))
            if sm.any():
                row['straight_f_out_launch'] = round(float(np.nanmean(prof['f_out'][sm])), 1)
                row['straight_f_out_fly'] = round(float(np.nanmean(F['f_out'][sm])), 1)
                row['straight_f_raw_launch'] = round(float(np.nanmean(prof['f_raw'][sm])), 1)
                row['straight_f_raw_fly'] = round(float(np.nanmean(F['f_raw'][sm])), 1)
                row['straight_s_abs_launch'] = round(float(np.nanmean(np.abs(prof['s_out'][sm]))), 1)
                row['straight_s_abs_fly'] = round(float(np.nanmean(np.abs(F['s_out'][sm]))), 1)
                row['straight_ay_launch'] = round(float(np.nanmean(prof['ay'][sm])), 2)
                row['straight_v_launch'] = round(float(np.nanmean(prof['v'][sm])), 2)
                row['straight_v_fly'] = round(float(np.nanmean(F['v'][sm])), 2)
            lm = x >= (x_lh if x_lh is not None else dist)
            if lm.any():
                row['lh_f_out_launch'] = round(float(np.nanmean(prof['f_out'][lm])), 1)
                row['lh_f_out_fly'] = round(float(np.nanmean(F['f_out'][lm])), 1)
                row['lh_v_launch'] = round(float(np.nanmean(prof['v'][lm])), 2)
                row['lh_v_fly'] = round(float(np.nanmean(F['v'][lm])), 2)
            row['grid'] = {'x': x.round(2).tolist(), 't_launch': prof['t'].round(3).tolist(), 't_fly': tF.round(3).tolist(),
                           'v_launch': prof['v'].round(3).tolist(), 'v_fly': F['v'].round(3).tolist(),
                           'f_out_launch': np.round(prof['f_out'], 1).tolist(), 'f_out_fly': np.round(F['f_out'], 1).tolist(),
                           'f_model_launch': np.round(prof['f_model'], 1).tolist(),
                           'f_raw_launch': np.round(prof['f_raw'], 1).tolist(), 'f_raw_fly': np.round(F['f_raw'], 1).tolist(),
                           's_out_launch': np.round(prof['s_out'], 1).tolist(), 's_out_fly': np.round(F['s_out'], 1).tolist(),
                           'ay_launch': np.round(prof['ay'], 2).tolist(), 'ay_fly': np.round(F['ay'], 2).tolist(),
                           'loc_fly': loc_mode.tolist(), 'assist_launch': prof['assist'].astype(int).tolist()}
        # 時間の格子(発進から 0〜3 秒、1/30 秒): 速さ・加速(車輪・IMU)・指令
        tg = t0 + np.arange(0, min(3.0, a1 - t0), 1 / 30.0)
        fvT = frame_vals(d, tg)
        vw = np.interp(tg, vt, d['v'])
        vt_s = smooth(d['v'], 9)
        aw = np.gradient(vt_s, vt)
        row['tgrid'] = {'t': (tg - t0).round(3).tolist(), 'v': vw.round(3).tolist(),
                        'a_wheel': np.interp(tg, vt, aw).round(2).tolist(),
                        'a_imu': np.interp(tg, it, smooth(d['ay'], 9)).round(2).tolist(),
                        'f_out': np.round(fvT['f_out'], 1).tolist(), 'f_model': np.round(fvT['f_model'], 1).tolist(),
                        'f_raw': np.round(fvT['f_raw'], 1).tolist(), 's_out': np.round(fvT['s_out'], 1).tolist(),
                        'assist': fvT['assist'].astype(int).tolist(),
                        'gz': np.interp(tg, it, smooth(d['gz'], 9)).round(1).tolist()}
        # 空転の目安: 動き出し(0.2 m/s)から 0.6 秒、または LH の手前まで
        if t_m1 is not None:
            tb = min(t_m1 + 0.6, t0 + (row.get('t_launch_to_lh') or 99), a1)
            row['slip'] = slip_window(d, t_m1, tb)
        rows.append(row)
    return rows


def flying_slip(d, laps):
    """参考: 周の中の加速(ヘアピンの出口など)で、車輪の速さの増え と IMU の加速の積分を比べる"""
    out = []
    vt = d['vt']
    vs = smooth(d['v'], 9)
    aw = np.gradient(vs, vt)
    for a, b in laps:
        m = (vt >= a) & (vt <= b)
        idx = np.nonzero(m)[0]
        if not len(idx):
            continue
        acc = aw[idx] > 1.5
        for i0, i1 in le.runs(acc):
            ta, tb = float(vt[idx[i0]]), float(vt[idx[i1]])
            if tb - ta < 0.25:
                continue
            s = slip_window(d, ta, tb)
            if s and s['ratio'] is not None:
                fv = frame_vals(d, np.array([ta, (ta + tb) / 2]))
                s.update({'v_a': round(float(np.interp(ta, vt, d['v'])), 2), 'v_b': round(float(np.interp(tb, vt, d['v'])), 2),
                          'f_out_mid': round(float(fv['f_out'][1]), 1), 's_out_mid': round(float(fv['s_out'][1]), 1)})
                out.append(s)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data, laps = {}, {}
    for log in logs():
        try:
            d = load(log)
        except Exception as e:      # noqa: BLE001
            print('skip', log, e)
            continue
        if not d['z']['ai'].any():
            continue
        data[log] = d
        laps[log] = laps_of(d)
    # 同じ設定(モデル・ゲイン・回)の周
    def key(d):
        lab = d['lab']
        no = ai_no(lab.get('description'))
        g = float(np.median(d['z']['gains'][d['z']['ai'].astype(bool), 1]))
        return (MODEL.get(lab.get('run') or '', lab.get('run')), round(g, 2), session_of(d['log'], no))
    by_key = {}
    for log, d in data.items():
        by_key.setdefault(key(d), []).append(log)
    rows, fly_slip, lap_rows = [], [], []
    for log, d in data.items():
        if laps[log]:
            ref = [(a, b, d) for a, b in laps[log]]
            kind = 'same_log'
        else:
            ref = [(a, b, data[o]) for o in by_key[key(d)] for a, b in laps[o]]
            kind = 'same_setting' if ref else 'none'
        rows += launch_rows(d, ref, kind)
        fly_slip += [dict(s, log=log) for s in flying_slip(d, laps[log])]
        for a, b in laps[log]:
            lap_rows.append({'log': log, 'a': round(a, 3), 'b': round(b, 3), 'lap_s': round(b - a, 3)})
    res = {'created_by': 'experiments/20261026_fix/diag_launch/collect.py', 'launches': rows,
           'flying_accel_slip': fly_slip, 'laps': lap_rows}
    (OUT / 'launches.json').write_text(json.dumps(res, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
    print('logs', len(data), 'launches', len(rows), 'with R1', sum(r.get('to_r1_s') is not None for r in rows),
          'laps', len(lap_rows), 'flying accel windows', len(fly_slip))
    for r in rows:
        print('%s No.%-3s %-10s g%.2f kick=%d(%3dfr) mv%-6s m80=%-6s toR1=%-6s d=%-5s loss=%-6s parts=%s fly=%s/%s %s' % (
            r['log'], r['no'], r['model'], r['gain'], r['kick'], r['kick_frames'], r['move_0p4_s'], r['model80_s'],
            r.get('to_r1_s'), r.get('dist_m'), r.get('loss_s'), r.get('loss_parts'), r.get('fly_n'), r.get('fly_ref'),
            r.get('start_loc')))


if __name__ == '__main__':
    main()
