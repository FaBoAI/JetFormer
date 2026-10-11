"""1本のログについて、推測航法の軌跡をコース図に合わせ、フレームごとの (t, x, z, heading, 進行度) を書き出す。
使い方: python fit_run.py <log> [出力フォルダ]
"""
import sys, os, json
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import course_geom as cg, track_fit as tf
from loader import L

LINES_X = np.array([2.50, 4.30, 6.10])
GATE_X = 4.90
G_LINE, G_GATE = 0, 1
# 画面の端までの距離 D の事前分布 (平均, 標準偏差) [m]。周回の多い 172910 / 173445 を自由に合わせた値
# (線 0.39 / 0.20、ゲート 1.47 / 1.53) から決めた。周回の少ないログで D が暴れないようにする
D_PRIOR = {0: (0.30, 0.10), 1: (1.50, 0.15)}
FIT_KW = dict(wall_w=20, margin=0.12, look=0.5, cw=0.3, dzw=1.0, dz=0.45)

def runs(mask):
    m = np.r_[False, mask, False].astype(int); dd = np.diff(m)
    return list(zip(np.nonzero(dd == 1)[0], np.nonzero(dd == -1)[0]))

def segments(d, psi):
    """手で持ち上げて置き直した(止まっている間に 20° 以上向きが変わった)ところで区間を切る"""
    still = np.abs(d.erpm) < 100
    cuts = []
    for a, b in runs(still):
        ta, tb = d.vt[a], d.vt[b - 1]
        ia, ib = np.searchsorted(d.it, [ta, tb])
        rot = np.abs(np.diff(psi[ia:ib + 1])).sum() if ib > ia else 0.0
        if rot > 20.0:
            cuts.append((ta, tb, rot))
    edges = [d.it[0]]
    for ta, tb, _ in cuts:
        edges += [ta, tb]
    edges.append(d.it[-1])
    segs = [(edges[k], edges[k + 1]) for k in range(0, len(edges), 2)]
    return segs, cuts

def detect_lines(d, t, X, Z, h):
    """手前の直線を +X に走っている間のカメラ画像で、白いスタートラインの帯が画面の下端(行 212)に届いた瞬間"""
    j = np.clip(np.searchsorted(t, d.ct), 0, len(t) - 1)
    Xc, Zc = X[j], Z[j]; hd = np.degrees(np.angle(np.exp(1j * h[j])))
    cand = (Xc > 1.4) & (Xc < 7.8) & (Zc < 1.8) & (np.abs(hd) < 50) & (d.ct >= t[0]) & (d.ct <= t[-1])
    rows = []
    for i in np.nonzero(cand)[0]:
        im = cv2.imread(d.cf[i])
        if im is None: continue
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        wm = (hsv[..., 2] > 175) & (hsv[..., 1] < 60)
        w = wm[:, 60:164].mean(1)
        rr = np.nonzero(w[90:] > 0.45)[0]
        rows.append((i, 90 + rr.max() if len(rr) else -1))
    ev = []
    for (i0, b0), (i1, b1) in zip(rows[:-1], rows[1:]):
        if i1 == i0 + 1 and 120 <= b0 < 212 and (b1 >= 212 or b1 == -1 or b1 < b0 - 40):
            ev.append(i0)
    ev = np.array(ev, int)
    if not len(ev):
        return np.zeros(0), np.zeros(0, int)
    te = d.ct[ev]; xe = Xc[ev]
    # 1.2 s 以内に続くものを1回の通過とみなし、3本そろえば順番で線を決める。そうでなければ近い線
    groups, cur = [], [0]
    for k in range(1, len(te)):
        if te[k] - te[k - 1] < 1.2: cur.append(k)
        else: groups.append(cur); cur = [k]
    groups.append(cur)
    lab = np.full(len(te), -1)
    trip = [g for g in groups if len(g) == 3]
    off = np.median([xe[g].mean() - LINES_X.mean() for g in trip]) if trip else 0.0
    sc = np.median([(xe[g[2]] - xe[g[0]]) / 3.6 for g in trip]) if trip else 1.0
    for g in groups:
        if len(g) == 3:
            lab[g] = [0, 1, 2]
        else:
            for k in g:
                pred = LINES_X.mean() + off + (LINES_X - LINES_X.mean()) * sc
                m = np.argmin(np.abs(pred - xe[k]))
                if abs(pred[m] - xe[k]) < 0.5: lab[k] = m
    ok = lab >= 0
    return te[ok], lab[ok]

def detect_gate(d, t, X, Z, h, row=8):
    """奥の直線を -X に走っている間、LED 矢印ゲートの緑の表示の下端が画面の上端(行 8)より上へ抜けた瞬間"""
    j = np.clip(np.searchsorted(t, d.ct), 0, len(t) - 1)
    Xc, Zc = X[j], Z[j]; hd = np.degrees(np.angle(np.exp(1j * h[j])))
    cand = (Xc > 4.3) & (Xc < 8.8) & (Zc > 3.8) & (np.abs(np.abs(hd) - 180) < 50) & (d.ct >= t[0]) & (d.ct <= t[-1])
    low = {}
    for i in np.nonzero(cand)[0]:
        im = cv2.imread(d.cf[i])
        if im is None: continue
        top = im[:45].astype(int); b, g, r = top[..., 0], top[..., 1], top[..., 2]
        m = (g > r + 25) & (g > b + 10) & (g > 110)
        rr = np.nonzero(m.sum(1) >= 3)[0]
        low[i] = rr.max() if (m.sum() >= 20 and len(rr)) else -1
    ev = []
    for i in sorted(low):
        if low[i] >= row and low.get(i + 1, None) is not None and low[i + 1] < row:
            ev.append(i)
    ev = np.array(ev, int)
    if not len(ev):
        return np.zeros(0)
    te = d.ct[ev]
    # 1.5 s 以内に複数あれば(衝突で車体が揺れたときなど)その通過は使わない。直前 1 s に衝突があったものも使わない
    imp = np.array([e['t'] for e in d.events()])
    keep = []
    for k, tk in enumerate(te):
        if np.sum(np.abs(te - tk) < 1.5) > 1: continue
        if len(imp) and np.any((imp > tk - 1.0) & (imp < tk + 0.1)): continue   # 直前 1 s に衝突(車体が揺れている)
        keep.append(tk)
    return np.array(keep)

def lap_gyro_scale(t, psi):
    """R1 開始から次の R1 開始までの向きの変化(生のジャイロ)の中央値が 360° になる倍率"""
    r1 = r1_idx = tf.r1_starts(t, psi)
    inc = np.diff(psi[r1]) if len(r1) >= 2 else np.zeros(0)
    inc = inc[(inc > 300) & (inc < 420)]
    return (360.0 / np.median(inc) if len(inc) >= 3 else None), inc

def fit_segment(d, t, psi, s, gz, gimp, cl, walls, lines=True, kw=FIT_KW, log=None, g0=None, holdout=None, g_sd=0.007, speed_k=True):
    """holdout: None または (群, 保持する通過の偶奇) — 交差検証用に目印の半分を使わない"""
    r1 = tf.r1_starts(t, psi)
    if len(r1) == 0:
        return None
    v = np.gradient(s, t); vs = np.convolve(v, np.ones(15) / 15, 'same')
    near_imp = np.zeros(len(t), bool)
    for i in np.nonzero(gimp > 2.5)[0]:
        near_imp[max(0, i - 40):i + 40] = True
    steps = []
    for i in np.nonzero(np.abs(gz) >= 990)[0]:
        if not steps or t[i] - steps[-1] > 0.5: steps.append(t[i])
    use = np.zeros(len(t), bool); use[::10] = True; use &= (vs > 0.5) & ~near_imp
    knots = sorted(set(np.round([t[0]] + [t[i] for i in r1] + [t[-1]], 3)))
    kk = [knots[0]]
    for a, b in zip(knots[:-1], knots[1:]):
        n = int(np.ceil((b - a) / 11.0)); kk += list(np.linspace(a, b, n + 1)[1:])
    gl, inc = lap_gyro_scale(t, psi)
    if g0 is None:
        g0 = gl if gl is not None else 1.0
    F = tf.Fitter(t, psi, s, np.array(kk), steps, r1, cl, use, g0=g0, g_sd=g_sd, n_groups=2, speed_k=speed_k)
    F.set_events(G_LINE, [], [], *D_PRIOR[G_LINE]); F.set_events(G_GATE, [], [], *D_PRIOR[G_GATE])
    x0 = F.init((7.70, 2.10), np.pi)
    lg1 = []
    x = F.fit(x0, walls=walls, log=lg1, **kw)
    info = {'r1': len(r1), 'knots': len(kk), 'gyro_steps': len(steps), 'gyro_scale_prior': float(g0),
            'lap_heading_raw_deg': inc.round(1).tolist(), 'stage1': lg1[-1]}
    info['x_stage1'] = x.copy()
    if lines:
        X, Z, h = F.pose(x)
        te, lab = detect_lines(d, t, X, Z, h)
        tg = detect_gate(d, t, X, Z, h)
        info['line_events'] = int(len(te)); info['gate_events'] = int(len(tg))
        ei = np.clip(np.searchsorted(t, te), 0, len(t) - 1); gi = np.clip(np.searchsorted(t, tg), 0, len(t) - 1)
        # 通過番号(1.2 s 以上あいたら次の通過)
        pl = np.cumsum(np.r_[1, np.diff(te) > 1.2]) if len(te) else np.zeros(0, int)
        pg = np.arange(len(tg))
        keep_l = np.ones(len(te), bool); keep_g = np.ones(len(tg), bool)
        if holdout is not None:
            keep_l = (pl % 2) == holdout; keep_g = (pg % 2) == holdout
        F.set_events(G_LINE, ei[keep_l], LINES_X[lab[keep_l]], *D_PRIOR[G_LINE])
        F.set_events(G_GATE, gi[keep_g], np.full(keep_g.sum(), GATE_X), *D_PRIOR[G_GATE])
        lg2 = []
        x = F.fit(x, walls=walls, log=lg2, **kw)
        info['stage2'] = lg2[-1]
        X, Z, h = F.pose(x); D = F.unpack(x)[6]
        rl = X[ei] + D[G_LINE] * np.cos(h[ei]) - LINES_X[lab] if len(te) else np.zeros(0)
        rg = X[gi] + D[G_GATE] * np.cos(h[gi]) - GATE_X if len(tg) else np.zeros(0)
        info['line_residual_m'] = rl.round(3).tolist(); info['gate_residual_m'] = rg.round(3).tolist()
        info['line_used'] = keep_l.tolist(); info['gate_used'] = keep_g.tolist()
        info['D_line_m'] = float(D[G_LINE]); info['D_gate_m'] = float(D[G_GATE])
        info['line_t'] = te.tolist(); info['line_id'] = lab.tolist(); info['gate_t'] = tg.tolist()
    g, sg, A, Bx, Bz, c, D = F.unpack(x)
    info['speed_k'] = float(F.kval(x))
    info.update({'gyro_scale': float(g), 'dist_scale': float(sg), 'm_per_hall': float(sg * tf.KH),
                 'heading_corr_step_per_knot_deg_abs_median': float(np.median(np.abs(np.degrees(np.diff(A))))) if len(A) > 1 else 0.0,
                 'pos_corr_step_per_knot_m_median': float(np.median(np.hypot(np.diff(Bx), np.diff(Bz)))) if len(A) > 1 else 0.0,
                 'gyro_step_deg': np.degrees(c).round(1).tolist()})
    return F, x, use, info

def progress(cl, X, Z, h, valid):
    P = np.stack([X, Z], 1)
    idx, off = cl.match(P, h, max_dh=np.radians(75), max_d=2.5)
    sp = np.where(idx >= 0, cl.s[np.clip(idx, 0, None)], np.nan)
    sec = np.where(idx >= 0, cl.sec[np.clip(idx, 0, None)], -1)
    # 周をまたいで積算(進行度が L/2 以上戻ったら1周)
    cum = np.full(len(sp), np.nan); lap = np.zeros(len(sp), int); k = 0; last = None
    for i in range(len(sp)):
        if not valid[i] or np.isnan(sp[i]):
            continue
        if last is not None:
            dsp = sp[i] - last
            if dsp < -cl.L / 2: k += 1
            elif dsp > cl.L / 2: k -= 1
        last = sp[i]; lap[i] = k; cum[i] = k * cl.L + sp[i]
    return sp, cum, lap, sec, off

def main(log, outdir, lines=True):
    geo = cg.build(); C = cg.centerline(); cl = tf.Centerline(C); walls = cg.wall_segments(geo)
    d = L(log)
    psi, s, bias, nst, nbad = tf.raw_dr(d)
    segs, cuts = segments(d, psi)
    ev = d.events()
    T = d.ct                                            # フレーム = カメラ画像の時刻
    nF = len(T)
    out = {k: np.full(nF, np.nan) for k in ('x', 'z', 'heading', 'speed')}
    seg_id = np.full(nF, -1)
    infos = []
    full_X = np.full(len(d.it), np.nan); full_Z = np.full(len(d.it), np.nan); full_h = np.full(len(d.it), np.nan); full_use = np.zeros(len(d.it), bool)
    for k, (a, b) in enumerate(segs):
        m = (d.it >= a) & (d.it <= b)
        t = d.it[m]
        if m.sum() < 100:
            continue
        dist = s[m][-1] - s[m][0]
        info = {'segment': k, 't0': float(a), 't1': float(b), 'moving_distance_m_raw': float(dist)}
        if dist < 8.0:
            info['fitted'] = False; info['reason'] = '走行距離が短い (< 8 m)'; infos.append(info); continue
        r = fit_segment(d, t, psi[m], s[m] - s[m][0], d.gyr[m, 2], d.g[m], cl, walls, lines=lines)
        if r is None:
            info['fitted'] = False; info['reason'] = 'R1 が見つからない(1周未満)'; infos.append(info); continue
        F, x, use, fi = r
        fi.pop('x_stage1', None)
        X, Z, h = F.pose(x)
        ev_m, dw = tf.evaluate(X, Z, use, cl, walls, h)
        cr = cg.crossings(np.stack([X[::5], Z[::5]], 1), walls)
        fi.update(ev_m); fi['wall_crossings_5'] = int(len(cr))
        info.update(fi); info['fitted'] = True
        infos.append(info)
        full_X[m] = X; full_Z[m] = Z; full_h[m] = h; full_use[m] = use
        mf = (T >= a) & (T <= b)
        out['x'][mf] = np.interp(T[mf], t, X); out['z'][mf] = np.interp(T[mf], t, Z)
        out['heading'][mf] = np.interp(T[mf], t, np.unwrap(h))
        seg_id[mf] = k
    sp_ = np.gradient(s, d.it)
    out['speed'] = np.interp(T, d.it, np.convolve(sp_, np.ones(9) / 9, 'same')) * np.nanmedian([i.get('dist_scale', np.nan) for i in infos if i.get('fitted')] or [1.0])
    valid = seg_id >= 0
    sp, cum, lap, sec, off = progress(cl, out['x'], out['z'], out['heading'], valid)
    ai = np.interp(T, d.at, d.sel.astype(float)) > 0.5
    # 衝突
    imp = []
    for e in ev:
        i = int(np.argmin(np.abs(d.it - e['t'])))
        if np.isfinite(full_X[i]):
            dd, kw = cg.dist_to_segments(np.array([[full_X[i], full_Z[i]]]), walls)
            fi_ = int(np.argmin(np.abs(T - e['t'])))
            imp.append({'t': e['t'], 'g': e['g'], 'x': float(full_X[i]), 'z': float(full_Z[i]), 'heading_deg': float(np.degrees(full_h[i])),
                        'wall_dist_m': float(dd[0]), 'ai': bool(ai[fi_]), 'section': C['section_ids'][sec[fi_]] if sec[fi_] >= 0 else '',
                        'accel_lat_g': float(d.acc[i, 0] / 9.80665), 'accel_lon_g': float(d.acc[i, 1] / 9.80665)})
        else:
            imp.append({'t': e['t'], 'g': e['g'], 'x': None, 'z': None})
    os.makedirs(outdir, exist_ok=True)
    np.savez_compressed(os.path.join(outdir, 'tracks_%s.npz' % log),
                        t=T, t_us=(T * 1e6).astype(np.int64) + d.t0, x=out['x'], z=out['z'], heading=out['heading'],
                        speed=out['speed'], progress_m=sp, progress_cum_m=cum, lap=lap, section=sec, cross_track_m=off,
                        valid=valid, segment=seg_id, ai=ai,
                        imu_t=d.it, imu_x=full_X, imu_z=full_Z, imu_heading=full_h, imu_used_in_fit=full_use,
                        section_names=np.array(C['section_ids']), centerline_xz=C['xz'], centerline_s=C['s'],
                        impacts_t=np.array([e['t'] for e in imp]), impacts_g=np.array([e['g'] for e in imp]),
                        impacts_x=np.array([np.nan if e['x'] is None else e['x'] for e in imp]),
                        impacts_z=np.array([np.nan if e['z'] is None else e['z'] for e in imp]))
    meta = {'log': log, 'description': d.meta.get('description'), 't0_us': d.t0,
            'time_base': '0 s = このログ(残っている最初のシーン)の最初の AI_CMD',
            'gyro_bias_dps': bias, 'gyro_bias_still_samples': nst, 'hall_gaps_filled': nbad,
            'carried_stops': [{'t0': a, 't1': b, 'rotation_deg': r} for a, b, r in cuts],
            'segments': infos, 'impacts': imp, 'centerline_length_m': C['length'],
            'frames': int(nF), 'frames_valid': int(valid.sum())}
    json.dump(meta, open(os.path.join(outdir, 'tracks_%s.json' % log), 'w'), ensure_ascii=False, indent=1, default=float)
    return meta

if __name__ == '__main__':
    log = sys.argv[1]
    outdir = sys.argv[2] if len(sys.argv) > 2 else '/home/jetson/jetformer_reports/overnight_20261011/course'
    m = main(log, outdir, lines='--nolines' not in sys.argv)
    for sgi in m['segments']:
        print(json.dumps({k: v for k, v in sgi.items() if k not in ('line_t', 'line_id', 'gate_t', 'line_used', 'gate_used', 'lap_heading_raw_deg')}, ensure_ascii=False, default=float))
