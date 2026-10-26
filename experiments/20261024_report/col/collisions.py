#!/usr/bin/env python3
"""解析 C: 10/24 の衝突は「どこで」「どう」当たったか(読むだけ。車・基板には触れない)

    /home/jetson/bin/jf-heavy --lock none --mem 1500M python3 -I collisions.py
    → col_data.json(図と表のもと)・col_tables.md(表)

元:
  衝突       ~/jetformer_reports/bench_20261024_tune/since_1300/nightly/scorecard/scorecard.json の
             per_log.*.collision_events(class=main が数えたもの 64 件。restart = 立て直し中の 2 件は別に出す)
             場所は loc_detail(R1・GM_in/GM_out → GM・DE・SW・HS・LH)
  当たる前   生の記録 data/raw/<ログ>/<シーン>/AI_CMD.npz(out・model_out)と VEHICLE.npz(erpm、100 Hz)
             時刻は data/labels/<ログ>.json の events.collisions の t_us(scorecard の t_s と 0.01 秒で合わせる)
  通過       data/labels/<ログ>.npz(30 fps。loc_code・loc_detail・out・ai・collision・speed_mps)。
             DE の通過 = tools/de_swing.py と同じ(loc_code 3 が 30 フレーム以上続き、ずっと AI。
             区間の中か終わりから 0.3 秒以内の AI の衝突 = 当たった)
             SW の通過 = loc_detail 5 が 15 フレーム以上続き、ずっと AI。当たった = SW の中か出口から 0.6 秒以内の
             AI の衝突(場所 SW か HS)
  ステア     out − 中央(ref_us.steer_center)。+ = 左・− = 右(tof_guard.py・de_swing.py と同じ)
  スロットル 前進 = 中立 − out(throttle_neutral)。+ = 前へ(中立より小さい µs が前進。発進の補助 = 中立 − 80)
  速さ       −eRPM × 0.5e-3 m/s(label_events.KV)
回: 1 回目 = 13:46〜(No.1〜4)・2 回目 = 16:01〜(No.5〜10)・3 回目 = 18:31〜(No.11〜20)
"""
import json
from pathlib import Path

import numpy as np

HOME = Path('/home/jetson')
SC = HOME / 'jetformer_reports/bench_20261024_tune/since_1300/nightly/scorecard/scorecard.json'
RAW = HOME / 'jetformer/data/raw'
LAB = HOME / 'jetformer/data/labels'
OUT = Path(__file__).resolve().parent
KV = 0.5e-3
FPS = 30
LOCS = ['R1', 'GM', 'DE', 'SW', 'HS', 'LH']
DETAIL_TO = {'R1': 'R1', 'GM_in': 'GM', 'GM_out': 'GM', 'DE': 'DE', 'SW': 'SW', 'HS': 'HS', 'LH': 'LH'}
GRID = np.round(np.arange(-0.5, 1.2 + 1e-9, 1 / FPS), 4)        # DE の形: 入る 0.5 秒前 〜 入って 1.2 秒
SW_GRID = np.round(np.arange(-1.5, 0.6 + 1e-9, 1 / FPS), 4)      # SW の形: 出口の 1.5 秒前 〜 出口 +0.6 秒


def session_of(log):
    hh = int(log.split('-')[1][:2])
    return 1 if hh < 15 else (2 if hh < 17 else 3)


def load_raw(log):
    """シーンをつないだ AI_CMD(30 fps)と VEHICLE(100 Hz)・IMU(100 Hz)・カメラのファイル"""
    scenes = sorted(p for p in (RAW / log).iterdir() if p.is_dir())
    ai = {k: [] for k in ('t', 'out', 'model_out', 'raw', 'select')}
    ve = {k: [] for k in ('t', 'erpm')}
    im = {k: [] for k in ('t', 'gz')}
    cam = {'t': [], 'path': []}
    for s in scenes:
        try:
            with np.load(s / 'AI_CMD.npz') as z:
                ai['t'].append(z['timestamps_us']); ai['out'].append(z['out']); ai['model_out'].append(z['model_out'])
                ai['raw'].append(z['raw'])
                ai['select'].append(z['select'])
        except Exception:
            pass
        try:
            with np.load(s / 'VEHICLE.npz') as z:
                ve['t'].append(z['timestamps_us']); ve['erpm'].append(z['erpm'])
        except Exception:
            pass
        try:
            with np.load(s / 'IMU.npz') as z:
                im['t'].append(z['timestamps_us']); im['gz'].append(z['gyro'][:, 2])
        except Exception:
            pass
        try:
            with np.load(s / 'CAM_FRONT.npz') as z:
                cam['t'].append(z['timestamps_us'])
                cam['path'] += [str(s / f) for f in z['files']]
        except Exception:
            pass
    cat = lambda d: {k: (np.concatenate(v) if v and not isinstance(v[0], str) else v) for k, v in d.items()}
    return cat(ai), cat(ve), cat(im), {'t': np.concatenate(cam['t']) if cam['t'] else np.zeros(0), 'path': cam['path']}


def win(t, x, a, b):
    m = (t >= a) & (t <= b)
    return x[m]


def med(xs, nd=1):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return round(float(np.median(xs)), nd) if xs else None


def q(xs, p, nd=1):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return round(float(np.percentile(xs, p)), nd) if xs else None


def segments(mask, min_len):
    e = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return [(int(a), int(b)) for a, b in zip(e[::2], e[1::2]) if b - a >= min_len]


def track_from_anchor(L, hd_f, t_anchor, t_entry, k_end, it, hd_i):
    """R1 の起点から k_end までの推測の位置 [m](速さ × 向き。起点の向き = +x、+y = 左)。起点が無ければ None"""
    if t_anchor is None or t_entry - t_anchor > 8.0 or not len(it):
        return None
    t = L['t_s']
    k0 = int(np.searchsorted(t, t_anchor))
    if k_end <= k0 + 2:
        return None
    tt = t[k0:k_end + 1]
    psi = np.deg2rad(hd_f[k0:k_end + 1] - float(np.interp(t_anchor, it, hd_i)))
    v = L['speed_mps'][k0:k_end + 1].astype(float)
    dt = np.diff(tt, prepend=tt[0])
    dt[(dt <= 0) | (dt > 0.2)] = 0.0
    x = np.cumsum(v * np.cos(psi) * dt)
    y = np.cumsum(v * np.sin(psi) * dt)
    rel = tt - t_entry
    return {'rel_s': [round(float(r), 3) for r in rel], 'x': [round(float(u), 3) for u in x], 'y': [round(float(u), 3) for u in y]}


def main():
    sc = json.loads(SC.read_text(encoding='utf-8'))
    events, de_passes, sw_passes, logs_info = [], [], [], {}
    for log, p in sorted(sc['per_log'].items()):
        lj = json.loads((LAB / (log + '.json')).read_text(encoding='utf-8'))
        center = float(lj['ref_us']['steer_center'])
        neutral = float(lj['ref_us']['throttle_neutral'])
        t0 = int(lj['t0_us'])
        prm = p.get('params_at_start') or {}
        ai_no = p['desc'].get('ai_no')
        sess = session_of(log)
        model = p.get('run') or ''
        short = {'clean_driving_20261011-nocmd-s3-20261011-024626': 'B', 'bench_20261022-slow-sec-s1': '遅く入る型',
                 'bench_20261021-ens-p2': '2本の平均'}.get(model, model)
        setting = '%s 速さ %.2f ステア %.2f' % (short, prm.get('throttle_gain', 1.0), prm.get('steer_gain', 1.0))
        logs_info[log] = {'ai_no': ai_no, 'session': sess, 'model': short, 'setting': setting,
                          'throttle_gain': prm.get('throttle_gain'), 'steer_gain': prm.get('steer_gain'),
                          'lap_equiv': p.get('lap_equiv'), 'ai_s': p.get('ai_s'), 'center': center, 'neutral': neutral}
        ai, ve, im, cam = load_raw(log)
        vt = (ve['t'] - t0) / 1e6 if len(ve['t']) else np.zeros(0)
        v_mps = -ve['erpm'].astype(float) * KV if len(ve['t']) else np.zeros(0)
        at = (ai['t'] - t0) / 1e6 if len(ai['t']) else np.zeros(0)
        it = (im['t'] - t0) / 1e6 if len(im['t']) else np.zeros(0)
        logs_info[log]['cam_t_s'] = None
        lab_ev = lj['events']['collisions']
        # ---- 衝突 1 件ずつ
        for e in p['collision_events']:
            m = [c for c in lab_ev if abs(c['t_s'] - e['t_s']) < 0.02]
            c = m[0] if m else {}
            t = e['t_s']
            st = win(at, ai['out'][:, 0], t - 0.5, t - 0.034) - center if len(at) else np.zeros(0)
            st_m = win(at, ai['model_out'][:, 0], t - 0.5, t - 0.034) - center if len(at) else np.zeros(0)
            th = neutral - win(at, ai['out'][:, 1], t - 0.5, t - 0.034) if len(at) else np.zeros(0)
            th_m = neutral - win(at, ai['model_out'][:, 1], t - 0.5, t - 0.034) if len(at) else np.zeros(0)
            th_r = neutral - win(at, ai['raw'][:, 1], t - 0.5, t - 0.034) if len(at) else np.zeros(0)
            vv = win(vt, v_mps, t - 0.5, t - 0.02)
            vmax1 = win(vt, v_mps, t - 1.5, t - 0.02)
            gz = win(it, im['gz'], t - 0.5, t - 0.05) if len(it) else np.zeros(0)
            events.append({
                'log': log, 'ai_no': ai_no, 'session': sess, 'setting': setting, 't_s': t, 't_us': c.get('t_us'),
                'frame': c.get('frame'), 'loc': DETAIL_TO.get(e['loc_detail'], e['loc_detail']),
                'loc_detail': e['loc_detail'], 'class': e['class'], 'g': e['g'], 'strong': e['strong'],
                'side': c.get('side'), 'lat_g': c.get('lat_g'), 'lon_g': c.get('lon_g'),
                'speed_before_mps': e.get('speed_before_mps'),
                'v_max_0p5_mps': round(float(vv.max()), 2) if len(vv) else None,
                'v_max_1p5_mps': round(float(vmax1.max()), 2) if len(vmax1) else None,
                'steer_last_us': round(float(st[-1]), 1) if len(st) else None,
                'steer_min_us': round(float(st.min()), 1) if len(st) else None,
                'steer_max_us': round(float(st.max()), 1) if len(st) else None,
                'steer_model_last_us': round(float(st_m[-1]), 1) if len(st_m) and np.isfinite(st_m[-1]) else None,
                'thr_last_us': round(float(th[-1]), 1) if len(th) else None,
                'thr_mean_us': round(float(th.mean()), 1) if len(th) else None,
                'thr_model_mean_us': round(float(np.nanmean(th_m)), 1) if np.isfinite(th_m).any() else None,
                'thr_raw_mean_us': round(float(np.nanmean(th_r)), 1) if np.isfinite(th_r).any() else None,
                'steer_model_minus_out_us': round(float(np.nanmax(np.abs(st_m - st))), 1) if np.isfinite(st_m).any() else None,
                'yaw_mean_dps': round(float(gz.mean()), 1) if len(gz) else None,
            })
        # ---- 通過(ラベル)
        with np.load(LAB / (log + '.npz')) as z:
            L = {k: z[k] for k in ('t_s', 't_us', 'loc_code', 'loc_detail', 'out', 'model_out', 'ai', 'collision',
                                   'collision_g', 'speed_mps', 'lap_id')}
        n = len(L['t_s'])
        steer = L['out'][:, 0] - center
        thr = neutral - L['out'][:, 1]
        coll_ai = L['collision'] & L['ai']
        gz_f = np.interp(L['t_s'], it, im['gz']) if len(it) else np.zeros(n)
        # 向き(label_events.heading_deg と同じ: 止まっているときのジャイロの中央値をずれとして引いて積分)
        if len(it) > 1:
            ie = np.interp(it, vt, ve['erpm']) if len(vt) else np.zeros(len(it))
            still = (np.abs(ie) < 50) & (np.abs(im['gz']) < 5)
            bias = float(np.median(im['gz'][still])) if still.sum() > 50 else -0.09
            dti = np.diff(it, prepend=it[0])
            dti[(dti <= 0) | (dti > 0.2)] = 0.0
            hd_i = np.cumsum((im['gz'] - bias) * dti)
        else:
            hd_i = np.zeros(len(it))
        hd_f = np.interp(L['t_s'], it, hd_i) if len(it) else np.zeros(n)
        anchors = np.array(sorted(pp['ta_s'] for pp in p['passes'])) if p['passes'] else np.zeros(0)
        cam_t = (cam['t'] - t0) / 1e6 if len(cam['t']) else np.zeros(0)

        def cam_at(ts):
            if not len(cam_t):
                return None
            j = int(np.argmin(np.abs(cam_t - ts)))
            return {'path': cam['path'][j], 'dt_s': round(float(cam_t[j] - ts), 3), 't_s': round(float(cam_t[j]), 3)}

        for a, b in segments(L['loc_code'] == 3, 30):
            if not L['ai'][a:b].all():
                continue
            b2 = min(n, b + 9)
            h = np.flatnonzero(coll_ai[a:b2])
            hit = bool(len(h))
            k_hit = a + int(h[0]) if hit else None
            ta = float(L['t_s'][a])
            rel = L['t_s'] - ta
            # 形: 入る 0.5 秒前から(衝突の後は NaN)
            lo = max(0, a - 16)
            hi = min(n, a + 38)
            end = k_hit if hit else hi
            tt, ss = rel[lo:end], steer[lo:end]
            shape = np.interp(GRID, tt, ss, left=np.nan, right=np.nan) if len(tt) > 1 else np.full(len(GRID), np.nan)
            vshape = np.interp(GRID, tt, L['speed_mps'][lo:end], left=np.nan, right=np.nan) if len(tt) > 1 else np.full(len(GRID), np.nan)
            # 向き(ジャイロの積分、入ったときを 0。+ = 左)
            gzz = gz_f[lo:end]
            psi = np.r_[0, np.cumsum(0.5 * (gzz[1:] + gzz[:-1]) * np.diff(tt))] if len(tt) > 1 else np.zeros(len(tt))
            if len(tt) > 1:
                psi = psi - np.interp(0.0, tt, psi)
            pshape = np.interp(GRID, tt, psi, left=np.nan, right=np.nan) if len(tt) > 1 else np.full(len(GRID), np.nan)
            ka = np.searchsorted(anchors, ta, side='right') - 1
            if ka >= 0 and ta - anchors[ka] < 8.0:
                h_a = float(np.interp(anchors[ka], it, hd_i))
                cshape = np.interp(GRID, tt, hd_f[lo:end] - h_a, left=np.nan, right=np.nan) if len(tt) > 1 else np.full(len(GRID), np.nan)
                t_from_anchor = round(ta - float(anchors[ka]), 3)
            else:
                cshape = np.full(len(GRID), np.nan)
                t_from_anchor = None
            r0 = (rel[a:end] >= 0) & (rel[a:end] <= 0.5)
            r1 = (rel[a:end] >= 0.6) & (rel[a:end] <= 1.0)
            s_in = steer[a:end]
            k_os = int(np.argmin(np.where(r0, s_in, np.inf))) if r0.any() else None
            dsteer = np.diff(s_in) * FPS if len(s_in) > 1 else np.zeros(0)
            de_passes.append({
                'log': log, 'ai_no': ai_no, 'session': sess, 'setting': setting, 't_s': round(ta, 3),
                'frames': b - a, 'hit': hit, 'hit_rel_s': round(float(rel[k_hit]), 3) if hit else None,
                'hit_g': round(float(L['collision_g'][k_hit]), 2) if hit else None,
                'entry_v_mps': round(float(L['speed_mps'][a]), 2),
                'vmax_0_1p2_mps': round(float(np.nanmax(vshape[GRID >= 0])), 2) if np.isfinite(vshape[GRID >= 0]).any() else None,
                'gm_exit_steer_us': round(float(np.nanmean(shape[(GRID >= -0.5) & (GRID < -0.2)])), 1),
                'steer_at0_us': round(float(steer[a]), 1),
                'overshoot_us': round(float(s_in[r0].min()), 1) if r0.any() else None,
                'overshoot_t_s': round(float(rel[a + k_os]), 3) if k_os is not None else None,
                'late_left_us': round(float(s_in[r1].max()), 1) if r1.any() else None,
                'swing_us': round(float(s_in[r1].max() - s_in[r0].min()), 1) if (r0.any() and r1.any()) else None,
                'max_rate_left_us_s': round(float(dsteer[(rel[a:end - 1] >= 0.3)].max()), 0) if len(dsteer) and (rel[a:end - 1] >= 0.3).any() else None,
                'psi_min_deg': round(float(np.nanmin(pshape[GRID >= 0])), 1) if np.isfinite(pshape[GRID >= 0]).any() else None,
                'psi_at_0p8_deg': round(float(pshape[np.argmin(np.abs(GRID - 0.8))]), 1) if np.isfinite(pshape[np.argmin(np.abs(GRID - 0.8))]) else None,
                'shape': [None if not np.isfinite(x) else round(float(x), 1) for x in shape],
                'vshape': [None if not np.isfinite(x) else round(float(x), 2) for x in vshape],
                'pshape': [None if not np.isfinite(x) else round(float(x), 1) for x in pshape],
                'track': track_from_anchor(L, hd_f, anchors[ka] if ka >= 0 else None, ta, k_hit if hit else min(n - 1, a + 36), it, hd_i),
                'cshape': [None if not np.isfinite(x) else round(float(x), 1) for x in cshape],
                'psi_course_entry_deg': None if not np.isfinite(cshape[15]) else round(float(cshape[15]), 1),
                't_from_anchor_s': t_from_anchor,
                'frame_a': a, 'frame_hit': k_hit,
                'cam_hit_m0p2': cam_at(float(L['t_s'][k_hit]) - 0.2) if hit else None,
                'cam_rel': {('%.1f' % x): cam_at(ta + x) for x in (0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9)},
            })
        # SW: 出口でそろえる
        for a, b in segments(L['loc_detail'] == 5, 15):
            if not L['ai'][a:b].all():
                continue
            tb = float(L['t_s'][b - 1])
            b2 = min(n, b + 18)                    # 出口から 0.6 秒
            h = np.flatnonzero(coll_ai[a:b2])
            hit = bool(len(h))
            k_hit = a + int(h[0]) if hit else None
            rel = L['t_s'] - tb
            lo = max(0, b - 1 - 45)
            hi = min(n, b - 1 + 19)
            end = k_hit if hit else hi
            tt = rel[lo:end]
            shape = np.interp(SW_GRID, tt, steer[lo:end], left=np.nan, right=np.nan) if len(tt) > 1 else np.full(len(SW_GRID), np.nan)
            vshape = np.interp(SW_GRID, tt, L['speed_mps'][lo:end], left=np.nan, right=np.nan) if len(tt) > 1 else np.full(len(SW_GRID), np.nan)
            tshape = np.interp(SW_GRID, tt, thr[lo:end], left=np.nan, right=np.nan) if len(tt) > 1 else np.full(len(SW_GRID), np.nan)
            vin = L['speed_mps'][a:b]
            sw_passes.append({
                'log': log, 'ai_no': ai_no, 'session': sess, 'setting': setting, 't_exit_s': round(tb, 3),
                'dur_s': round(float(L['t_s'][b - 1] - L['t_s'][a]), 2), 'hit': hit,
                'hit_rel_exit_s': round(float(rel[k_hit]), 3) if hit else None,
                'hit_loc': int(L['loc_detail'][k_hit]) if hit else None,
                'hit_g': round(float(L['collision_g'][k_hit]), 2) if hit else None,
                'v_max_mps': round(float(vin.max()), 2), 'v_exit_mps': round(float(L['speed_mps'][b - 1]), 2),
                'steer_mean_us': round(float(steer[a:b].mean()), 1),
                'steer_last0p5_us': round(float(steer[max(a, b - 15):b].mean()), 1),
                'thr_mean_us': round(float(thr[a:b].mean()), 1),
                'shape': [None if not np.isfinite(x) else round(float(x), 1) for x in shape],
                'vshape': [None if not np.isfinite(x) else round(float(x), 2) for x in vshape],
                'tshape': [None if not np.isfinite(x) else round(float(x), 1) for x in tshape],
                'frame_a': a, 'frame_exit': b - 1, 'frame_hit': k_hit,
                'cam_hit_m0p2': cam_at(float(L['t_s'][k_hit]) - 0.2) if hit else None,
                'cam_rel_exit': {('%.1f' % x): cam_at(tb + x) for x in (-0.6, -0.3, 0.0, 0.2)},
            })
    res = {'source': {'scorecard': str(SC), 'labels': str(LAB), 'raw': str(RAW)}, 'grid_de': GRID.tolist(),
           'grid_sw': SW_GRID.tolist(), 'logs': logs_info, 'events': events, 'de_passes': de_passes, 'sw_passes': sw_passes}
    (OUT / 'col_data.json').write_text(json.dumps(res, ensure_ascii=False, indent=0), encoding='utf-8')
    print('events', len(events), 'de', len(de_passes), 'sw', len(sw_passes))


if __name__ == '__main__':
    main()
