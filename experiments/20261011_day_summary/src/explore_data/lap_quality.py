#!/usr/bin/env python3
"""周ごとの品質表(人・AI)— explore_data.md の根拠(読み取りのみ、CPU)

    source ~/miniconda/etc/profile.d/conda.sh && conda activate jetracer
    nice -n 15 python lap_quality.py > lap_quality.json

生ログ ~/jetformer/data/raw/<log>/ の IMU・VEHICLE・AI_CMD だけを読む(画像は読まない)。
定義は collision_map.md(cmap_src)にそろえた:
  - 向き: gyro z(deg/s)から静止時の中央値を引いて積分(0.2秒を超える途切れは積分しない)
  - R1 の起点: 平滑化したヨーレート(11サンプル平均)が −30 deg/s を下回る区間のうち、
    最小 ≤ −120、向きの変化 ≤ −40、直前 2秒に 90度超の左旋回(または区間の変化 ≤ −90)、
    直前 1.2〜0.1秒に 60度超の右旋回がない(cmap は 3〜0.6秒前。夜の人のログで R1 を落とすので狭めた)、前の起点から 5秒以上。さらに 1.5秒以内のヨー最小 ≤ −170 かつ
    (深さ ≤ −90度 または 1秒以内に衝突)
  - R1 の進入速度: 起点の 0.4秒前〜0.15秒後の最大の前進 eRPM
  - 緑マット出口(tg): 起点から 2秒以内の向きの最小(≤ −90)の後、4.5秒以内に −25度まで戻った時刻(cmap は 3.5秒)
  - DE の向き: tg+0.6秒の向き − 起点の向き(+ = 左)
  - 衝突: 水平 hypot(ax, ay) ≥ 2.5G、0.6秒以内はまとめる、±50ms に 1G 超が1サンプルだけのもの(跳ね)は除く。≥4G を強い衝突
  - 衝突の場所: R1 = [起点−0.05, 起点+1.0)、GM = [起点+1.0, tg+0.3)、DE = [tg+0.3, tg+2.0)、その他 = 周の残り
  - 周: 起点から次の起点まで(6〜14秒、IMU の途切れ ≤ 0.25秒、向きの変化 300〜420度)
  - 区間の時間: R1+GM = 起点→tg、DE+SW = tg→向き +150度(tg+1秒以降で最初)、HS+LH = +150度→次の起点(向きは起点基準の目安)
  - 速度: |eRPM| × 0.5e-3 m/s(65mm タイヤ、空転を含む目安)
"""

import json
import os
import sys

import numpy as np

RAW = '/home/jetson/jetformer/data/raw'
G = 9.80665

HUMAN = {  # ログ: (グループ, ジャンル)
    '20261010-135449': ('afternoon', 5), '20261010-144028': ('afternoon', 5), '20261010-150424': ('afternoon', 5),
    '20261010-152405': ('afternoon', 5), '20261010-152707': ('afternoon', 5),
    '20261010-193638': ('evening', 4), '20261010-200916': ('evening', 4), '20261010-201626': ('evening', 5),
    '20261010-202633': ('evening', 4),
}
AI = {'20261010-171639': 'ai_run_003', '20261010-172910': 'ai_run_006', '20261010-173445': 'ai_run_007',
      '20261010-181504': 'ai_run_008', '20261010-203924': 'ai_run_009', '20261010-204208': 'ai_run_010',
      '20261010-204605': 'ai_run_101'}
# clean_driving_20261011 に入っているシーン(dataset_card.json の source_scenes)
CD = json.load(open('/home/jetson/jetformer/data/datasets/clean_driving_20261011/dataset_card.json'))['source_scenes']
CD_SPLIT = json.load(open('/home/jetson/jetformer/data/datasets/clean_driving_20261011/splits.json'))


def load(log):
    root = os.path.join(RAW, log)
    scenes = sorted(s for s in os.listdir(root) if s.startswith(log + '-'))
    parts = {k: [] for k in ('IMU', 'VEHICLE', 'AI_CMD', 'CAM')}
    sc_ranges = []
    for s in scenes:
        try:
            imu = np.load(os.path.join(root, s, 'IMU.npz'))
            veh = np.load(os.path.join(root, s, 'VEHICLE.npz'))
            cam = np.load(os.path.join(root, s, 'CAM_FRONT.npz'), allow_pickle=True)
            it, acc, gyr = imu['timestamps_us'], imu['accel'], imu['gyro']
            vt, pwm, erpm, sel = veh['timestamps_us'], veh['pwm'], veh['erpm'], veh['select']
            ct = cam['timestamps_us']
            ai = None
            p = os.path.join(root, s, 'AI_CMD.npz')
            if os.path.exists(p):
                z = np.load(p)
                ai = (z['timestamps_us'], z['select'], z['out'], z['raw'], z['gains'], z['lookahead'])
        except Exception:
            continue
        parts['IMU'].append((it, acc, gyr))
        parts['VEHICLE'].append((vt, pwm, erpm, sel))
        parts['CAM'].append(ct)
        if ai is not None:
            parts['AI_CMD'].append(ai)
        sc_ranges.append((s, int(ct[0]) if len(ct) else int(it[0]), int(ct[-1]) if len(ct) else int(it[-1])))
    it = np.concatenate([p[0] for p in parts['IMU']]).astype(np.int64)
    t0 = it[0]
    d = {'log': log, 't0_us': int(t0)}
    d['it'] = (it - t0) / 1e6
    d['acc'] = np.concatenate([p[1] for p in parts['IMU']]).astype(float)
    d['gyr'] = np.concatenate([p[2] for p in parts['IMU']]).astype(float)
    d['vt'] = (np.concatenate([p[0] for p in parts['VEHICLE']]).astype(np.int64) - t0) / 1e6
    d['pwm'] = np.concatenate([p[1] for p in parts['VEHICLE']]).astype(float)
    d['erpm'] = np.concatenate([p[2] for p in parts['VEHICLE']]).astype(float)
    d['vsel'] = np.concatenate([p[3] for p in parts['VEHICLE']]).astype(bool)
    d['ct'] = (np.concatenate(parts['CAM']).astype(np.int64) - t0) / 1e6
    d['scenes'] = [(s, (a - t0) / 1e6, (b - t0) / 1e6) for s, a, b in sc_ranges]
    if parts['AI_CMD']:
        d['at'] = (np.concatenate([p[0] for p in parts['AI_CMD']]).astype(np.int64) - t0) / 1e6
        d['asel'] = np.concatenate([p[1] for p in parts['AI_CMD']]).astype(bool)
        d['out'] = np.concatenate([p[2] for p in parts['AI_CMD']]).astype(float)
        d['raw'] = np.concatenate([p[3] for p in parts['AI_CMD']]).astype(float)
        d['gains'] = np.concatenate([p[4] for p in parts['AI_CMD']]).astype(float)
        d['la'] = np.concatenate([p[5] for p in parts['AI_CMD']]).astype(int)
    return d


def heading(d):
    gz = d['gyr'][:, 2]
    er = np.interp(d['it'], d['vt'], d['erpm'])
    m = (np.abs(er) < 1) & (np.abs(gz) < 5)
    bias = float(np.median(gz[m])) if m.sum() > 50 else -0.0915
    dt = np.diff(d['it'], prepend=d['it'][0])
    dt = np.where((dt > 0) & (dt <= 0.2), dt, 0.0)
    d['bias'] = bias
    d['gz'] = gz - bias
    d['hd'] = np.cumsum(d['gz'] * dt)
    d['gs'] = np.convolve(d['gz'], np.ones(11) / 11, 'same')


def impacts(d, thr=2.5):
    hz = np.hypot(d['acc'][:, 0], d['acc'][:, 1]) / G
    ev = []
    for i in np.nonzero(hz >= thr)[0]:
        t = d['it'][i]
        if ev and t - ev[-1]['t'] <= 0.6:
            if hz[i] > ev[-1]['g']:
                ev[-1]['g'] = float(hz[i])
            continue
        near = (np.abs(d['it'] - t) <= 0.05) & (hz > 1.0)
        ev.append({'t': float(t), 'g': float(hz[i]), 'spike': bool(near.sum() <= 1)})
    ev = [e for e in ev if not e['spike']]
    for e in ev:
        j = min(np.searchsorted(d['vt'], e['t']), len(d['vt']) - 1)
        e['ai'] = bool(d['vsel'][j])
    d['ev'] = ev
    return ev


def rsegs(d, thr=-30):
    neg = d['gs'] < thr
    out, i, n = [], 0, len(neg)
    while i < n:
        if neg[i]:
            j = i
            while j < n and neg[j]:
                j += 1
            out.append((d['it'][i], d['it'][j - 1], d['hd'][j - 1] - d['hd'][i], d['gs'][i:j].min()))
            i = j
        else:
            i += 1
    return out


def r1_anchors(d):
    it, hd = d['it'], d['hd']
    out = []
    for (a, b, dh, mn) in rsegs(d):
        if mn > -120 or dh > -40:
            continue
        h_prev = np.interp(a, it, hd) - np.interp(a - 2.0, it, hd)
        m = (it >= a - 1.2) & (it < a - 0.1)   # cmap は 3.0〜0.6秒前。人の夜のログでは LH 手前の右の切り返しで R1 を落としたので狭めた
        hw = hd[m]
        drop = np.min(hw - np.maximum.accumulate(hw)) if m.any() else 0
        if drop < -60:
            continue
        if h_prev > 90 or dh < -90:
            out.append(a)
    # 深さ・ヨーの条件を先に当て、そのあと 5秒の間隔をとる(cmap は間隔を先にとっていたので、
    # 浅い右の切り返しのすぐ後の本物の R1 を落とすことがあった)
    res = []
    for a in out:
        m = (it >= a) & (it < a + 1.5)
        depth = float(np.min(hd[m] - np.interp(a, it, hd)))
        ymin = float(d['gs'][m].min())
        hit = [e for e in d['ev'] if a - 0.05 <= e['t'] < a + 1.0]
        if ymin <= -170 and (depth <= -90 or hit):
            if res and a - res[-1] < 5.0:
                continue
            res.append(a)
    return res


def first_cross(t, h, thr, t_from):
    m = np.nonzero((t >= t_from) & (h >= thr))[0]
    return float(t[m[0]]) if len(m) else None


def lap_rows(d, ai_log):
    it, hd, vt = d['it'], d['hd'], d['vt']
    fwd = -d['erpm']
    an = r1_anchors(d)
    passes = []
    for k, a in enumerate(an):
        h0 = np.interp(a, it, hd)
        m = (it >= a) & (it < a + 4.5)
        hr, tt = hd[m] - h0, it[m]
        P = {'a': round(float(a), 2)}
        mv = (vt >= a - 0.4) & (vt < a + 0.15)
        P['v_in'] = float(np.max(fwd[mv])) if mv.any() else None
        P['r1_hits'] = [round(e['g'], 1) for e in d['ev'] if a - 0.05 <= e['t'] < a + 1.0]
        tg = None
        if len(hr):
            imin = int(np.argmin(np.where(tt < a + 2.0, hr, np.inf)))
            if hr[imin] <= -90:
                j = np.nonzero(hr[imin:] >= -25)[0]
                if len(j):
                    tg = float(tt[imin + j[0]])
        P['tg'] = tg
        if tg is not None:
            P['de_psi'] = float(np.interp(tg + 0.6, it, hd) - h0)
            mm = (vt >= tg) & (vt < tg + 0.8)
            P['de_v'] = float(np.mean(fwd[mm])) if mm.any() else None
            mm = (vt >= tg) & (vt < tg + 2.0)
            P['de_vmax'] = float(np.max(fwd[mm])) if mm.any() else None
            P['gm_hits'] = [round(e['g'], 1) for e in d['ev'] if a + 1.0 <= e['t'] < tg + 0.3]
            P['de_hits'] = [round(e['g'], 1) for e in d['ev'] if tg + 0.3 <= e['t'] < tg + 2.0]
            P['de_psi12'] = float(np.interp(tg + 1.2, it, hd) - h0)
            mm = (it >= tg + 0.2) & (it < tg + 0.8)
            P['de_yaw'] = float(np.mean(d['gz'][mm])) if mm.any() else None
            # 緑マット出口の後の右への戻し(向きの最小、tg〜tg+1.5)
            mm = (it >= tg) & (it < tg + 1.5)
            P['de_psi_min'] = float(np.min(hd[mm]) - h0) if mm.any() else None
        # AI 側か: R1 は 2.5秒前〜1.0秒後、DE は R1 の 0.5秒前〜tg+2.0 がずっと AI
        if ai_log:
            mm = (vt >= a - 2.5) & (vt < a + 1.0)
            P['ai'] = bool(mm.any() and d['vsel'][mm].all())
            if tg is not None:
                mm = (vt >= a - 0.5) & (vt < tg + 2.0)
                P['ai_de'] = bool(mm.any() and d['vsel'][mm].all())
            else:
                P['ai_de'] = False
            ma = (d['at'] >= a - 0.5) & (d['at'] < a + 0.5)
            if ma.any():
                P['gain'] = [round(float(x), 2) for x in np.median(d['gains'][ma], axis=0)]
                P['la'] = int(np.median(d['la'][ma]))
        else:
            P['ai'] = False
            P['ai_de'] = False
        passes.append(P)
    laps = []
    for k in range(len(an) - 1):
        a, b = an[k], an[k + 1]
        dur = b - a
        m = (it >= a) & (it <= b)
        gaps = np.diff(it[m])
        dh = np.interp(b, it, hd) - np.interp(a, it, hd)
        L = {'a': round(float(a), 2), 'time_s': round(float(dur), 2), 'dh': round(float(dh))}
        L['valid'] = bool(6.0 <= dur <= 14.0 and (len(gaps) == 0 or gaps.max() <= 0.25) and 300 <= dh <= 420)
        evs = [e for e in d['ev'] if a - 0.05 <= e['t'] < b - 0.05]
        L['impacts'] = len(evs)
        L['strong'] = sum(1 for e in evs if e['g'] >= 4.0)
        P = passes[k]
        L['v_in'] = P['v_in']
        L['de_psi'] = P.get('de_psi')
        L['r1_hit'] = bool(P['r1_hits'])
        L['gm_hit'] = bool(P.get('gm_hits'))
        L['de_hit'] = bool(P.get('de_hits'))
        tg = P['tg']
        L['other_hit'] = sum(1 for e in evs if not (a - 0.05 <= e['t'] < a + 1.0) and not (tg is not None and a + 1.0 <= e['t'] < tg + 2.0))
        if ai_log:
            mm = (vt >= a) & (vt <= b)
            L['ai_only'] = bool(mm.any() and d['vsel'][mm].all())
            ma = (d['at'] >= a) & (d['at'] < b)
            if ma.any():
                L['gain'] = [round(float(x), 2) for x in np.median(d['gains'][ma], axis=0)]
                L['la'] = int(np.median(d['la'][ma]))
        # 区間の時間
        hr = hd[m] - np.interp(a, it, hd)
        tt = it[m]
        sec = None
        if tg is not None and L['valid']:
            t150 = first_cross(tt, hr, 150.0, tg + 1.0)
            if t150 and t150 < b:
                sec = {'R1GM': tg - a, 'DESW': t150 - tg, 'HSLH': b - t150}
                spd = {}
                for nm, (x, y) in {'R1GM': (a, tg), 'DESW': (tg, t150), 'HSLH': (t150, b)}.items():
                    mm = (vt >= x) & (vt < y)
                    spd[nm] = float(np.median(fwd[mm])) if mm.any() else None
                L['sector_s'] = {k2: round(v, 3) for k2, v in sec.items()}
                L['sector_erpm'] = {k2: (round(v) if v is not None else None) for k2, v in spd.items()}
            mm = (vt >= a) & (vt < b)
            L['erpm_p50'] = float(np.percentile(fwd[mm], 50))
            L['erpm_p90'] = float(np.percentile(fwd[mm], 90))
            dtv = np.diff(vt[mm], prepend=vt[mm][0])
            L['dist_m'] = float(np.sum(np.clip(fwd[mm], 0, None) * np.clip(dtv, 0, 0.05)) * 0.5e-3)
        # どのシーンか(clean_driving に入っているか)
        scn = [s for s, x, y in d['scenes'] if x - 0.05 <= a <= y + 0.05]
        L['scene'] = scn[0] if scn else None
        L['in_clean_driving'] = bool(scn and scn[0] in CD)
        L['split'] = ('val' if scn and scn[0] in CD_SPLIT['val'] else 'train' if scn and scn[0] in CD_SPLIT['train'] else None)
        laps.append(L)
    return an, passes, laps


def frame_budget(d, an, passes):
    """カメラフレームを場所で分ける(動いているフレームのみ): R1 = 起点−1.0〜+1.0秒、GM = +1.0〜tg+0.3、DE = tg+0.3〜tg+2.0"""
    ct = d['ct']
    fwd = np.interp(ct, d['vt'], -d['erpm'])
    moving = np.abs(fwd) > 200
    zone = np.array(['other'] * len(ct), dtype=object)
    for P in passes:
        a = P['a']
        zone[(ct >= a - 1.0) & (ct < a + 1.0)] = 'R1'
        if P['tg'] is not None:
            zone[(ct >= a + 1.0) & (ct < P['tg'] + 0.3)] = 'GM'
            zone[(ct >= P['tg'] + 0.3) & (ct < P['tg'] + 2.0)] = 'DE'
    out = {}
    for z in ('R1', 'GM', 'DE', 'other'):
        out[z] = int(np.sum((zone == z) & moving))
    out['moving'] = int(moving.sum())
    # clean_driving のシーンだけ
    incd = np.zeros(len(ct), bool)
    for s, x, y in d['scenes']:
        if s in CD:
            incd |= (ct >= x) & (ct <= y)
    out['cd'] = {z: int(np.sum((zone == z) & moving & incd)) for z in ('R1', 'GM', 'DE', 'other')}
    return out


def main():
    res = {'definitions': __doc__, 'logs': {}}
    for log in list(HUMAN) + list(AI):
        d = load(log)
        heading(d)
        impacts(d)
        an, passes, laps = lap_rows(d, log in AI)
        fb = frame_budget(d, an, passes)
        res['logs'][log] = {'kind': 'human' if log in HUMAN else 'ai',
                            'group': HUMAN[log][0] if log in HUMAN else AI[log],
                            'genre': HUMAN[log][1] if log in HUMAN else 5,
                            'bias_dps': round(d['bias'], 4), 'anchors': len(an), 't0_us': d['t0_us'],
                            'passes': passes, 'laps': laps, 'frames': fb,
                            'impacts': len(d['ev'])}
        print(log, 'anchors', len(an), 'laps', len(laps), 'valid', sum(L['valid'] for L in laps), file=sys.stderr)
    json.dump(res, sys.stdout, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
