#!/usr/bin/env python3
"""解析 B(2026-10-25): 10/24 の AI 走行 No.1〜20 で「同じ設定(B・速さ 0.90・ステア 1.0)が 3 回目に再現しなかった」理由を、
電池・明るさ・後輪の修理の 3 つの印で数にする。読むだけ(numpy・PIL)。生ログ・ラベル・設定は変えない。

    jf-heavy --lock none --mem 1500M -- python3 -I cond_metrics.py

出力: /home/jetson/jetformer_reports/bench_20261024_report/cond_data/cond_metrics.json(走行ごと・区間ごと・時間の並び)
      /home/jetson/jetformer_reports/bench_20261024_report/cond_data/frames_<log>.npz(フレームごとの明るさなど。図用)

印の定義(フレーム = カメラの 30 fps。AI 側・dry run でない・中立止め(hold)でない・補助(assist)なし):
  前進の指令 cmd = 1496 − 車に出したスロットル out(µs)。モデルの指令 raw_cmd = 1496 − モデルの出力 raw(ゲインの前)
  直線 = ラベルの場所 DE(GM の出口の後の直線)・HS(奥の直線)
  落ち着いたフレーム = 直前 0.5 秒の cmd の平均 > 60 µs・ばらつき(標準偏差)< 30 µs・0.3 秒で eRPM の変化 < 500・前進 eRPM > 1500
  (a) 電池の効き(そのまま)= 前進 eRPM ÷ 直前 0.5 秒の cmd の平均(落ち着いた直線のフレームの中央値)
  (a') 電池の効き(指令をそろえた)= 前進 eRPM ÷ f(cmd)。f は 20 本をまとめた直線の当てはめ(eRPM = a·cmd + b)。
       1.00 = 20 本の平均の効き。指令が大きいほど (a) が下がる(切片がある)ので、指令の違う走行の比べには (a') を使う
  (b) 直線の最高速 = AI 中(前進 > 300 eRPM)の前進 eRPM の 90% 点。HS だけの 90% 点も出す
  (c) 1 本の中の変化 = AI の時間 20 秒ごとの (a') と (b) の並び、AI の時間 1 分あたりの傾き(最小二乗)
  明るさ = カメラ画像の輝度 Y(0.299R+0.587G+0.114B、0〜255)の平均、コントラスト = Y の標準偏差(AI のフレーム 2 枚に 1 枚)
  直線のヨーの偏り = 直線(DE・HS)で前進 > 3,000 eRPM・直前 0.2 秒の舵の平均が中央 ±25 µs・今の舵が ±40 µs のときの
       ジャイロ gz(止まっているときの中央値を引いたもの、+ = 左)の平均 [deg/s]
  まっすぐ進む舵の位置 s0 = 直線で |舵| < 150 µs・速さ > 1.5 m/s の κ = gz[rad/s] ÷ 速さ を 舵(直前 0.2 秒の平均)で
       直線に当てはめ、κ = 0 になる舵 [µs](中央 1441 からのずれ。+ = 左)
  左右の旋回の効き = |舵| ≥ 150 µs・速さ > 1.5 m/s・衝突の前後 0.5 秒を除くフレームの κ ÷ 舵 の中央値を左右で(1/m/µs ×1e4)
"""
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

RAW = Path('/home/jetson/jetformer/data/raw')
LAB = Path('/home/jetson/jetformer/data/labels')
DS = Path('/home/jetson/jetformer/data/datasets/clean_driving_20261011')
SC = Path('/home/jetson/jetformer_reports/bench_20261024_tune/since_1300/nightly')
OUT = Path('/home/jetson/jetformer_reports/bench_20261024_report/cond_data')
NEUTRAL, CENTER = 1496.0, 1441.0
DET = ['?', 'R1', 'GM_in', 'GM_out', 'DE', 'SW', 'HS', 'LH']
STRAIGHT = (4, 6)
W_CMD = 15          # 0.5 秒
W_DE = 9            # 0.3 秒
W_ST = 6            # 0.2 秒
BIN_S = 20.0


def kv(desc, k):
    m = re.search(r'(?:^|\s)%s=(\S+)' % re.escape(k), desc or '')
    return m.group(1) if m else None


def short(run):
    return {'clean_driving_20261011-nocmd-s3-20261011-024626': 'B', 'bench_20261022-slow-sec-s1': 'SLOW-SEC1',
            'bench_20261021-ens-p2': 'ENS'}.get(run, run)


def cat(lg, name, keys):
    parts = {k: [] for k in keys}
    for s in sorted(glob.glob(str(RAW / lg / (lg + '-*')))):
        try:
            with np.load(s + '/' + name + '.npz') as z:
                for k in keys:
                    parts[k].append(z[k])
        except Exception:
            continue
    o = {k: np.concatenate(v) for k, v in parts.items()}
    idx = np.argsort(o['timestamps_us'], kind='stable')
    return {k: v[idx] for k, v in o.items()}


def trailing(x, w, fn='mean'):
    n = len(x)
    if fn == 'mean':
        c = np.cumsum(np.r_[0.0, x])
        i0 = np.maximum(np.arange(n) - w + 1, 0)
        return (c[np.arange(n) + 1] - c[i0]) / (np.arange(n) + 1 - i0)
    out = np.empty(n)
    for i in range(n):
        out[i] = np.std(x[max(0, i - w + 1):i + 1])
    return out


def trailing_all(m, w):
    c = np.cumsum(np.r_[0, (~m).astype(int)])
    i0 = np.maximum(np.arange(len(m)) - w + 1, 0)
    return (c[np.arange(len(m)) + 1] - c[i0]) == 0


def med(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else None


def pct(x, q):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, q)) if len(x) else None


def r(x, n=3):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def brightness(files, step=2):
    """[(Y の平均, Y の標準偏差, R, G, B の平均)]。読めないファイルは NaN"""
    out = np.full((len(files), 5), np.nan, np.float32)
    for i in range(0, len(files), step):
        try:
            a = np.asarray(Image.open(files[i]).convert('RGB'), np.float32)
        except Exception:
            continue
        y = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
        out[i] = (y.mean(), y.std(), a[..., 0].mean(), a[..., 1].mean(), a[..., 2].mean())
    return out


def load_one(lg):
    j = json.loads((RAW / lg / 'log.json').read_text(encoding='utf-8'))
    info = json.loads((LAB / (lg + '.json')).read_text(encoding='utf-8'))
    with np.load(LAB / (lg + '.npz')) as z:
        lab = {k: z[k] for k in ('t_us', 't_s', 'ai', 'dry_run', 'hold', 'assist', 'out', 'raw', 'erpm', 'speed_mps',
                                 'loc_detail', 'collision', 'lap_id')}
    t0 = info['t0_us']
    I = cat(lg, 'IMU', ['timestamps_us', 'gyro'])
    V = cat(lg, 'VEHICLE', ['timestamps_us', 'erpm'])
    it = (I['timestamps_us'] - t0) / 1e6
    vt = (V['timestamps_us'] - t0) / 1e6
    gz = I['gyro'][:, 2].astype(np.float64)
    ve = np.interp(it, vt, V['erpm'].astype(np.float64))
    still = (np.abs(ve) < 50) & (np.abs(gz) < 5)
    bias = float(np.median(gz[still])) if still.sum() > 50 else -0.09
    gzs = np.convolve(gz - bias, np.ones(5) / 5, 'same')
    # カメラのファイル(ラベルの t_us と同じ時刻)
    ct, cf = [], []
    for s in sorted(glob.glob(str(RAW / lg / (lg + '-*')))):
        try:
            with np.load(s + '/CAM_FRONT.npz') as c:
                ct.append(c['timestamps_us'])
                cf += [s + '/' + f for f in c['files']]
        except Exception:
            continue
    ct = np.concatenate(ct)
    cf = np.array(cf)
    k = np.clip(np.searchsorted(ct, lab['t_us']), 0, len(ct) - 1)
    files = np.where(ct[k] == lab['t_us'], cf[k], '')
    A = cat(lg, 'AI_CMD', ['timestamps_us', 'latency_ms', 'infer_ms'])
    return j, info, lab, {'it': it, 'gz': gz - bias, 'gzs': gzs, 'bias': bias, 'still_n': int(still.sum()),
                          'still_bias_first': float(np.median(gz[still][:300])) if still.sum() > 300 else None,
                          'still_bias_last': float(np.median(gz[still][-300:])) if still.sum() > 300 else None}, files, A


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sc = json.loads((SC / 'scorecard' / 'scorecard.json').read_text(encoding='utf-8'))['per_log']
    dsw = {r_['log']: r_ for r_ in json.loads((SC / 'de_swing' / 'de_swing.json').read_text(encoding='utf-8'))['rows']}
    logs = sorted(p.name for p in RAW.glob('20261024-*'))
    per = []
    steady_pool = []
    for lg in logs:
        j, info, lab, imu, files, A = load_one(lg)
        desc = j.get('description') or ''
        p = (j.get('drive') or {}).get('params') or {}
        no = int(kv(desc, 'ai_no'))
        hhmm = lg.split('-')[1]
        sess = 1 if hhmm < '1500' else (2 if hhmm < '1800' else 3)
        t = lab['t_s']
        n = len(t)
        ai = lab['ai'] & ~lab['dry_run']
        drv = ai & ~lab['hold'] & (lab['assist'] == 0)
        cmd = NEUTRAL - lab['out'][:, 1].astype(np.float64)
        raw_cmd = NEUTRAL - lab['raw'][:, 1].astype(np.float64)
        e = -lab['erpm'].astype(np.float64)
        v = lab['speed_mps'].astype(np.float64)
        ld = lab['loc_detail'].astype(int)
        steer = lab['out'][:, 0].astype(np.float64) - CENTER
        gz_f = np.interp(t, imu['it'], imu['gzs'])
        cs = trailing(np.nan_to_num(cmd), W_CMD)
        csd = trailing(np.nan_to_num(cmd), W_CMD, 'std')
        drvw = trailing_all(drv, W_CMD)
        de = e - np.r_[np.full(W_DE, np.nan), e[:-W_DE]]
        straight = np.isin(ld, STRAIGHT)
        steady = drvw & (np.abs(de) < 500) & (e > 1500) & (cs > 60) & (csd < 30) & straight
        moving = ai & (e > 300)
        # 衝突の前後 0.5 秒
        ct = t[lab['collision'] & ai]
        near_c = np.zeros(n, bool)
        for c in ct:
            near_c |= (t >= c - 0.5) & (t <= c + 0.5)
        # AI の経過時間(AI のフレームの dt の和)
        dt = np.clip(np.diff(t, append=t[-1]), 0, 0.2)
        ai_el = np.cumsum(dt * ai)
        # 明るさ(AI のフレームだけ・2 枚に 1 枚)
        fsel = np.where(ai, files, '')
        br = np.full((n, 5), np.nan, np.float32)
        idx = np.flatnonzero(fsel != '')
        b = brightness(list(fsel[idx]), step=2)
        br[idx] = b
        # 舵とヨー
        st_avg = trailing(steer, W_ST)
        ok_y = drvw & straight & (e > 3000) & (np.abs(st_avg) < 25) & (np.abs(steer) < 40) & ~near_c
        kappa = np.radians(gz_f) / np.maximum(v, 0.1)
        ok_s0 = drvw & straight & (np.abs(st_avg) < 150) & (v > 1.5) & ~near_c
        s0 = slope = None
        if ok_s0.sum() > 50:
            slope, icpt = np.polyfit(st_avg[ok_s0], kappa[ok_s0], 1)
            s0 = -icpt / slope
        ok_t = drvw & (np.abs(st_avg) >= 150) & (v > 1.5) & ~near_c
        kl = kappa[ok_t & (st_avg > 0)] / st_avg[ok_t & (st_avg > 0)] * 1e4
        kr = kappa[ok_t & (st_avg < 0)] / st_avg[ok_t & (st_avg < 0)] * 1e4
        np.savez_compressed(OUT / ('frames_%s.npz' % lg), t=t, ai=ai, drv=drv, ai_el=ai_el, cmd=cmd, raw_cmd=raw_cmd, erpm=e,
                            speed=v, loc=ld, steer=steer, gz=gz_f, cs=cs, steady=steady, ok_y=ok_y, br=br, files=files)
        row = {'no': no, 'log': lg, 'clock': '%s:%s' % (hhmm[:2], hhmm[2:4]), 'session': sess, 'model': short(j['drive'].get('run')),
               'throttle_gain': p.get('throttle_gain'), 'steer_gain': p.get('steer_gain'), 'desc_battery': kv(desc, 'battery'),
               'desc_free': desc if 'ai_no=' in desc and 'bench=' not in desc else None,
               'ai_s': r(float((dt * ai).sum()), 1), 'gyro_bias_dps': r(imu['bias'], 3), 'gyro_still_n': imu['still_n'],
               'gyro_bias_first_dps': r(imu['still_bias_first'], 3), 'gyro_bias_last_dps': r(imu['still_bias_last'], 3),
               'erpm_p90_ai': r(pct(e[moving], 90), 0), 'erpm_p90_HS': r(pct(e[moving & (ld == 6)], 90), 0),
               'erpm_p50_HS': r(pct(e[moving & (ld == 6)], 50), 0),
               'speed_per_erpm': r(med(v[moving] / e[moving]), 6),
               'steady_n': int(steady.sum()),
               'eff_raw': r(med(e[steady] / cs[steady]), 2), 'steady_cmd_p50': r(med(cs[steady]), 1),
               'steady_erpm_p50': r(med(e[steady]), 0),
               'yaw_bias_dps': r(float(np.mean(gz_f[ok_y])) if ok_y.sum() > 20 else None, 2), 'yaw_bias_n': int(ok_y.sum()),
               'yaw_bias_dps_DE': r(float(np.mean(gz_f[ok_y & (ld == 4)])) if (ok_y & (ld == 4)).sum() > 15 else None, 2),
               'yaw_bias_dps_HS': r(float(np.mean(gz_f[ok_y & (ld == 6)])) if (ok_y & (ld == 6)).sum() > 15 else None, 2),
               'steer_straight_p50_us': r(med(steer[drvw & straight & (e > 3000) & ~near_c]), 1),
               's0_us': r(s0, 1), 's0_slope': r(slope * 1e4 if slope is not None else None, 3), 's0_n': int(ok_s0.sum()),
               'turn_gain_left': r(med(kl), 3), 'turn_gain_right': r(med(kr), 3), 'turn_n_left': int(len(kl)),
               'turn_n_right': int(len(kr)),
               'bright_mean': r(med(br[ai, 0]), 1), 'bright_std': r(med(br[ai, 1]), 1),
               'bright_rgb': [r(med(br[ai, 2]), 1), r(med(br[ai, 3]), 1), r(med(br[ai, 4]), 1)],
               'latency_ms_p50': r(med(A['latency_ms']), 1), 'infer_ms_p50': r(med(A['infer_ms']), 1),
               'latency_ms_p95': r(pct(A['latency_ms'], 95), 1)}
        # 区間ごと
        secs = {}
        for c, nm in enumerate(DET):
            m = drv & (ld == c)
            if m.sum() < 10:
                continue
            secs[nm] = {'n': int(m.sum()), 'cmd_p50': r(med(cmd[m]), 1), 'raw_cmd_p50': r(med(raw_cmd[m]), 1),
                        'erpm_p50': r(med(e[m]), 0), 'erpm_p90': r(pct(e[m], 90), 0), 'bright_mean': r(med(br[m, 0]), 1),
                        'bright_std': r(med(br[m, 1]), 1), 'steer_p50': r(med(steer[m]), 1)}
        row['sections'] = secs
        # ラップ・衝突(成績表から)
        L = sc.get(lg) or {}
        laps = L.get('laps') or []
        clean = [x for x in laps if x.get('clean')]
        row['laps_clean'] = len(clean)
        row['lap_clean_best'] = r(min([x['lap_s'] for x in clean]), 3) if clean else None
        row['lap_clean_med'] = r(med([x['lap_s'] for x in clean]), 3) if clean else None
        row['lap_sections_med'] = {k: r(med([x['sections_s'].get(k, np.nan) for x in clean]), 3)
                                   for k in ('R1', 'GM', 'DE', 'other', 'HS', 'LH')} if clean else None
        ce = L.get('collision_events') or []
        row['collisions'] = len(ce)
        row['collisions_by_detail'] = {}
        for c in ce:
            row['collisions_by_detail'][c.get('loc_detail') or c.get('loc')] = row['collisions_by_detail'].get(c.get('loc_detail') or c.get('loc'), 0) + 1
        row['de_collision_speed'] = [c.get('speed_before_mps') for c in ce if c.get('loc') == 'DE']
        row['passes_ai'] = sum(1 for x in (L.get('passes') or []) if x.get('ai'))
        ds = dsw.get(lg) or {}
        row['de'] = {k: ds.get(k) for k in ('passes', 'hits', 'overshoot_med_us', 'late_left_med_us')}
        row['de_pass_list'] = ds.get('pass_list') or []
        # 時間の並び(AI の時間 20 秒ごと)
        bins = []
        if ai.any():
            nb = int(np.ceil(ai_el[ai].max() / BIN_S))
            for bi in range(nb):
                m = ai & (ai_el > bi * BIN_S) & (ai_el <= (bi + 1) * BIN_S)
                ms = m & steady
                bins.append({'t_ai_s': r((bi + 0.5) * BIN_S, 1), 'steady_n': int(ms.sum()),
                             'erpm_p90': r(pct(e[m & moving], 90), 0) if (m & moving).sum() > 30 else None,
                             'bright_mean': r(med(br[m, 0]), 1)})
                bins[-1]['_eff_e'] = e[ms].tolist()
                bins[-1]['_eff_c'] = cs[ms].tolist()
        row['bins'] = bins
        steady_pool.append((no, e[steady], cs[steady]))
        per.append(row)
        print(no, lg, row['ai_s'], row['erpm_p90_ai'], row['eff_raw'], row['bright_mean'], row['yaw_bias_dps'], flush=True)
    # 指令をそろえた電池の効き: eRPM = s_run × (a·cmd + b)。a・b は全部をまとめて当てはめ(2 回くり返す)
    scale = {no: 1.0 for no, _, _ in steady_pool}
    for _ in range(3):
        E = np.concatenate([ee / scale[no] for no, ee, _ in steady_pool])
        C = np.concatenate([cc for _, _, cc in steady_pool])
        a, b = np.polyfit(C, E, 1)
        for no, ee, cc in steady_pool:
            if len(ee) >= 20:
                scale[no] = float(np.median(ee / (a * cc + b)))
        g = np.exp(np.mean(np.log([scale[no] for no, ee, _ in steady_pool if len(ee) >= 20])))
        scale = {k: v_ / g for k, v_ in scale.items()}
    fit = {'a': round(float(a), 3), 'b': round(float(b), 1), 'n': int(len(C))}
    for row in per:
        ok = [x for x in steady_pool if x[0] == row['no']][0]
        row['eff_adj'] = r(scale[row['no']], 3) if len(ok[1]) >= 20 else None
        for bn in row['bins']:
            ee, cc = np.array(bn.pop('_eff_e')), np.array(bn.pop('_eff_c'))
            bn['eff_adj'] = r(float(np.median(ee / (a * cc + b)) / g), 3) if len(ee) >= 15 else None
        xs = [bn['t_ai_s'] / 60 for bn in row['bins'] if bn['eff_adj'] is not None]
        ys = [bn['eff_adj'] for bn in row['bins'] if bn['eff_adj'] is not None]
        row['eff_adj_slope_per_min'] = r(float(np.polyfit(xs, ys, 1)[0]), 4) if len(xs) >= 3 else None
        xs = [bn['t_ai_s'] / 60 for bn in row['bins'] if bn['erpm_p90'] is not None]
        ys = [bn['erpm_p90'] for bn in row['bins'] if bn['erpm_p90'] is not None]
        row['erpm_p90_slope_per_min'] = r(float(np.polyfit(xs, ys, 1)[0]), 0) if len(xs) >= 3 else None
    # 学習データ(B の学習に使った 10/10 の 9 本)の明るさと人の速さ
    train = []
    for s in sorted(glob.glob(str(DS / 'aligned' / '*.npz'))):
        with np.load(s) as z:
            cfs, veh, ok = z['cam_files'], z['vehicle'], z['vehicle_valid']
        sel = np.arange(0, len(cfs), 6)
        b = brightness([str(DS / str(cfs[i])) for i in sel], step=1)
        er = -veh[:, 3][ok]
        name = Path(s).stem
        train.append({'scene': name, 'clock': name.split('-')[1][:2] + ':' + name.split('-')[1][2:4],
                      'bright_mean': r(med(b[:, 0]), 1), 'bright_std': r(med(b[:, 1]), 1),
                      'erpm_p90': r(pct(er[er > 300], 90), 0), 'thr_cmd_p50': r(med(NEUTRAL - veh[:, 1][ok & (-veh[:, 3] > 300)]), 1),
                      'frames': int(len(cfs))})
    res = {'created': '2026-10-25', 'definitions': __doc__, 'eff_fit': fit, 'runs': per, 'train_scenes': train}
    (OUT / 'cond_metrics.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print('fit', fit)


if __name__ == '__main__':
    sys.exit(main())
