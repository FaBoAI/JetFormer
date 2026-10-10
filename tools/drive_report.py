#!/usr/bin/env python3
"""AI走行の記録(走行中に収集したログ)の評価。レポートを data/reports/drive_<ログ>/ に作る

    python tools/drive_report.py 20261010-160306 [20261010-155820 ...]

ログごとに: AI側(CH3)だった時間と連続区間・人が取り返した回数・中立で止めた(ホールド)回数と原因の推定
(各センサーの値の途切れ)・遅れ・出力の分布と上下限への張り付き・ステアリングの揺れ・速度・衝撃(ぶつかった可能性)。
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from jetformer import config  # noqa: E402
from jetformer.laps import LapCounter  # noqa: E402

G = 9.80665


def _load(scene_dir, name):
    p = os.path.join(scene_dir, name + '.npz')
    return dict(np.load(p, allow_pickle=True)) if os.path.exists(p) else None


def _cat(parts, key):
    parts = [p for p in parts if p is not None and key in p]
    return np.concatenate([p[key] for p in parts]) if parts else np.zeros(0)


def _segments(mask, t):
    """True が続く区間の長さ [秒] のリスト"""
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if (not m or i == len(mask) - 1) and start is not None:
            end = i if m else i - 1
            out.append(float(t[end] - t[start]) + 1 / 30.0)
            start = None
    return out


def _gaps(t_us, limit_ms):
    d = np.diff(t_us) / 1000.0
    return d[d > limit_ms]


def analyze(log):
    root = config.RAW_DIR / log
    meta = json.load(open(root / 'log.json', encoding='utf-8'))
    scenes = sorted(s for s in os.listdir(root) if s.startswith(log + '-'))
    data = {k: [_load(root / s, k) for s in scenes] for k in ('AI_CMD', 'VEHICLE', 'IMU', 'CAM_FRONT', 'TOF_FRONT')}
    ai = {k: _cat(data['AI_CMD'], k) for k in ('timestamps_us', 'raw', 'out', 'hold', 'select', 'latency_ms', 'infer_ms', 'gains')}
    veh = {k: _cat(data['VEHICLE'], k) for k in ('timestamps_us', 'pwm', 'select', 'erpm')}
    imu = {k: _cat(data['IMU'], k) for k in ('timestamps_us', 'accel', 'gyro')}
    t = (ai['timestamps_us'] - ai['timestamps_us'][0]) / 1e6
    sel = ai['select'].astype(bool)
    hold = ai['hold'].astype(bool)
    drive = meta.get('drive') or {}
    center = drive.get('steer_center_us', 1500.0)
    r = {'log': log, 'scenes': len(scenes), 'description': meta.get('description', ''),
         'run': drive.get('run'), 'params': drive.get('params'), 'steer_center_us': center,
         'throttle_neutral_us': drive.get('throttle_neutral_us'),
         'duration_s': round(float(t[-1]), 1), 'frames': int(len(t))}
    r['ai_side_s'] = round(float(sel.sum() / 30.0), 1)
    r['ai_side_ratio'] = round(float(sel.mean()), 3)
    segs = sorted(_segments(sel, t), reverse=True)
    r['ai_segments'] = len(segs)
    r['ai_segments_longest_s'] = [round(s, 1) for s in segs[:5]]
    r['human_takeovers'] = int(np.sum(sel[:-1] & ~sel[1:]))       # AI側 → 人の側 に切り替えた回数
    r['holds_on_ai_side'] = int((hold & sel).sum())
    r['hold_events'] = int(np.sum(~(hold & sel)[:-1] & (hold & sel)[1:]) + int((hold & sel)[0]))
    # センサーの途切れ(走行のホールド条件: カメラ・ToF・車両 150ms、IMU 100ms)
    gaps = {'VEHICLE': _gaps(veh['timestamps_us'], config.DRIVE['max_sensor_age_ms'] - 10),
            'IMU': _gaps(imu['timestamps_us'], config.DRIVE['imu_max_age_ms'] - 10),
            'CAM_FRONT': _gaps(_cat(data['CAM_FRONT'], 'timestamps_us'), config.DRIVE['max_sensor_age_ms'] - 10),
            'TOF_FRONT': _gaps(_cat(data['TOF_FRONT'], 'timestamps_us'), config.DRIVE['max_sensor_age_ms'] - 10)}
    r['sensor_gaps_ms'] = {k: [round(float(x)) for x in v] for k, v in gaps.items()}
    lat = ai['latency_ms'][np.isfinite(ai['latency_ms'])]          # 中立で止めたフレームは NaN
    r['latency_ms'] = {'p50': round(float(np.median(lat)), 1), 'p99': round(float(np.percentile(lat, 99)), 1)} if len(lat) else None
    r['infer_ms_p50'] = round(float(np.nanmedian(ai['infer_ms'])), 1) if np.isfinite(ai['infer_ms']).any() else None
    r['camera_fps'] = round(float((len(t) - 1) / t[-1]), 1)
    a = sel & ~hold
    if a.any():
        so, to = ai['out'][a, 0], ai['out'][a, 1]
        r['steer_out_us'] = {k: round(float(np.percentile(so, p))) for k, p in (('p5', 5), ('p50', 50), ('p95', 95))}
        r['throttle_out_us'] = {k: round(float(np.percentile(to, p))) for k, p in (('p5', 5), ('p50', 50), ('p95', 95))}
        r['steer_at_limit_ratio'] = round(float(np.mean((so <= config.DRIVE['min_us'] + 1) | (so >= config.DRIVE['max_us'] - 1))), 3)
        dev = so - center
        r['steer_abs_dev_us_mean'] = round(float(np.abs(dev).mean()), 1)
        flips = np.sum(np.sign(dev[:-1]) * np.sign(dev[1:]) < 0) if len(dev) > 1 else 0
        r['steer_sign_flips_per_s'] = round(float(flips / max(a.sum() / 30.0, 1e-6)), 2)
        r['steer_step_us_p95'] = round(float(np.percentile(np.abs(np.diff(so)), 95)), 1) if len(so) > 1 else 0.0
    # 速度(AI側のとき)と衝撃
    if len(veh['timestamps_us']):
        vsel = veh['select'].astype(bool)
        e = np.abs(veh['erpm'][vsel]) if vsel.any() else np.zeros(0)
        r['erpm_ai'] = {'median': int(np.median(e)) if len(e) else 0, 'p95': int(np.percentile(e, 95)) if len(e) else 0,
                        'moving_ratio': round(float(np.mean(e > 0)), 3) if len(e) else 0.0}
        eh = np.abs(veh['erpm'][~vsel])
        r['erpm_human'] = {'median': int(np.median(eh)) if len(eh) else 0, 'p95': int(np.percentile(eh, 95)) if len(eh) else 0}
    if len(imu['timestamps_us']):
        acc = np.abs(np.linalg.norm(imu['accel'], axis=1) - G)
        ti = imu['timestamps_us']
        hits = ti[acc > 2 * G]
        ev = []
        for h in hits:                             # 0.5秒以内は同じ衝撃
            if not ev or h - ev[-1] > 500000:
                ev.append(int(h))
        # 衝撃の時に AI側だったか
        on_ai = [bool(sel[np.clip(np.searchsorted(ai['timestamps_us'], h), 0, len(sel) - 1)]) for h in ev]
        r['impacts'] = [{'t_s': round(float((h - ai['timestamps_us'][0]) / 1e6), 1), 'ai_side': o} for h, o in zip(ev, on_ai)]
        # 人が取り返した時刻と、その直前(3秒以内)の衝撃
        tk = t[1:][sel[:-1] & ~sel[1:]]
        r['takeovers'] = [{'t_s': round(float(x), 1),
                           'impact_before_s': next((round(float(x - i['t_s']), 1) for i in reversed(r['impacts'])
                                                    if 0 <= x - i['t_s'] <= 3), None)} for x in tk]
        r['accel_peak_g'] = round(float(acc.max() / G), 1)
    # 周回(ジャイロで360度ごと)と1周のタイム
    lc = LapCounter()
    vsel = veh['select'].astype(bool) if len(veh['timestamps_us']) else np.zeros(0, bool)
    hg = np.hypot(imu['accel'][:, 0], imu['accel'][:, 1]) / G
    for ti, g, h in zip(imu['timestamps_us'] / 1e6, imu['gyro'][:, 2], hg):
        k = max(int(np.searchsorted(veh['timestamps_us'] / 1e6, ti, side='right')) - 1, 0)
        lap = lc.add(float(ti), float(g), float(veh['erpm'][k]), bool(vsel[k]), float(h))
    s = lc.summary()
    # best/median は「AIだけ・ぶつからず・リカバリーなし」の周、median_ai_all はAIだけの周すべて
    r['laps'] = {k: s[k] for k in ('laps', 'direction', 'best_ai_s', 'median_ai_s', 'clean_laps', 'ai_laps', 'median_ai_all_s')}
    r['lap_times'] = [{k: x[k] for k in ('n', 'time_s', 'ai_only', 'takeovers', 'impacts', 'clean', 'first')} for x in lc.laps]
    del lap
    return r, (t, ai, veh, imu, center)


def plot(r, arrays, out_png):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    fp = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp)
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=fp).get_name()
    t, ai, veh, imu, center = arrays
    t0 = ai['timestamps_us'][0]
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    sel = ai['select'].astype(bool)
    hold = ai['hold'].astype(bool) & sel
    for ax in axes:
        ax.fill_between(t, 0, 1, where=sel, transform=ax.get_xaxis_transform(), color='#a855f7', alpha=.10, lw=0)
        ax.fill_between(t, 0, 1, where=hold, transform=ax.get_xaxis_transform(), color='#ef4444', alpha=.35, lw=0)
    vt = (veh['timestamps_us'] - t0) / 1e6
    axes[0].plot(vt, veh['pwm'][:, 0], color='#111827', lw=1, label='プロポの操作(人・基板で計測)')
    axes[0].plot(t, ai['out'][:, 0], color='#2563eb', lw=1, label='AIの出力(ゲイン後)')
    axes[0].plot(t, ai['raw'][:, 0], color='#93c5fd', lw=.8, label='モデルの予測(ゲイン前)')
    axes[0].axhline(center, color='#9ca3af', ls='--', lw=.8)
    axes[0].set_ylabel('ステアリング µs')
    axes[0].legend(fontsize=8, loc='upper right')
    axes[1].plot(vt, veh['pwm'][:, 1], color='#111827', lw=1, label='プロポの操作(人)')
    axes[1].plot(t, ai['out'][:, 1], color='#dc2626', lw=1, label='AIの出力')
    axes[1].set_ylabel('スロットル µs')
    axes[1].legend(fontsize=8, loc='upper right')
    axes[2].plot(vt, veh['erpm'], color='#16a34a', lw=1, label='eRPM')
    axes[2].set_ylabel('eRPM')
    ax2 = axes[2].twinx()
    it = (imu['timestamps_us'] - t0) / 1e6
    ax2.plot(it, np.abs(np.linalg.norm(imu['accel'], axis=1) - G) / G, color='#f59e0b', lw=.7, label='衝撃 [g]')
    ax2.set_ylabel('衝撃 [g]')
    axes[2].set_xlabel('秒(紫=AI側 / 赤=中立で止めた)')
    axes[0].set_title('%s  モデル %s  ゲイン ステア%.1f・速度%.1f' % (
        r['log'], r['run'], (r['params'] or {}).get('steer_gain', 1), (r['params'] or {}).get('throttle_gain', 1)))
    fig.tight_layout()
    fig.savefig(out_png, dpi=85)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('logs', nargs='+', help='走行時に収集したログ(data/raw/ の名前)')
    args = ap.parse_args()
    for log in args.logs:
        out_dir = config.DATA_DIR / 'reports' / ('drive_' + log)
        out_dir.mkdir(parents=True, exist_ok=True)
        r, arrays = analyze(log)
        plot(r, arrays, out_dir / 'drive_timeline.png')
        with open(out_dir / 'report.json', 'w', encoding='utf-8') as f:
            json.dump(r, f, ensure_ascii=False, indent=1)
        print(json.dumps(r, ensure_ascii=False))
        print('レポート: %s' % out_dir)


if __name__ == '__main__':
    main()
