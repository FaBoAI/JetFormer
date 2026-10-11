"""共通ローダー(読み取りのみ・CPU)。explore_control の解析スクリプトから使う。

軸と単位(既存レポートで確認済み): IMU accel [m/s^2]、gyro [deg/s]、+x = 右、+y = 前、+z = 上、gz + = 左旋回。
速度 v = -eRPM * 0.5e-3 m/s(前進で eRPM 負)。ステア中心 1443us(+ = 左)、スロットル中立 1498us(- = 前進)。
"""
import json
import os

import numpy as np

RAW = '/home/jetson/jetformer/data/raw'
G = 9.80665
KV = 0.5e-3
STEER_C, THR_N = 1443.0, 1498.0

HUMAN_LOGS = ['20261010-135449', '20261010-144028', '20261010-150424', '20261010-152405', '20261010-152707',
              '20261010-193638', '20261010-200916', '20261010-201626', '20261010-202633']
AI_LOGS = {'20261010-171639': 'ai_run_003', '20261010-172505': 'ai_run_004', '20261010-172910': 'ai_run_006',
           '20261010-173445': 'ai_run_007', '20261010-181504': 'ai_run_008', '20261010-203924': 'ai_run_009',
           '20261010-204208': 'ai_run_010', '20261010-204605': 'ai_run_101'}


def load(log, keys=('IMU', 'VEHICLE', 'AI_CMD')):
    root = os.path.join(RAW, log)
    scenes = sorted(s for s in os.listdir(root) if s.startswith(log + '-'))
    out = {}
    for k in keys:
        parts = []
        for s in scenes:
            p = os.path.join(root, s, k + '.npz')
            if not os.path.exists(p) or os.path.getsize(p) == 0:
                continue
            try:
                z = np.load(p, allow_pickle=True)
                parts.append({n: z[n] for n in z.files if np.ndim(z[n]) > 0})
            except Exception:
                pass
        if not parts:
            out[k] = None
            continue
        keys_common = set(parts[0])
        for q in parts[1:]:
            keys_common &= set(q)
        out[k] = {n: np.concatenate([q[n] for q in parts]) for n in keys_common}
        o = np.argsort(out[k]['timestamps_us'], kind='stable')
        out[k] = {n: v[o] for n, v in out[k].items()}
    meta = json.load(open(os.path.join(root, 'log.json'), encoding='utf-8'))
    return meta, out


def prev_index(t_src, t_query):
    """t_query の各時刻について、それ以前で最後の t_src の添字(無ければ 0)"""
    k = np.searchsorted(t_src, t_query, side='right') - 1
    return np.clip(k, 0, len(t_src) - 1)


def impacts(imu_t, acc, thr_g=2.5, merge_s=0.6):
    hg = np.hypot(acc[:, 0], acc[:, 1]) / G
    ev = []
    for i in np.nonzero(hg >= thr_g)[0]:
        t = imu_t[i]
        if ev and t - ev[-1][0] <= merge_s * 1e6:
            ev[-1][1] = max(ev[-1][1], float(hg[i]))
            continue
        ev.append([int(t), float(hg[i])])
    return ev


def smooth(x, n):
    if n <= 1:
        return x
    k = np.ones(n) / n
    return np.convolve(x, k, mode='same')
