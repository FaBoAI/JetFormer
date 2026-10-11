"""旋回中の横加速度(IMU x = 右)と、ヨー×速度(eRPM換算)の上限を、人と AI のログで見る。CPUのみ・読み取りのみ。
衝突(水平2.5G以上)の前1秒〜後2秒は除く。速度 v = |eRPM| * 0.5e-3 m/s(空転を含む目安)。
出力: lateral_grip.json(このフォルダ)
"""
import json, os, sys, numpy as np
sys.path.insert(0, os.path.expanduser('~/jetformer_reports/summary_20261010/src'))
import runs_table as rt  # load_log / cat / impacts_from を再利用
G = 9.80665
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lateral_grip.json')
LOGS = {'20261010-152707': '人(午後・速い)', '20261010-200916': '人(夜・高速)', '20261010-201626': '人(夜・低速)',
        '20261010-172910': 'AI 006', '20261010-173445': 'AI 007'}
res = {}
for log, label in LOGS.items():
    meta, ok, broken, missing, sj, data = rt.load_log(log)
    it = rt.cat(data['IMU'], 'timestamps_us'); acc = rt.cat(data['IMU'], 'accel'); gyr = rt.cat(data['IMU'], 'gyro')
    vt = rt.cat(data['VEHICLE'], 'timestamps_us'); erpm = rt.cat(data['VEHICLE'], 'erpm'); sel = rt.cat(data['VEHICLE'], 'select')
    o = np.argsort(it); it, acc, gyr = it[o], acc[o], gyr[o]
    o = np.argsort(vt); vt, erpm, sel = vt[o], erpm[o], sel[o]
    hg = np.hypot(acc[:, 0], acc[:, 1]) / G
    bad = np.zeros(len(it), bool)
    for i in np.nonzero(hg >= 2.5)[0]:
        bad |= (it >= it[i] - 1.0e6) & (it <= it[i] + 2.0e6)
    k = np.clip(np.searchsorted(vt, it) - 1, 0, len(vt) - 1)
    v = np.abs(erpm[k].astype(float)) * 0.5e-3
    s = sel[k]
    keep = ~bad & (v > 1.0)
    if label.startswith('AI'):
        keep &= s
    else:
        keep &= ~s
    # 50ms の移動平均で振動を落とす
    w = 5
    ax = np.convolve(acc[:, 0], np.ones(w) / w, 'same') / G
    gz = np.convolve(gyr[:, 2], np.ones(w) / w, 'same')
    turn = keep & (np.abs(gz) > 90)
    ay_c = np.deg2rad(np.abs(gz)) * v / G   # ヨー×速度(空転があると過大)
    r = {'samples_turning': int(turn.sum()),
         'lat_acc_imu_g_p50_p90_p99': [round(float(np.percentile(np.abs(ax[turn]), q)), 2) for q in (50, 90, 99)] if turn.any() else None,
         'yaw_x_speed_g_p50_p90_p99': [round(float(np.percentile(ay_c[turn], q)), 2) for q in (50, 90, 99)] if turn.any() else None,
         'speed_ms_p50_p90_moving': [round(float(np.percentile(v[keep], q)), 2) for q in (50, 90)]}
    res[log] = {'label': label, **r}
    print(log, label, r)
json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1)
