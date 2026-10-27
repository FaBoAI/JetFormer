"""10/10 の AI の記録(読むだけ)で、AI 側・前へ走っている間の当たり(水平 2.5G 以上)のあと、
処理したフレーム(hold でない行)が 0.5 秒より長く来なかった所を探す。最後の行が当たりの前か後か、次の処理まで何秒か。"""
import glob
import numpy as np

out = []
for seg in sorted(glob.glob('/home/jetson/jetformer/data/raw/20261010-*/20261010-*-*')):
    try:
        a = np.load(seg + '/AI_CMD.npz')
        i = np.load(seg + '/IMU.npz')
        v = np.load(seg + '/VEHICLE.npz')
    except Exception:
        continue
    ta, hold, sel = a['timestamps_us'], a['hold'], a['select']
    proc = ta[~hold]
    ti = i['timestamps_us']
    g = np.hypot(i['accel'][:, 0], i['accel'][:, 1]) / 9.80665
    tv, erpm = v['timestamps_us'], v['erpm']
    last_hit = -1e18
    for k in np.where(g >= 2.5)[0]:
        t = ti[k]
        if t - last_hit < 1.0e6:
            continue
        last_hit = t
        if len(proc) == 0 or t < proc[0] or t > proc[-1]:
            continue
        j = np.searchsorted(proc, t)          # proc[j-1] <= t < proc[j]
        before, after = proc[j - 1], proc[j]
        gap = (after - before) / 1e6
        if gap <= 0.5:
            continue
        kv = np.searchsorted(tv, before) - 1
        e_before = int(erpm[max(kv, 0)])
        row = np.searchsorted(ta, before)
        out.append((seg.split('/')[-1], round(float(g[k]), 1), round((t - before) / 1e3, 1), round((after - t) / 1e6, 2),
                    round(gap, 2), e_before, bool(sel[row])))
for r in out:
    print('seg=%s g=%.1fG last_row_before_hit_ms=%.1f next_proc_after_hit_s=%.2f gap_s=%.2f erpm_at_last=%d select=%s' % r)
