"""10/10 20261010-181504-0004(7.5G の衝突の 0.43 秒あとに中立止め 2 フレーム)を Assist に通す(tests/test_rec_hold_keep._replay_1010 と
同じ入力)。直す 2 回目のあとは、残した中立止めのあとの最初のフレームに hold_since(最初の中立止めの行の時刻)・hold_back_s(車両の値。
drive と同じ数え方)を渡す。読むだけ。戻り値 最初のリカバリーの行"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '/home/jetson/jetformer')
from jetformer import assist as A          # noqa: E402

REC = Path('/home/jetson/jetformer/data/raw/20261010-181504/20261010-181504-0004')


def replay(keep, pass_hold, no_noise=False):
    z = np.load(REC / 'AI_CMD.npz'); v = np.load(REC / 'VEHICLE.npz'); imu = np.load(REC / 'IMU.npz')
    t = z['timestamps_us'] / 1e6; tv = v['timestamps_us'] / 1e6; ti = imu['timestamps_us'] / 1e6
    hg = np.hypot(imu['accel'][:, 0], imu['accel'][:, 1]) / 9.80665
    h = A.Assist((1443.0, 1498.0))
    hold_t0, last_t = None, None
    for i in range(100, 200):
        if not z['select'][i]:
            return None, None
        if z['hold'][i]:
            h.launch_hold(); h.reset(keep_recovery=keep)
            if hold_t0 is None:
                hold_t0 = float(t[i])
            continue
        e = float(v['erpm'][np.searchsorted(tv, t[i], side='right') - 1])
        if no_noise and i >= 143 and abs(e) >= 1000:
            e = 0.0
        m = (ti > t[i] - 0.15) & (ti <= t[i] + 0.01)
        g = float(hg[m].max()) if m.any() else 0.0
        kw = {}
        if pass_hold and keep and hold_t0 is not None:
            mm = (tv > last_t - 0.0) & (tv <= t[i])
            ev = v['erpm'][mm].astype(float)
            ts = tv[mm]
            back = sum(b - a for (a, b), (x, y) in zip(zip(ts, ts[1:]), zip(ev, ev[1:])) if x > A.BACKING_ERPM or y > A.BACKING_ERPM or b - a > 0.06)
            kw = dict(hold_since=hold_t0, hold_back_s=back)
        o, k = h.step(tuple(float(x) for x in z['model_out'][i]), e, True, float(t[i]), esc_on=True, rec_on=True, impact_g=g, **kw)
        hold_t0, last_t = None, float(t[i])
        if k == A.ASSIST_RECOVERY:
            return i, round(float(t[i] - t[100]), 3)
    return None, None


for name, args in (('OFF', (False, False)), ('ON(直す 2 回目の前の形: hold_since なし)', (True, False)), ('ON(今)', (True, True)),
                   ('OFF 雑音なし', (False, False, True)), ('ON 雑音なし(前の形)', (True, False, True)), ('ON 雑音なし(今)', (True, True, True))):
    print(name, replay(*args))
