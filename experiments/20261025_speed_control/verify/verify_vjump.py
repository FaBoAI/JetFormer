"""反証 V-drive: 基板の電圧の最初の値が「走り出したあと」に来たとき、掛ける m が 1 フレームで跳ぶか(閉ループ・偽物の車)。
説明欄の V_before は走行開始の画面が基板から入れた値(volt の段)。基板の値が壊れて低い(0.5 V)・ふつう(7.8 V)の 2 つ"""
import sys

import numpy as np

sys.path.insert(0, '/home/jetson/jetformer')
sys.path.insert(0, '/home/jetson/jetformer/experiments/20261025_speed_control/verify')
import vdrive_harness as H  # noqa: E402
from sim_source import SimSource  # noqa: E402
from jetformer import drive  # noqa: E402
from verify_off import PB  # noqa: E402

P0 = dict(PB, throttle_gain=0.9, start_kick=0, speed_ceiling=1, speed_auto=1)
for label, vb, desc, stand in (('ふつう 7.8 V・V_before=7.80', lambda t: 7.8, 'V_before=7.80 V_src=board', 0.2),
                               ('壊れて 0.5 V・V_before なし(6〜9 V の外で捨てる)', lambda t: 0.5, 'V_before=0.50 V_src=board', 0.2),
                               ('壊れて 0.5 V・止まって 3 秒待つ', lambda t: 0.5, 'V_before=0.50 V_src=board', 3.0)):
    src = SimSource(t_end_s=12.0, b=1.0, vbat=vb, stand_s=stand)
    r = H.run_loop(drive, P0, src, src.pred_of, floor_ref=None, description=desc)
    rows = [((t - src.t0) / 1e9, x) for t, x in r['rec'].rows if not x['hold']]
    ms = np.array([x['sa_m'] for _, x in rows])
    k = int(np.argmax(np.abs(np.diff(ms)))) + 1
    t, x = rows[k]
    v = x['ceil_v']
    print('%-40s 最大の 1 フレームの m の跳び %.3f(t %.2f 秒・速さ %.2f m/s・源 %d)ゲイン %.3f → %.3f' % (
        label, ms[k] - ms[k - 1], t, v, x['sa_src'], rows[k - 1][1]['gains'][1], x['gains'][1]))
