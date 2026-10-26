"""反証 V-drive: 自動合わせの「上げない」決まりを閉ループで(遅い電池 b 0.85 = 上げたい状態)
  1) R1 の板 7G → 上げない(板の衝撃)になるか
  2) 衝突 → 人が持って 3 秒止める → 再開: 衝突の次の周は上げないか(位相が止まって不確かに戻る場合)
  3) 衝突 → 止めずに続ける(止まらない衝突): 次の周は上げないか
"""
import collections
import sys

import numpy as np

sys.path.insert(0, '/home/jetson/jetformer')
sys.path.insert(0, '/home/jetson/jetformer/experiments/20261025_speed_control/verify')
import vdrive_harness as H  # noqa: E402
from sim_source import SimSource  # noqa: E402
from jetformer import drive  # noqa: E402
from jetformer import speed_auto as sa  # noqa: E402
from verify_off import PB  # noqa: E402

P0 = dict(PB, throttle_gain=0.9, start_kick=0, speed_ceiling=1, speed_auto=1)
NAMES = {k: v for k, v in sa.EVENT_NAMES.items()}


def laps(r):
    out = []
    for t, x in r['rec'].rows:
        if x['sa_event'] not in (-1, 0):
            out.append(((t - 1_800_000_000 * 10**9) / 1e9, NAMES.get(int(x['sa_event']), x['sa_event']),
                        round(float(x['sa_lap_p90']), 2), round(float(x['sa_m_fb']), 3)))
    return out


for name, kw in (('板 7G', dict(bump_g=7.0)),
                 ('衝突 → 3 秒止める', dict(crash=[(30.0, 3.0)])),
                 ('衝突 → 0.2 秒(止まらない)', dict(crash=[(30.0, 0.2)]))):
    src = SimSource(t_end_s=80.0, b=0.85, **kw)
    r = H.run_loop(drive, P0, src, src.pred_of, floor_ref=None)
    print('==', name, 'end', {k: r['st']['auto_end'].get(k) for k in ('m_fb_end', 'laps_used', 'laps_skipped')})
    for x in laps(r):
        print('   t %.1f %s p90 %s m_fb %s' % x)
