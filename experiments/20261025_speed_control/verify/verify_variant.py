"""反証 V-drive: 仕様からの 2 つの直し(start_from_hold・release_bumpless)を閉ループの偽物の車で比べる。
仕様の文のまま(両方 False)と、作ったまま(両方 True)。区間の最高の速さ・SW の後半〜出口の最高・周の時間・中立の数"""
import sys

import numpy as np

sys.path.insert(0, '/home/jetson/jetformer')
sys.path.insert(0, '/home/jetson/jetformer/experiments/20261025_speed_control/verify')
import vdrive_harness as H  # noqa: E402
import sim_source as S  # noqa: E402
from jetformer import drive, safety_caps  # noqa: E402
from verify_off import PB  # noqa: E402

P0 = dict(PB, throttle_gain=0.9, start_kick=0, speed_ceiling=1)
real = safety_caps.ceil_params


def sw_exit_max(src, t_from=10.0):
    """SW の後半(位置 60% 以降)と、その後の直線 X の最高"""
    acc = 0.0
    lo = hi = None
    for ln, deg, name in S.SEG:
        if name == 'SW':
            lo, hi = acc + 0.6 * ln, acc + ln + 1.05
        acc += ln
    xs = [v for t, s, v, n in src.log if t >= t_from and lo <= (s % S.L) < hi]
    return max(xs)


for b in (1.0, 1.15, 1.3):
    for label, flags in (('作ったまま', dict()), ('仕様の文のまま', dict(start_from_hold=False, release_bumpless=False))):
        safety_caps.ceil_params = lambda a=4.5, l=4.0, **kw: real(a, l, **dict(kw, **flags))
        try:
            src = S.SimSource(t_end_s=60.0, b=b)
            r = H.run_loop(drive, P0, src, src.pred_of, floor_ref=None)
        finally:
            safety_caps.ceil_params = real
        lt = src.lap_times()
        print('b %.2f %-8s SW 出口 %.2f LH %.2f HS %.2f DE %.2f 周 %.2f 中立 %d 変えた %d' % (
            b, label, sw_exit_max(src), src.seg_max('LH', 10), src.seg_max('HS', 10), src.seg_max('DE', 10), np.median(lt),
            r['st']['stats']['ceil_neutral'], r['st']['stats']['ceil_applied']))
