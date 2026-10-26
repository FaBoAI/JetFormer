"""反証 V-drive: 自動合わせの 1 フレームの時間の内訳(どのフレームが遅いか)。No.9・ON・電圧あり"""
import collections
import sys

import numpy as np

sys.path.insert(0, '/home/jetson/jetformer')
sys.path.insert(0, '/home/jetson/jetformer/experiments/20261025_speed_control/verify')
import vdrive_harness as H  # noqa: E402
from jetformer import drive  # noqa: E402
from verify_off import PB, vb_good  # noqa: E402

log = []
orig = drive._SpeedAuto._timed


def timed(self, t0, close):
    if close:
        ms = self._frame_ms + (drive.time.perf_counter() - t0) * 1000.0
        log.append((ms, int(self.event), len(self.frames), self.eff_n))
    return orig(self, t0, close)


drive._SpeedAuto._timed = timed
for rep in range(2):
    log.clear()
    src = H.LogSource('20261024-162033', vbat=vb_good)
    r = H.run_loop(drive, dict(PB, speed_ceiling=1, speed_auto=1), src, src.pred_of, description='V_before=8.30')
    a = np.array([x[0] for x in log])
    by = collections.defaultdict(list)
    for ms, ev, n, e in log:
        by[ev].append(ms)
    top = sorted(log, reverse=True)[:6]
    print('回 %d: 平均 %.3f ms・99%% 点 %.3f・最大 %.3f。出来事ごとの最大 %s。遅い 6 フレーム (ms, 出来事, 周のフレーム数) %s' % (
        rep, a.mean(), np.percentile(a, 99), a.max(), {k: round(max(v), 3) for k, v in by.items()},
        [(round(x[0], 3), x[1], x[2]) for x in top]))
