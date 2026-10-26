"""反証 V-drive: 周ごとにモデルの強さがばらつく(lap_noise)ときの自動合わせ(閉ループ、b 1.0、tg 0.9、150 秒 ≈ 17 周)"""
import sys

import numpy as np

sys.path.insert(0, '/home/jetson/jetformer')
sys.path.insert(0, '/home/jetson/jetformer/experiments/20261025_speed_control/verify')
import vdrive_harness as H  # noqa: E402
from sim_source import SimSource  # noqa: E402
from jetformer import drive  # noqa: E402
from verify_off import PB  # noqa: E402

P0 = dict(PB, throttle_gain=0.9, start_kick=0, speed_ceiling=1, speed_auto=1)
for noise in (0.02, 0.04):
    for seed in (1, 2, 3):
        src = SimSource(t_end_s=150.0, b=1.0, lap_noise=noise, seed=seed)
        r = H.run_loop(drive, P0, src, src.pred_of, floor_ref=None)
        laps = [(x['sa_lap_p90'], x['sa_m_fb'], int(x['sa_event'])) for _, x in r['rec'].rows if np.isfinite(x['sa_lap_s'])]
        mfb = np.array([l[1] for l in laps])
        p90 = np.array([l[0] for l in laps])
        print('ばらつき %.2f seed %d 周 %d p90 %.2f±%.2f m_fb %.3f〜%.3f(最後 %.3f)知らせ %s 周ごとの p90 %s' % (
            noise, seed, len(laps), p90.mean(), p90.std(), mfb.min(), mfb.max(), mfb[-1], r['st']['auto_end']['warn_seen'],
            ' '.join('%.2f' % x for x in p90)))
