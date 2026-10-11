"""Print what the renderer (make_digest.Drive) will show for each log: impacts, laps, AI on/off, holds, board resets."""
import sys
sys.path.insert(0, '/home/jetson/jetformer_reports/overnight_20261011/video/tools')
import numpy as np
from make_digest import Drive
for log in sys.argv[1:]:
    d = Drive(log)
    print('==', log, d.description, 'dur %.1f' % d.duration, 'gains', d.steer_gain, d.thr_gain, d.lookahead)
    print(' ai_on ', [round(x, 2) for x in d.ai_on])
    print(' ai_off', [round(x, 2) for x in d.ai_off])
    print(' segs  ', [(round(a, 2), round(b, 2)) for a, b in d.segments])
    print(' holds ', [(round(a, 2), round(b, 2)) for a, b in d.holds])
    print(' recov ', d.recoveries)
    print(' board ', d.board_resets)
    print(' impacts(renderer, 3D>2G):', [(round(t, 2), round(g, 1), s) for t, g, s in d.impacts])
    for l in d.laps:
        print(' lap', {k: (round(v, 2) if isinstance(v, float) else v) for k, v in l.items()})
    print(' assist>0 frames', int((d.assist > 0).sum()), 'assist==1 times', np.round(d.ai_t[d.assist == 1], 2)[:40])
