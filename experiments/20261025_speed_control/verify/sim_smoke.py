import sys, collections
sys.path.insert(0, '/home/jetson/jetformer')
sys.path.insert(0, '/home/jetson/jetformer/experiments/20261025_speed_control/verify')
import numpy as np
import vdrive_harness as H
from sim_source import SimSource
from jetformer import drive
from verify_off import PB
for b in (0.9, 1.0, 1.15):
    src = SimSource(t_end_s=45.0, b=b)
    r = H.run_loop(drive, dict(PB, throttle_gain=1.0, start_kick=0), src, src.pred_of, floor_ref=None)
    rows = [x for _, x in r['rec'].rows]
    lp = [x['sa_lap_p90'] for x in rows if np.isfinite(x['sa_lap_p90'])]
    print('b', b, 'laps', np.round(src.lap_times(), 2), 'p90', np.round(lp, 2), {n: round(src.seg_max(n, 10), 2) for n in ('R1','GM','DE','HS','LH','SW','X')},
          'ev', collections.Counter(int(x['sa_event']) for x in rows), r['st']['auto_end'])
