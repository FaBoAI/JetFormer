"""Dump the renderer's state (make_digest.Drive.state) over a time window: python dump_window.py LOG T0 T1 [STEP]"""
import sys, os
sys.path.insert(0, '/home/jetson/jetformer_reports/overnight_20261011/video/tools')
import numpy as np
from make_digest import Drive
log, t0, t1 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
step = float(sys.argv[4]) if len(sys.argv) > 4 else 0.1
d = Drive(log)
root = os.path.join('/home/jetson/jetformer/data/raw', log)
sc = sorted(s for s in os.listdir(root) if s.startswith(log + '-'))
ai = [dict(np.load(os.path.join(root, s, 'AI_CMD.npz'), allow_pickle=True)) for s in sc]
mo = np.concatenate([a['model_out'] for a in ai]) if all('model_out' in a for a in ai) else None
gains = np.concatenate([a['gains'] for a in ai]) if all('gains' in a for a in ai) else None
look = np.concatenate([a['lookahead'] for a in ai]) if all('lookahead' in a for a in ai) else None
for t in np.arange(t0, t1 + 1e-9, step):
    st = d.state(t)
    a = d.idx(d.ai_t, t)
    extra = ''
    if mo is not None:
        extra += ' mo=%s' % np.round(mo[a]).astype(int).tolist()
    if gains is not None:
        extra += ' g=%s' % np.round(gains[a], 2).tolist()
    if look is not None:
        extra += ' la=%s' % look[a]
    lap = '%d周 el=%s last=%s best=%s' % (st['laps_done'] + 1, None if st['lap_elapsed'] is None else round(st['lap_elapsed'], 1),
                                       None if not st['last_lap'] else st['last_lap']['time_s'], st['best_lap'])
    print('%6.2f sel=%d hold=%d as=%d st=%4.0f thr=%4.0f erpm=%6d g=%4.1f ttc=%s guard=%d %s%s veh=%d' % (
        t, st['sel'], st['hold'], int(d.assist[a]), st['steer'], st['thr'], st['erpm'], st['gmag'],
        None if st['ttc'] is None else round(st['ttc'], 2), st['guard'], lap, extra, st['veh_ok']))
