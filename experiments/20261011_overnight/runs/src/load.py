"""Loader for the 2026-10-10 evening AI runs (ai_run_009/010/101).

Time base: t = 0 at the first AI_CMD timestamp of the log (same as tools/drive_report.py).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, '/home/jetson/jetformer')
RAW = '/home/jetson/jetformer/data/raw'
G = 9.80665
KV = 0.5e-3          # m/s per eRPM (forward = negative eRPM)
CENTER, NEUTRAL = 1443.0, 1498.0
NAMES = {'20261010-203924': 'ai_run_009', '20261010-204208': 'ai_run_010', '20261010-204605': 'ai_run_101',
         '20261010-181504': 'ai_run_008', '20261010-173445': 'ai_run_007', '20261010-172910': 'ai_run_006'}


def seg_mask(mask):
    """list of (i0, i1) inclusive index runs where mask is True"""
    m = np.asarray(mask, bool)
    if not m.any():
        return []
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    s = np.nonzero(d == 1)[0]
    e = np.nonzero(d == -1)[0] - 1
    return list(zip(s.tolist(), e.tolist()))


class L(object):
    def __init__(self, log, tof=False):
        self.log = log
        self.name = NAMES.get(log, log)
        root = os.path.join(RAW, log)
        self.root = root
        self.meta = json.load(open(os.path.join(root, 'log.json'), encoding='utf-8'))
        self.scenes = sorted(s for s in os.listdir(root) if s.startswith(log + '-'))

        def load(s, n):
            p = os.path.join(root, s, n + '.npz')
            if not os.path.exists(p):
                return None
            try:
                d = np.load(p, allow_pickle=True)
                return {k: d[k] for k in d.files}
            except Exception as e:  # noqa
                print('skip', s, n, e)
                return None
        parts = {n: [load(s, n) for s in self.scenes] for n in ('AI_CMD', 'VEHICLE', 'IMU', 'CAM_FRONT')}

        def cat(n, k):
            xs = [p[k] for p in parts[n] if p is not None and k in p and len(p['timestamps_us'])]
            return np.concatenate(xs) if xs else np.zeros(0)
        a = {k: cat('AI_CMD', k) for k in ('timestamps_us', 'pred', 'raw', 'out', 'gains', 'lookahead', 'hold', 'select',
                                           'latency_ms', 'infer_ms', 'assist', 'model_out', 'ttc', 'guard',
                                           'guard_applied')}
        self.t0 = int(a['timestamps_us'][0])
        sec = lambda x: (np.asarray(x, np.int64) - self.t0) / 1e6
        self.at = sec(a['timestamps_us'])
        self.out = a['out'].astype(float)
        self.raw = a['raw'].astype(float)
        self.pred = a['pred']
        self.gains = a['gains'].astype(float)
        self.la = a['lookahead'].astype(int)
        self.hold = a['hold'].astype(bool)
        self.sel = a['select'].astype(bool)
        self.lat = a['latency_ms'].astype(float)
        self.assist = a['assist'].astype(int)
        self.mout = a['model_out'].astype(float)
        self.ttc = a['ttc'].astype(float)
        self.guard = a['guard'].astype(int)
        self.gapplied = a['guard_applied'].astype(bool)
        # scene index per AI frame (for video lookup)
        self.a_scene = np.concatenate([np.full(len(p['timestamps_us']), i) for i, p in enumerate(parts['AI_CMD'])
                                       if p is not None])
        v = {k: cat('VEHICLE', k) for k in ('timestamps_us', 'pwm', 'select', 'erpm', 'board_t_us', 'seq', 'valid',
                                            'hall_count')}
        self.vt = sec(v['timestamps_us'])
        self.erpm = v['erpm'].astype(float)
        self.pwm = v['pwm'].astype(float)
        self.vsel = v['select'].astype(bool)
        self.board = v['board_t_us'].astype(np.int64)
        self.vseq = v['seq'].astype(np.int64)
        self.valid = v['valid'].astype(bool)
        self.hall = v['hall_count'].astype(np.int64)
        i = {k: cat('IMU', k) for k in ('timestamps_us', 'accel', 'gyro')}
        self.it = sec(i['timestamps_us'])
        self.acc = i['accel'].astype(float)
        self.gyr = i['gyro'].astype(float)
        self.hg = np.hypot(self.acc[:, 0], self.acc[:, 1]) / G          # horizontal |a| [g] (as drive.py / LapCounter)
        self.g3 = np.abs(np.linalg.norm(self.acc, axis=1) - G) / G       # ||a|-1g| (as collision_map)
        self.ct = sec(cat('CAM_FRONT', 'timestamps_us'))
        self.cf = [os.path.join(root, s, str(f)) for s, p in zip(self.scenes, parts['CAM_FRONT']) if p is not None
                   for f in p['files']]
        self.c_scene = [s for s, p in zip(self.scenes, parts['CAM_FRONT']) if p is not None for _ in p['files']]
        self.scene_t0 = {}
        for s, p in zip(self.scenes, parts['AI_CMD']):
            if p is not None and len(p['timestamps_us']):
                self.scene_t0[s] = float(sec(p['timestamps_us'][:1])[0])
        self.params = (self.meta.get('drive') or {}).get('params') or {}
        if tof:
            self.load_tof()

    def load_tof(self):
        ts, ds = [], []
        for s in self.scenes:
            p = os.path.join(self.root, s, 'TOF_FRONT.npz')
            if os.path.exists(p):
                d = np.load(p)
                ts.append(d['timestamps_us']); ds.append(d['depth'])
        self.tt = (np.concatenate(ts).astype(np.int64) - self.t0) / 1e6
        self.depth = np.concatenate(ds)
        return self

    # --- helpers
    def at_t(self, ts, arr, t):
        i = np.clip(np.searchsorted(ts, t, side='right') - 1, 0, len(ts) - 1)
        return arr[i]

    def erpm_at(self, t):
        return self.at_t(self.vt, self.erpm, t)

    def sel_at(self, t):
        return self.at_t(self.at, self.sel, t)

    def frame_at(self, t):
        """camera file nearest to t"""
        i = int(np.clip(np.searchsorted(self.ct, t), 0, len(self.ct) - 1))
        return self.cf[i]

    def events(self, gthr=2.5, merge=0.6, horiz=True):
        """impacts: horizontal |a| >= gthr g (as drive.py/LapCounter); merged within `merge` s; flags single-sample spikes"""
        g = self.hg if horiz else self.g3
        ev = []
        for i in np.nonzero(g >= gthr)[0]:
            if ev and self.it[i] - ev[-1]['t'] <= merge:
                if g[i] > ev[-1]['g']:
                    ev[-1]['g'] = float(g[i]); ev[-1]['i'] = int(i)
                continue
            near = (np.abs(self.it - self.it[i]) <= 0.05) & (g > 1.0)
            ev.append({'t': float(self.it[i]), 'g': float(g[i]), 'i': int(i), 'suspect': bool(near.sum() <= 1)})
        for e in ev:
            e['ai'] = bool(self.sel_at(e['t']))
            e['v_before'] = float(-np.min(self.erpm[(self.vt >= e['t'] - 0.3) & (self.vt < e['t'])], initial=0) * KV)
            e['erpm_before'] = float(np.median(self.erpm[(self.vt >= e['t'] - 0.2) & (self.vt < e['t'])])) \
                if ((self.vt >= e['t'] - 0.2) & (self.vt < e['t'])).any() else 0.0
            ax = self.acc[e['i'], 0] / G; ay = self.acc[e['i'], 1] / G
            e['ax'], e['ay'] = float(ax), float(ay)
        return ev
