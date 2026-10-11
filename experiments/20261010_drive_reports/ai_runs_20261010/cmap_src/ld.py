import os, json, numpy as np, sys
sys.path.insert(0, '/home/jetson/jetformer')
from jetformer import tof_format
SNAP = '/tmp/claude-1000/-home-jetson-jetracer/f44d13ba-776f-4620-a46a-2574918483a9/scratchpad/cmap/snap'
G = 9.80665
# original time base (first AI_CMD timestamp of the original scene 0001), absolute us
T0 = {}

class D:
    def __init__(self, log, t0_us=None):
        root = os.path.join(SNAP, log); self.root = root; self.log = log
        self.meta = json.load(open(os.path.join(root, 'log.json'), encoding='utf-8'))
        sc = sorted(s for s in os.listdir(root) if s.startswith(log + '-'))
        good = []
        for s in sc:
            try:
                for n in ('VEHICLE', 'IMU', 'CAM_FRONT', 'TOF_FRONT'):
                    np.load(os.path.join(root, s, n + '.npz'), allow_pickle=True)['timestamps_us']
                good.append(s)
            except Exception as e:
                print('skip', s, e)
        self.scenes = good
        def cat(name, key):
            parts = []
            for s in good:
                p = os.path.join(root, s, name + '.npz')
                if os.path.exists(p):
                    parts.append(np.load(p, allow_pickle=True)[key])
            return np.concatenate(parts) if parts else None
        self.has_ai = os.path.exists(os.path.join(root, good[0], 'AI_CMD.npz'))
        vt = cat('VEHICLE', 'timestamps_us')
        if t0_us is None:
            t0_us = int(cat('AI_CMD', 'timestamps_us')[0]) if self.has_ai else int(vt[0])
        self.t0 = t0_us
        sec = lambda a: (a.astype(np.int64) - self.t0) / 1e6
        if self.has_ai:
            self.at = sec(cat('AI_CMD', 'timestamps_us')); self.out = cat('AI_CMD', 'out'); self.sel = cat('AI_CMD', 'select').astype(bool)
            self.raw = cat('AI_CMD', 'raw'); self.pred = cat('AI_CMD', 'pred')
        self.vt = sec(vt); self.pwm = cat('VEHICLE', 'pwm').astype(float); self.erpm = cat('VEHICLE', 'erpm').astype(float)
        self.vsel = cat('VEHICLE', 'select').astype(bool)
        self.it = sec(cat('IMU', 'timestamps_us')); self.acc = cat('IMU', 'accel'); self.gyr = cat('IMU', 'gyro')
        self.ct = sec(cat('CAM_FRONT', 'timestamps_us'))
        self.cf = [os.path.join(root, s, str(f)) for s in good for f in np.load(os.path.join(root, s, 'CAM_FRONT.npz'), allow_pickle=True)['files']]
        self.g = np.abs(np.linalg.norm(self.acc, axis=1) - G) / G
        self.mode_id = self.meta.get('mode_id')

    def tof(self):
        self.tt = (np.concatenate([np.load(os.path.join(self.root, s, 'TOF_FRONT.npz'))['timestamps_us'] for s in self.scenes]).astype(np.int64) - self.t0) / 1e6
        self.depth = np.concatenate([np.load(os.path.join(self.root, s, 'TOF_FRONT.npz'))['depth'] for s in self.scenes])
        return self

    def heading(self, bias=None):
        gz = self.gyr[:, 2].astype(float)
        if bias is None:
            er = np.interp(self.it, self.vt, self.erpm)
            m = (np.abs(er) < 1) & (np.abs(gz) < 5)
            bias = np.median(gz[m]) if m.sum() > 50 else 0.0
        self.bias = bias
        dt = np.diff(self.it, prepend=self.it[0])
        self.hd = np.cumsum((gz - bias) * dt)
        return self.hd

    def at_t(self, ts, arr, t):
        i = np.clip(np.searchsorted(ts, t, side='right') - 1, 0, len(ts) - 1)
        return arr[i]

    def events(self, gthr=2.5):
        ev = []
        for i in np.nonzero(self.g > gthr)[0]:
            if ev and self.it[i] - ev[-1]['t'] <= 0.6:
                if self.g[i] > ev[-1]['g']:
                    ev[-1]['g'] = float(self.g[i]); ev[-1]['i'] = int(i)
                continue
            near = (np.abs(self.it - self.it[i]) <= 0.05) & (self.g > 1.0)
            ev.append({'t': float(self.it[i]), 'g': float(self.g[i]), 'i': int(i), 'suspect': bool(near.sum() <= 1)})
        return ev

def load(log):
    if log == '20261010-172910':
        root = os.path.join(SNAP, log)
        sc = sorted(s for s in os.listdir(root) if s.startswith(log + '-'))
        t = int(np.load(os.path.join(root, sc[0], 'AI_CMD.npz'))['timestamps_us'][0])
        return D(log, t - 20022000)
    return D(log)

import cv2
def sheet(paths, labels, ncol, out, scale=1.0):
    ims = []
    for p, lb in zip(paths, labels):
        im = cv2.imread(p) if p and os.path.exists(p) else np.zeros((224, 224, 3), np.uint8)
        if scale != 1.0: im = cv2.resize(im, None, fx=scale, fy=scale)
        for k, line in enumerate(lb.split('\n')):
            cv2.putText(im, line, (3, 14 + 14 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(im, line, (3, 14 + 14 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        ims.append(im)
    h, w = ims[0].shape[:2]
    while len(ims) % ncol: ims.append(np.zeros_like(ims[0]))
    rows = [np.hstack(ims[i:i + ncol]) for i in range(0, len(ims), ncol)]
    cv2.imwrite(out, np.vstack(rows))
