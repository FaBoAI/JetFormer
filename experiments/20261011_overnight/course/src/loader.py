import os, json, numpy as np
RAW = '/home/jetson/jetformer/data/raw'
G = 9.80665

class L:
    def __init__(self, log, imgs=True):
        root = os.path.join(RAW, log); self.root = root; self.log = log
        self.meta = json.load(open(os.path.join(root, 'log.json'), encoding='utf-8'))
        sc = sorted(s for s in os.listdir(root) if s.startswith(log + '-'))
        good = []
        for s in sc:
            try:
                for n in ('VEHICLE', 'IMU', 'AI_CMD'):
                    np.load(os.path.join(root, s, n + '.npz'), allow_pickle=True)['timestamps_us']
                good.append(s)
            except Exception as e:
                print('skip', s, e)
        self.scenes = good
        def cat(name, key):
            return np.concatenate([np.load(os.path.join(root, s, name + '.npz'), allow_pickle=True)[key] for s in good])
        at = cat('AI_CMD', 'timestamps_us').astype(np.int64)
        self.t0 = int(at[0])
        sec = lambda a: (a.astype(np.int64) - self.t0) / 1e6
        self.at = sec(at)
        keys = np.load(os.path.join(root, good[0], 'AI_CMD.npz')).files
        for k in ('out', 'raw', 'model_out', 'gains', 'lookahead', 'assist', 'ttc', 'guard', 'select', 'hold'):
            setattr(self, 'a_' + k, cat('AI_CMD', k) if k in keys else None)
        self.sel = self.a_select.astype(bool)
        self.vt = sec(cat('VEHICLE', 'timestamps_us')); self.pwm = cat('VEHICLE', 'pwm').astype(float)
        self.erpm = cat('VEHICLE', 'erpm').astype(float); self.vsel = cat('VEHICLE', 'select').astype(bool)
        self.hall = cat('VEHICLE', 'hall_count').astype(np.int64); self.board = cat('VEHICLE', 'board_t_us').astype(np.int64)
        self.it = sec(cat('IMU', 'timestamps_us')); self.acc = cat('IMU', 'accel').astype(float); self.gyr = cat('IMU', 'gyro').astype(float)
        self.ct = sec(cat('CAM_FRONT', 'timestamps_us'))
        self.cf = [os.path.join(root, s, str(f)) for s in good for f in np.load(os.path.join(root, s, 'CAM_FRONT.npz'), allow_pickle=True)['files']]
        self.g = np.abs(np.linalg.norm(self.acc, axis=1) - G) / G

    def events(self, gthr=2.5):
        ev = []
        for i in np.nonzero(self.g > gthr)[0]:
            if ev and self.it[i] - ev[-1]['t'] <= 0.6:
                if self.g[i] > ev[-1]['g']:
                    ev[-1]['g'] = float(self.g[i]); ev[-1]['i'] = int(i); ev[-1]['tp'] = float(self.it[i])
                continue
            near = (np.abs(self.it - self.it[i]) <= 0.05) & (self.g > 1.0)
            ev.append({'t': float(self.it[i]), 'tp': float(self.it[i]), 'g': float(self.g[i]), 'i': int(i), 'suspect': bool(near.sum() <= 1)})
        return [e for e in ev if not e['suspect']]

    def at_t(self, ts, arr, t):
        i = np.clip(np.searchsorted(ts, t, side='right') - 1, 0, len(ts) - 1)
        return arr[i]
