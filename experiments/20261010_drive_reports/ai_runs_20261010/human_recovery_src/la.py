import os, json, numpy as np, sys
sys.path.insert(0, '/home/jetson/jetformer')
from jetformer import tof_format
RAW = '/home/jetson/jetformer/data/raw'
G = 9.80665

def _ld(p, keys):
    try:
        z = np.load(p, allow_pickle=True)
        out = {k: z[k] for k in keys if k in z.files}
        return out
    except Exception as e:
        return None

class LA:
    """Generic loader for any log (human-only or AI). Skips scenes whose npz fail to load."""
    def __init__(self, log, tof=True):
        self.log = log
        root = os.path.join(RAW, log)
        self.meta = json.load(open(os.path.join(root, 'log.json'), encoding='utf-8'))
        sc = sorted(s for s in os.listdir(root) if s.startswith(log + '-'))
        self.bad = []
        streams = {'VEHICLE': ['timestamps_us', 'pwm', 'erpm', 'select', 'hall_count', 'valid', 'moving'],
                   'IMU': ['timestamps_us', 'accel', 'gyro'],
                   'AI_CMD': ['timestamps_us', 'out', 'select', 'hold'],
                   'TOF_FRONT': ['timestamps_us', 'depth'] if tof else [],
                   'CAM_FRONT': ['timestamps_us', 'files']}
        data = {k: [] for k in streams}
        self.scene_ranges = []
        for s in sc:
            for name, keys in streams.items():
                if not keys: continue
                p = os.path.join(root, s, name + '.npz')
                if not os.path.exists(p): continue
                d = _ld(p, keys)
                if d is None or 'timestamps_us' not in d or any(k not in d for k in keys if name != 'VEHICLE' or k in ('timestamps_us','pwm','erpm','select')):
                    self.bad.append((s, name)); continue
                if name == 'CAM_FRONT':
                    d['files'] = np.array([os.path.join(root, s, str(f)) for f in d['files']])
                data[name].append(d)
        def cat(name, key):
            parts = [d[key] for d in data[name] if key in d]
            return np.concatenate(parts) if parts else None
        vts = cat('VEHICLE', 'timestamps_us')
        o = np.argsort(vts, kind='stable')
        self.t0 = int(vts[o[0]])
        sec = lambda a: (a - self.t0) / 1e6
        self.vt = sec(vts[o]); self.pwm = cat('VEHICLE', 'pwm')[o].astype(float); self.erpm = cat('VEHICLE', 'erpm')[o].astype(float)
        self.vsel = cat('VEHICLE', 'select')[o].astype(bool)
        hc = cat('VEHICLE', 'hall_count'); self.hall = hc[o].astype(float) if hc is not None else None
        va = cat('VEHICLE', 'valid'); self.valid = va[o] if va is not None else np.ones((len(o), 3), bool)
        its = cat('IMU', 'timestamps_us')
        if its is not None:
            o2 = np.argsort(its, kind='stable')
            self.it = sec(its[o2]); self.acc = cat('IMU', 'accel')[o2].astype(float); self.gyr = cat('IMU', 'gyro')[o2].astype(float)
            self.g = np.abs(np.linalg.norm(self.acc, axis=1) - G) / G
        else:
            self.it = None
        ats = cat('AI_CMD', 'timestamps_us')
        if ats is not None:
            o3 = np.argsort(ats, kind='stable')
            self.at = sec(ats[o3]); self.out = cat('AI_CMD', 'out')[o3].astype(float); self.sel = cat('AI_CMD', 'select')[o3].astype(bool)
            self.hold = cat('AI_CMD', 'hold')[o3].astype(bool)
        else:
            self.at = None
        tts = cat('TOF_FRONT', 'timestamps_us') if tof else None
        if tts is not None:
            o4 = np.argsort(tts, kind='stable')
            self.tt = sec(tts[o4]); self.depth = cat('TOF_FRONT', 'depth')[o4]
            d = tof_format.distance_mm(self.depth[:, 8:30, 12:42].reshape(len(self.depth), -1).astype(np.uint16)).astype(float)
            d[d <= 0] = np.nan
            with np.errstate(all='ignore'):
                import warnings; warnings.simplefilter('ignore')
                self.front = np.nanpercentile(d, 5, axis=1) / 1000.0
        else:
            self.tt = None
        cts = cat('CAM_FRONT', 'timestamps_us')
        if cts is not None:
            o5 = np.argsort(cts, kind='stable'); self.ct = sec(cts[o5]); self.cf = cat('CAM_FRONT', 'files')[o5]
        else:
            self.ct = None
        dr = self.meta.get('drive') or {}
        self.center = dr.get('steer_center_us', 1443.0); self.neutral = dr.get('throttle_neutral_us', 1498.0)

    @staticmethod
    def at_t(ts, arr, t):
        i = np.clip(np.searchsorted(ts, t, side='right') - 1, 0, len(ts) - 1)
        return arr[i]
