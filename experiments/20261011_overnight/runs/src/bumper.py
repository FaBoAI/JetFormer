"""Camera pitch proxy: height of the car's own blue bumper at the bottom of the 224x224 front image.
For each log, 150 frames spread over the log; per frame, count rows (from the bottom) whose centre columns
(80-144) are >= 50 % 'blue' (HSV hue 95-130, S >= 80, V >= 60). Median per log (and per scene)."""
import json, os, sys
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load import L

LOGS = ['20261010-172910', '20261010-173445', '20261010-181504', '20261010-192141', '20261010-200916',
        '20261010-201626', '20261010-202633', '20261010-203924', '20261010-204208', '20261010-204605']


def bumper_rows(path):
    im = cv2.imread(path)
    if im is None:
        return np.nan
    hsv = cv2.cvtColor(im[150:, 80:144], cv2.COLOR_BGR2HSV)
    m = (hsv[..., 0] >= 95) & (hsv[..., 0] <= 130) & (hsv[..., 1] >= 80) & (hsv[..., 2] >= 60)
    frac = m.mean(axis=1)[::-1]          # from the bottom row up
    n = 0
    for f in frac:
        if f >= 0.5:
            n += 1
        else:
            break
    return n


out = {}
for log in LOGS:
    try:
        root = '/home/jetson/jetformer/data/raw/' + log
        sc = sorted(s for s in os.listdir(root) if s.startswith(log + '-'))
        files = []
        for s in sc:
            p = os.path.join(root, s, 'CAM_FRONT.npz')
            if os.path.exists(p):
                f = np.load(p, allow_pickle=True)['files']
                files += [os.path.join(root, s, str(x)) for x in f]
        idx = np.linspace(0, len(files) - 1, 150).astype(int)
        r = np.array([bumper_rows(files[i]) for i in idx], float)
        meta = json.load(open(root + '/log.json'))
        out[log] = {'desc': meta.get('description'), 'median_rows': float(np.nanmedian(r)),
                    'p25': float(np.nanpercentile(r, 25)), 'p75': float(np.nanpercentile(r, 75)), 'n': int(np.isfinite(r).sum())}
        print(log, out[log])
    except Exception as e:
        print('fail', log, e)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bumper.json'), 'w'), indent=1, ensure_ascii=False)
