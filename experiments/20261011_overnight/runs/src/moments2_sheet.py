"""fig/video_moments_v2.jpg: 6 frames per moment of moments2.json"""
import json
import sys
import cv2
import numpy as np
sys.path.insert(0, '/home/jetson/jetformer_reports/overnight_20261011/runs/src')
from load import L
from sheets import label
mo = json.load(open('moments2.json'))
rows, cache = [], {}
for m in mo:
    d = cache.get(m['log']) or L(m['log']); cache[m['log']] = d
    ims = []
    for t in np.linspace(m['t_start'], m['t_end'], 6):
        k = int(np.clip(np.searchsorted(d.ct, t), 0, len(d.ct) - 1))
        a = int(np.clip(np.searchsorted(d.at, t), 0, len(d.at) - 1))
        im = cv2.imread(d.cf[k])
        tg = ('v %.1f ttc %.2f g%d' % (-d.erpm_at(t) * 5e-4, min(d.ttc[a], 9.9), d.guard[a]) if len(d.ttc) == len(d.at)
              else 'v %.1f out %d/%d' % (-d.erpm_at(t) * 5e-4, d.out[a, 0], d.out[a, 1]))
        im = label(im, ['%s %.2fs' % (d.name, t), tg, 'sel %d assist %d' % (d.sel[a], d.assist[a])] +
                   ([m['id']] if t == m['t_start'] else []))
        ims.append(im)
    rows.append(np.hstack(ims))
cv2.imwrite('/home/jetson/jetformer_reports/overnight_20261011/runs/fig/video_moments_v2.jpg', np.vstack(rows),
            [cv2.IMWRITE_JPEG_QUALITY, 75])
print('ok', len(rows))
