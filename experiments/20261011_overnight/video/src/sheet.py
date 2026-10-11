"""Camera contact sheet: python sheet.py LOG T0 T1 STEP OUT.jpg  (frames as the renderer picks them)"""
import sys
sys.path.insert(0, '/home/jetson/jetformer_reports/overnight_20261011/video/tools')
import cv2
import numpy as np
from make_digest import Drive
log, t0, t1, step, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
d = Drive(log)
tiles = []
for t in np.arange(t0, t1 + 1e-9, step):
    st = d.state(t)
    im = cv2.resize(d.frame(t), (240, 240))
    txt = ['%.2fs sel%d hold%d' % (t, st['sel'], st['hold']), 'erpm %d ttc %s g%d' % (st['erpm'], None if st['ttc'] is None else round(st['ttc'], 2), st['guard']),
           'st %d th %d gm %.1f' % (st['steer'] - d.center, st['thr'] - d.neutral, st['gmag'])]
    for i, s in enumerate(txt):
        cv2.putText(im, s, (4, 16 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(im, s, (4, 16 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    tiles.append(im)
cols = 8
while len(tiles) % cols:
    tiles.append(np.zeros_like(tiles[0]))
rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])
print(out)
