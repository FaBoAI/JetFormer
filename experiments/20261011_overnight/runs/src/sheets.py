"""Contact sheets: camera frames around AI-side impacts (for location classification)."""
import os
import sys

import cv2
import numpy as np

from load import L, KV

OUT = '/home/jetson/jetformer_reports/overnight_20261011/runs/fig'


def label(im, lines, color=(255, 255, 255)):
    for k, line in enumerate(lines):
        cv2.putText(im, line, (3, 14 + 14 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(im, line, (3, 14 + 14 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    return im


def frame(d, t):
    i = int(np.clip(np.searchsorted(d.ct, t), 0, len(d.ct) - 1))
    im = cv2.imread(d.cf[i])
    if im is None:
        im = np.zeros((224, 224, 3), np.uint8)
    return im, i


def row(d, t, offs, head):
    ims = []
    for o in offs:
        im, i = frame(d, t + o)
        a = int(np.clip(np.searchsorted(d.at, t + o), 0, len(d.at) - 1))
        lines = ['%+.1fs' % o] if o != offs[0] else head + ['%+.1fs' % o]
        lines.append('st %+.0f th %+.0f' % (d.out[a, 0] - 1443, d.out[a, 1] - 1498))
        lines.append('erpm %.0f ttc %.2f g%d' % (d.erpm_at(t + o), min(d.ttc[a], 9.9), d.guard[a]))
        lines.append('sel %d as %d' % (d.sel[a], d.assist[a]))
        ims.append(label(im.copy(), lines))
    return np.hstack(ims)


def sheet(items, offs, out):
    rows = [row(d, t, offs, head) for d, t, head in items]
    cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == '__main__':
    logs = ['20261010-203924', '20261010-204208', '20261010-204605']
    items = []
    for log in logs:
        d = L(log)
        for e in d.events(2.5):
            if e['ai']:
                items.append((d, e['t'], ['%s %.2fs %.1fg' % (d.name, e['t'], e['g'])]))
    offs = [-1.0, -0.5, -0.2, 0.0, 0.5]
    for k in range(0, len(items), 6):
        sheet(items[k:k + 6], offs, os.path.join(OUT, 'impacts_%d.jpg' % (k // 6)))
    print(len(items))
