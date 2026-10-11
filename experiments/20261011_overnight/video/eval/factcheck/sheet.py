"""Contact sheet of camera frames with telemetry, for fact-checking storyboard clips.
usage: sheet.py LOG T0 T1 STEP OUT.jpg"""
import sys, numpy as np, cv2
sys.path.insert(0,'/home/jetson/jetformer_reports/overnight_20261011/runs/src')
from load import L, KV
log, a, b, step, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
l = L(log)
ev = l.events()
ts = np.arange(a, b + 1e-6, step)
tiles = []
for t in ts:
    f = cv2.imread(l.frame_at(t)); f = cv2.resize(f, (224, 224))
    ai = np.clip(np.searchsorted(l.at, t, side='right') - 1, 0, len(l.at) - 1)
    sel = l.sel[ai]; hold = l.hold[ai]
    if len(l.vt):
        vi = np.clip(np.searchsorted(l.vt, t, side='right') - 1, 0, len(l.vt) - 1)
        er = l.erpm[vi]; pw = l.pwm[vi]
    else:
        er = np.nan; pw = [np.nan, np.nan]
    if sel:
        st, th = l.out[ai]
    else:
        st, th = pw[0], pw[1]
    near = [e for e in ev if abs(e['t'] - t) < step / 2 + 1e-9]
    canvas = np.zeros((224 + 64, 224, 3), np.uint8)
    canvas[:224] = f
    col = (247, 85, 168) if sel else (200, 200, 200)
    if hold: col = (68, 68, 239)
    cv2.rectangle(canvas, (0, 0), (223, 223), col, 3)
    txt = ['t=%.2f %s%s g%d' % (t, 'AI' if sel else 'HU', ' HOLD' if hold else '', l.guard[ai]),
           'erpm %d v%.1f ttc %.2f' % (er, -er * KV, l.ttc[ai] if np.isfinite(l.ttc[ai]) else -1),
           'st %d th %d la%d' % (st, th, l.la[ai])]
    if near: txt.append('IMPACT %.1fG %.2f' % (near[0]['g'], near[0]['t']))
    for k, s in enumerate(txt):
        cv2.putText(canvas, s, (3, 238 + 14 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255) if 'IMPACT' in s else (255, 255, 255), 1)
    tiles.append(canvas)
cols = 8
while len(tiles) % cols: tiles.append(np.zeros_like(tiles[0]))
rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
print(out, len(ts))
