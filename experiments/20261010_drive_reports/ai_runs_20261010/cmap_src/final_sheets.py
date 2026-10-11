import json, cv2, numpy as np
from ld import *
from hum2 import analyze_h
from sheets import put, frame
REP = '/home/jetson/jetformer/data/reports/ai_runs_20261010'
C = json.load(open('classified.json'))
SEC_JA = {'R1': 'R1 right hairpin', 'GM_in': 'GM entry (inner board)', 'GM_out': 'GM exit (outer board)', 'DE': 'DE back-straight end wall',
          'SW': 'SW divider lane', 'LH': 'LH curtain hairpin', 'REC': 'recovery (off-line)'}
_cache = {}
def getd(log):
    if log not in _cache:
        d = load(log) if log in ('20261010-172910', '20261010-173445') else D(log)
        d.heading(bias=None if d.has_ai else -0.0915)
        _cache[log] = d
    return _cache[log]
def ev(log, t):
    return [e for e in C[log] if abs(e['t'] - t) < 0.05][0]
def panel(d, e, sec):
    W, H = 300, 224; im = np.full((H, W, 3), 32, np.uint8); t = e['t']
    a, b = t - 2.0, t + 1.0
    X = lambda tt: int((tt - a) / (b - a) * (W - 10)) + 5
    Y = lambda v: int(H / 2 + 10 - v * 0.15)
    cv2.line(im, (5, Y(0)), (W - 5, Y(0)), (90, 90, 90), 1); cv2.line(im, (X(t), 0), (X(t), H), (0, 0, 255), 1)
    def poly(ts, vals, col):
        if len(ts) < 2: return
        pts = np.stack([[X(x) for x in ts], [Y(v) for v in vals]], 1).astype(np.int32); cv2.polylines(im, [pts], False, col, 1, cv2.LINE_AA)
    if d.has_ai:
        m = (d.at >= a) & (d.at < b)
        poly(d.at[m], d.out[m, 0] - 1443, (255, 255, 0)); poly(d.at[m], d.out[m, 1] - 1498, (0, 165, 255))
        j = np.searchsorted(d.at, t - 0.5)
        if j < len(d.at) and 'sg' in e:
            for k in range(10):
                cv2.circle(im, (X(d.at[j] + (k + 1) / 30), Y(e['sg'] * (d.pred[j, k, 0] - 1443))), 2, (255, 0, 255), -1)
    else:
        m = (d.vt >= a) & (d.vt < b)
        poly(d.vt[m][::2], d.pwm[m, 0][::2] - 1443, (255, 255, 0)); poly(d.vt[m][::2], d.pwm[m, 1][::2] - 1498, (0, 165, 255))
    m = (d.vt >= a) & (d.vt < b); poly(d.vt[m][::3], -d.erpm[m][::3] / 15, (0, 200, 0))
    m = (d.it >= a) & (d.it < b); poly(d.it[m], d.gyr[m, 2] - d.bias, (200, 200, 200))
    who = ('AI' if e['sel'] else 'HUMAN-side') if d.has_ai else 'HUMAN demo'
    put(im, '%s %.2fs %s' % (e['log'][-6:], t, who), 4, 14, 0.45)
    put(im, sec, 4, 29, 0.4, (120, 230, 255))
    put(im, '|a| %.1fg lon %+.1f lat %+.1f(R+)' % (e['g'], e['ay'], e['ax']), 4, 44, 0.4)
    put(im, 'steer %+d thr %+d us (-0.5..0s)' % (e['steer_b'], e['thr_b']), 4, 59, 0.4)
    put(im, 'eRPM %d -> %d' % (e['erpm_b'], e['erpm_a']), 4, 74, 0.4)
    if 'sg' in e:
        put(im, 'gain st %.2f thr %.2f la %d' % (e['sg'], e['tg_gain'], e['la']), 4, 89, 0.4)
        tk = e.get('takeover'); put(im, 'takeover %s' % ('+%.1fs' % tk if tk is not None else '-'), 4, 104, 0.4)
    put(im, 'cyan steer(L+) org thr(fwd-)', 4, H - 34, 0.36); put(im, 'grn speed  wht yaw(L+)', 4, H - 20, 0.36)
    put(im, 'mag: pred steer @-0.5s' if d.has_ai else '-2s .. +1s', 4, H - 6, 0.36)
    return im
def make(items, out, offs=(-1.0, -0.5, 0.0, 0.5), secs=None):
    lines = []
    for k, (log, t) in enumerate(items):
        d = getd(log); e = ev(log, t); sec = SEC_JA.get((secs or {}).get((log, t), e['sec']).rstrip('?'), e['sec'])
        ims = [panel(d, e, sec)]
        for o in offs:
            im = frame(d, t + o).copy(); put(im, '%+.1fs' % o, 4, 14, 0.5, (0, 255, 255) if o == 0 else (255, 255, 255)); ims.append(im)
        lines.append(np.hstack(ims))
    cv2.imwrite(out, np.vstack(lines), [cv2.IMWRITE_PNG_COMPRESSION, 9])
A6, A7 = '20261010-172910', '20261010-173445'
make([(A7, t) for t in (35.12, 58.19, 92.99, 143.17)], REP + '/cs_R1_007_a.png')
make([(A7, t) for t in (157.78, 210.43, 244.30, 271.11)], REP + '/cs_R1_007_b.png')
make([(A6, t) for t in (28.97, 39.35, 57.78, 86.72, 106.00, 115.21)], REP + '/cs_DE_006_a.png')
make([(A6, t) for t in (124.26, 133.28, 152.31, 161.56, 170.89)], REP + '/cs_DE_006_b.png')
make([(A7, t) for t in (41.73, 66.51, 86.47, 114.83, 125.72)], REP + '/cs_DE_007_a.png')
make([(A7, t) for t in (136.04, 174.17, 236.86, 277.49)], REP + '/cs_DE_007_b.png')
make([(A6, 36.48), (A6, 46.87), (A7, 74.77), (A7, 235.25), (A6, 122.73), (A6, 131.90)], REP + '/cs_GM_006_007.png')
make([(A6, 55.44), (A6, 83.19), (A6, 102.87), (A6, 139.02), (A7, 132.12)], REP + '/cs_R1entry_mild.png')
make([(A7, 252.96), (A7, 265.36), (A7, 259.03), (A7, 121.30), (A7, 100.17), (A7, 106.20)], REP + '/cs_other_007.png',
     secs={(A7, 100.17): 'REC', (A7, 106.20): 'REC'})
H = ['20261010-150424', '20261010-152405', '20261010-152707', '20261010-135449']
for h in H: C.setdefault(h, C.get(h, []))
make([('20261010-150424', 116.45), ('20261010-152707', 67.47), ('20261010-152707', 77.20), ('20261010-152405', 86.82),
      ('20261010-152405', 28.67), ('20261010-135449', 49.76), ('20261010-135449', 96.48), ('20261010-152405', 17.09)], REP + '/cs_human_same_spots.png',
     secs={('20261010-152405', 17.09): 'GM_in'})
print('ok')
