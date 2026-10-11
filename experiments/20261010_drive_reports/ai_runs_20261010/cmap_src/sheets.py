from ev import *
import cv2, sys
REP = '/home/jetson/jetformer/data/reports/ai_runs_20261010'
def put(im, txt, x, y, s=0.42, c=(255,255,255)):
    cv2.putText(im, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, s, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(im, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, s, c, 1, cv2.LINE_AA)
def frame(d, t):
    i = min(np.searchsorted(d.ct, t), len(d.ct)-1)
    im = cv2.imread(d.cf[i]); return im if im is not None else np.zeros((224,224,3),np.uint8)
def panel(d, t, r, tag):
    W, H = 300, 224
    im = np.full((H, W, 3), 32, np.uint8)
    # mini plot -2..+1 s : steer out (cyan), steer raw (gray), throttle out (orange), erpm/20 (green), gyro z (white)
    a, b = t-2.0, t+1.0
    def X(tt): return int((tt-a)/(b-a)*(W-10))+5
    def Y(v): return int(H/2 - v*0.17)   # +-600 range
    cv2.line(im, (5, Y(0)), (W-5, Y(0)), (90,90,90), 1)
    cv2.line(im, (X(t), 0), (X(t), H), (0,0,255), 1)
    m = (d.at>=a)&(d.at<b)
    for arr, col in ((d.out[m,0]-1443, (255,255,0)), (d.out[m,1]-1498, (0,165,255))):
        pts = np.stack([[X(x) for x in d.at[m]], [Y(v) for v in arr]],1).astype(np.int32)
        cv2.polylines(im, [pts], False, col, 1)
    m = (d.vt>=a)&(d.vt<b)
    pts = np.stack([[X(x) for x in d.vt[m][::3]], [Y(v) for v in (-d.erpm[m][::3]/15)]],1).astype(np.int32)
    cv2.polylines(im, [pts], False, (0,200,0), 1)
    m = (d.it>=a)&(d.it<b)
    pts = np.stack([[X(x) for x in d.it[m]], [Y(v) for v in (d.gyr[m,2]-d.bias)]],1).astype(np.int32)
    cv2.polylines(im, [pts], False, (200,200,200), 1)
    # pred at t-0.5 (steer after gain): draw as dots ahead
    j = np.searchsorted(d.at, t-0.5)
    if j < len(d.at):
        sg = d.meta['drive']['params']['steer_gain']
        for k in range(10):
            cv2.circle(im, (X(d.at[j]+(k+1)/30), Y(sg*(d.pred[j,k,0]-1443))), 2, (255,0,255), -1)
    put(im, '%s t=%.2fs %s' % (tag, t, 'AI' if r['sel'] else 'HUMAN'), 4, 14, 0.45)
    put(im, '|a|=%.1fg ax=%+.1f ay=%+.1f' % (r['g'], r['ax'], r['ay']), 4, 30)
    put(im, 'steer %+d thr %+d (us, -0.5..0s)' % (r['steer_b'], r['thr_b']), 4, 46)
    put(im, 'eRPM %d -> %d  yaw %+d' % (r['erpm_b'], r['erpm_a'], r['gz_b']), 4, 62)
    tk = r['takeover']; put(im, 'takeover %s' % ('+%.1fs' % tk if tk is not None else '-'), 4, 78)
    put(im, 'cyan=steer org=thr grn=spd', 4, H-20, 0.36)
    put(im, 'wht=yaw mag=pred(t-0.5)', 4, H-6, 0.36)
    return im
def make(log, rows, tags, out, offs=(-1.0, -0.5, 0.0, 0.5)):
    d = load(log); d.heading()
    lines = []
    for r, tag in zip(rows, tags):
        ims = [panel(d, r['t'], r, tag)]
        for o in offs:
            im = frame(d, r['t']+o).copy(); put(im, '%+.1fs' % o, 4, 14, 0.5, (0,255,255) if o==0 else (255,255,255)); ims.append(im)
        lines.append(np.hstack(ims))
    cv2.imwrite(out, np.vstack(lines))
if __name__ == '__main__':
    log = sys.argv[1]; sel = [float(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else None
    d, rows, anchors, tk, segs = analyze(log)
    if sel: rows = [r for r in rows if any(abs(r['t']-s) < 0.05 for s in sel)]
    make(log, rows, [log[-6:] for _ in rows], sys.argv[2])
