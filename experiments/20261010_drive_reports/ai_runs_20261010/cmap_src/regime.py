from r1stat import *
import os
def gains_tl(d):
    G_, LA, T = [], [], []
    for s in d.scenes:
        z = np.load(os.path.join(d.root, s, 'AI_CMD.npz'), allow_pickle=True)
        G_.append(z['gains']); LA.append(z['lookahead']); T.append(z['timestamps_us'])
    return (np.concatenate(T).astype(np.int64) - d.t0) / 1e6, np.concatenate(G_), np.concatenate(LA)
if __name__ == '__main__':  # _MAINGUARD_
    out = {}
    for log in ['20261010-172910', '20261010-173445']:
        d, rows, _, tk, segs = analyze(log)
        gt, gg, gl = gains_tl(d)
        an = [a for a, dep, hit in true_r1(d, r1_anchors(d), rows)]
        laps = []
        for k in range(len(an) - 1):
            a, b = an[k], an[k+1]
            m = (d.at >= a) & (d.at < b)
            if not d.sel[m].all() or b - a > 12: 
                laps.append(dict(a=a, b=b, ai=False)); continue
            gs_ = gg[(gt >= a) & (gt < b)]
            ev = [r for r in rows if a - 0.05 <= r['t'] < b - 0.05]
            mv = (d.vt >= a) & (d.vt < b)
            laps.append(dict(a=a, b=b, ai=True, sg=float(gs_[:,0].mean()), tg=float(gs_[:,1].mean()), la=int(gl[(gt>=a)][0]),
                v_med=float(np.median(-d.erpm[mv])), v_p90=float(np.percentile(-d.erpm[mv], 90)),
                r1=[r['g'] for r in ev if r['t']-a < 1.0], gm=[r['g'] for r in ev if 1.0 <= r['t']-a < 2.6],
                de=[r['g'] for r in ev if 2.6 <= r['t']-a < 4.8], oth=[r['g'] for r in ev if r['t']-a >= 4.8]))
        out[log] = laps
        print(log)
        for L in laps:
            if not L['ai']: print('  %6.1f-%6.1f  (not full AI lap / broken)' % (L['a'], L['b'])); continue
            print('  %6.1f %5.2fs sg %.2f tg %.2f la %d v_med %5.0f p90 %5.0f R1 %s GM %s DE %s other %s' % (L['a'], L['b']-L['a'], L['sg'], L['tg'], L['la'], L['v_med'], L['v_p90'], L['r1'], L['gm'], L['de'], L['oth']))
    import pickle; pickle.dump(out, open('regime.pkl', 'wb'))
