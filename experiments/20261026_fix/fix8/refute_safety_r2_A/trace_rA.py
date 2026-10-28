"""票 A: 1 場面で、止まりなし(ON・OFF)と、'reverse' が終わるフレームから k フレームずらした所から L フレームのカメラの止まりを並べる。
    python -I -B trace_rA.py ESC XF M K L [PHASE] [PROC] → 標準出力
"""
import hashlib
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rA  # noqa: E402

DS = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8]


def show(tag, sim, rec, pca, ph, t_ref):
    t0, t1 = rA.first_rec(ph)
    x0, back, vpk, hits = rA.profile(sim, t0, DS)
    te = rA.rev_end(ph, t0)
    phs, pp = [], None
    for tt, p0, p1, *_ in ph:
        if t0 - 0.01 <= tt <= (t1 or 99) and p1 != pp:
            phs.append((tt, p1))
            pp = p1
    holds = [(round((t - rA.CL.T0) / 1e9, 3), r['out'][1]) for t, r in rec.rows if r['hold'] and t0 - 0.2 <= (t - rA.CL.T0) / 1e9 <= t0 + 3]
    print('%-10s rec %.3f-%s rev_end %s back %.3f m peak %.2f m/s hits %s' % (tag, t0, t1, te, back, vpk, hits))
    print('           writes', rA.writes(pca, t_ref - 0.1, t_ref + 0.5))
    print('           phases', [p for p in phs if p[0] >= t_ref - 0.6][:8])
    print('           hold rows', holds[:6], 'events', [e[1]['what'] for e in rec.events if e[0] == 'recovery_guard'])
    return back, vpk, hits


def main():
    for fn, h in rA.NOW.items():
        assert hashlib.sha256((rA.ROOT / 'jetformer' / fn).read_bytes()).hexdigest() == h, fn
    esc, xf, m, k, ln = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    phase = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
    proc = float(sys.argv[7]) if len(sys.argv) > 7 else 0.0
    try:
        t_s = time.time()
        base = {}
        for sw in ('on', 'off'):
            base[sw] = rA.run(esc, sw, rA.model_fwd(m), xf=xf, phase=phase, proc=proc)
        sim, rec, pca, ph = base['on']
        t0, _ = rA.first_rec(ph)
        te = rA.rev_end(ph, t0)
        fi = rA.frame_at(sim, te)
        print('esc %s xf %.2f m %.0f phase %.4f proc %.3f: no-stall rev_end %.3f = frame %s; drop frames %d..%d' % (
            esc, xf, m, phase, proc, te, fi, fi + k, fi + k + ln - 1))
        same = [(r[1]['out'], r[1]['assist']) for r in base['on'][1].rows] == [(r[1]['out'], r[1]['assist']) for r in base['off'][1].rows]
        print('no-stall ON == OFF rows(out, assist):', same, ' writes same:', base['on'][2].w == base['off'][2].w)
        show('base_on', *base['on'], te)
        show('base_off', *base['off'], te)
        for sw in ('on', 'off', 'guard'):
            r = rA.run(esc, sw, rA.model_fwd(m), xf=xf, phase=phase, proc=proc, drop_frames=[(fi + k, ln)])
            show('drop_' + sw, *r, te)
        print('(%.1fs)' % (time.time() - t_s))
    finally:
        rA.cleanup()


if __name__ == '__main__':
    main()
