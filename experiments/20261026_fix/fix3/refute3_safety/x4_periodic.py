"""X4: 短い中立止めがくり返す(カメラが周期的に止まる)ときの残したリカバリー。下がる距離・後ろの当たり・後退を書いた秒"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cl3                                    # noqa: E402
from cl3 import C, N, PARAMS                  # noqa: E402


def one(keep, period, length, phase0=0.0, xr_back=1.45, rev_s=0.4, rev_us=300.0, t_from=1.6, dur=8.0, guard=1):
    xf = 2.0
    drops = []
    t = t_from + phase0
    while t < dur:
        drops.append((t, t + length))
        t += period
    sc_ = {'dur': dur, 'xf': xf, 'model': (C, N - 120.0), 'drops': drops}
    if xr_back is not None:
        sc_['xr'] = xf - xr_back
    p = dict(PARAMS, recovery=1, start_kick=0, rec_back_guard=guard, rec_hold_keep=keep, rev_s=rev_s, rev_us=rev_us)
    sim, rec, pca, ph = cl3.run(p, sc_)
    xs = [r[1] for r in sim.log if r[0] > 1.5]
    rev_written = sum(t1 - t0 for (t0, s, th), (t1, *_r) in zip(pca.w, pca.w[1:]) if th > N + 40 and t0 > 1.5)
    rear = [h for h in sim.car.hits if h[1] == 'rear']
    ev = [e[1]['what'] for e in rec.events if e[0] == 'recovery_guard']
    tr = [(t, p0, p1) for t, p0, p1 in cl3.phases(ph)][:12]
    return {'back_m': round(xf - min(xs), 3), 'rear_hits': rear, 'rev_side_written_s': round(rev_written, 3),
            'guard_events': ev, 'giveups': sum(1 for _, r in rec.rows if not r['hold'] and int(r['assist']) == 3),
            'phases': tr}


def main():
    res = {}
    for keep in (0, 1):
        res['nohold keep=%d' % keep] = one(keep, 99.0, 0.0)
    for period, length in ((0.25, 0.12), (0.3, 0.15), (0.4, 0.2), (0.5, 0.3), (0.6, 0.45), (0.8, 0.45)):
        for ph0 in (0.0, 0.07, 0.15):
            for keep in (0, 1):
                k = 'per=%.2f len=%.2f ph=%.2f keep=%d' % (period, length, ph0, keep)
                res[k] = one(keep, period, length, ph0)
    for k, v in res.items():
        print(k, {kk: v[kk] for kk in ('back_m', 'rear_hits', 'rev_side_written_s', 'guard_events', 'giveups')})
    out = Path(__file__).resolve().parent / 'out'
    out.mkdir(exist_ok=True)
    (out / 'x4_periodic.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    shutil.rmtree(cl3.TMP, ignore_errors=True)


if __name__ == '__main__':
    main()
