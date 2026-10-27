"""反証 6-1 の指摘 w1-safety-r61-safety-1 の 'brake'(back_early)版を独りで確かめる(読むだけ)。
モデルが壁の手前でブレーキ(後退側)を出して止まる → ESC は後退できる状態 → リカバリーの 'brake' ですぐ下がり始める。
その入りぎわに車両の値が来ない区間を置き、下がった距離を比べる。一時の場所は fix6/tmp_r61s1。
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import verify_s3_rev_start as V   # noqa: E402

R, C, N, PARAMS = V.R, V.C, V.N, V.PARAMS
XF = 2.4


def model(t, car):
    return (C, N - 120.0) if car.x < XF - 0.45 and t < 6.0 else (C, N + 200.0)


def run(p, skip, mod=None):
    sc = {'dur': 6.5, 'xf': XF, 'model': model, 'drops': [], 'veh_skip': [skip] if skip else []}
    kw = {} if mod is None else {'mod': mod[0], 'amod': mod[1]}
    del V.CALLS[:]
    sim, rec, pca, d = R.run_closed(p, sc, wrap=V.wrap, **kw)
    calls = list(V.CALLS)
    tb = next((c['t'] for c in calls if c['p1'] == 'brake'), None)
    x_b = next((x for t, x, *_ in sim.log if tb is not None and t >= tb), None)
    win = [x for t, x, *_ in sim.log if tb is not None and tb <= t <= tb + 2.5]
    rv = next((t for t, x, v, thr, con, r, sl in sim.log if tb is not None and t >= tb and r), None)
    return {'t_brake': tb, 't_rev_on': rv, 'backed_m': round(x_b - min(win), 3) if win else None, 'hits': sim.car.hits[:3],
            'events': [(e[1].get('what'), e[1].get('phase'), e[1].get('backed_s')) for e in rec.events if e[0] == 'recovery_guard'][:3],
            'after_hold': [c for c in calls if c['held']][:1]}


def main():
    BEF = R.load_snapshot('fix5_safety_before')
    out = {}
    try:
        p1 = dict(PARAMS, recovery=1, start_kick=0, rec_back_guard=1, rec_hold_keep=1, tof_guard=1, rev_s=0.4, rev_us=300.0)
        p0 = dict(p1, rec_hold_keep=0)
        base = run(p1, None)
        out['base'] = base
        print('base', base)
        t0 = base['t_rev_on']
        for d0 in (-0.10, -0.06, -0.02, 0.0):
            for L in (0.45, 0.52, 0.6):
                sk = (round(t0 + d0, 3), round(t0 + d0 + L, 3))
                r = {nm: run(p, sk, m) for nm, p, m in (('keep1', p1, None), ('keep0', p0, None), ('before', p1, BEF))}
                out['d0=%+.2f L=%.2f' % (d0, L)] = r
                print('d0=%+.2f L=%.2f' % (d0, L), {nm: (x['backed_m'], x['events'][:2]) for nm, x in r.items()},
                      r['keep1']['after_hold'])
    finally:
        R.cleanup()
    (HERE / 'out' / 'verify_brake_ready.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main()
