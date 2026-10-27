"""反証 6-1 の指摘 w1-safety-r61-safety-1 を打ち消す試み(読むだけ・直さない・装置に触れない・GPU を使わない)。
本物の drive._loop を fix3/refute2_safety/closed_loop の偽物の ESC・車で回す(fix4/review_safety/rs_common.run_closed)。
指摘の台本は使わず、ここで場面を組み直して確かめる。一時の場所は fix6/tmp_r61s1(終わったら消す)。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I \
        experiments/20261026_fix/fix6/refute_r61_s1/verify_s3_rev_start.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIX = HERE.parents[1]
sys.path.insert(0, str(FIX / 'fix4' / 'review_safety'))
import rs_common as R          # noqa: E402

R.TMP = FIX / 'fix6' / 'tmp_r61s1' / 'rs'
R.CL.TMP = R.TMP
R.SNAP_SHA['fix5_safety_before'] = {'drive.py': 'c575af1262836c3e10bfe32b2ee34293117a2144b3e568acb326ada57d59a471',
                                    'assist.py': 'b2bc818c5d8c8d410269f3aaf852d458fcc0c31d8f514002d8e34aa2c163fae6'}
A, C, N, PARAMS, T0 = R.A, R.C, R.N, R.PARAMS, R.T0
XF = 2.0
CALLS = []


def wrap(orig):
    def step(self, model_us, erpm, now, impact_g=0.0, *a, **kw):
        p0, b0 = self.phase, self.backed_s
        r = orig(self, model_us, erpm, now, impact_g, *a, **kw)
        CALLS.append({'t': round(now - 100.0, 3), 'p0': p0, 'p1': self.phase, 'erpm': float(erpm),
                      'held': kw.get('hold_since') is not None, 'hb': round(float(kw.get('hold_back_s', 0.0)), 3),
                      'hu': round(float(kw.get('hold_unseen_s', 0.0)), 3), 'b0': round(b0, 3), 'b1': round(self.backed_s, 3),
                      'out': None if r[0] is None else [round(float(r[0][0]), 1), round(float(r[0][1]), 1)]})
        return r
    return step


def run(p, skip, mod=None, xr=None, dur=6.0):
    del CALLS[:]
    sc = {'dur': dur, 'xf': XF, 'model': (C, N - 120.0), 'drops': [], 'veh_skip': [skip] if skip else []}
    if xr is not None:
        sc['xr'] = xr
    kw = {} if mod is None else {'mod': mod[0], 'amod': mod[1]}
    sim, rec, pca, d = R.run_closed(p, sc, wrap=wrap, **kw)
    calls = list(CALLS)
    t_hit = next((h[0] for h in sim.car.hits if h[1] == 'front'), None)
    rev_on = next((t for t, x, v, thr, con, rv, sl in sim.log if t_hit and t > t_hit and rv), None)
    after_hit = [(t, x, v, thr, con) for t, x, v, thr, con, rv, sl in sim.log if t_hit and t > t_hit]
    xmin = min(x for t, x, *_ in after_hit) if after_hit else None
    # 前の壁へ前進を書いていた秒(当たりのあと・接している間)
    push = sum(0.01 for t, x, v, thr, con in after_hit if con == 'front' and thr < N - 20 and t > (rev_on or 1e9))
    # 後退側(> N+40)を書いていた秒(当たりのあと)
    rev_side = sum(0.01 for t, x, v, thr, con in after_hit if thr > N + 40 and t >= (rev_on or 1e9) - 0.2)
    holds = [((t - T0) / 1e9) for t, r in rec.rows if r['hold']]
    return {'t_front_hit': t_hit, 't_rev_on': rev_on, 'backed_m': None if xmin is None else round(XF - xmin, 3),
            'front_push_after_rev_s': round(push, 2), 'rev_side_s': round(rev_side, 2),
            'hits': sim.car.hits, 'hold_rows': [round(x, 3) for x in holds[:20]],
            'events': [(e[1].get('what'), e[1].get('phase'), e[1].get('backed_s')) for e in rec.events if e[0] == 'recovery_guard'],
            'calls': calls, 'log': [(round(t, 2), round(x, 3), round(v, 2), thr, rv) for t, x, v, thr, con, rv, sl in sim.log]}


def short(r):
    return {k: r[k] for k in ('t_front_hit', 't_rev_on', 'backed_m', 'front_push_after_rev_s', 'rev_side_s', 'hits', 'events')}


def main():
    BEF = R.load_snapshot('fix5_safety_before')
    out = {}
    try:
        p1 = dict(PARAMS, recovery=1, start_kick=0, rec_back_guard=1, rec_hold_keep=1, tof_guard=1, rev_s=0.4, rev_us=300.0)
        p0 = dict(p1, rec_hold_keep=0)
        poff = dict(p1, rec_hold_keep=0, rec_back_guard=0)
        print('defaults', A.DEFAULT_REV_S, A.DEFAULT_REV_US, 'PARAMS rev', PARAMS.get('rev_s'), PARAMS.get('rev_us'))
        base = run(p1, None)
        t0 = base['t_rev_on']
        out['base_keep1'] = short(base)
        out['base_keep0'] = short(run(p0, None))
        out['base_off'] = short(run(poff, None))
        print('base', out['base_keep1'])
        # 1 場面を細かく(指摘の 0.0 / 0.55)
        skip = (round(t0, 3), round(t0 + 0.55, 3))
        tr = run(p1, skip)
        out['trace_keep1'] = short(tr)
        out['trace_keep1']['calls'] = [c for c in tr['calls'] if t0 - 0.4 <= c['t'] <= t0 + 1.6]
        out['trace_keep1']['log'] = [x for x in tr['log'] if t0 - 0.1 <= x[0] <= t0 + 1.6 and int(round(x[0] * 100)) % 2 == 0]
        out['trace_keep1']['hold_rows'] = tr['hold_rows']
        out['trace_keep0'] = short(run(p0, skip))
        out['trace_before_keep1'] = short(run(p1, skip, BEF))
        # 窓を振る: 始め d0 × 長さ L
        grid = {}
        for d0 in (-0.2, -0.15, -0.12, -0.1, -0.08, -0.06, -0.04, -0.02, 0.0, 0.02, 0.05, 0.1):
            for L in (0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7):
                sk = (round(t0 + d0, 3), round(t0 + d0 + L, 3))
                a = run(p1, sk)
                b = run(p0, sk)
                c = run(p1, sk, BEF)
                grid['d0=%+.2f L=%.2f' % (d0, L)] = {'keep1': a['backed_m'], 'keep0': b['backed_m'], 'before': c['backed_m'],
                                                     'keep1_push': a['front_push_after_rev_s'], 'before_push': c['front_push_after_rev_s'],
                                                     'keep1_ev': a['events'][:2]}
        out['grid'] = grid
        out['base_backed_m'] = base['backed_m']
        # 後ろの物
        rear = {}
        for rr in (1.30, 1.35, 1.40, 1.45, 1.50, 1.60, 1.70):
            xr = round(XF - rr, 3)
            for nm, p, sk, mod in (('keep1_gap', p1, skip, None), ('keep0_gap', p0, skip, None), ('keep1_nogap', p1, None, None),
                                   ('off_gap', poff, skip, None), ('before_keep1_gap', p1, skip, BEF)):
                r = run(p, sk, mod, xr=xr)
                rear['rear=%.2f %s' % (rr, nm)] = {'rear_hits': [h for h in r['hits'] if h[1] == 'rear'],
                                                   'backed_m': r['backed_m'], 'push': r['front_push_after_rev_s']}
        out['rear'] = rear
    finally:
        R.cleanup()
    (HERE / 'out' / 'verify_s3_rev_start.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    for k in ('base_keep1', 'base_keep0', 'base_off', 'trace_keep1', 'trace_keep0', 'trace_before_keep1'):
        print(k, {kk: out[k][kk] for kk in ('t_rev_on', 'backed_m', 'front_push_after_rev_s', 'rev_side_s', 'events')},
              [h for h in out[k]['hits']])
    print('--- trace calls (keep1)')
    for c in out['trace_keep1']['calls']:
        if c['p0'] != c['p1'] or c['held'] or abs(c['t'] - t0) < 0.3:
            print(c)
    print('hold rows', out['trace_keep1']['hold_rows'])
    print('--- log')
    for x in out['trace_keep1']['log']:
        print(x)
    print('--- grid (base %.3f)' % base['backed_m'])
    for k, v in grid.items():
        print(k, v)
    print('--- rear')
    for k, v in rear.items():
        print(k, v)


if __name__ == '__main__':
    main()
