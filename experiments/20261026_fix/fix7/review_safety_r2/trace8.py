"""反証 7-2 回目・safety(直さない)。1 つの場面の段の並び・PCA9685 に書いた値・偽物の ESC の後退を時刻で並べる(読むだけ)。
    python -I trace8.py <variant on|off|f7b_on|g_on> <scene json>
scene json の例: '{"kind": "rev", "drops": [[2.45, 2.58]], "glitch": [[2.62, 1322]], "esc": null, "veh_dt": null}'
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault('RV8_TAG', 'trace')
import h8  # noqa: E402
from g8 import scene, first_rec  # noqa: E402

N = h8.N
VAR = {'on': ('now', h8.ON), 'off': ('now', h8.OFF), 'f7b_on': ('f7b', h8.ON),
       'g_on': ('now', lambda p: dict(p, rec_back_guard=1, rec_hold_keep=0))}


def build(js):
    d = json.loads(js)
    sc = scene(d.get('kind', 'rev'), skips=[tuple(x) for x in d.get('skips', [])],
               drops=[tuple(x) for x in d.get('drops', [])], rear=d.get('rear'),
               dur=d.get('dur', 9.0), esc=tuple(d['esc']) if d.get('esc') else None)
    if d.get('glitch'):
        sc['glitch'] = [tuple(x) for x in d['glitch']]
    if d.get('veh_dt'):
        sc['veh_dt'] = d['veh_dt']
    if d.get('cam0'):
        sc['cam0'] = d['cam0']
    return sc


ESCREV = []


def _wrap(orig, ph, force=None):
    """h8._wrap と同じ。esc_rev(drive が渡した値)と esc_ran も控える"""
    def step(self, model_us, erpm, now, impact_g=0.0, *a, **kw):
        p0 = self.phase
        r = orig(self, model_us, erpm, now, impact_g, *a, **kw)
        ph.append((round(now - 100.0, 4), p0, self.phase, kw.get('hold_since') is not None,
                   round(float(kw.get('hold_rev_s', 0.0)), 4), round(float(getattr(self, 'est_s', 0.0)), 4),
                   round(float(self.backed_s), 4), round(float(erpm), 1)))
        ESCREV.append((kw.get('esc_rev', 'n/a'), getattr(self, 'esc_ran', 'n/a'), round(float(getattr(self, 'seen_rev_s', 0.0)), 3)))
        return r
    return step


def main(var, js, t0=2.2, t1=5.5, kw='{}'):
    h8._wrap = _wrap
    name, f = VAR[var]
    p = h8.base_params(**json.loads(kw))
    sc = build(js)
    try:
        res = h8.run(name, f(p), sc)
    finally:
        h8.cleanup()
    sim, rec, pca, ph = res
    print('first_rec', json.dumps(first_rec(res), default=str))
    print('events', [(e[1]['what'], e[1]['phase'], e[1]['backed_s']) for e in rec.events if e[0] == 'recovery_guard'])
    ws = {round(t, 3): th for t, s, th in pca.w}
    last = None
    for (t, p0, p1, hs, hr, est, bs, e), (er, eran, srs) in zip(ph, ESCREV):
        if t0 <= t <= t1:
            th = None
            for tw in sorted(ws):
                if tw >= t - 1e-6:
                    th = ws[tw]
                    break
            mark = '' if (p0, p1) == last else '<-'
            last = (p0, p1)
            print('%.3f %-8s -> %-8s hold=%d rev_s=%.3f est=%.3f backed=%.3f seen=%.3f erpm=%7.1f esc_rev=%-5s ran=%-5s thr=%s %s' %
                  (t, p0, p1, hs, hr, est, bs, srs, e, er, eran, None if th is None else round(th - N, 1), mark))
    rv = [t for t, r, c in sim.car.revlog if r]
    print('esc_rev_on first', rv[:1], 'n', len(rv))
    print('hits', sim.car.hits)
    xs = [(round(t, 2), round(x, 3)) for t, x, *_ in sim.log if t0 <= t <= t1 and abs(round(t * 100) % 10) == 0]
    print('x', xs)


if __name__ == '__main__':
    main(*sys.argv[1:3], *[float(x) for x in sys.argv[3:5]], *sys.argv[5:6])
