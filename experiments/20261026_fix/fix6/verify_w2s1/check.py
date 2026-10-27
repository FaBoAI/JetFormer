"""反証の確かめ(w2-safety-w2-safety-r62-safety-1。読むだけ・直さない)。refute62_safety の部品で、指摘の場面を
長め(12 秒)に回し、最初のリカバリーのあと(2 回目のリカバリー・後ろの当たり・中止)まで見る。一時の場所は fix6/tmp_verify_w2s1。
    python -I check.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'refute62_safety'))
import c62 as K          # noqa: E402
MY = HERE.parent / 'tmp_verify_w2s1'
K.TMPROOT = MY
K.R.TMP = MY / 'rs'
K.R.CL.TMP = K.R.TMP
K.E.TMP = MY / 'exp'
import tf62 as T         # noqa: E402
E = K.E


def summary(sim, rec, pca):
    from test_launch_drive import N
    w = [(round(t, 2), round(th)) for t, s, th in pca.w]
    esc = [t for t, th in w if th == round(N - 130)]
    xs = [(round(t, 2), round(x, 3)) for t, x, v, th in sim.log]
    x0 = sim.s['xf']
    xmin = min(x for t, x in xs if t > 2.0)
    revon = []
    prev = False
    for t, r, c in sim.car.revlog:
        if r and not prev:
            revon.append(round(t, 2))
        prev = r
    gu = [round((t - K.T0) / 1e9, 2) for t, r in rec.rows if r.get('assist') == 3][:1]
    return {'guards': E.guards(rec)[:6], 'escape_starts': sorted(set(round(t, 1) for t in esc))[:12],
            'max_back_m': round(x0 - xmin, 3), 'esc_rev_on': revon[:6], 'hits': sim.car.hits[:6], 'giveup_t': gu}


def main():
    p = E.P1()
    out = {}
    try:
        for esc_neu in (0.1, 0.12, 0.07):
            T.ESC.update(neu=esc_neu)
            base = dict(T.scene('rev', 2.0, 120.0, 0.5), dur=12.0)
            for a, b in (((2.11, 2.26), (2.26, 2.45)), ((2.14, 2.29), (2.29, 2.46)), ((2.11, 2.28), (2.28, 2.45))):
                sc = dict(base, drops=[a], veh_skip=[b])
                for nm, mod, pp in (('now1', 'now', p), ('bef1', 'fix6_before', p),
                                    ('now0', 'now', dict(p, rec_hold_keep=0)),
                                    ('off', 'now', dict(p, rec_hold_keep=0, rec_back_guard=0))):
                    sim, rec, pca = T.run(mod, pp, sc)
                    k = 'neu%.2f A=%s B=%s %s' % (esc_neu, a, b, nm)
                    out[k] = summary(sim, rec, pca)
                    print(k, out[k], flush=True)
            sim, rec, pca = T.run('now', p, base)
            out['neu%.2f nogap now1' % esc_neu] = summary(sim, rec, pca)
            print('neu%.2f nogap now1' % esc_neu, out['neu%.2f nogap now1' % esc_neu], flush=True)
    finally:
        K.cleanup()
    (HERE / 'check.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main()
