"""反証 6-2・安全: probe_false_ready.py の印のついた場面を、閉じたループの元の偽物の ESC(tests/test_fix3_r2_1.Car そのまま。
fix6/safety_r1/exp_common.run と同じ)で回す(読むだけ)。中立止めの中立 0.10 秒(A)のあと 'brake' の途中に車両の値の止まり(B)。
    python -I probe_false_ready_orig.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c62 as K          # noqa: E402
import tf62 as T         # noqa: E402
import probe_false_ready as P   # noqa: E402
E = K.E

CELLS = [((2.11, 2.26), (2.26, 2.43)), ((2.11, 2.26), (2.26, 2.45)), ((2.11, 2.26), (2.29, 2.46)),
         ((2.11, 2.28), (2.28, 2.45)), ((2.14, 2.29), (2.29, 2.46)), ((2.14, 2.29), (2.29, 2.50)),
         ((2.11, 2.26), (2.26, 2.49)), ((2.14, 2.29), (2.29, 2.52))]


def main():
    p = E.P1()
    out = {}
    try:
        base = T.scene('rev', 2.0, 120.0, 0.5)
        sim, rec, pca = E.run('now', p, base)
        out['ng1'] = P.metrics(sim, rec, pca, p)
        print('ng1', out['ng1'], flush=True)
        for a, b in CELLS:
            sc = dict(base, drops=[a], veh_skip=[b])
            row = {}
            for nm, mod, pp in (('now1', 'now', p), ('bef1', 'fix6_before', p), ('now0', 'now', dict(p, rec_hold_keep=0)),
                                ('off', 'now', dict(p, rec_hold_keep=0, rec_back_guard=0))):
                sim, rec, pca = E.run(mod, pp, sc)
                row[nm] = P.metrics(sim, rec, pca, pp)
                row[nm]['revlog_on'] = next((t for t, r, c in sim.car.revlog if t > 2.0 and r), None)
            k = 'A=%s B=%s' % (a, b)
            out[k] = row
            print(k, {nm: (r['back_m'], r['esc_push_s'], r['rev_dead_s'], r['guards'][:1], r['revlog_on'],
                           r['neutral_runs'][:2]) for nm, r in row.items()}, flush=True)
    finally:
        K.cleanup()
    (HERE / 'out' / 'probe_false_ready_orig.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main()
