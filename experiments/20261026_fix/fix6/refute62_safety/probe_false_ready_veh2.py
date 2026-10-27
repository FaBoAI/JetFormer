"""反証 6-2・安全: probe_false_ready の A をカメラでなく車両の値の止まりにした形(車両の値が 2 回続けて途切れる)を、元の偽物の
ESC(tests/test_fix3_r2_1.Car)で振る(読むだけ)。場面 rev・既定の rev_s 0.4・rev_us 300。
    python -I probe_false_ready_veh2.py
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


def main():
    p = E.P1()
    out = {}
    n_bad = n = 0
    try:
        base = T.scene('rev', 2.0, 120.0, 0.5)
        for a0 in (2.10, 2.12, 2.14):
            for aL in (0.22, 0.24, 0.26, 0.28):
                for b0 in (0.02, 0.05, 0.08):
                    for bL in (0.18, 0.21, 0.24):
                        a = (a0, round(a0 + aL, 3))
                        b = (round(a[1] + b0, 3), round(a[1] + b0 + bL, 3))
                        sc = dict(base, veh_skip=[a, b])
                        row = {}
                        for nm, mod in (('now1', 'now'), ('bef1', 'fix6_before')):
                            sim, rec, pca = E.run(mod, p, sc)
                            row[nm] = P.metrics(sim, rec, pca, p)
                        k = 'A=%s B=%s' % (a, b)
                        out[k] = row
                        n += 1
                        n1, b1 = row['now1'], row['bef1']
                        bad = n1['back_m'] < 0.1 and n1['esc_push_s'] > 0
                        n_bad += bad
                        if bad or b1['esc_push_s'] > 0:
                            print(k, 'now1', (n1['back_m'], n1['esc_push_s'], n1['rev_dead_s'], n1['guards'][:1],
                                              n1['neutral_runs'][:2]),
                                  'bef1', (b1['back_m'], b1['esc_push_s'], b1['rev_dead_s']), flush=True)
    finally:
        K.cleanup()
    print('cells', n, 'now1 no-back + escape push', n_bad)
    (HERE / 'out' / 'probe_false_ready_veh2.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main()
