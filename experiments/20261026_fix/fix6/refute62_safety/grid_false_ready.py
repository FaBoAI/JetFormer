"""反証 6-2・安全: カメラの止まり A(最初のリカバリーの 'brake' の始めから a0 秒・長さ aL)と、その終わりから b0 秒あとの車両の値の
止まり B(長さ bL)の組を、元の偽物の ESC(tests/test_fix3_r2_1.Car)・既定の rev_s 0.4・rev_us 300 で振る(読むだけ)。場面 rev・brake。
今(now1)と直す前(bef1 = fix5 のあと)で、最初のリカバリーで下がらずに escape が前の壁を押した場面を数える。
    python -I grid_false_ready.py <rev|brake>
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


def main(kind='rev'):
    p = E.P1()
    if kind == 'rev':
        base = T.scene('rev', 2.0, 120.0, 0.5)
    else:
        base = T.scene('brake', 2.4, 120.0, 0.45)
    sim, rec, pca = E.run('now', p, base)
    ng = P.metrics(sim, rec, pca, p)
    tb = ng['t_brake']
    print('ng1', ng, flush=True)
    out = {'kind': kind, 'ng1': ng, 'cells': {}}
    cnt = {'n': 0, 'now1_fail': 0, 'bef1_fail': 0, 'now1_push': 0, 'bef1_push': 0}
    try:
        for a0 in (0.0, 0.03, 0.06, 0.1, 0.15, 0.2):
            for aL in (0.13, 0.15, 0.17, 0.19, 0.21):
                for b0 in (0.0, 0.03, 0.06, 0.1):
                    for bL in (0.17, 0.2, 0.25, 0.3):
                        a = (round(tb + a0, 3), round(tb + a0 + aL, 3))
                        b = (round(a[1] + b0, 3), round(a[1] + b0 + bL, 3))
                        sc = dict(base, drops=[a], veh_skip=[b])
                        row = {}
                        for nm, mod in (('now1', 'now'), ('bef1', 'fix6_before')):
                            s_, r_, pc_ = E.run(mod, p, sc)
                            row[nm] = P.metrics(s_, r_, pc_, p)
                        cnt['n'] += 1
                        for nm in ('now1', 'bef1'):
                            m = row[nm]
                            if m['esc_push_s'] > 0:
                                cnt[nm + '_push'] += 1
                            if m['esc_push_s'] > 0 and m['back_m'] < 0.1:
                                cnt[nm + '_fail'] += 1
                        k = 'A=%s B=%s' % (a, b)
                        out['cells'][k] = row
                        if row['now1']['esc_push_s'] > 0 or row['bef1']['esc_push_s'] > 0:
                            print(k, {nm: (m['back_m'], m['esc_push_s'], m['rev_dead_s'], m['guards'][:1],
                                           m['neutral_runs'][:2]) for nm, m in row.items()}, flush=True)
    finally:
        K.cleanup()
    out['count'] = cnt
    print(cnt)
    (HERE / 'out' / ('grid_false_ready_%s.json' % kind)).write_text(json.dumps(out, ensure_ascii=False, indent=1,
                                                                               default=str))


if __name__ == '__main__':
    main(*sys.argv[1:2])
