"""反証 6-2・安全: 'reverse' の「下がり始めなければ中立を長くしてやり直す」(CHECK_S・RETRY_GAP_S)が要る ESC(最初の中立 0.15 秒
では下がれず、0.16 秒以上の中立が要る = 本物の ESC の 11% の「後退に入らない」に寄せた偽物)で、後退の入りぎわに車両の値が
1 回途切れる場面を振る(読むだけ)。場面 rev・既定の rev_s 0.4・rev_us 300。今(now1)と直す前(bef1 = fix5 のあと)と止まりなし。
    python -I probe_retry_est.py
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
    T.ESC.update(neu=float(sys.argv[1]) if len(sys.argv) > 1 else 0.2, rdelay=0.08)
    p = E.P1()
    base = T.scene('rev', 2.0, 120.0, 0.5)
    out = {'cells': {}}
    cnt = {'n': 0, 'now1_fail': 0, 'bef1_fail': 0}
    try:
        sim, rec, pca = T.run('now', p, base)
        ng = P.metrics(sim, rec, pca, p)
        out['ng1'] = ng
        print('ng1', ng, 'writes', [(round(t, 2), round(th)) for t, s, th in pca.w if 2.3 <= t <= 3.4][:40], flush=True)
        tg = next(t for t, s, th in pca.w if t > ng['t_brake'] + 0.2 and abs(th - (P.N + 10)) < 1e-6)   # 'gap' の始め
        tr = next(t for t, s, th in pca.w if t > tg and abs(th - (P.N + 300)) < 1e-6)                   # 'reverse' の始め
        out['t_gap'], out['t_rev'] = tg, tr
        for d0 in (-0.1, -0.05, -0.02, 0.0, 0.02, 0.05, 0.1, 0.15, 0.2):
            for L in (0.16, 0.18, 0.2, 0.22, 0.25, 0.3, 0.35, 0.4, 0.45):
                b = (round(tr + d0, 3), round(tr + d0 + L, 3))
                sc = dict(base, veh_skip=[b])
                row = {}
                for nm, mod in (('now1', 'now'), ('bef1', 'fix6_before')):
                    s_, r_, pc_ = T.run(mod, p, sc)
                    row[nm] = P.metrics(s_, r_, pc_, p)
                cnt['n'] += 1
                for nm in ('now1', 'bef1'):
                    if row[nm]['back_m'] < 0.1 and row[nm]['esc_push_s'] > 0:
                        cnt[nm + '_fail'] += 1
                k = 'd0=%+.2f L=%.2f' % (d0, L)
                out['cells'][k] = row
                print(k, {nm: (m['back_m'], m['esc_push_s'], m['rev_dead_s'], m['retry'], m['guards'][:1])
                          for nm, m in row.items()}, flush=True)
    finally:
        K.cleanup()
    out['count'] = cnt
    print(cnt)
    (HERE / 'out' / ('probe_retry_est_%s.json' % T.ESC['neu'])).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main()
