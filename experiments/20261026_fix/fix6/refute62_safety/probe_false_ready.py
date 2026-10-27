"""反証 6-2・安全: EscTrack の「下がれる」(中立 READY_NEUTRAL_S = 0.08 秒)が ESC より早いときの形を狙って振る(読むだけ)。
場面 rev(前の壁 2.0 m・モデル 前進 120 µs。fix6/safety_r1/exp_grid と同じ)。最初のリカバリーの 'brake'(ESC は前進のあとの
ブレーキ = 下がれない)の途中に
  A: カメラの止まり(中立止めの中立が 0.08〜0.12 秒になる長さ)
  B: そのあと、'brake'・'wait' の間に始まる車両の値の止まり(中立止めの中立が短い長さ)
を入れる。ESC は 'fake'(閉じたループの偽物: 中立 0.1 秒で下がれる)と 'slow'(中立 0.12 秒・後退側から 0.10 秒)。
今(now1)・直す前(bef1 = fix5 のあと)・止まりなし(ng1)を比べる: 最初のリカバリーの escape までに下がった距離・escape が
前の壁を押した秒・'reverse' で後退側を出したのに下がらなかった秒・やり直し(+50 µs)の有無。

    python -I probe_false_ready.py [esc]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c62 as K          # noqa: E402
import tf62 as T         # noqa: E402
E = K.E
from test_launch_drive import C, N   # noqa: E402

TB = 2.09     # 場面 rev の 'brake' の始め(止まりなし)


def metrics(sim, rec, pca, p):
    m = T.win(sim, pca, p, t_from=1.5)
    if m is None:
        return None
    tb, te = m['t_brake'], m['t_escape']
    end = te if te is not None else tb + 3.0
    # 後退側(> N + 40)を書いているのに車が止まっている(|v| < 0.05)秒
    dead = sum(T.DT for t, x, v, th in sim.log if tb <= t < end and th > N + 40 and abs(v) < 0.05)
    retry = any(abs(th - (N + p['rev_us'] + 50)) < 1e-6 for t, s, th in pca.w if tb <= t <= end)
    holds = [(round((t - K.T0) / 1e9, 3)) for t, r in rec.rows if r['hold'] and tb - 0.2 <= (t - K.T0) / 1e9 <= end]
    neu = []
    # 中立(N)を書き続けた区間の長さ
    run0 = None
    for t, s, th in pca.w:
        if tb - 0.1 <= t <= end:
            if abs(th - N) < 1e-6 and run0 is None:
                run0 = t
            elif abs(th - N) >= 1e-6 and run0 is not None:
                neu.append((round(run0, 3), round(t - run0, 3)))
                run0 = None
    return {'t_brake': tb, 't_escape': te, 'back_m': m['back_m'], 'esc_push_s': m['esc_push_s'],
            'rev_dead_s': round(dead, 2), 'retry': retry, 'guards': E.guards(rec)[:4], 'neutral_runs': neu[:6],
            'front_hits': m['front_hits'][:3]}


def main(esc='fake'):
    if esc == 'slow':
        T.ESC.update(neu=0.12, rdelay=0.10)
    p = E.P1()
    out = {'esc': esc, 'cells': {}}
    try:
        base = T.scene('rev', 2.0, 120.0, 0.5)
        for nm, mod in (('ng1', 'now'),):
            sim, rec, pca = T.run(mod, p, base)
            out['ng1'] = metrics(sim, rec, pca, p)
            print('ng1', out['ng1'], flush=True)
        for a0 in (0.02, 0.05, 0.08):                    # A の始め(TB から)
            for aL in (0.15, 0.17, 0.19, 0.21):          # A の長さ
                for b0 in (0.0, 0.03, 0.06):             # B の始め(A の終わりから)
                    for bL in (0.17, 0.19, 0.21, 0.23):  # B の長さ
                        a = (round(TB + a0, 3), round(TB + a0 + aL, 3))
                        b = (round(a[1] + b0, 3), round(a[1] + b0 + bL, 3))
                        sc = dict(base, drops=[a], veh_skip=[b])
                        row = {}
                        for nm, mod in (('now1', 'now'), ('bef1', 'fix6_before')):
                            sim, rec, pca = T.run(mod, p, sc)
                            row[nm] = metrics(sim, rec, pca, p)
                        k = 'A=%s B=%s' % (a, b)
                        out['cells'][k] = row
                        n1, b1 = row['now1'], row['bef1']
                        print(k, 'now1', (n1['back_m'], n1['esc_push_s'], n1['rev_dead_s'], n1['retry'], n1['guards'][:2]),
                              'bef1', (b1['back_m'], b1['esc_push_s'], b1['rev_dead_s'], b1['retry'], b1['guards'][:2]),
                              flush=True)
    finally:
        K.cleanup()
    (HERE / 'out' / ('probe_false_ready_%s.json' % esc)).write_text(json.dumps(out, ensure_ascii=False, indent=1,
                                                                               default=str))


if __name__ == '__main__':
    main(*sys.argv[1:2])
