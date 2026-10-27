"""反証の打ち消し(w2-safety-r62-safety-2): 指摘の場面('reverse' の始め 2.52 秒 + d0 から車両の値が L 秒来ない)を
ESC の形を変えて回す(読むだけ)。
  esc: neu0.2 = 指摘の偽物(中立 0.2 秒が要る = 0.15 秒の 'gap' では毎回下がれない)
       neu0.1 = 閉じたループの偽物の既定 / slow = 中立 0.12 秒・下がり始め 0.10 秒(10/10 の人の値)
       nofire = 中立 0.1 秒で下がれるが、最初に下がれるはずの後退側を 1 回だけ無視する(10/10 の原因不明の不発 1/37 の形)
  組: now1 = 今 keep 1・見張り 1 / now0 = 今 keep 0(既定 OFF)/ bef1 = fix6 の直す前 keep 1
  見るもの: 最初のリカバリーの下がった距離・escape が前の壁を押した秒・10 秒の間の後ろの当たり・前の壁へ前進を出して
  押していた秒(モデル・escape のどちらも)・後ろへ下がった合計
    python -I probe_reach.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
R62 = HERE.parent / 'refute62_safety'
sys.path.insert(0, str(R62))
import c62 as K          # noqa: E402
import tf62 as T         # noqa: E402
import probe_false_ready as P   # noqa: E402
E = K.E
N = P.N


class NoFireCar(T.Car):
    """最初に後退へ入るはずの 1 回だけ入らない(ブレーキのまま。次の中立 ESC['neu'] 秒で下がれる)"""
    def __init__(self, s):
        super().__init__(s)
        self.nofire_left = 1

    def step(self, t, thr):
        was = self.rev_on
        acc, g = super().step(t, thr)
        if self.rev_on and not was and self.nofire_left > 0:
            self.nofire_left -= 1
            self.rev_on = False
            self.ready = False
            self.braked = True
        return acc, g


def whole(sim, p):
    xf = sim.s['xf']
    push = sum(T.DT for t, x, v, th in sim.log if t > 1.6 and x >= xf - 1e-6 and th < N - 20)
    rear = [h for h in sim.car.hits if h[1] == 'rear']
    xs = [x for t, x, v, th in sim.log if t > 1.6]
    return {'push_front_fwd_s': round(push, 2), 'rear_hits': rear[:3], 'min_x_from_wall': round(xf - min(xs), 3),
            'front_hits': [h for h in sim.car.hits if h[1] == 'front'][:4]}


def main():
    out = {}
    try:
        for esc in ('neu0.2', 'neu0.1', 'slow', 'nofire'):
            if esc == 'neu0.2':
                T.ESC.update(neu=0.2, rdelay=0.08)
            elif esc == 'slow':
                T.ESC.update(neu=0.12, rdelay=0.10)
            else:
                T.ESC.update(neu=0.1, rdelay=0.08)
            T.Car_orig = getattr(T, 'Car_orig', T.Car)
            Car = NoFireCar if esc == 'nofire' else T.Car_orig
            for d0, L in ((None, None), (0.0, 0.2), (0.0, 0.16), (0.05, 0.25), (-0.05, 0.22)):
                sc = dict(T.scene('rev', 2.0, 120.0, 0.5), dur=10.0)
                if d0 is not None:
                    sc['veh_skip'] = [(round(2.52 + d0, 3), round(2.52 + d0 + L, 3))]
                row = {}
                for nm, mod, pp in (('now1', 'now', E.P1()), ('now0', 'now', dict(E.P1(), rec_hold_keep=0)),
                                    ('bef1', 'fix6_before', E.P1())):
                    T.Car = Car

                    s_, r_, pc_ = T.run(mod, pp, sc)
                    T.Car = T.Car_orig
                    m = P.metrics(s_, r_, pc_, pp)
                    row[nm] = {'first': None if m is None else {k: m[k] for k in ('back_m', 'esc_push_s', 'retry', 'rev_dead_s')},
                               'whole': whole(s_, pp)}
                k = '%s d0=%s L=%s' % (esc, d0, L)
                out[k] = row
                print(k, json.dumps(row, ensure_ascii=False, default=str), flush=True)
    finally:
        K.cleanup()
    (HERE / 'out' / 'probe_reach.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main()
