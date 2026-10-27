"""指摘 w1-safety-r61-safety-1 の場面を、今の版(now)と直しの候補(cand)で振る(読むだけ・装置に触れない)。

  場面 rev: 前の壁 2.0 m・モデル 前進 120 µs(壁を押す)→ 最初のリカバリーの 'reverse' で偽物の ESC が後退に入る時刻 t_on
            (値の止まりなし)から d0 秒ずらして車両の値が L 秒来ない
  場面 brake: 前の壁 2.4 m・壁の 0.45 m 手前からモデルがブレーキ側(中立 + 200)= ESC は後退を待っている → 'brake' ですぐ下がる
            (back_early)。同じく t_on から d0・L
  既定の rev_s 0.4・rev_us 300(引数で変えられる)

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I \\
        experiments/20261026_fix/fix6/safety_r1/exp_grid.py rev now,cand [rev_s] [rev_us] [out_name]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp_common as E    # noqa: E402
from exp_common import C, N   # noqa: E402

D0 = [-0.2, -0.15, -0.12, -0.1, -0.08, -0.06, -0.04, -0.02, 0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
LS = [0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]


def scene(kind, skip=None, rear=None):
    if kind == 'rev':
        xf, model = 2.0, (C, N - 120.0)
    else:
        xf = 2.4

        def model(t, car):
            return (C, N - 120.0) if car.x < xf - 0.45 and t < 6.0 else (C, N + 200.0)
    sc = {'dur': 6.5, 'xf': xf, 'model': model, 'drops': [], 'veh_skip': [skip] if skip else []}
    if rear is not None:
        sc['xr'] = xf - rear
    return sc, xf


def one(name, p, kind, skip=None, rear=None):
    sc, xf = scene(kind, skip, rear)
    sim, rec, pca = E.run(name, p, sc)
    m = E.first_rec(sim, pca, xf, rev_us=p['rev_us'])
    m['guards'] = E.guards(rec)[:4]
    return m


def main():
    kind = sys.argv[1]
    names = sys.argv[2].split(',')
    rev_s = float(sys.argv[3]) if len(sys.argv) > 3 else 0.4
    rev_us = float(sys.argv[4]) if len(sys.argv) > 4 else 300.0
    out_name = sys.argv[5] if len(sys.argv) > 5 else 'grid_%s_%g_%g.json' % (kind, rev_s, rev_us)
    p = E.P1(rev_s=rev_s, rev_us=rev_us)
    out = {'kind': kind, 'rev_s': rev_s, 'rev_us': rev_us, 'ref': {}, 'grid': {}}
    try:
        for nm in names:
            base = nm.partition('/')[0]
            out['ref'][nm] = one(base, p, kind)
            out['ref'][nm + '_keep0'] = one(base, dict(p, rec_hold_keep=0), kind)
        t_on = out['ref'][names[0]]['t_rev_on']
        out['t_on'] = t_on
        print('ref', out['ref'])
        for d0 in D0:
            for L in LS:
                sk = (round(t_on + d0, 3), round(t_on + d0 + L, 3))
                row = {}
                for nm in names:
                    base, _, opt = nm.partition('/')
                    pp = dict(p, rec_hold_keep=0) if opt == 'k0' else (dict(p, rec_hold_keep=0, rec_back_guard=0)
                                                                      if opt == 'off' else p)
                    m = one(base, pp, kind, sk)
                    row[nm] = m
                k = 'd0=%+.2f L=%.2f' % (d0, L)
                out['grid'][k] = row
                print(k, {nm: (row[nm]['back_m'], row[nm]['esc_push_s'], row[nm]['guards'][:2]) for nm in names}, flush=True)
    finally:
        E.cleanup()
    (HERE / 'out').mkdir(exist_ok=True)
    (HERE / 'out' / out_name).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main()
