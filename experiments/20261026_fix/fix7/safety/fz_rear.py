"""直す 7-1 回目・係 safety: fuzz_fix7.py の場面を作り直して(同じ乱数の並び)、後ろの当たりを今(now1)・直す前(bef1)・OFF で比べる。
now1 の後ろの当たりの最高の速さが bef1 より 0.05 m/s を超えて速い場面を出し、OFF とも比べる(OFF より速いか)。
    python -I fz_rear.py <種> <本数>
出力は標準出力。装置・GPU・API には触れない
"""
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED, NRUN = int(sys.argv[1]), int(sys.argv[2])
sys.argv = [sys.argv[0], str(NRUN), str(SEED)]
sys.path.insert(0, str(HERE))
import fuzz_fix7 as Z  # noqa: E402

R, FC = Z.R, Z.FC
R.TMP = Z.FIX / 'fix7' / 'tmp_safety' / ('fzr_%d' % SEED)
R.CL.TMP = R.TMP


def rv(sim):
    return max([-h[2] for h in sim.car.hits if h[1] == 'rear'], default=0.0)


def main():
    rng = random.Random(SEED)
    BEF = R.load_snapshot('fix7_safety_before')
    orig = R.CL.Sim
    R.CL.Sim = Z.GSim
    n = {'scenes': 0, 'now_faster_than_bef': 0, 'now_faster_than_off': 0, 'bef_faster_than_off': 0,
         'now_faster_than_bef_and_off': 0}
    try:
        for i in range(NRUN):
            sc = Z.add_glitch(rng, FC.rand_scene(rng))
            p1 = FC.rand_params(rng)
            if sc.get('xr') is None:
                continue
            n['scenes'] += 1
            a = rv(R.run_closed(p1, sc)[0])
            b = rv(R.run_closed(p1, sc, mod=BEF[0], amod=BEF[1])[0])
            o = rv(R.run_closed(dict(p1, **Z.OFF4), sc)[0])
            n['now_faster_than_bef'] += a > b + 0.05
            n['now_faster_than_off'] += a > o + 0.05
            n['bef_faster_than_off'] += b > o + 0.05
            n['now_faster_than_bef_and_off'] += a > b + 0.05 and a > o + 0.05
            if a > b + 0.05 or a > o + 0.05:
                print('run', i, 'rear v now %.2f bef %.2f off %.2f' % (a, b, o), 'rear_m', round(sc['xf'] - sc['xr'], 2),
                      {k: p1[k] for k in ('rev_s', 'rev_us')}, flush=True)
    finally:
        R.CL.Sim = orig
        R.cleanup()
    print(n)


if __name__ == '__main__':
    main()
