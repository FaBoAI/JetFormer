"""直す 7-1 回目・係 safety: fuzz_fix7.py の場面を作り直し、跳ね(glitch)を抜いた同じ場面で今(now1)と直す前(bef1)の後ろの当たりを
見る(跳ねありの今が「跳ねなしの今」と同じかを確かめる)。 python -I fz_clean.py <種> <本数> <場面の番号 ...>。出力は標準出力"""
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED, NRUN = int(sys.argv[1]), int(sys.argv[2])
WANT = [int(x) for x in sys.argv[3:]]
sys.argv = [sys.argv[0], str(NRUN), str(SEED)]
sys.path.insert(0, str(HERE))
import fuzz_fix7 as Z  # noqa: E402

R, FC = Z.R, Z.FC
R.TMP = Z.FIX / 'fix7' / 'tmp_safety' / ('fzc_%d' % SEED)
R.CL.TMP = R.TMP


def rear(sim):
    return [h for h in sim.car.hits if h[1] == 'rear']


def main():
    rng = random.Random(SEED)
    BEF = R.load_snapshot('fix7_safety_before')
    orig = R.CL.Sim
    R.CL.Sim = Z.GSim
    try:
        for i in range(NRUN):
            sc = Z.add_glitch(rng, FC.rand_scene(rng))
            p1 = FC.rand_params(rng)
            if i not in WANT:
                continue
            clean = dict(sc, glitch=[])
            print('run', i, 'now(glitch)', rear(R.run_closed(p1, sc)[0]), '| now(clean)', rear(R.run_closed(p1, clean)[0]),
                  '| bef(clean)', rear(R.run_closed(p1, clean, mod=BEF[0], amod=BEF[1])[0]), flush=True)
    finally:
        R.CL.Sim = orig
        R.cleanup()


if __name__ == '__main__':
    main()
