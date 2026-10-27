"""fuzz_onoff の 1 場面を作り直し、ON(か OFF)の全フレームの段・出した値・車の位置と速さを出す(読むだけ)。
  python -I replay_detail.py <種> <番号> <on|off> <t_from> <t_to>"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rv7  # noqa: E402
from replay_fz import scene_of  # noqa: E402


def main(seed, idx, v, t_from, t_to):
    sc, p = scene_of(int(seed), int(idx))
    pp = rv7.ON(p) if v == 'on' else rv7.OFF(p)
    try:
        sim, rec, pca, ph = rv7.run('now', pp, sc)
        phd = {round(t, 3): x for t, *x in ph}
        rows = {round((t - rv7.T0) / 1e9, 3): r for t, r in rec.rows}
        for t, s, th in pca.w:
            if float(t_from) <= t <= float(t_to):
                xx = next((q for q in sim.log if q[0] >= t - 1e-9), sim.log[-1])
                k = min(phd, key=lambda q: abs(q - t))
                info = phd[k] if abs(k - t) < 0.02 else ''
                print('%.3f thr %.0f st %.0f x %.3f v %+.2f con %s sel %s %s' % (t, th, s, xx[1], xx[2], xx[4], xx[6], info))
    finally:
        rv7.cleanup()


if __name__ == '__main__':
    main(*sys.argv[1:])
