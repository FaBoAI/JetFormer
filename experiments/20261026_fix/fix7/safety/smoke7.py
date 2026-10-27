"""直す 7-1 回目・係 safety: 指摘 w1-safety-rv7-safety-r1-1 の入力 1 つ(場面 rev・回 1 の形・'brake' の間に 1 サンプル 1322)を
直す前の写し(f7b)と今で、ON / OFF。段の移り・下がった距離・escape の壁押し"""
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import h7  # noqa: E402
from g7 import first_rec, scene  # noqa: E402


def phases(ph):
    out, last = [], None
    for t, p0, p1, *_r in ph:
        if p1 != last:
            out.append((t, p1))
            last = p1
    return out[:14]


def main():
    print(h7.check_now())
    p = h7.base_params()
    try:
        for name in sys.argv[1:] or ['f7b', 'now']:
            for tg in (None, 2.54):
                sc = scene('rev', dur=12.0)
                if tg is not None:
                    sc['glitch'] = [(tg, 1322)]
                for lab, pp in (('on', h7.ON(p)), ('off', h7.OFF(p))):
                    t0 = time.time()
                    r = h7.run(name, pp, sc)
                    m = first_rec(r)
                    print(name, tg, lab, 'back', m['back'], 'esc_push', m['esc_push'], 'front_push', m['front_push'],
                          'guards', m['guards'][:3], 'rev_on', next((t for t, ro, c in r[0].car.revlog if ro), None),
                          '%.1fs' % (time.time() - t0))
                    print('   ', phases(r[3]))
    finally:
        h7.cleanup()


if __name__ == '__main__':
    main()
