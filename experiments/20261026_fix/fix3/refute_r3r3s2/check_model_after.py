"""check_giveup_coast.py の続き: 中立止めが明けたあと、モデルが前進でない(中立・ブレーキ側)日の keep=0 と keep=1。
明ける前はモデル前進(N − 120。前の壁に当たってリカバリーが始まる)。惰性 1.5 m/s²。後ろの物は中立止めなしの届く所 + 0.15 m"""
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import check_giveup_coast as G                # noqa: E402
from cl3 import C, N                          # noqa: E402


def main():
    G.DEC[0] = 1.5
    nh = G.one(1, None, None)
    reach = round(G.XF - nh['min_x'], 3)
    tr = next(t for t, p0, p1 in nh['phases'] if p1 == 'reverse')
    drop = (tr + 0.40, tr + 0.95)
    for name, after in (('fwd120', N - 120.0), ('neutral', N), ('brake60', N + 60.0)):
        def model(t, car, after=after):
            return (C, N - 120.0) if t < drop[1] else (C, after)
        for keep in (0, 1):
            r = G.one(keep, drop, reach + 0.15, model)
            print(name, 'keep=%d' % keep, {k: r[k] for k in ('rear_hits', 'min_x', 'ev', 'v_at_cam_back', 'backed_after_m',
                                                              'thr_after')})
        r = G.one(1, None, reach + 0.15, model)
        print(name, 'nohold', {k: r[k] for k in ('rear_hits', 'min_x', 'ev')})


if __name__ == '__main__':
    try:
        main()
    finally:
        shutil.rmtree(G.cl3.TMP, ignore_errors=True)
