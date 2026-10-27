"""反証の確かめ: カメラの止まり A と車両の値の止まり B の組(grid_false_ready と同じ並び)を、自分の閉じたループ(repro.run)で回す。
最初のリカバリーで下がった距離 < 0.1 m かつ escape が前の壁を押した組を、今(now)と直す前(before)で数える。
    python -I grid.py [neu]   (neu = 偽物の ESC が中立で ready になる秒。省くと R2.Car のまま)
"""
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import repro as R  # noqa: E402
from test_launch_drive import C, N  # noqa: E402


def main(neu=None, step=1):
    tb = 2.09
    cnt = {'n': 0, 'now_fail': 0, 'before_fail': 0, 'now_push': 0, 'before_push': 0}
    fails = []
    t0 = time.time()
    i = 0
    try:
        for a0 in (0.0, 0.03, 0.06, 0.1, 0.15, 0.2):
            for aL in (0.13, 0.15, 0.17, 0.19, 0.21):
                for b0 in (0.0, 0.03, 0.06, 0.1):
                    for bL in (0.17, 0.2, 0.25, 0.3):
                        i += 1
                        if (i - 1) % step:
                            continue
                        a = (round(tb + a0, 3), round(tb + a0 + aL, 3))
                        b = (round(a[1] + b0, 3), round(a[1] + b0 + bL, 3))
                        sc = {'dur': 7.0, 'xf': 2.0, 'xr': None, 'model': (C, N - 120.0), 'drops': [a], 'veh_skip': [b]}
                        row = {}
                        for nm in ('now', 'before'):
                            sim, rec, pca, PH, CAR = R.run(nm, sc, neu)
                            m = R.metrics(sim, pca)
                            m['guards'] = [(e[1]['what'], e[1]['phase'], e[1]['backed_s']) for e in rec.events if e[0] == 'recovery_guard'][:2]
                            row[nm] = m
                            if m['esc_push_wall_s'] > 0:
                                cnt[nm + '_push'] += 1
                                if m['back_m'] < 0.1:
                                    cnt[nm + '_fail'] += 1
                        cnt['n'] += 1
                        if any(row[k]['esc_push_wall_s'] > 0 and row[k]['back_m'] < 0.1 for k in row):
                            fails.append(('A=%s B=%s' % (a, b), row))
                            print('A=%s B=%s' % (a, b), row, flush=True)
    finally:
        shutil.rmtree(R.TMP, ignore_errors=True)
    print('neu', neu, cnt, 'sec', round(time.time() - t0, 1))
    (HERE / 'out').mkdir(exist_ok=True)
    (HERE / 'out' / ('grid_%s.json' % ('orig' if neu is None else neu))).write_text(json.dumps({'count': cnt, 'fails': fails}, indent=1, default=str))


if __name__ == '__main__':
    a = sys.argv[1:]
    main(None if not a or a[0] == 'orig' else float(a[0]), int(a[1]) if len(a) > 1 else 1)
