"""指摘の場面に後ろの物を置く(反証の probe_s3_rear_hit.py と同じ形を、この係の台本で組み直したもの。読むだけ)。
場面 rev・brake(exp_grid.py)・既定の rev_s 0.4・rev_us 300・値の止まり (d0, L) = ESC が後退に入る時刻からのずれと長さ。
比べ: final = 今(直したあと)の keep 1・k0 = 今の keep 0・nogap = 今の keep 1 で値の止まりなし・before = 直す前の写しの keep 1

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I \\
        experiments/20261026_fix/fix6/safety_r1/exp_rear.py
出力: out/exp_rear.json
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp_common as E    # noqa: E402
import exp_grid as G      # noqa: E402

CELLS = [(0.0, 0.55), (-0.02, 0.6), (-0.04, 0.45), (-0.06, 0.5), (0.0, 0.6)]
REARS = [1.30, 1.35, 1.40, 1.45, 1.50, 1.60, 1.70]
T_ON = {'rev': 2.60, 'brake': 2.73}


def main():
    out = {}
    p = E.P1(rev_s=0.4, rev_us=300.0)
    try:
        for kind in ('rev', 'brake'):
            for rear in REARS:
                for d0, L in CELLS:
                    sk = (round(T_ON[kind] + d0, 3), round(T_ON[kind] + d0 + L, 3))
                    row = {}
                    for nm, name, pp, skip in (('final', 'now', p, sk), ('k0', 'now', dict(p, rec_hold_keep=0), sk),
                                               ('nogap', 'now', p, None), ('before', 'fix6_before', p, sk)):
                        m = G.one(name, pp, kind, skip, rear=rear)
                        row[nm] = {'back_m': m['back_m'], 'rear_hits': m['rear_hits'], 'esc_push_s': m['esc_push_s'],
                                   'front_hits': m['front_hits']}
                    k = '%s rear=%.2f d0=%+.2f L=%.2f' % (kind, rear, d0, L)
                    out[k] = row
                    print(k, {nm: (r['back_m'], [h[2] for h in r['rear_hits']]) for nm, r in row.items()}, flush=True)
    finally:
        E.cleanup()
    (HERE / 'out' / 'exp_rear.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main()
