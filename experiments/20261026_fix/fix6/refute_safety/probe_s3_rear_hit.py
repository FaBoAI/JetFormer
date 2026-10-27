"""反証 6-1・安全: probe_s3_reverse_start.py の場面(既定の rev_s 0.4・rev_us 300、車両の値の止まりが ESC の後退の入り
ぎわから 0.45〜0.6 秒)に、後ろの物を置く。値の止まりなし・keep 0 では当たらない距離で、今の keep 1 だけ当たるか。
probe_s3_reverse_start と同じ閉じたループ(本物の drive._loop)。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I \
        experiments/20261026_fix/fix6/refute_safety/probe_s3_rear_hit.py
出力: out/probe_s3_rear_hit.json
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_s3_reverse_start as P   # noqa: E402
from r6_common import PARAMS, R      # noqa: E402


def main():
    BEF = R.load_snapshot('fix5_safety_before')
    out = {}
    try:
        p1 = dict(PARAMS, recovery=1, start_kick=0, rec_back_guard=1, rec_hold_keep=1, tof_guard=1, rev_s=0.4, rev_us=300.0)
        base = P.run(p1, None)
        t0 = base['t_first_rev_on']
        out['ref'] = {'t_first_rev_on': t0, 'no_gap_backed_m': base['backed_m']}
        for rear in (1.35, 1.40, 1.45, 1.50):
            xr = round(P.XF - rear, 3)
            for d0, L in ((0.0, 0.55), (-0.02, 0.60), (-0.04, 0.45), (-0.06, 0.50)):
                skip = (round(t0 + d0, 3), round(t0 + d0 + L, 3))
                k = 'rear=%.2fm gap=rev_on%+.2f..+%.2f' % (rear, d0, L)
                for nm, p, mod, sk in (('now_keep1', p1, None, skip), ('now_keep0', dict(p1, rec_hold_keep=0), None, skip),
                                       ('now_keep1_nogap', p1, None, None), ('before_keep1', p1, BEF, skip)):
                    r = P.run(p, sk, mod, xr=xr)
                    out[k + ' ' + nm] = {'rear_hits': r['rear_hits'], 'backed_m': r['backed_m'],
                                         'after_hold': r['after_hold'][:1], 'front_hits': r['front_hits'][:3],
                                         't_escape': r['t_escape']}
    finally:
        R.cleanup()
    (HERE / 'out').mkdir(exist_ok=True)
    (HERE / 'out' / 'probe_s3_rear_hit.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    for k, v in out.items():
        print(k, v)


if __name__ == '__main__':
    main()
