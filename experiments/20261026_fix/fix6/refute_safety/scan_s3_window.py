"""反証 6-1・安全: 本物の 10/10 に AI 側で 10 回あった 0.52 秒の車両の値の止まり(fix4/review_safety/out/scan_veh_gaps2_20261010.json)
を、既定の rev_s 0.4・rev_us 300 のリカバリーの 'reverse' の入りぎわに 10 ms 刻みでずらして置き、下がった距離を値の止まりなし
と比べる(今の keep 1・keep 0・直す前の写し keep 1)。probe_s3_reverse_start と同じ閉じたループ。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I \
        experiments/20261026_fix/fix6/refute_safety/scan_s3_window.py
出力: out/scan_s3_window.json
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
    out = {'rows': []}
    try:
        p1 = dict(PARAMS, recovery=1, start_kick=0, rec_back_guard=1, rec_hold_keep=1, tof_guard=1, rev_s=0.4, rev_us=300.0)
        base = P.run(p1, None)
        t0 = base['t_first_rev_on']
        out['no_gap_backed_m'] = base['backed_m']
        out['t_reverse'] = base['t_reverse']
        out['t_first_rev_on'] = t0
        L = 0.52
        for i in range(-20, 5):
            d0 = round(i * 0.01, 2)
            skip = (round(t0 + d0, 3), round(t0 + d0 + L, 3))
            a = P.run(p1, skip)
            b = P.run(dict(p1, rec_hold_keep=0), skip)
            c = P.run(p1, skip, BEF)
            out['rows'].append({'d0': d0, 'gap': skip, 'now_keep1': a['backed_m'], 'now_keep0': b['backed_m'],
                                'before_keep1': c['backed_m'], 'now_after_hold': a['after_hold'][:1],
                                'now_escape': a['t_escape'], 'before_escape': c['t_escape']})
            print(out['rows'][-1], flush=True)
    finally:
        R.cleanup()
    (HERE / 'out').mkdir(exist_ok=True)
    (HERE / 'out' / 'scan_s3_window.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main()
