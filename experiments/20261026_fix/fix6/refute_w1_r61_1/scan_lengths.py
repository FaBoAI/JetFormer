"""反証 w1-safety-r61-safety-1 の打ち消し: 今の車両データの読み取り(sensors/vehicle.py: 30 回続けて読めない ≈ 0.3 秒 → 0.05 秒待って
開き直す。開き直しに失敗すると 0.25 秒待つ)で出うる止まりの長さ(0.36 秒 = 開き直し 1 回・10/10 16:10 のあとに 2 回あった形、
0.61 秒 = 開き直しの失敗 1 回)と、指摘の 0.55 秒を、ESC の後退の入りぎわに 10 ms 刻みで置き、下がった距離を値の止まりなしと比べる。
fix6/refute_safety の probe_s3_reverse_start.run を使う(変えない)。一時の場所は fix6/tmp_w1r61(終わったら消す)。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I \
        experiments/20261026_fix/fix6/refute_w1_r61_1/scan_lengths.py
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'refute_safety'))
import probe_s3_reverse_start as P   # noqa: E402
from r6_common import PARAMS, R      # noqa: E402

R.TMP = HERE.parents[0] / 'tmp_w1r61' / 'rs'
R.CL.TMP = R.TMP


def main():
    out = {'rows': []}
    t_start = time.time()
    try:
        p1 = dict(PARAMS, recovery=1, start_kick=0, rec_back_guard=1, rec_hold_keep=1, tof_guard=1, rev_s=0.4, rev_us=300.0)
        base = P.run(p1, None)
        t0 = base['t_first_rev_on']
        out['no_gap_backed_m'] = base['backed_m']
        out['t_first_rev_on'] = t0
        print('no gap', base['backed_m'], 't0', t0, 'sec', round(time.time() - t_start, 1), flush=True)
        for L in (0.36, 0.55, 0.61):
            for i in range(-12, 3):
                d0 = round(i * 0.01, 2)
                skip = (round(t0 + d0, 3), round(t0 + d0 + L, 3))
                a = P.run(p1, skip)
                row = {'L': L, 'd0': d0, 'now_keep1': a['backed_m'], 'excess': round(a['backed_m'] - base['backed_m'], 3),
                       'escape': a['t_escape'], 'giveup': a['t_giveup'], 'front_hits': len(a['front_hits']),
                       'after_hold': a['after_hold'][:1]}
                out['rows'].append(row)
                print(row, flush=True)
    finally:
        R.cleanup()
    (HERE / 'out').mkdir(exist_ok=True)
    (HERE / 'out' / 'scan_lengths.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    for L in (0.36, 0.55, 0.61):
        ex = [r['excess'] for r in out['rows'] if r['L'] == L]
        print('L=%.2f max excess %.3f m, n>0.05: %d / %d' % (L, max(ex), sum(e > 0.05 for e in ex), len(ex)))


if __name__ == '__main__':
    main()
