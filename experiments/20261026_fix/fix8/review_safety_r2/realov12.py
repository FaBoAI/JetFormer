"""本物の記録(10/10・10/23・10/24。data/raw は読むだけ。場面をつなぐ = tests/test_rec_fix8_safety_r1 の読み方)で、リカバリーの
後退側(AI_CMD の out が中立 + 200 µs より後ろ・assist 2)を書いたフレームの次のフレームまでの間(撮像の時刻)を数える。
間が 0.1 秒より長い(= カメラが止まって中立止めになる形)ものが、後退の最後のフレーム(次の行が後退側でない)の直後にあったか。
あわせて、AI 側の処理したフレームどうしの間が 0.1 秒より長い割合(1 フレームあたり)。
    python -I -B realov12.py → out/realov12.json
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/jetson/jetformer')
sys.path.insert(0, str(ROOT / 'tests'))
sys.path.insert(0, str(ROOT))
import test_rec_fix8_safety_r1 as T  # noqa: E402


def main():
    out = {'logs': 0, 'ai_intervals': 0, 'ai_gaps_gt_0p1': 0, 'rev_rows': 0, 'rev_last_rows': 0, 'rev_gap_gt_0p1': [],
           'rev_last_gap_gt_0p1': [], 'recoveries': 0}
    for log in T.raw_logs():
        for scenes in T.units(log, 'concat'):
            cat, meta, sk = T.load_unit(log, scenes)
            if cat is None:
                continue
            out['logs'] += 1
            z = cat['AI_CMD']
            t0 = T.log_t0_us(log)
            ts = (z['timestamps_us'].astype(np.int64) - t0) / 1e6
            n = float((meta.get('drive') or {}).get('throttle_neutral_us', 1496))
            sel = z['select'].astype(bool)
            hold = z['hold'].astype(bool)
            dry = z['dry_run'].astype(bool) if 'dry_run' in z else np.zeros(len(ts), bool)
            a = z['assist'].astype(int) if 'assist' in z else np.zeros(len(ts), int)
            thr = z['out'][:, 1].astype(float)
            ok = sel & ~hold & ~dry
            k = np.nonzero(ok)[0]
            if len(k) > 1:
                dd = np.diff(ts[k])
                out['ai_intervals'] += int(len(dd))
                out['ai_gaps_gt_0p1'] += int(np.sum(dd > 0.1))
            prev_rec = False
            for j in range(len(k)):
                i = k[j]
                isrec = a[i] == 2
                if isrec and not prev_rec:
                    out['recoveries'] += 1
                prev_rec = isrec
                if not (isrec and thr[i] > n + 200):
                    continue
                out['rev_rows'] += 1
                if j + 1 >= len(k):
                    continue
                nx = k[j + 1]
                gap = float(ts[nx] - ts[i])
                last = not (a[nx] == 2 and thr[nx] > n + 200)
                if last:
                    out['rev_last_rows'] += 1
                if gap > 0.1:
                    e = {'log': log, 's': round(float(ts[i]), 3), 'gap_s': round(gap, 3), 'next_thr': float(thr[nx]),
                         'next_assist': int(a[nx])}
                    out['rev_gap_gt_0p1'].append(e)
                    if last:
                        out['rev_last_gap_gt_0p1'].append(e)
    out['p_gap_per_frame'] = round(out['ai_gaps_gt_0p1'] / max(out['ai_intervals'], 1), 5)
    (HERE / 'out' / 'realov12.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in out.items()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
