"""反証 1 回目・safety: 本物の AI の記録(data/raw の AI_CMD.npz がある本)で、AI 側の衝撃(水平 2.5G 以上)のあと、
走行ループが衝撃を見られた(衝撃の IMU のサンプルから 100 ms 以内に処理したフレームがある = drive の impact_g の窓
「直近 10 サンプル・150 ms」に入る)かを数える。見られなかった衝撃 = その間が中立止め(カメラが来ない・車両の値が古いなど)か
フレームの間が空いた。読むだけ。

    /home/jetson/miniconda/envs/jetracer/bin/python -I scan_impact_gaps.py /home/jetson/jetformer/data/raw [出力の json]
"""
import json
import sys
from pathlib import Path

import numpy as np

G = 9.80665


def scan(rec_dir):
    a = np.load(rec_dir / 'AI_CMD.npz')
    im = np.load(rec_dir / 'IMU.npz')
    t_ai = a['timestamps_us'].astype(np.int64)
    hold = a['hold'].astype(bool)
    sel = a['select'].astype(bool)
    dry = a['dry_run'].astype(bool)
    assist = a['assist'].astype(int)
    t_im = im['timestamps_us'].astype(np.int64)
    acc = im['accel']
    hg = np.hypot(acc[:, 0], acc[:, 1]) / G
    if len(t_ai) == 0 or len(t_im) == 0:
        return []
    t_lo, t_hi = t_ai[0], t_ai[-1]
    idx = np.where((hg >= 2.5) & (t_im >= t_lo) & (t_im <= t_hi))[0]
    out = []
    last_t = -10 ** 18
    for i in idx:
        t = int(t_im[i])
        if t - last_t < 300_000:                 # 同じ当たりの続きのサンプル(0.3 秒以内)はまとめる
            continue
        last_t = t
        j = int(np.searchsorted(t_ai, t))        # 衝撃のあとの最初の AI の行
        prev = j - 1
        if prev < 0 or not sel[prev] or dry[prev]:
            continue                             # AI 側(dry run でない)のときの衝撃だけ
        # 衝撃のあと 100 ms 以内に、中立止めでない行があるか
        seen = False
        k = j
        first_hold = None
        while k < len(t_ai) and t_ai[k] - t <= 100_000:
            if hold[k] and first_hold is None:
                first_hold = k
            if not hold[k]:
                seen = True
                break
            k += 1
        nxt = j
        while nxt < len(t_ai) and hold[nxt]:
            nxt += 1
        gap_ms = (t_ai[nxt] - t) / 1000.0 if nxt < len(t_ai) else None
        n_hold = int(hold[j:nxt].sum()) if nxt <= len(t_ai) else 0
        out.append({'rec': rec_dir.name, 'impact_g': round(float(hg[i: i + 30].max()), 2), 't_us': t,
                    'assist_before': int(assist[prev]), 'seen_within_100ms': seen,
                    'next_processed_ms': None if gap_ms is None else round(gap_ms, 1), 'hold_rows_between': n_hold,
                    'prev_row_to_impact_ms': round((t - t_ai[prev]) / 1000.0, 1)})
    return out


def main():
    root = Path(sys.argv[1])
    res = []
    for p in sorted(root.glob('2026*/*/AI_CMD.npz')):
        d = p.parent
        if not (d / 'IMU.npz').exists():
            continue
        try:
            res += scan(d)
        except Exception as e:      # noqa: BLE001  読めない記録は飛ばして名前を出す
            print('skip', d.name, type(e).__name__, e)
    missed = [r for r in res if not r['seen_within_100ms']]
    summary = {'impacts_ai': len(res), 'missed': len(missed),
               'missed_in_recovery': sum(1 for r in missed if r['assist_before'] == 2), 'missed_list': missed,
               'recs': len({r['rec'] for r in res})}
    print(json.dumps({k: v for k, v in summary.items() if k != 'missed_list'}, ensure_ascii=False))
    for r in missed:
        print(r)
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
