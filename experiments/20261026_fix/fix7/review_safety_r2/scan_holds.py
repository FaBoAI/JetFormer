"""反証 7-2 回目・safety: 本物の記録(data/raw の AI_CMD.npz、読むだけ)で、AI 側の中立止めがどれだけあるか。
- 処理したフレームの間(timestamps_us の差): 0.1 秒より空くと hold('カメラの映像が来ていません')が中立を書く(行は残らない)
- hold の行(処理が遅れています・センサーが古い など。cam があるときだけ行が残る)の続き
をまとめ、とくに中立を書いた長さが 0.1 秒より短い(閉じたループの偽物の ESC はまだ下がれない)ものを数える。
リカバリー(assist = 2)の行の間・すぐあとの止まりも数える。出力 out/scan_holds.json
    python -I scan_holds.py
"""
import glob
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = '/home/jetson/jetformer/data/raw'


def main():
    res = defaultdict(lambda: Counter())
    ex = defaultdict(list)
    for fp in sorted(glob.glob(ROOT + '/2026*/*/AI_CMD.npz')):
        if os.path.getsize(fp) == 0:
            continue
        try:
            d = np.load(fp)
            t = d['timestamps_us'].astype(np.int64) / 1e6
            sel = d['select'].astype(bool)
            hold = d['hold'].astype(bool)
            dry = d['dry_run'].astype(bool)
            assist = d['assist'].astype(int)
        except Exception:      # noqa: BLE001
            continue
        day = fp.split('/')[-3][:8]
        r = res[day]
        ai = sel & ~dry
        r['ai_rows'] += int((ai & ~hold).sum())
        n = len(t)
        i = 0
        while i < n - 1:
            if not (ai[i] and ai[i + 1]):
                i += 1
                continue
            dt = t[i + 1] - t[i]
            rec = assist[i] == 2 and not hold[i]
            if not hold[i] and not hold[i + 1] and dt > 0.1:
                # カメラが来ない hold: 中立は前のフレームの 0.1 秒あとから次のフレームまで(約 dt − 0.1)
                k = 'camgap_neu<0.1' if dt - 0.1 < 0.1 else 'camgap_neu>=0.1'
                r[k] += 1
                if rec:
                    r['rec_' + k] += 1
                    ex[day].append((fp.split('/')[-2], round(float(t[i] - t[0]), 3), round(float(dt), 3), 'cam'))
            if not hold[i] and hold[i + 1]:
                j = i + 1
                while j < n and hold[j] and ai[j]:
                    j += 1
                if j < n and ai[j]:
                    neu = t[j] - t[i + 1]        # 中立の hold の行の始めから次の処理したフレームまで(下限)
                    k = 'holdrows_neu<0.1' if neu < 0.1 else 'holdrows_neu>=0.1'
                    r[k] += 1
                    if rec:
                        r['rec_' + k] += 1
                        ex[day].append((fp.split('/')[-2], round(float(t[i] - t[0]), 3), round(float(neu), 3), 'rows'))
                i = j
                continue
            if rec:
                r['rec_rows'] += 1
            i += 1
    out = {day: dict(sorted(r.items())) for day, r in res.items()}
    out['_examples_in_recovery'] = {k: v[:40] for k, v in ex.items()}
    (HERE / 'out').mkdir(exist_ok=True)
    (HERE / 'out' / 'scan_holds.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
    for day, r in out.items():
        print(day, r if day != '_examples_in_recovery' else {k: len(v) for k, v in r.items()})


if __name__ == '__main__':
    main()
