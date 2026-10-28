"""fix8 係 safety-decided: 本物の AI 走行の記録(data/raw の AI_CMD がある記録。読むだけ)で、AI 側の処理したフレーム(中立止めの行でない)の
間(撮像時刻の差。この間は走行ループが中立止めか前の値のまま。assist.Recovery の「見ていない時間」と同じ見方)を数える。0.5 秒(assist.Recovery.HOLD_KEEP_MAX_S)より長い間について、
始めと終わりの速さ(VEHICLE の eRPM。1000 eRPM = 0.5 m/s)と、終わりで止まっている(|eRPM| < 300)かを出す
(= 両方 ON の日に、当たりが無くても「長い中立止めのあと止まっていたので中止」になりうる形)。
    python -I scan_gaps10.py   → out/scan_gaps10.json
"""
import json
from pathlib import Path

import numpy as np

RAW = Path('/home/jetson/jetformer/data/raw')
OUT = Path(__file__).resolve().parent / 'out'
KV = 0.0005


def main():
    res = {'logs': 0, 'intervals': 0, 'gt': {'0.1': 0, '0.15': 0, '0.3': 0, '0.5': 0}, 'long': [], 'by_day': {}}
    for root in sorted(RAW.glob('2026*')):
        if not root.is_dir():
            continue
        ts, sel, hold, vt, ve = [], [], [], [], []
        for s in sorted(d for d in root.iterdir() if d.is_dir()):
            try:
                z = np.load(s / 'AI_CMD.npz')
                v = np.load(s / 'VEHICLE.npz')
                t_ = z['timestamps_us'].astype(np.int64)
                sl = z['select'].astype(bool) & ~(z['dry_run'].astype(bool) if 'dry_run' in z else np.zeros(len(t_), bool))
                ts.append(t_)
                sel.append(sl)
                hold.append(z['hold'].astype(bool))
                vt.append(v['timestamps_us'].astype(np.int64))
                ve.append(v['erpm'].astype(float))
            except Exception:       # noqa: BLE001  無い・壊れた npz は飛ばす
                continue
        if not ts:
            continue
        ts, sel, hold = np.concatenate(ts), np.concatenate(sel), np.concatenate(hold)
        vt, ve = np.concatenate(vt), np.concatenate(ve)
        o = np.argsort(ts, kind='stable')
        ts, sel, hold = ts[o], sel[o], hold[o]
        o = np.argsort(vt, kind='stable')
        vt, ve = vt[o], ve[o]
        if not sel.any():
            continue
        res['logs'] += 1
        # 処理したフレーム(AI 側・中立止めの行でない)どうしの間。間に人側の行があれば数えない(人側で手順は戻る)。間の中立止めの行
        # (センサーが古い・処理の遅れなど)は「処理していない」= Recovery.step を呼ばない時間に入る
        proc = np.nonzero(sel & ~hold)[0]
        pairs = []
        for a, b in zip(proc, proc[1:]):
            if sel[a:b + 1].all():
                pairs.append((a, b))
        for a_, b_ in pairs:
            i = a_
            res['intervals'] += 1
            g = (ts[b_] - ts[a_]) / 1e6
            day = res['by_day'].setdefault(root.name[:8], {'intervals': 0, '0.1': 0, '0.15': 0, '0.3': 0, '0.5': 0})
            day['intervals'] += 1
            for k in res['gt']:
                if g > float(k):
                    res['gt'][k] += 1
                    day[k] += 1
            if g > 0.5:
                def v_at(t):
                    k = np.searchsorted(vt, t, side='right') - 1
                    return float(ve[k]) if k >= 0 else None
                e0, e1 = v_at(ts[a_]), v_at(ts[b_])
                res['long'].append({'log': root.name, 't_s': round((ts[i] - ts[0]) / 1e6, 3), 'gap_s': round(g, 3),
                                    'v0_mps': None if e0 is None else round(-e0 * KV, 2),
                                    'v1_mps': None if e1 is None else round(-e1 * KV, 2),
                                    'stopped_at_end': e1 is not None and abs(e1) < 300,
                                    'hold_rows_between': int(hold[a_:b_ + 1].sum())})
    (OUT / 'scan_gaps10.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print('AI の記録 %d 本・AI 側のフレームの間 %d' % (res['logs'], res['intervals']), '長い間:', res['gt'])
    for k, v in sorted(res['by_day'].items()):
        print('  ', k, v)
    for x in res['long']:
        print('  ', x)


if __name__ == '__main__':
    main()
