"""fix7 係 tools 直す 7-1 回目(読むだけ): 本物の記録(data/labels にラベルのある全部の日)で、AI の区間の最後のフレーム
(中立止め・AI_CMD が人側のフレームを除く)がリカバリー中止(assist 3)・リカバリー(assist 2)の区間と、そのあとの AI の区間の始め
(止まっていたか)を数える。data/raw・labels は読むだけ。出力は ../out/scan_real_abort.json"""
import json
import sys
from pathlib import Path

import numpy as np

JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF / 'tools'))
import label_events as le  # noqa: E402

LAB = JF / 'data' / 'labels'
OUT = Path(__file__).resolve().parent.parent / 'out' / 'scan_real_abort.json'


def main():
    res = {'books': 0, 'segments': 0, 'ends': {}, 'cases': [], 'giveup_books': []}
    for pj in sorted(LAB.glob('2026*.json')):
        log = pj.stem
        pn = LAB / (log + '.npz')
        if not pn.exists():
            continue
        try:
            with np.load(pn) as z:
                z = {k: z[k] for k in z.files}
        except Exception:      # noqa: BLE001
            continue
        if 'assist' not in z or 'ai' not in z:
            continue
        res['books'] += 1
        t = z['t_s'].astype(float)
        ai = z['ai'].astype(bool)
        if 'dry_run' in z:
            ai = ai & ~z['dry_run'].astype(bool)
        a = z['assist'].astype(int)
        h = z['hold'].astype(bool) if 'hold' in z else np.zeros(len(t), bool)
        s = z['ai_cmd_select'].astype(bool) if 'ai_cmd_select' in z else np.ones(len(t), bool)
        if (a == 3).any():
            res['giveup_books'].append(log)
        segs = le.runs(ai)
        res['segments'] += len(segs)
        for j, (i0, i1) in enumerate(segs):
            seg = np.arange(i0, i1 + 1)
            k = seg[~h[seg] & s[seg]]
            last = int(a[k[-1]]) if len(k) else None
            last_raw = int(a[i1])
            res['ends'][str(last)] = res['ends'].get(str(last), 0) + 1
            if last in (2, 3) or last_raw in (2, 3):
                nxt = segs[j + 1] if j + 1 < len(segs) else None
                res['cases'].append({'log': log, 'seg_end_s': round(float(t[i1]), 3), 'last_assist': last,
                                     'last_raw_assist': last_raw, 'n_giveup_frames_in_seg': int((a[seg] == 3).sum()),
                                     'next_t_ai': round(float(t[nxt[0]]), 3) if nxt else None,
                                     'next_seg_s': round(float(t[nxt[1]] - t[nxt[0]]), 2) if nxt else None,
                                     'human_s': round(float(t[nxt[0]] - t[i1]), 2) if nxt else None})
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print('books', res['books'], 'segments', res['segments'], 'ends', res['ends'])
    print('giveup books', len(res['giveup_books']))
    for c in res['cases']:
        print(c)


if __name__ == '__main__':
    main()
