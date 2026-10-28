"""fix8 係 tools-decided(読むだけ): 本物の記録(data/labels にラベルのある全部)で、AI の区間の最後のフレーム(中立止め・AI_CMD が人側の
フレームを除く)がリカバリーの途中(assist 2)の区間と、その次の AI の区間の始め(止まっていたか = 発進か)、今の abort_resumes の印を並べる。
data/raw・labels は読むだけ。出力は ../out/scan_real_recovery.json と標準出力"""
import json
import sys
from pathlib import Path

import numpy as np

JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF / 'tools'))
sys.path.insert(0, str(JF))
import label_events as le  # noqa: E402
import standing_3lap as s3  # noqa: E402

LAB = JF / 'data' / 'labels'
OUT = Path(__file__).resolve().parent.parent / 'out' / 'scan_real_recovery.json'


def main():
    res = {'books': 0, 'segments': 0, 'ends': {}, 'cases': [], 'marks': []}
    for pj in sorted(LAB.glob('2026*.json')):
        log = pj.stem
        pn = LAB / (log + '.npz')
        if not pn.exists() or not (JF / 'data' / 'raw' / log / 'log.json').exists():
            continue
        with np.load(pn) as z:
            z = {k: z[k] for k in z.files}
        if 'assist' not in z or 'ai' not in z:
            continue
        res['books'] += 1
        lab = json.loads(pj.read_text(encoding='utf-8'))
        t = z['t_s'].astype(float)
        ai = z['ai'].astype(bool) & ~z['dry_run'].astype(bool)
        a = z['assist'].astype(int)
        h = z['hold'].astype(bool) if 'hold' in z else np.zeros(len(t), bool)
        s = z['ai_cmd_select'].astype(bool) if 'ai_cmd_select' in z else np.ones(len(t), bool)
        V = le.load_log(log, JF / 'data' / 'raw')['VEHICLE']
        vt = (V['timestamps_us'].astype(np.int64) - int(lab['t0_us'])) / 1e6
        ve = V['erpm'].astype(np.float64)
        _, Df = le.cumulative_distance(vt, ve)
        segs = le.runs(ai)
        res['segments'] += len(segs)
        for j, (i0, i1) in enumerate(segs):
            seg = np.arange(i0, i1 + 1)
            k = seg[~h[seg] & s[seg]]
            last = int(a[k[-1]]) if len(k) else None
            res['ends'][str(last)] = res['ends'].get(str(last), 0) + 1
            if last == 2:
                nxt = segs[j + 1] if j + 1 < len(segs) else None
                e0 = float(np.interp(t[nxt[0]], vt, ve)) if nxt else None
                res['cases'].append({'log': log, 'seg_end_s': round(float(t[i1]), 3), 'next_t_ai': round(float(t[nxt[0]]), 3) if nxt else None,
                                     'next_erpm_at_ai': round(e0) if e0 is not None else None,
                                     'next_is_launch': bool(nxt) and abs(e0) < s3.STOP_ERPM and float(t[nxt[1]] - t[nxt[0]]) >= 0.1,
                                     'human_s': round(float(t[nxt[0]] - t[i1]), 2) if nxt else None})
        marks = s3.abort_resumes(t, ai, z['assist'], z.get('hold'), z.get('ai_cmd_select'), vt, Df, ve)
        for i, v in sorted(marks.items()):
            res['marks'].append(dict(v, log=log, t_ai=round(float(t[i]), 3)))
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print('books', res['books'], 'segments', res['segments'], 'ends', res['ends'])
    for c in res['cases']:
        print('case', c)
    for m in res['marks']:
        print('mark', m)


if __name__ == '__main__':
    main()
