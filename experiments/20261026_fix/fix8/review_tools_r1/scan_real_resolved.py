"""反証 8-1 回目・係 tools・台本 4(読むだけ): 本物の記録(data/labels にラベルのある全部。data/raw・labels は読むだけ)で、AI の区間の
最後のフレーム(中立止め・AI_CMD が人側を除く)が assist 2 でない(= 今の abort_resumes は 'recovery' の印を付けない)のに、
  - その区間の終わりの 3 秒の中にリカバリー(assist 2)があり、
  - 区間の終わりで車が止まっていた(|eRPM| < 400)
区間と、その次の AI の区間が止まった状態からの発進か、今の印(abort_resumes)を並べる。台本 3(probe_t3_recovery_edge)の形が本物にあるか。
出力 out/scan_real_resolved.json と標準出力

  jf-heavy --lock none --mem 1500M -- python -B scan_real_resolved.py
"""
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
OUT = Path(__file__).resolve().parent / 'out' / ('scan_real_resolved%s.json' % ('' if len(sys.argv) < 2 else '_' + '_'.join(sys.argv[1:])))
WIN_S = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
STOP_ONLY = (sys.argv[2] != "any") if len(sys.argv) > 2 else True


def main():
    res = {'books': 0, 'segments': 0, 'cases': []}
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
        marks = s3.abort_resumes(t, ai, z['assist'], z.get('hold'), z.get('ai_cmd_select'), vt, Df, ve)
        segs = le.runs(ai)
        res['segments'] += len(segs)
        for j, (i0, i1) in enumerate(segs):
            seg = np.arange(i0, i1 + 1)
            k = seg[~h[seg] & s[seg]]
            if not len(k):
                continue
            last = int(a[k[-1]])
            if last in (2, 3):
                continue
            rk = k[(a[k] == 2) & (t[k] >= t[i1] - WIN_S)]
            if not len(rk):
                continue
            e_end = float(np.interp(t[i1], vt, ve))
            if STOP_ONLY and abs(e_end) >= s3.STOP_ERPM:
                continue
            nxt = segs[j + 1] if j + 1 < len(segs) else None
            e0 = float(np.interp(t[nxt[0]], vt, ve)) if nxt else None
            res['cases'].append({
                'log': log, 'seg_end_s': round(float(t[i1]), 3), 'last_assist': last,
                'rec_last_s': round(float(t[rk[-1]]), 3), 'after_rec_s': round(float(t[i1] - t[rk[-1]]), 3),
                'assist_after_rec': sorted({int(x) for x in a[k[k > rk[-1]]]}),
                'erpm_at_end': round(e_end), 'next_t_ai': round(float(t[nxt[0]]), 3) if nxt else None,
                'next_erpm_at_ai': round(e0) if e0 is not None else None,
                'next_is_launch': bool(nxt) and abs(e0) < s3.STOP_ERPM and float(t[nxt[1]] - t[nxt[0]]) >= 0.1,
                'human_s': round(float(t[nxt[0]] - t[i1]), 2) if nxt else None,
                'next_mark': marks.get(nxt[0]) if nxt else None})
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print('books', res['books'], 'segments', res['segments'], 'cases', len(res['cases']))
    for c in res['cases']:
        print(c)


if __name__ == '__main__':
    main()
