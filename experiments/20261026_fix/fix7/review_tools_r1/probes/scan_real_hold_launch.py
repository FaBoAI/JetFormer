"""反証 7-1 回目・係 tools(読むだけ): 本物の 10/10・10/23・10/24 の止まってからの発進で、AI 側 → 最初の R1 の起点の間に
中立止め(AI_CMD の hold)・フレームの途切れ(> 0.2 秒)があった発進を数える。⑤ collect.py の launch_rows は、その間の hold・途切れを
見ずに to_r1_s を出す(③ standing_3lap は 3 周の valid で途切れを見る)。data/raw・labels は読むだけ。出力は ../out/scan_real_hold_launch.json
"""
import json
import sys
from pathlib import Path

import numpy as np

JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF / 'tools'))
sys.path.insert(0, str(JF))
import label_events as le  # noqa: E402

RAW = JF / 'data' / 'raw'
LAB = JF / 'data' / 'labels'
OUT = Path(__file__).resolve().parent.parent / 'out' / 'scan_real_hold_launch.json'


def main():
    rows = []
    for p in sorted(RAW.iterdir()):
        if p.name[:8] not in ('20261010', '20261023', '20261024') or len(p.name) != 15:
            continue
        lj, lz = LAB / (p.name + '.json'), LAB / (p.name + '.npz')
        if not (lj.exists() and lz.exists()):
            continue
        try:
            meta = json.loads((p / 'log.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        dr = meta.get('drive') or {}
        if not dr or dr.get('dry_run'):
            continue
        lab = json.loads(lj.read_text())
        z = dict(np.load(lz))
        if 'hold' not in z:
            continue
        L = le.load_log(p.name, RAW)
        T0 = int(lab['t0_us'])
        V = L['VEHICLE']
        if V is None:
            continue
        vt = (V['timestamps_us'].astype(np.int64) - T0) / 1e6
        verpm = V['erpm'].astype(np.float64)
        t, ai, hold = z['t_s'], z['ai'].astype(bool), z['hold'].astype(bool)
        anchors = np.asarray([a['t_s'] if isinstance(a, dict) else a for a in (lab.get('events') or {}).get('r1_anchors', [])] or
                             [], float)
        for i0, i1 in le.runs(ai):
            t0, t1 = float(t[i0]), float(t[i1])
            e0 = float(np.interp(t0, vt, verpm))
            if abs(e0) >= 400 or t1 - t0 < 0.1:
                continue
            seg = slice(i0, i1 + 1)
            h = int(hold[seg].sum())
            gaps = np.diff(t[seg])
            rows.append({'log': p.name, 't_ai': round(t0, 3), 'ai_seg_s': round(t1 - t0, 2), 'hold_frames_seg': h,
                         'max_gap_seg': round(float(gaps.max()) if len(gaps) else 0.0, 3),
                         'hold_frames_first5s': int(hold[seg][(t[seg] - t0) < 5.0].sum()),
                         'max_gap_first5s': round(float(np.diff(t[seg][(t[seg] - t0) < 5.0]).max())
                                                  if ((t[seg] - t0) < 5.0).sum() > 1 else 0.0, 3)})
    bad = [r for r in rows if r['hold_frames_first5s'] or r['max_gap_first5s'] > 0.2]
    OUT.write_text(json.dumps({'n': len(rows), 'with_hold_or_gap_first5s': bad, 'rows': rows}, ensure_ascii=False, indent=1),
                   encoding='utf-8')
    print('launches', len(rows), 'with hold/gap in first 5 s', len(bad))
    for r in bad:
        print(r)


if __name__ == '__main__':
    main()
