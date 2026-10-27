"""fix7 tools r2(読むだけ): 本物の 10/10・10/23・10/24 の AI の区間の終わりの形を数える(中立止めの長さ・次のフレームまでの
途切れ・終わりで止まっているか・区間の前進)。新しい abort_resumes の hold・stall の決まりで、どの区間が「止まって終わった」に
なり、どの区間が何で外れたか。出力は ../out/scan_real_tails.json"""
import json
import sys
from pathlib import Path

import numpy as np

JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF / 'tools'))
sys.path.insert(0, str(JF))
import label_events as le  # noqa: E402
import standing_3lap as s3  # noqa: E402

RAW, LAB = JF / 'data' / 'raw', JF / 'data' / 'labels'
OUT = Path(__file__).resolve().parent.parent / 'out' / 'scan_real_tails.json'


def main():
    rows = []
    for p in sorted(RAW.iterdir()):
        if p.name[:8] not in ('20261010', '20261023', '20261024') or len(p.name) != 15:
            continue
        lj, lz = LAB / (p.name + '.json'), LAB / (p.name + '.npz')
        if not (lj.exists() and lz.exists()):
            continue
        z = dict(np.load(lz))
        lab = json.loads(lj.read_text(encoding='utf-8'))
        t = z['t_s'].astype(np.float64)
        ai = z['ai'].astype(bool) & ~z['dry_run'].astype(bool)
        if not ai.any():
            continue
        V = le.load_log(p.name, RAW).get('VEHICLE')
        if V is None:
            continue
        vt = (V['timestamps_us'].astype(np.int64) - int(lab['t0_us'])) / 1e6
        ve = V['erpm'].astype(np.float64)
        _, Df = le.cumulative_distance(vt, ve)
        hold, sel = z['hold'].astype(bool), z['ai_cmd_select'].astype(bool)
        runs = le.runs(ai)
        for n, (i0, i1) in enumerate(runs):
            k = np.arange(i0, i1 + 1)[sel[i0:i1 + 1]]
            hs = 0.0
            if len(k) and hold[k[-1]]:
                j = len(k) - 1
                while j > 0 and hold[k[j - 1]]:
                    j -= 1
                hs = float(t[k[-1]] - (t[k[j - 1]] if j > 0 else t[k[j]]))
            gap = float(t[i1 + 1] - t[i1]) if i1 + 1 < len(t) else None
            if hs < 0.2 and (gap is None or gap <= le.GAP_S):
                continue
            moved = float(np.interp(t[i1], vt, Df) - np.interp(t[i0], vt, Df))
            e_end = float(np.interp(t[i1], vt, ve))
            m = (vt >= t[i1]) & (vt <= (t[i1 + 1] if gap is not None else t[i1]))
            e_gap_min = float(np.min(np.abs(ve[m]))) if m.any() else None
            rows.append({'log': p.name, 'seg': [round(float(t[i0]), 2), round(float(t[i1]), 2)], 'last_seg': n == len(runs) - 1,
                         'hold_tail_s': round(hs, 2), 'gap_s': None if gap is None else round(gap, 2), 'moved_m': round(moved, 2),
                         'erpm_end': round(e_end), 'erpm_gap_min': e_gap_min,
                         'hold_kind': hs >= s3.HOLD_KEEP_MAX_S - 1e-6 and abs(e_end) < 400 and moved >= 0.5,
                         'stall_kind': gap is not None and gap > le.GAP_S and e_gap_min is not None and e_gap_min < 400 and moved >= 0.5})
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding='utf-8')
    for r in rows:
        print(r)


if __name__ == '__main__':
    main()
