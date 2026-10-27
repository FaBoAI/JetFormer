"""fix7 tools r2(読むだけ): 本物の 10/10・10/23・10/24 のラベルのある AI の記録で、abort_resumes の印を直す前(fix7 r1 の写し
tests/fixtures/fix7_tools_before/r2/standing_3lap.py.orig)と今で比べる。今は車輪の eRPM も渡す(③ analyze_log・⑤ launch_rows と同じ)。
data/raw・data/labels は読むだけ。出力は ../out/scan_real_marks.json
"""
import json
import sys
from pathlib import Path

import numpy as np

JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF / 'tools'))
sys.path.insert(0, str(JF))
sys.path.insert(0, str(JF / 'tests'))
import label_events as le  # noqa: E402
import standing_3lap as s3  # noqa: E402
from test_fix3_r1_2 import load_mod  # noqa: E402

s3old = load_mod('standing_3lap_r1', JF / 'tests/fixtures/fix7_tools_before/r2/standing_3lap.py.orig')
RAW = JF / 'data' / 'raw'
LAB = JF / 'data' / 'labels'
OUT = Path(__file__).resolve().parent.parent / 'out' / 'scan_real_marks.json'


def main():
    rows, n_sec, n_logs = [], 0, 0
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
        n_logs += 1
        L = le.load_log(p.name, RAW)
        V = L.get('VEHICLE')
        if V is None:
            continue
        T0 = int(lab['t0_us'])
        vt = (V['timestamps_us'].astype(np.int64) - T0) / 1e6
        ve = V['erpm'].astype(np.float64)
        _, Df = le.cumulative_distance(vt, ve)
        runs = le.runs(ai)
        n_sec += len(runs)
        old = s3old.abort_resumes(t, ai, z['assist'], z['hold'], z['ai_cmd_select'], vt, Df)
        new = s3.abort_resumes(t, ai, z['assist'], z['hold'], z['ai_cmd_select'], vt, Df, ve)
        for i0 in sorted(set(old) | set(new)):
            e0 = float(np.interp(t[i0], vt, ve))
            rows.append({'log': p.name, 't_ai': round(float(t[i0]), 3), 'stopped_at_ai': abs(e0) < 400,
                         'old': old.get(i0), 'new': new.get(i0)})
    res = {'logs_with_ai': n_logs, 'ai_sections': n_sec, 'rows': rows,
           'changed': [r for r in rows if r['old'] != r['new']],
           'abort_marks_same_except_kind': all(
               {k: v for k, v in (r['new'] or {}).items() if k != 'kind'} == r['old'] for r in rows if r['old'])}
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print('logs', n_logs, 'sections', n_sec, 'marks old', sum(1 for r in rows if r['old']), 'new', sum(1 for r in rows if r['new']))
    print('abort marks same except kind:', res['abort_marks_same_except_kind'])
    for r in rows:
        print(r)


if __name__ == '__main__':
    main()
