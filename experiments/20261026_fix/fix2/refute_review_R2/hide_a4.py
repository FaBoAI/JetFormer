#!/usr/bin/env python3
"""R2 の確かめ: 止まってから 3 周で A4 を隠す(A3+dist)と、A4 ありとどれだけずれるか。読むだけ(numpy)"""
import json, sys
from pathlib import Path
ROOT = Path('/home/jetson/jetformer')
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'tools'))
import standing_3lap as s3

HIDE = {'on': False, 'cut_t1': False}
orig_line_ends = s3._line_ends

def patched(vt, Df, A_, d_pre, t_line0, t1, Lrun):
    if HIDE['on'] and len(A_) == 4:
        A3 = A_[:3]
        if HIDE['cut_t1']:
            # 本当に「3 周目の線を越えた HS の上で人側」にした形: t1 = 3 周目の線(anchor の値)直後
            e, _ = orig_line_ends(vt, Df, A_, d_pre, t_line0, t1, Lrun)
            if len(e) == 3:
                t1 = e[2] + 0.5
        return orig_line_ends(vt, Df, A3, d_pre, t_line0, t1, Lrun)
    return orig_line_ends(vt, Df, A_, d_pre, t_line0, t1, Lrun)
s3._line_ends = patched

logs = []
for day in ('20261023', '20261024'):
    dl, _ = s3.day_logs(day)
    logs += dl
names = s3.candidate_names()
out = []
for lg in logs:
    HIDE['on'] = False
    r0 = s3.analyze_log(lg, names=names)
    HIDE['on'] = True; HIDE['cut_t1'] = False
    r1 = s3.analyze_log(lg, names=names)
    HIDE['cut_t1'] = True
    r2 = s3.analyze_log(lg, names=names)
    HIDE['on'] = False
    for a, b, c in zip(r0.get('standing') or [], r1.get('standing') or [], r2.get('standing') or []):
        if a.get('method') != 'anchor' or a.get('from_move_s') is None:
            continue
        out.append({'log': lg, 'ai_no': r0.get('ai_no'), 't_ai': a['t_ai'], 'Lrun': r0.get('lap_dist_median_m'),
                    'dist_anchor_m': a.get('dist_anchor_m'), 'd_pre': a.get('d_pre_m'),
                    'valid_a4': a['valid'], 'why_a4': a.get('why'), 'from_move_a4': a['from_move_s'],
                    'method_hid': b.get('method'), 'valid_hid': b['valid'], 'from_move_hid': b.get('from_move_s'),
                    'valid_cut': c['valid'], 'from_move_cut': c.get('from_move_s'), 'why_cut': c.get('why'),
                    'note': a.get('note')})
for o in out:
    d = None if o['from_move_hid'] is None else round(o['from_move_hid'] - o['from_move_a4'], 3)
    o['diff'] = d
    print('%s No.%s t_ai=%.2f Lrun=%s dist=%s d_pre=%s | A4: %.2f valid=%s | hid: %s %s valid=%s diff=%s | cut: %s valid=%s %s' % (
        o['log'], o['ai_no'], o['t_ai'], o['Lrun'], o['dist_anchor_m'], o['d_pre'], o['from_move_a4'], o['valid_a4'],
        o['method_hid'], o['from_move_hid'], o['valid_hid'], d, o['from_move_cut'], o['valid_cut'], o['why_cut']))
Path(sys.argv[1]).write_text(json.dumps(out, ensure_ascii=False, indent=1))
