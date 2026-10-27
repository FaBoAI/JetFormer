"""本物の落ちた本(読むだけ)の lost_tail と ON の手がかり(⑤ collect.py の今の版)。 → ../out/real_tail.json"""
import json
import sys
from pathlib import Path
JF = Path('/home/jetson/jetformer')
for p in (JF, JF / 'tools', JF / 'tests'):
    sys.path.insert(0, str(p))
from test_fix3_r1_2 import load_mod  # noqa: E402
m = load_mod('collect_real_tail', JF / 'experiments/20261026_fix/diag_launch/collect.py')
blocks, names = m.bench_blocks()
plans = m.bench_plan_launch()
res = {}
for lg in ('20261010-163247', '20261024-162305'):
    d = m.load(lg, True, blocks, names)
    ls = m.lost_scenes(d)
    u = m.unreadable_info(lg, '', blocks, names, plans)
    res[lg] = {'problems_n': len(d['problems']), 'lost_scenes': ls, 'lost_tail': m.lost_tail(ls),
               'launch_on_from': m.launch_on_from(d, u), 'launch_param': u['launch_param'], 'plan_launch': u['plan_launch'],
               'ai_frames': int(d['z']['ai'].sum())}
out = Path(__file__).resolve().parent.parent / 'out' / 'real_tail.json'
out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print(json.dumps(res, ensure_ascii=False, default=str))
