#!/usr/bin/env python3
"""build_ai_dataset.scene_problem の直し(2026-10-26 fix2)の前と後で、本物の data/raw の全シーンの答えが同じかを確かめる(読むだけ)。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- ionice -c3 nice -n 19 \\
        /home/jetson/miniconda/envs/jetracer/bin/python -I experiments/20261026_fix/fix2/check_scene_problem_same.py <前の build_ai_dataset.py>
前の build_ai_dataset.py は git show 81a4d34:tools/build_ai_dataset.py の中身(10/26 の直しの前の写しと同じ)をファイルにして渡す。
結果: experiments/20261026_fix/fix2/out/check_scene_problem_same.json
"""
import json
import sys
import types
from pathlib import Path

JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF / 'tools'))
sys.path.insert(0, str(JF))
import build_ai_dataset as new  # noqa: E402

old_src = Path(sys.argv[1])
old = types.ModuleType('build_ai_dataset_old')
old.__file__ = str(JF / 'tools' / 'build_ai_dataset.py')
exec(compile(old_src.read_text(encoding='utf-8'), str(old_src), 'exec'), old.__dict__)
raw = JF / 'data' / 'raw'
rows, diff = 0, []
reasons = {}
for log in sorted(d.name for d in raw.iterdir() if d.is_dir()):
    for s in new.log_scenes(log, raw):
        p = raw / log / s
        a, b = old.scene_problem(p), new.scene_problem(p)
        rows += 1
        if b:
            reasons[s] = b
        if a != b:
            diff.append({'scene': s, 'old': a, 'new': b})
out = {'scenes': rows, 'differ': diff, 'problems_new': reasons}
od = Path(__file__).resolve().parent / 'out'
od.mkdir(exist_ok=True)
(od / 'check_scene_problem_same.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
print('シーン %d・違い %d・問題のあるシーン(後)%d' % (rows, len(diff), len(reasons)))
for d in diff:
    print(d)
