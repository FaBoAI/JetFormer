#!/usr/bin/env python3
"""確認用の静止画を summary_20261010/video/check/ に描く(make_digest.py の --stills と同じ描画。work/stills は使わない)。

    cd ~/jetformer_reports/ai_drive_digest && .venv/bin/python ../summary_20261010/video/src/render_stills.py 5,20,45
"""
import json
import os
import sys

DIGEST = '/home/jetson/jetformer_reports/ai_drive_digest'
sys.path.insert(0, DIGEST)
import make_digest as md  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SB = os.path.join(HERE, '..', 'summary_storyboard.json')
OUT = os.path.join(HERE, '..', 'check')

story = json.load(open(SB, encoding='utf-8'))
story['log'] = story.get('log') or '20261010-172910'
logs = {story['log']} | {sc['log'] for sc in story['scenes'] if sc.get('log')}
drives = {lg: md.Drive(lg) for lg in sorted(logs)}
scenes, total, sr = md.build(story, drives, os.path.join(DIGEST, 'work'))
cdir = os.path.join(DIGEST, 'chars', 'final')
char = md.Character(cdir) if os.path.exists(os.path.join(cdir, 'manifest.json')) else None
r = md.Renderer(drives, story, scenes, total, char)
os.makedirs(OUT, exist_ok=True)
print('全体 %.1f秒' % total)
for s in sys.argv[1].split(','):
    t = float(s)
    sc = md.scene_at(scenes, t)
    rel = t - sc['t_out']
    src = sc['src_start'] + min(rel * sc['speed_eff'], sc['src_end'] - sc['src_start'])
    p = os.path.join(OUT, 'still_%06.2f_%s.png' % (t, sc['id']))
    r.render(t).resize((960, 540)).save(p)
    print('%s  場面 %s  元 %.2f秒' % (p, sc['id'], src))
