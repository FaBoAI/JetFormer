#!/usr/bin/env python3
"""R2-rec-6 の確かめ: 回 4(ブロック 20〜23)の本で、発進の前進が ON なのに始まらなかった・本どうしで形が違う、を
runs_1103.py が印にするか。偽の記録を一時フォルダに作り、runs_1103 の RAW だけを差し替えて main() を流す。
data/raw・bench_candidates.json は読むだけ(bench は本物)。"""
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

PLAN = Path('/home/jetson/jetformer/experiments/20261026_fix/plan/runs_1103.py')
spec = importlib.util.spec_from_file_location('runs_1103', PLAN)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

tmp = Path(tempfile.mkdtemp(prefix='r2rec6_', dir=sys.argv[1] if len(sys.argv) > 1 else None))
BASE = {'steer_gain': 1.0, 'throttle_gain': 0.9, 'lookahead': 3, 'tof_guard': 1, 'zone_caps': 0, 'start_kick': 1,
        'speed_auto': 0, 'launch_us': 250, 'launch_mps': 2.5, 'launch_max_s': 1.2}


def rec(name, ai_no, block, tune, params, events):
    d = tmp / name
    (d / ('%s-0001' % name)).mkdir(parents=True)
    desc = 'ai_no=%d bench=x day=2026-11-03 block=%d cand=EXP tune=%s battery=#A V_before=8.30 memo=' % (ai_no, block, tune)
    j = {'description': desc, 'drive': {'run': 'clean_driving_20261011-nocmd-s3-20261011-024626', 'dry_run': False, 'params': dict(BASE, **params)}}
    (d / 'log.json').write_text(json.dumps(j, ensure_ascii=False), encoding='utf-8')
    (d / 'events.jsonl').write_text(''.join(json.dumps(dict(e, kind='launch_boost'), ensure_ascii=False) + '\n'
                                            for e in events), encoding='utf-8')


SKIP = [{'event': 'skip', 'why': 'near', 'reason': '前が近い(TTC の中央値 < 3.3 秒)'}]
START = [{'event': 'start'}, {'event': 'end', 'why': 'mps', 'reason': '速さ 2.5 m/s'}]
try:
    # 対照: 回 3 の ON の本(expect launch_boost=1)で始まらない → 印が付くはず
    rec('20261103-100000', 1, 13, '3-2', {'speed_ceiling': 1, 'launch_boost': 1}, SKIP)
    # 回 4: ON で走ったが始まらない(skip near)
    rec('20261103-110000', 2, 20, '4-1', {'speed_ceiling': 1, 'launch_boost': 1}, SKIP)
    # 回 4: ON で始まった(本番の形)
    rec('20261103-111000', 3, 21, '4-2', {'speed_ceiling': 1, 'launch_boost': 1}, START)
    # 回 4: 天井 OFF・発進 OFF(4-1・4-2 と形が違う)
    rec('20261103-112000', 4, 22, '4-3', {'speed_ceiling': 0, 'launch_boost': 0, 'throttle_gain': 1.0}, [])
    m.RAW = tmp
    sys.argv = ['runs_1103.py', '--day', '20261103', '--json']
    m.main()
finally:
    shutil.rmtree(tmp, ignore_errors=True)
    print('tmp removed:', not tmp.exists())
