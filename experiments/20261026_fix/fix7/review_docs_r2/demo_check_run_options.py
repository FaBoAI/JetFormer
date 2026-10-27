"""反証 7-2・docs: 手順書 3 章の「発進の前進の確かめ」を収集したまま(手順書が許す形。memo に「試し」)で走り、そのあと 3 章の順に
試験の選択肢(見張り → 残す)を入れて 1-1・1-2 を走った日に、② runs_1103 が出す行を本物の runs_1103.py で見る(読むだけ)。
data/raw は使わない: runs_1103 の RAW をこの台本の作業場所の作り物に向ける(bench_candidates.json は本物を読むだけ)。

    python3 -I demo_check_run_options.py <作業場所>
"""
import importlib.util
import json
import sys
from pathlib import Path

work = Path(sys.argv[1])
raw = work / 'raw'
spec = importlib.util.spec_from_file_location('runs_1103', '/home/jetson/jetformer/experiments/20261026_fix/plan/runs_1103.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.RAW = raw

BASE = {'throttle_gain': 0.9, 'speed_ceiling': 0, 'ceil_left_mps': 4.1, 'speed_auto': 0, 'launch_boost': 0, 'launch_us': 250.0,
        'steer_trim_us': 0.0, 'zone_caps': 0, 'rev_s': 0.4, 'rev_us': 300.0,
        'rec_back_guard': 0, 'rec_hold_keep': 0, 'launch_no_kick': 0}


def put(name, desc, **kw):
    d = raw / name
    d.mkdir(parents=True, exist_ok=True)
    p = dict(BASE, **kw)
    (d / 'log.json').write_text(json.dumps({'description': desc, 'drive': {'run': 'clean_driving_20261011-nocmd-s3-20261011-024626',
                                                                            'params': p, 'dry_run': False}}), encoding='utf-8')
    if p['launch_boost']:
        (d / 'events.jsonl').write_text(json.dumps({'kind': 'launch_boost', 'event': 'start'}) + '\n'
                                        + json.dumps({'kind': 'launch_boost', 'event': 'end', 'reason': '速さに届いた'}) + '\n',
                                        encoding='utf-8')


# 3 章の順: 発進の前進の確かめ(基準の設定のあと。block の無い説明欄・収集したまま・memo に「試し」。試験の選択肢はまだ OFF)
put('20261103-081500', 'ai_no=1 memo=試し', launch_boost=1)
# そのあと 3 章の「試験の選択肢」で見張り → 残すを ON(11/3 の形)。1-1(No.2)・1-2(No.3)
put('20261103-083000', 'day=2026-11-03 block=1 official=1 ai_no=2 battery=#A', rec_back_guard=1, rec_hold_keep=1)
put('20261103-084000', 'day=2026-11-03 block=2 official=1 ai_no=3 battery=#A', rec_back_guard=1, rec_hold_keep=1)
sys.argv = ['runs_1103.py', '--day', '20261103']
m.main()
