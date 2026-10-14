#!/usr/bin/env python3
"""準備の段: 記録 D1・D2・D3 の「連続試験」の計画(各 30 シーン)を作る(2026-10-14、sim_dart_20261014)

    python -I ~/jetformer/experiments/20261014_sim_dart_margin/record/make_plans.py

  元: ベストラップの記録の計画 plan_rlbest60.json(ユーザーの qa/rc-collection-plan-60.json の写し)の 0〜23 番(train)と 48〜53 番(val)。
  開始・物理・センサー・種・train / val は A0 と同じで、名前だけ変える(protocol.json recording.plan_source・datasets.*.scene_names)。
  書く場所: /home/jetson/jetformer_reports/sim_dart_20261014/record_prep/plan_D{1,2,3}.json
"""
from __future__ import annotations

import json
from pathlib import Path

SRC = Path('/home/jetson/jetformer_reports/sim_rl_bestlap_20261013/recording/plan_rlbest60.json')
OUT = Path('/home/jetson/jetformer_reports/sim_dart_20261014/record_prep')
PREFIX = {'D1': 'dartfast10', 'D2': 'margin10', 'D3': 'margindart10'}
INDICES = list(range(0, 24)) + list(range(48, 54))


def main():
    src = json.loads(SRC.read_text(encoding='utf-8'))
    assert len(src) == 60
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for d, prefix in PREFIX.items():
        plan = []
        for i in INDICES:
            c = dict(src[i])
            old = c['name']
            assert old.startswith('rlbest20-mode8-'), old
            c['name'] = old.replace('rlbest20-mode8-', prefix + '-mode8-')
            assert c['dataset-split'] == ('train' if i < 48 else 'val')
            assert c['teacher-profile'] == 'speed-rc' and c['episode-mode'] == 'continuous-laps-v1' and c['limit'] == 20 and c['dataset-mode'] == 8
            plan.append(c)
        (OUT / f'plan_{d}.json').write_text(json.dumps(plan, ensure_ascii=False, indent=1), encoding='utf-8')
        summary[d] = {'scenes': len(plan), 'train': sum(c['dataset-split'] == 'train' for c in plan),
                      'val': sum(c['dataset-split'] == 'val' for c in plan), 'first': plan[0]['name'], 'last': plan[-1]['name'],
                      'scenarios': sorted({c['scenario'] for c in plan}), 'seeds': [c['seed'] for c in plan]}
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
