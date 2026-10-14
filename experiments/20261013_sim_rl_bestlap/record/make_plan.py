#!/usr/bin/env python3
"""記録の段: 60 シーンの計画(アプリの「連続試験」の JSON)を作る(2026-10-13、design.md 4.3 節)

    python -I ~/jetformer/experiments/20261013_sim_rl_bestlap/record/make_plan.py

  ユーザーの qa/rc-collection-plan-60.json(20 分の教師データの計画。sha256 1c5bb220…)の 60 条件をそのまま写し、
  変えるのは name だけ(rc20-mode8-{train|val}-NN → rlbest20-mode8-{train|val}-NN)。
  開始(ゲート 0・横 −2.9〜+2.8 cm・向き ±2.9°)・種(2026101200+i)・baseline / noise(ToF 25 mm)の交互・train 0〜47 / val 48〜59・
  センサー・RC 応答の物理・controller demo・teacher-profile speed-rc・連続周回 20 秒・ジャンル 8 は元と同じ。
  出力: /home/jetson/jetformer_reports/sim_rl_bestlap_20261013/recording/plan_rlbest60.json
"""
import hashlib
import json
from pathlib import Path

SRC = Path('/home/jetson/sim/extracted/course-simulator/qa/rc-collection-plan-60.json')
EXPECTED = '1c5bb22087b5913531e6ebc76e6ae3da99c04ef1feaebfb1f2e74eaa68160dfb'
OUT = Path('/home/jetson/jetformer_reports/sim_rl_bestlap_20261013/recording/plan_rlbest60.json')


def main():
    raw = SRC.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED:
        raise SystemExit('元の計画の sha256 が違います')
    plan = json.loads(raw)
    assert len(plan) == 60 and len({c['name'] for c in plan}) == 60
    out = []
    for c in plan:
        assert c['name'].startswith('rc20-mode8-'), c['name']
        d = dict(c)
        d['name'] = 'rlbest20-mode8-' + c['name'][len('rc20-mode8-'):]
        assert d['name'].split('-')[2] == d['dataset-split']
        out.append(d)
    # name 以外は元と同じ
    for a, b in zip(plan, out):
        assert {k: v for k, v in a.items() if k != 'name'} == {k: v for k, v in b.items() if k != 'name'}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({'plan': str(OUT), 'conditions': len(out), 'sha256': hashlib.sha256(OUT.read_bytes()).hexdigest(),
                      'source_sha256': EXPECTED, 'train': sum(c['dataset-split'] == 'train' for c in out),
                      'val': sum(c['dataset-split'] == 'val' for c in out), 'noise': sum(c['scenario'] == 'noise' for c in out),
                      'first': out[0]['name'], 'last': out[-1]['name']}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
