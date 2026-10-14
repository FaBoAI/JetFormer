#!/usr/bin/env python3
"""閉ループ評価(sim_dart_20261014 の評価の段)の共通の定義(2026-10-14)

ベストラップの eval/eval_common.py の写し。変えたところ:
  - 評価するモデル: A1-s0・A2-s0(この作業で学習)、A0-s0(= ベストラップの S-R 種 0。E6〜E11 だけ新しく走らせる)、
    EXT(作者の ONNX。陽性対照の E0 だけ)
  - 開始: protocol.json eval.starts の E0〜E11(E0〜E5 はベストラップと同じ。E6〜E11 は今日足した)
  - 計画を「段」に分けた(電源断・遅れに備え、削る順番どおりに走らせる):
      core   = E0〜E5 の単周(never_cut)+ A1 は E0 の 2 回目(決定性の確かめ)
      ext    = E6〜E11 の単周(遅れたら protocol の順で削る)
      flying = E0 の連続周回 20 秒(参考。最初に削る)
  - ブリッジの口 8812(protocol eval.where)
物理・センサーの欄はベストラップと同じく、ユーザーの REC60 の計画(qa/rc-collection-plan-60.json)の 1 条件目から
教師・収集の欄を除いて使う。ユーザーのファイルは読むだけで変更しない。
"""
from __future__ import annotations

import json
from pathlib import Path

REPORT = Path('/home/jetson/jetformer_reports/sim_dart_20261014')
CODE = Path('/home/jetson/jetformer/experiments/20261014_sim_dart_margin')
BESTLAP_REPORT = Path('/home/jetson/jetformer_reports/sim_rl_bestlap_20261013')
USER_SIM = Path('/home/jetson/sim/extracted/course-simulator')
EVAL = REPORT / 'eval'
CL = EVAL / 'closed_loop'
EXT_DIR = USER_SIM / 'models' / 'shortcut-rc-finetuned'
EXT_ZIP = EVAL / 'bundles' / 'shortcut-rc-finetuned.zip'
REC60_PLAN = USER_SIM / 'qa' / 'rc-collection-plan-60.json'
BRIDGE_PORT = 8812

# 評価するモデル。label は結果のフォルダー名にも使う。teacher = その腕の先生(ずれ・引き継ぎの基準)
MODELS = [
    {'label': 'EXT', 'arm': '作者の ONNX', 'seed': None, 'run': 'shortcut-rc-finetuned', 'mode': 4, 'throttle_gain': 1.01,
     'teacher': 'fast', 'phases': ['core_ext_e0']},
    {'label': 'A1-s0', 'arm': 'A1', 'seed': 0, 'run': 'simdart_20261014-A1-s0', 'mode': 8, 'throttle_gain': 1.0,
     'teacher': 'fast', 'phases': ['core', 'ext', 'flying']},
    {'label': 'A2-s0', 'arm': 'A2', 'seed': 0, 'run': 'simdart_20261014-A2-s0', 'mode': 8, 'throttle_gain': 1.0,
     'teacher': 'margin', 'phases': ['core', 'ext', 'flying']},
    {'label': 'A0-s0', 'arm': 'A0', 'seed': 0, 'run': 'simrl_bestlap_20261013-SR-s0', 'mode': 8, 'throttle_gain': 1.0,
     'teacher': 'fast', 'phases': ['ext']},
]
_runs = json.loads((REPORT / 'train' / 'runs.json').read_text(encoding='utf-8'))['runs']
for m in MODELS:
    if m['label'] == 'EXT':
        m['zip'] = str(EXT_ZIP)
        m['onnx'] = str(EXT_DIR / 'jetformer.onnx')
    else:
        r = _runs[m['label'].replace('-', '/')]
        assert r['run'] == m['run'], m
        m['zip'] = r['bundle_zip']
        m['onnx'] = str(Path(r['bundle_dir']) / 'jetformer.onnx')
        m['zip_sha256_expected'] = r['bundle_zip_sha256']
        m['onnx_sha256_expected'] = r['onnx_sha256']
MODEL_BY_LABEL = {m['label']: m for m in MODELS}

# protocol.json eval.starts([ゲート, 左ずれ m, 向き 度])
STARTS = {
    'E0': (0, 0.0, 0.0), 'E1': (0, 0.03, 3.0), 'E2': (0, -0.03, -3.0), 'E3': (0, 0.08, -5.0), 'E4': (0, -0.08, 5.0), 'E5': (20, 0.0, 0.0),
    'E6': (0, 0.05, 2.0), 'E7': (0, -0.05, -2.0), 'E8': (0, 0.02, -4.0), 'E9': (0, -0.02, 4.0), 'E10': (10, 0.0, 0.0), 'E11': (28, 0.0, 0.0),
}
CORE = ['E0', 'E1', 'E2', 'E3', 'E4', 'E5']
EXT_STARTS = ['E6', 'E7', 'E8', 'E9', 'E10', 'E11']
GATE0_STARTS = ['E0', 'E1', 'E2', 'E3', 'E4', 'E6', 'E7', 'E8', 'E9']   # ゲート 0 の開始(最初の直線を抜けたかを数える。protocol の 10 本の数え方は analyze で)
SINGLE_LIMIT_S = 15
DROP_KEYS = {'controller', 'teacher-profile', 'episode-mode', 'start-gate', 'limit', 'dataset-mode', 'name', 'dataset-split',
             'scenario', 'seed', 'spawn-offset', 'spawn-yaw'}


def base_condition():
    plan = json.loads(REC60_PLAN.read_text(encoding='utf-8'))
    base = {k: v for k, v in plan[0].items() if k not in DROP_KEYS}
    for c in plan:
        for k, v in base.items():
            assert c[k] == v, f'REC60 の欄 {k} が条件ごとに違います'
    return base


def condition(model, name, start, *, continuous=False, delay=1, lookahead=1, steer_gain=1.0, throttle_gain=None):
    """アプリの連続試験の 1 条件(controller external)。ベストラップと同じ作り方"""
    gate, off, yaw = STARTS[start]
    c = dict(base_condition())
    c.update({
        'name': name,
        'controller': 'external',
        'scenario': 'baseline',
        'seed': 1,
        'episode-mode': 'continuous-laps-v1' if continuous else 'single-lap',
        'limit': 20 if continuous else SINGLE_LIMIT_S,
        'start-gate': gate, 'spawn-offset': off, 'spawn-yaw': yaw,
        'control-delay-frames': delay,
        'model-mode': model['mode'],
        'model-lookahead': lookahead,
        'model-steer-gain': steer_gain,
        'model-throttle-gain': model['throttle_gain'] if throttle_gain is None else throttle_gain,
    })
    return c


def phase_plan(model, phase):
    lab = model['label']
    if phase == 'core_ext_e0':
        return [condition(model, f'{lab}-E0-single', 'E0')]
    if phase == 'core':
        plan = [condition(model, f'{lab}-{e}-single', e) for e in CORE]
        if lab == 'A1-s0':
            plan.append(condition(model, f'{lab}-E0-single-rep2', 'E0'))
        return plan
    if phase == 'ext':
        return [condition(model, f'{lab}-{e}-single', e) for e in EXT_STARTS]
    if phase == 'flying':
        return [condition(model, f'{lab}-E0-flying', 'E0', continuous=True)]
    raise ValueError(phase)


def run_meta(name):
    """計画の name から(モデル、開始、種類、変種)を読む。例: A1-s0-E0-single-rep2 / EXT-E0-single / SR-s0-E3-single"""
    p = name.split('-')
    if p[0] in ('A0', 'A1', 'A2', 'A3', 'A4', 'SR', 'ST'):
        label, rest = f'{p[0]}-{p[1]}', p[2:]
    else:
        label, rest = p[0], p[1:]
    return label, rest[0], rest[1], '-'.join(rest[2:]) or 'protocol'


if __name__ == '__main__':
    for m in MODELS:
        for ph in m['phases']:
            p = phase_plan(m, ph)
            print(m['label'], ph, len(p), [c['name'] for c in p])
    print(json.dumps(base_condition(), ensure_ascii=False))
