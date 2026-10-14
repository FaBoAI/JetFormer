#!/usr/bin/env python3
"""閉ループ評価(段 E)の共通の定義(2026-10-14、強化学習のベストラップ)

  - 評価するモデル(学習した 4 本 + 作者の外部の参照 shortcut-rc-finetuned)と、その一式の ZIP
  - 開始の 6 条件(E0〜E5)とフライングラップ(E0・E1 の連続周回 20 秒)。protocol.json の closed_loop_eval のとおり
  - アプリの「連続試験」の計画(JSON の配列)の作り方。物理・センサーは記録の 60 条件(REC60)と同じ欄をそのまま使う
ユーザーのファイルは読むだけで変更しない。
"""
from __future__ import annotations

import json
from pathlib import Path

REPORT = Path('/home/jetson/jetformer_reports/sim_rl_bestlap_20261013')
CODE = Path('/home/jetson/jetformer/experiments/20261013_sim_rl_bestlap')
USER_SIM = Path('/home/jetson/sim/extracted/course-simulator')
EVAL = REPORT / 'eval'
BUNDLES = REPORT / 'bundles'
EXT_DIR = USER_SIM / 'models' / 'shortcut-rc-finetuned'
EXT_ZIP = EVAL / 'bundles' / 'shortcut-rc-finetuned.zip'
REC60_PLAN = USER_SIM / 'qa' / 'rc-collection-plan-60.json'
BRIDGE_PORT = 8792

# 評価するモデル。label は結果のフォルダー名にも使う
MODELS = [
    {'label': 'SR-s0', 'arm': 'S-R', 'seed': 0, 'run': 'simrl_bestlap_20261013-SR-s0', 'mode': 8, 'throttle_gain': 1.0},
    {'label': 'SR-s1', 'arm': 'S-R', 'seed': 1, 'run': 'simrl_bestlap_20261013-SR-s1', 'mode': 8, 'throttle_gain': 1.0},
    {'label': 'ST-s0', 'arm': 'S-T', 'seed': 0, 'run': 'simrl_bestlap_20261013-ST-s0', 'mode': 8, 'throttle_gain': 1.0},
    {'label': 'ST-s1', 'arm': 'S-T', 'seed': 1, 'run': 'simrl_bestlap_20261013-ST-s1', 'mode': 8, 'throttle_gain': 1.0},
    {'label': 'EXT', 'arm': '外部の参照', 'seed': None, 'run': 'shortcut-rc-finetuned', 'mode': 4, 'throttle_gain': 1.01},
]
for m in MODELS:
    if m['label'] == 'EXT':
        m['zip'] = str(EXT_ZIP)
        m['onnx'] = str(EXT_DIR / 'jetformer.onnx')
    else:
        m['zip'] = str(BUNDLES / f"{m['run']}.zip")
        m['onnx'] = str(BUNDLES / m['run'] / 'jetformer.onnx')
MODEL_BY_LABEL = {m['label']: m for m in MODELS}

# 開始の 6 条件(protocol.json closed_loop_eval.starts_single_lap)
STARTS = {
    'E0': (0, 0.0, 0.0),
    'E1': (0, 0.03, 3.0),
    'E2': (0, -0.03, -3.0),
    'E3': (0, 0.08, -5.0),
    'E4': (0, -0.08, 5.0),
    'E5': (20, 0.0, 0.0),
}
SINGLE_LIMIT_S = 15
FLYING = ['E0', 'E1']           # 連続周回 20 秒
# REC60 の 1 条件から、物理・センサーの欄だけを使う(教師・収集の欄は外す)
DROP_KEYS = {'controller', 'teacher-profile', 'episode-mode', 'start-gate', 'limit', 'dataset-mode', 'name', 'dataset-split',
             'scenario', 'seed', 'spawn-offset', 'spawn-yaw'}


def base_condition():
    plan = json.loads(REC60_PLAN.read_text(encoding='utf-8'))
    base = {k: v for k, v in plan[0].items() if k not in DROP_KEYS}
    # 30 条件目(雑音なしの別の開始)とも物理・センサーの欄が同じことを確かめる(計画の全条件で共通の欄だけを使う)
    for c in plan:
        for k, v in base.items():
            assert c[k] == v, f'REC60 の欄 {k} が条件ごとに違います'
    return base


def condition(model, name, start, *, continuous=False, delay=1, lookahead=1, steer_gain=1.0, throttle_gain=None):
    """アプリの連続試験の 1 条件(controller external)"""
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


def protocol_plan(model):
    """protocol の計画: E0〜E5 の単周 → (S-R 種 0 は)E0 の繰り返し → フライング E0・E1 → (S-R 種 0 は)ゲインの感度"""
    lab = model['label']
    plan = [condition(model, f'{lab}-{e}-single', e) for e in STARTS]
    if lab == 'SR-s0':
        plan.append(condition(model, f'{lab}-E0-single-rep2', 'E0'))
    plan += [condition(model, f'{lab}-{e}-flying', e, continuous=True) for e in FLYING]
    if lab == 'SR-s0':
        plan += [condition(model, f'{lab}-E0-single-thr1.01', 'E0', throttle_gain=1.01),
                 condition(model, f'{lab}-E0-single-steer0.95', 'E0', steer_gain=0.95),
                 condition(model, f'{lab}-E0-single-steer1.05', 'E0', steer_gain=1.05)]
    return plan


def extra_plan(model):
    """事後に足した参考の条件(合否に使わない。protocol の評価のあとで、結果を見てから決めた。deviations.md に書く)。
    すべて E0 の単周。ねらい:
      - x-d0:     制御遅延 0(記録の教師と同じ遅延。遅延 1 + 先読み 1 と比べ、1 フレームの扱いが効いているかを見る)
      - x-la2:    遅延 1 のまま、2 フレーム先の予測を使う(先読み 2。train.md 8 章の提案)
      - x-thr0.9 / x-thr0.8: 前進ゲインを下げて遅く走らせる(速さの余裕でぶつからなくなるか)
      - x-thr1.01: 作者の外部の参照と同じ前進ゲイン 1.01(S-T の止まり方が、作者の言う前進の弱さかを見る)"""
    lab = model['label']
    if lab.startswith('SR'):
        spec = [('x-d0', dict(delay=0)), ('x-la2', dict(lookahead=2)), ('x-thr0.9', dict(throttle_gain=0.9)),
                ('x-thr0.8', dict(throttle_gain=0.8))]
    elif lab.startswith('ST'):
        spec = [('x-thr1.01', dict(throttle_gain=1.01)), ('x-la2', dict(lookahead=2)), ('x-d0', dict(delay=0))]
    else:
        spec = [('x-d0', dict(delay=0))]
    return [condition(model, f'{lab}-E0-single-{tag}', 'E0', **kw) for tag, kw in spec]


def run_meta(name):
    """計画の name から(モデル、開始、種類)を読む"""
    parts = name.split('-')
    label = '-'.join(parts[:2]) if parts[0] in ('SR', 'ST') else parts[0]
    rest = parts[2:] if parts[0] in ('SR', 'ST') else parts[1:]
    return {'label': label, 'start': rest[0], 'kind': rest[1], 'variant': '-'.join(rest[2:]) or 'protocol'}


if __name__ == '__main__':
    for m in MODELS:
        p = protocol_plan(m)
        print(m['label'], len(p), [c['name'] for c in p])
    print(json.dumps(base_condition(), ensure_ascii=False))
