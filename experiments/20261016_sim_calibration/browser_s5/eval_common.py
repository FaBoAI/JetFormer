#!/usr/bin/env python3
# (sim_calibration_20261016 の写し、2026-10-16: 出力を sim_calibration_20261016/browser、ブリッジ 8831 に変えた。モデルの一式・開始・物理の欄は 10/15 のまま)
"""閉ループ評価(sim_diverse_train_20261015 の Q1 の評価の段)の共通の定義(2026-10-15)

10/14 の experiments/20261014_sim_dart_margin/eval/eval_common.py の写し。変えたところ:
  - 評価するモデル(protocol.json eval_Q1.sessions_and_order):
      EXT      = 作者の ONNX(陽性対照の E0 だけ。10/14 は 9.022 秒で完走)
      A1-s0    = 10/14 の A1(前日との同一性の E0 だけ。10/14 は 衝突・169 フレーム)
      SDIV-s0  = 今日の主の腕 S-DIV(5 モードの見本 + DART + 余裕の線)。E0〜E11 + E0 の 2 回目 + フライング
      SMOD-s0  = 2 つ目の腕 S-MOD(5 モードの見本だけ)。E0〜E11 + フライング(削らなければ)
      A2-s0    = 前日と環境が違ったときだけ走らせ直す(protocol eval_Q1.same_as_yesterday_check)
      A4-s0    = DAgger(削らなければ)
  - ブリッジの口 8821・geckodriver 4450・プロファイル sim_diverse_train_20261015/ff_profile(protocol eval_Q1.ports_profile)
  - 一式(ZIP)は train/runs.json(学習の段が作る対応表)から読む。まだ無い腕は「無い」として飛ばす
物理・センサーの欄・開始・遅延・先読み・ゲインは 10/14 と同じ(ユーザーの REC60 の計画の 1 条件目から教師・収集の欄を除く)。
ユーザーのファイルは読むだけで変更しない。実車・PCA9685・I2C には触れない。
"""
from __future__ import annotations

import json
from pathlib import Path

REPORT = Path('/home/jetson/jetformer_reports/sim_diverse_train_20261015')
CODE = Path('/home/jetson/jetformer/experiments/20261015_sim_diverse_train')
PREV_REPORT = Path('/home/jetson/jetformer_reports/sim_dart_20261014')          # 10/14 の評価(A0・A1・A2・S-T・作者の ONNX の数字。読むだけ)
PREV_CODE = Path('/home/jetson/jetformer/experiments/20261014_sim_dart_margin')  # 10/14 の専用の venv(.venv-bridge。読むだけで使う)
BESTLAP_REPORT = Path('/home/jetson/jetformer_reports/sim_rl_bestlap_20261013')
USER_SIM = Path('/home/jetson/sim/extracted/course-simulator')
EVAL = Path('/home/jetson/jetformer_reports/sim_calibration_20261016') / 'browser'   # 2026-10-16: 出力はシミュレーター較正のフォルダー(モデルの一式は 10/15 の train/runs.json から読む)
CL = EVAL / 'closed_loop'
EXT_DIR = USER_SIM / 'models' / 'shortcut-rc-finetuned'
EXT_ZIP = PREV_REPORT / 'eval' / 'bundles' / 'shortcut-rc-finetuned.zip'      # 10/14 に作者のフォルダーから作った ZIP(読むだけ)
REC60_PLAN = USER_SIM / 'qa' / 'rc-collection-plan-60.json'
BRIDGE_PORT = 8831

# 評価するモデル。label は結果のフォルダー名にも使う。teacher = ずれ・引き継ぎの基準の先生(両方の先生の線を測るため、
# 今日の腕は margin にする: teacher_eval.mjs は fast と teacher の 2 つを測る)
MODEL_DEFS = [
    {'label': 'EXT', 'arm': '作者の ONNX', 'seed': None, 'key': None, 'run': 'shortcut-rc-finetuned', 'mode': 4, 'throttle_gain': 1.01,
     'teacher': 'fast', 'phases': ['e0']},
    {'label': 'A1-s0', 'arm': 'A1', 'seed': 0, 'key': 'A1/s0', 'run': 'simdart_20261014-A1-s0', 'mode': 8, 'throttle_gain': 1.0,
     'teacher': 'fast', 'phases': ['e0']},
    {'label': 'SDIV-s0', 'arm': 'S-DIV', 'seed': 0, 'key': 'S-DIV/s0', 'run': 'simdiv_20261015-SDIV-s0', 'mode': 8, 'throttle_gain': 1.0,
     'teacher': 'margin', 'phases': ['core', 'ext', 'flying']},
    {'label': 'SMOD-s0', 'arm': 'S-MOD', 'seed': 0, 'key': 'S-MOD/s0', 'run': 'simdiv_20261015-SMOD-s0', 'mode': 8, 'throttle_gain': 1.0,
     'teacher': 'margin', 'phases': ['core', 'ext', 'flying']},
    {'label': 'A2-s0', 'arm': 'A2', 'seed': 0, 'key': 'A2/s0', 'run': 'simdart_20261014-A2-s0', 'mode': 8, 'throttle_gain': 1.0,
     'teacher': 'margin', 'phases': ['core', 'ext']},
    {'label': 'A4-s0', 'arm': 'A4', 'seed': 0, 'key': 'A4/s0', 'run': 'simdiv_20261015-A4-s0', 'mode': 8, 'throttle_gain': 1.0,
     'teacher': 'margin', 'phases': ['core', 'ext']},
]


def load_models():
    """train/runs.json(学習の段の対応表)から ZIP・ONNX を付ける。一式がまだ無い腕は available False"""
    runs = {}
    p = REPORT / 'train' / 'runs.json'
    if p.exists():
        runs = json.loads(p.read_text(encoding='utf-8'))['runs']
    out = []
    for m0 in MODEL_DEFS:
        m = dict(m0)
        if m['label'] == 'EXT':
            m['zip'] = str(EXT_ZIP)
            m['onnx'] = str(EXT_DIR / 'jetformer.onnx')
            m['available'] = EXT_ZIP.exists()
        else:
            r = runs.get(m['key'])
            if r and r.get('bundle_zip') and Path(r['bundle_zip']).exists():
                assert r['run'] == m['run'], (r['run'], m['run'])
                m['zip'] = r['bundle_zip']
                m['onnx'] = str(Path(r['bundle_dir']) / 'jetformer.onnx')
                m['zip_sha256_expected'] = r.get('bundle_zip_sha256')
                m['onnx_sha256_expected'] = r.get('onnx_sha256')
                m['available'] = True
            else:
                m['available'] = False
        out.append(m)
    return out


MODELS = load_models()
MODEL_BY_LABEL = {m['label']: m for m in MODELS}

# protocol.json eval_Q1.starts([ゲート, 左ずれ m, 向き 度])
STARTS = {
    'E0': (0, 0.0, 0.0), 'E1': (0, 0.03, 3.0), 'E2': (0, -0.03, -3.0), 'E3': (0, 0.08, -5.0), 'E4': (0, -0.08, 5.0), 'E5': (20, 0.0, 0.0),
    'E6': (0, 0.05, 2.0), 'E7': (0, -0.05, -2.0), 'E8': (0, 0.02, -4.0), 'E9': (0, -0.02, 4.0), 'E10': (10, 0.0, 0.0), 'E11': (28, 0.0, 0.0),
}
# protocol.json dagger.states_starts(DAgger の状態集め。評価の開始とは別。結果は closed_loop_dagger/ に分けて置く)
DSTARTS = {
    'D0': (0, 0.04, 1.0), 'D1': (0, -0.04, -1.0), 'D2': (0, 0.01, 2.0), 'D3': (0, -0.01, -2.0), 'D4': (0, 0.06, -3.0), 'D5': (0, -0.06, 3.0),
    'D6': (0, 0.0, 1.0), 'D7': (0, 0.0, -1.0), 'D8': (5, 0.0, 0.0), 'D9': (15, 0.0, 0.0), 'D10': (24, 0.0, 0.0), 'D11': (31, 0.0, 0.0),
}
CL_DAGGER = EVAL / 'closed_loop_dagger'
CORE = ['E0', 'E1', 'E2', 'E3', 'E4', 'E5']
EXT_STARTS = ['E6', 'E7', 'E8', 'E9', 'E10', 'E11']
GATE0_STARTS = ['E0', 'E1', 'E2', 'E3', 'E4', 'E6', 'E7', 'E8', 'E9']   # protocol eval_Q1.starts.gate0_starts(9 本)
SINGLE_LIMIT_S = 15
DROP_KEYS = {'controller', 'teacher-profile', 'episode-mode', 'start-gate', 'limit', 'dataset-mode', 'name', 'dataset-split',
             'scenario', 'seed', 'spawn-offset', 'spawn-yaw'}
# 名前の頭(ラベルの 1 つ目の部分)。run_meta で使う
LABEL_HEADS = ('A0', 'A1', 'A2', 'A3', 'A4', 'SR', 'ST', 'SDIV', 'SMOD')


def base_condition():
    plan = json.loads(REC60_PLAN.read_text(encoding='utf-8'))
    base = {k: v for k, v in plan[0].items() if k not in DROP_KEYS}
    for c in plan:
        for k, v in base.items():
            assert c[k] == v, f'REC60 の欄 {k} が条件ごとに違います'
    return base


def condition(model, name, start, *, continuous=False, delay=1, lookahead=1, steer_gain=1.0, throttle_gain=None):
    """アプリの連続試験の 1 条件(controller external)。10/14 と同じ作り方"""
    gate, off, yaw = STARTS[start] if start in STARTS else DSTARTS[start]
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
    """段: e0 = E0 の単周 1 本(陽性対照・前日との同一性)、core = E0〜E5(S-DIV は E0 の 2 回目も)、ext = E6〜E11、flying = E0 の連続周回 20 秒、
    dstates = DAgger の状態集め D0〜D11(--out closed_loop_dagger で走らせる)"""
    lab = model['label']
    if phase == 'e0':
        return [condition(model, f'{lab}-E0-single', 'E0')]
    if phase == 'core':
        plan = [condition(model, f'{lab}-{e}-single', e) for e in CORE]
        if lab == 'SDIV-s0':
            plan.append(condition(model, f'{lab}-E0-single-rep2', 'E0'))   # 決定性の確かめ(protocol sessions_and_order 3)
        return plan
    if phase == 'ext':
        return [condition(model, f'{lab}-{e}-single', e) for e in EXT_STARTS]
    if phase == 'dstates':
        return [condition(model, f'{lab}-{d}-single', d) for d in DSTARTS]   # DAgger の状態集め(評価と同じ設定の単周)
    if phase == 'flying':
        return [condition(model, f'{lab}-E0-flying', 'E0', continuous=True)]
    raise ValueError(phase)


def run_meta(name):
    """計画の name から(モデル、開始、種類、変種)を読む。例: SDIV-s0-E0-single-rep2 / EXT-E0-single / A1-s0-E3-single"""
    p = name.split('-')
    if p[0] in LABEL_HEADS:
        label, rest = f'{p[0]}-{p[1]}', p[2:]
    else:
        label, rest = p[0], p[1:]
    return label, rest[0], rest[1], '-'.join(rest[2:]) or 'protocol'


if __name__ == '__main__':
    for m in MODELS:
        if not m['available']:
            print(m['label'], '一式なし(まだ学習していない・削った)')
            continue
        for ph in m['phases']:
            p = phase_plan(m, ph)
            print(m['label'], ph, len(p), [c['name'] for c in p])
    print(json.dumps(base_condition(), ensure_ascii=False))
