"""ens_average(学習済みの方策の actions の平均)の model_config.json を作る(torch を読まない。2026-10-21)

    from jetformer.policies import ens_config
    cfg = ens_config.make_config(['realacc_20261015-v0-s2', 'realacc_20261015-msf-s2'], runs_dir, io=...)

約束:
- メンバーは 2 本以上。走行の値(ジャンル・中心の値・入力の作り方)がすべて同じであること(SAME_KEYS)。違えば ValueError。
- 走行(drive.model_info)は model_config.json だけを読む。メンバーの学習結果は、torch で読み直すとき
  (runtime.load_run は、まとめた best.pth を読む)と記録のためだけ。
- アンサンブルの入れ子(メンバーが ens_average)は作らない。
"""

import json
from pathlib import Path

from . import CONTRACT, from_config, policy_of

POLICY = 'ens_average'
# メンバーどうしで同じでないといけない値(走行・正規化・ジャンル。違うと平均の意味が無い)
SAME_KEYS = ('num_modes', 'modes', 'future_steps', 'tof_channels', 'pwm_center', 'pwm_scale', 'tof_max_mm', 'erpm_scale',
             'trained_modes', 'steer_center_us', 'throttle_neutral_us', 'command_input')
# 先頭のメンバーからそのまま写す値(記録・表示用。違ってもよい)
COPY_KEYS = SAME_KEYS + ('dataset', 'stride', 'train_mode_counts', 'mask_impacts')


def _read(run_dir):
    with open(Path(run_dir) / 'model_config.json', encoding='utf-8') as f:
        return json.load(f)


def member_entry(run, cfg):
    """メンバー 1 本 → {'run', 'policy', 'policy_cfg'(全部のキー)}"""
    spec, pcfg = from_config(cfg)
    if spec.name == POLICY:
        raise ValueError('アンサンブルの入れ子は作りません: %s' % run)
    return {'run': run, 'policy': spec.name, 'policy_cfg': pcfg}


def make_config(member_runs, runs_dir, io=None, engine=None, note=''):
    """メンバーの学習結果の名前(runs_dir の下)→ ens_average の model_config(dict)。

    io: {'inputs', 'input_shapes', 'outputs', 'output_shapes'}(まとめた ONNX から読んだもの)。None なら先頭のメンバーの値。
    engine: {'precision', 'workspace_gb'}(None なら FP16・1 GB)"""
    member_runs = list(member_runs)
    if len(member_runs) < 2:
        raise ValueError('アンサンブルのメンバーは 2 本以上にしてください(%d 本)' % len(member_runs))
    if len(set(member_runs)) != len(member_runs):
        raise ValueError('同じメンバーが 2 回あります: %s' % member_runs)
    cfgs = [_read(Path(runs_dir) / r) for r in member_runs]
    base = cfgs[0]
    for r, c in zip(member_runs[1:], cfgs[1:]):
        for k in SAME_KEYS:
            if c.get(k) != base.get(k):
                raise ValueError('メンバーの %s が違います(%s: %r、%s: %r)' % (k, member_runs[0], base.get(k), r, c.get(k)))
    for r, c in zip(member_runs, cfgs):
        if c.get('postprocess'):
            raise ValueError('後処理のある方策(%s の %s)はアンサンブルに入れません' % (r, policy_of(c)))
        if int(c.get('contract', CONTRACT)) > CONTRACT:
            raise ValueError('新しい版の約束の学習結果です: %s' % r)
    members = [member_entry(r, c) for r, c in zip(member_runs, cfgs)]
    cfg = {k: base[k] for k in COPY_KEYS if k in base}
    io = io or {k: base.get(k) for k in ('inputs', 'input_shapes', 'outputs', 'output_shapes')}
    if any(not io.get(k) for k in ('inputs', 'input_shapes', 'outputs', 'output_shapes')):
        raise ValueError('入出力(inputs・input_shapes・outputs・output_shapes)が分かりません。まとめた ONNX から読んで渡してください')
    total = sum(float((c.get('params_m') or {}).get('total') or 0.0) for c in cfgs)
    cfg.update({
        'policy': POLICY, 'policy_cfg': {'members': members, 'member_runs': member_runs}, 'contract': CONTRACT,
        'members': member_runs,
        'inputs': list(io['inputs']), 'input_shapes': dict(io['input_shapes']),
        'outputs': list(io['outputs']), 'output_shapes': dict(io['output_shapes']),
        'engine': dict(engine or {'precision': 'fp16', 'workspace_gb': 1.0}),
        'params_m': {'total': round(total, 2), 'trainable': 0.0},
    })
    if note:
        cfg['note'] = note
    return cfg
