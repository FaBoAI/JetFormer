"""方策 ens_average(学習済みの方策の actions の平均・2026-10-21 段 S4)のテスト。CPU だけ・サーバーは起動しない・車は動かさない

- 登録簿: 走行・評価だけの方策(RUN_ONLY)。get・from_config・title_of で読める。学習の選択肢(names・list_specs・
  GET /api/policies・tools/train.py --policy)には出ない。ready() は学習できない理由を返す
- 構築: メンバーの構造を作り、forward はメンバーの actions の平均。state_dict のキーは ms.0.*・ms.1.*
  (10/15 の ens_gate.make_average_module と同じ = 10/15 のエンジンの元の形)
- model_config: ens_config.make_config はメンバーの走行の値が同じことを確かめる。drive.model_info で drivable True
- 置いた候補(data/runs/bench_20261021-ens-p2)があれば、走行タブで走れる状態か

    cd ~/jetformer && /home/jetson/bin/jf-heavy --mem 3G --lock big --need-mb 2500 -- \\
        env CUDA_VISIBLE_DEVICES= /home/jetson/miniconda/envs/jetracer/bin/python -m pytest -q tests/test_policy_ens.py
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
from jetformer import drive, policies  # noqa: E402
from jetformer.policies import ens_average, ens_config, runtime  # noqa: E402

SMALL_V0 = {'dim': 32, 'layers': 1, 'heads': 4}
SMALL_ACT = {'dim': 32, 'enc_layers': 1, 'dec_layers': 1, 'heads': 4, 'ffn': 64}
ENS_RUN = os.path.join(ROOT, 'data', 'runs', 'bench_20261021-ens-p2')
ENS_GATE = os.path.join(ROOT, 'experiments', '20261015_real_accuracy', 'ens', 'ens_gate.py')
BASE_CFG = {'num_modes': 6, 'modes': [{'id': i, 'name': str(i), 'description': ''} for i in range(6)], 'future_steps': 10,
            'tof_channels': 2, 'pwm_center': 1500.0, 'pwm_scale': 500.0, 'tof_max_mm': 4000.0, 'erpm_scale': 20000.0,
            'dataset': 'clean_driving_20261011', 'stride': 3, 'trained_modes': [4, 5], 'steer_center_us': 1441.0,
            'throttle_neutral_us': 1496.0, 'command_input': False, 'mask_impacts': True, 'contract': 1,
            'inputs': list(policies.BASE_INPUTS),
            'input_shapes': {'image': [1, 3, 224, 224], 'tof': [1, 2, 42, 54], 'imu': [1, 10, 6], 'state': [1, 3],
                             'mode': [1, 6]},
            'outputs': ['actions'], 'output_shapes': {'actions': [1, 10, 2]}, 'params_m': {'total': 1.0, 'trainable': 1.0}}


def _x(batch=2, seed=3):
    g = torch.Generator().manual_seed(seed)
    return dict(image=torch.randn(batch, 3, 224, 224, generator=g), tof=torch.rand(batch, 2, 42, 54, generator=g),
                imu=torch.randn(batch, 10, 6, generator=g) * 0.1, state=torch.randn(batch, 3, generator=g) * 0.1,
                mode=torch.nn.functional.one_hot(torch.tensor([4, 5][:batch]), 6).float())


def _member_run(runs, name, policy, pcfg, **over):
    d = runs / name
    d.mkdir(parents=True)
    (d / 'model_config.json').write_text(json.dumps(dict(BASE_CFG, policy=policy, policy_cfg=pcfg, **over)))
    return d


# ---------------------------------------------------------------- 登録簿
def test_registered_as_run_only_not_in_training_choices():
    spec = policies.get('ens_average')
    assert spec.name == 'ens_average' and spec.trainable is False and spec.genre_keys is None
    assert 'ens_average' in policies.run_only_names() and 'ens_average' not in policies.names()
    assert 'ens_average' not in [s['name'] for s in policies.list_specs()]
    assert [s['name'] for s in policies.list_specs()] == ['jetformer_v0', 'act', 'tcp', 'msfusion', 'dinov2']
    assert policies.title_of('ens_average') == spec.title and '平均' in spec.title
    ok, why = spec.ready()
    assert ok is False and '学習では作れません' in why
    for name in policies.names():                     # ほかの方策は今まで通り学習できる
        assert policies.get(name).trainable is True
    with pytest.raises(ValueError):
        policies.register_run_only(policies.PolicySpec('ens_average', 'x', 'm', 'x', '', trainable=False))
    with pytest.raises(ValueError):
        policies.register_run_only(policies.PolicySpec('_t', 'x', 'm', 'x', ''))        # trainable のまま
    with pytest.raises(KeyError):
        policies.get('no_such_policy')


def test_train_cli_does_not_offer_ens():
    out = subprocess.run([sys.executable, os.path.join(ROOT, 'tools', 'train.py'), '--policy', 'ens_average', '--help'],
                         cwd=ROOT, capture_output=True, text=True, timeout=120, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
    assert out.returncode != 0 and 'ens_average' in out.stderr      # choices の外


def test_ens_config_does_not_import_torch():
    code = 'import sys; import jetformer.policies.ens_config, jetformer.drive; print("torch" in sys.modules)'
    out = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True, timeout=60,
                         env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == 'False'


# ---------------------------------------------------------------- 構築・平均
def test_build_is_mean_of_members_with_loaded_weights(tmp_path):
    torch.manual_seed(0)
    a = policies.build_policy('jetformer_v0', 6, SMALL_V0).eval()
    b = policies.build_policy('act', 6, SMALL_ACT).eval()
    ens = ens_average.from_members([a, b]).eval()
    keys = list(ens.state_dict())
    assert all(k.startswith('ms.0.') or k.startswith('ms.1.') for k in keys)
    # model_config の policy_cfg から作り直し、まとめた重みを strict で読む(runtime.load_run と同じ道)
    pcfg = {'members': [{'run': 'a', 'policy': 'jetformer_v0', 'policy_cfg': policies.get('jetformer_v0').resolve(SMALL_V0)},
                        {'run': 'b', 'policy': 'act', 'policy_cfg': policies.get('act').resolve(SMALL_ACT)}],
            'member_runs': ['a', 'b']}
    d = tmp_path / 'ens'
    d.mkdir()
    (d / 'model_config.json').write_text(json.dumps({'num_modes': 6, 'policy': 'ens_average', 'policy_cfg': pcfg}))
    torch.save(ens.state_dict(), d / 'best.pth')
    model, cfg, spec, resolved = runtime.load_run(d, 'best')
    assert spec.name == 'ens_average' and resolved['member_runs'] == ['a', 'b']
    x = _x()
    out = runtime.predict(model, x)
    assert set(out) == {'actions'} and out['actions'].shape == (2, 10, 2)
    pa = runtime.predict(a, x)['actions']
    pb = runtime.predict(b, x)['actions']
    assert np.allclose(out['actions'], (pa + pb) / 2.0, atol=1e-6)
    assert not np.allclose(pa, pb)                    # 平均が意味を持つ(2 本が違う)
    runtime.check_forward(model, 6)


def test_build_rejects_bad_members():
    v0 = {'run': 'a', 'policy': 'jetformer_v0', 'policy_cfg': SMALL_V0}
    with pytest.raises(ValueError):
        policies.build_policy('ens_average', 6, {'members': [v0]})                 # 1 本だけ
    with pytest.raises(ValueError):
        policies.build_policy('ens_average', 6, {'members': [v0, {'run': 'e', 'policy': 'ens_average'}]})   # 入れ子
    with pytest.raises(ValueError):
        policies.build_policy('ens_average', 6, {'members': [v0, 'b']})


def test_keys_match_the_10_15_engine_module():
    """10/15 のエンジン(ens_p2_fp16.engine)を書き出したモジュール(ens_gate.make_average_module)と同じキー"""
    if not os.path.exists(ENS_GATE):
        pytest.skip('10/15 の ens_gate.py が無い')
    sys.path.insert(0, os.path.dirname(ENS_GATE))
    import ens_gate
    a = policies.build_policy('jetformer_v0', 6, SMALL_V0)
    b = policies.build_policy('jetformer_v0', 6, SMALL_V0)
    assert list(ens_gate.make_average_module([a, b]).state_dict()) == list(ens_average.from_members([a, b]).state_dict())


def test_export_passes_the_drive_contract(tmp_path):
    a = policies.build_policy('jetformer_v0', 6, SMALL_V0)
    b = policies.build_policy('jetformer_v0', 6, SMALL_V0)
    io = runtime.export_onnx(ens_average.from_members([a, b]), tmp_path / 'ens.onnx', 6)
    assert io['outputs'] == ['actions'] and io['output_shapes'] == {'actions': [1, 10, 2]}
    assert policies.check_io(io['input_shapes'], io['output_shapes'], 6) == (True, '')


# ---------------------------------------------------------------- model_config・走行タブ
def test_make_config_checks_members(tmp_path):
    runs = tmp_path / 'runs'
    _member_run(runs, 'v0', 'jetformer_v0', SMALL_V0)
    _member_run(runs, 'msf', 'msfusion', {'dim': 32})
    _member_run(runs, 'other_center', 'jetformer_v0', SMALL_V0, steer_center_us=1450.0)
    _member_run(runs, 'tcp', 'tcp', {}, postprocess={'kind': 'tcp_controller'})
    _member_run(runs, 'nm9', 'jetformer_v0', SMALL_V0, num_modes=9)
    cfg = ens_config.make_config(['v0', 'msf'], runs)
    assert cfg['policy'] == 'ens_average' and cfg['members'] == ['v0', 'msf'] and cfg['contract'] == 1
    assert [m['policy'] for m in cfg['policy_cfg']['members']] == ['jetformer_v0', 'msfusion']
    assert cfg['policy_cfg']['members'][1]['policy_cfg'] == policies.get('msfusion').resolve({'dim': 32})
    assert cfg['trained_modes'] == [4, 5] and cfg['steer_center_us'] == 1441.0 and cfg['throttle_neutral_us'] == 1496.0
    assert cfg['command_input'] is False and cfg['engine'] == {'precision': 'fp16', 'workspace_gb': 1.0}
    assert policies.drivable(cfg) == (True, '')
    assert policies.from_config(cfg)[0].name == 'ens_average'
    for bad, match in ((['v0'], '2 本以上'), (['v0', 'v0'], '2 回'), (['v0', 'other_center'], 'steer_center_us'),
                       (['v0', 'tcp'], '後処理'), (['v0', 'nm9'], 'num_modes')):
        with pytest.raises(ValueError, match=match):
            ens_config.make_config(bad, runs)
    (runs / 'ens').mkdir()
    (runs / 'ens' / 'model_config.json').write_text(json.dumps(dict(cfg)))
    with pytest.raises(ValueError, match='入れ子'):
        ens_config.make_config(['v0', 'ens'], runs)


def test_drive_model_info_drivable_and_engine_check(tmp_path, monkeypatch):
    runs = tmp_path / 'runs'
    _member_run(runs, 'v0', 'jetformer_v0', SMALL_V0)
    _member_run(runs, 'msf', 'msfusion', {})
    monkeypatch.setattr(drive, 'RUNS_DIR', runs)
    monkeypatch.setattr(drive, 'RECOMMENDED_PATH', tmp_path / 'drive_recommended.json')
    d = runs / 'ens'
    d.mkdir()
    (d / 'model_config.json').write_text(json.dumps(ens_config.make_config(['v0', 'msf'], runs)))
    assert drive.model_info('ens') is None                # エンジンが無ければ一覧に出ない(今まで通り)
    (d / drive.ENGINE_FILE).write_bytes(b'fake-engine')
    info = drive.model_info('ens')
    assert info['drivable'] is True and info['why_not'] == ''
    assert info['policy'] == 'ens_average' and info['policy_title'] == policies.get('ens_average').title
    assert info['trained_modes'] == [4, 5] and info['steer_center_us'] == 1441.0 and info['postprocess'] is None
    assert info['engine_check'] is None
    st = (d / drive.ENGINE_FILE).stat()
    chk = {'ok': True, 'accuracy_ok': True, 'speed_ok': True, 'actions_max_us': 0.9, 'gate_us': 2.5,
           'run_numpy_median_ms': 4.1, 'checked_at': '2026-10-21 12:00:00', 'note': '',
           'engine': {'size': st.st_size, 'mtime_ns': st.st_mtime_ns}}
    (d / drive.ENGINE_CHECK_FILE).write_text(json.dumps(chk))
    info = drive.model_info('ens')
    assert info['drivable'] is True and info['engine_check']['accuracy_ok'] is True
    (d / drive.ENGINE_CHECK_FILE).write_text(json.dumps(dict(chk, ok=False, accuracy_ok=False, actions_max_us=3.1)))
    assert drive.model_info('ens')['drivable'] is False           # 精度の確認に落ちたら走らせない(今まで通り)
    assert [m['name'] for m in drive.list_models()] == ['ens']


# ---------------------------------------------------------------- 置いた候補(あれば)
def test_placed_candidate_is_drivable():
    if not os.path.exists(os.path.join(ENS_RUN, drive.ENGINE_FILE)):
        pytest.skip('候補 ENS がまだ置かれていない(experiments/20261021_bench_prep/ens/place_ens.py all)')
    name = os.path.basename(ENS_RUN)
    info = drive.model_info(name)
    assert info is not None and info['drivable'] is True, info and info['why_not']
    assert info['policy'] == 'ens_average' and info['postprocess'] is None
    assert info['trained_modes'] == [4, 5] and info['command_input'] is False
    assert info['steer_center_us'] == 1441.0 and info['throttle_neutral_us'] == 1496.0
    ec = info['engine_check']
    assert ec is not None and ec['accuracy_ok'] is True and ec['actions_max_us'] <= 2.5
    cfg = json.load(open(os.path.join(ENS_RUN, 'model_config.json'), encoding='utf-8'))
    assert cfg['members'] == ['realacc_20261015-v0-s2', 'realacc_20261015-msf-s2']
    assert policies.check_io(cfg['input_shapes'], cfg['output_shapes'], cfg['num_modes']) == (True, '')
    rec = drive.recommended()
    assert rec is None or rec['run'] != name                      # 推奨のモデルは変えていない
