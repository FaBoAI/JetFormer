"""学習済みモデルの書き出し(エミュレーター用の zip)のテスト。小さな ONNX を作って使う(GPU は使わない)"""

import io
import json
import zipfile

import numpy as np
import pytest
import torch

from jetformer import model_export


class _Tiny(torch.nn.Module):
    def forward(self, image, tof, imu, state, mode):
        b = image.shape[0]
        return (image.mean() + tof.mean() + imu.mean() + state.mean() + mode.mean()).expand(b, 10, 2)


def _make_run(root, name, engine=True, pth=True, policy=None):
    d = root / name
    d.mkdir(parents=True)
    x = (torch.zeros(1, 3, 224, 224), torch.zeros(1, 2, 42, 54), torch.zeros(1, 10, 6), torch.zeros(1, 3), torch.zeros(1, 6))
    torch.onnx.export(_Tiny(), x, str(d / 'jetformer.onnx'), input_names=['image', 'tof', 'imu', 'state', 'mode'],
                      output_names=['actions'], opset_version=17, dynamo=False)
    cfg = {'num_modes': 6, 'future_steps': 10, 'trained_modes': [4, 5], 'command_input': False, 'dataset': 'ds',
           'steer_center_us': 1441.0, 'throttle_neutral_us': 1496.0,
           'modes': [{'id': 4, 'name': '高速走行'}, {'id': 5, 'name': '低速走行'}]}
    if policy:
        cfg['policy'] = policy
    (d / 'model_config.json').write_text(json.dumps(cfg), encoding='utf-8')
    (d / 'result.json').write_text(json.dumps({'best_val_us': 49.0, 'best_epoch': 25}), encoding='utf-8')
    (d / 'run.json').write_text(json.dumps({'created_at': '2026-10-12T10:00:00'}), encoding='utf-8')
    if engine:
        (d / 'jetformer_fp16.engine').write_bytes(b'engine')
    if pth:
        (d / 'best.pth').write_bytes(b'p' * 1000)
    return d


@pytest.fixture
def runs(tmp_path, monkeypatch):
    monkeypatch.setattr(model_export, 'RUNS_DIR', tmp_path)
    return tmp_path


def test_list_models(runs):
    _make_run(runs, 'model_b')
    _make_run(runs, 'rl_x', engine=False, pth=False, policy='act')
    (runs / 'no_onnx').mkdir()
    (runs / 'no_onnx' / 'model_config.json').write_text('{}', encoding='utf-8')
    _make_run(runs, '_smoke_hidden')
    ms = {m['name']: m for m in model_export.list_models()}
    assert set(ms) == {'model_b', 'rl_x'}
    assert ms['model_b']['policy'] == 'jetformer_v0' and ms['model_b']['drivable'] and ms['model_b']['has_pth']
    assert ms['rl_x']['policy'] == 'act' and not ms['rl_x']['drivable'] and not ms['rl_x']['has_pth']
    assert ms['model_b']['best_val_us'] == 49.0 and ms['model_b']['command_input'] is False


def test_io_spec_reads_onnx_and_config(runs):
    _make_run(runs, 'model_b')
    spec = model_export.io_spec('model_b')
    assert [i['name'] for i in spec['inputs']] == ['image', 'tof', 'imu', 'state', 'mode']
    assert spec['inputs'][0]['shape'] == [1, 3, 224, 224] and spec['outputs'][0]['shape'] == [1, 10, 2]
    state = next(i for i in spec['inputs'] if i['name'] == 'state')
    assert state['how']['command_input'] is False and '0' in state['how']['note']
    assert spec['outputs'][0]['how']['conventions']['steer_center_us'] == 1441.0


@pytest.mark.parametrize('pth', [False, True])
def test_bundle_zip(runs, pth):
    _make_run(runs, 'model_b')
    data = b''.join(model_export.iter_bundle_zip('model_b', pth=pth, chunk=1024))
    z = zipfile.ZipFile(io.BytesIO(data))
    assert z.testzip() is None
    names = set(z.namelist())
    want = {'model_b/jetformer.onnx', 'model_b/model_config.json', 'model_b/result.json', 'model_b/io_spec.json',
            'model_b/README.md', 'model_b/run_onnx_example.py'}
    assert want <= names
    assert ('model_b/best.pth' in names) == pth
    assert z.read('model_b/jetformer.onnx') == (runs / 'model_b' / 'jetformer.onnx').read_bytes()
    assert json.loads(z.read('model_b/io_spec.json'))['name'] == 'model_b'
    assert 'jetformer_fp16.engine' not in ' '.join(names)                 # Jetson 専用のエンジンは入れない


@pytest.mark.parametrize('bad', ['../x', 'a/b', '', '.hidden', 'missing'])
def test_bundle_rejects_bad_names(runs, bad):
    with pytest.raises(KeyError):
        model_export.iter_bundle_zip(bad)


def test_example_inputs_match_training_normalisation(runs):
    """zip に入れる動かす例(tools/emulator_run_onnx.py)の入力の作り方が、学習・走行と同じこと"""
    import importlib.util
    from jetformer.train_data import normalize_inputs
    spec = importlib.util.spec_from_file_location('emu', model_export.EXAMPLE)
    emu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(emu)
    rng = np.random.default_rng(0)
    m = emu.JetFormerOnnx.__new__(emu.JetFormerOnnx)          # onnxruntime 無しで、入力の作り方だけを確かめる
    m.cfg = {'command_input': False}
    m.center, m.scale, m.tof_max, m.erpm_scale, m.num_modes, m.use_command, m.size = 1500.0, 500.0, 4000.0, 20000.0, 6, False, 224
    m.input_names = ['image', 'tof', 'imu', 'state', 'mode']
    img = rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
    tof = rng.integers(0, 65535, (42, 54), dtype=np.uint16)
    tof[0, :5] = 0x1FFF | (3 << 13)
    imu = rng.normal(size=(10, 6)).astype(np.float32)
    veh = np.array([1600.0, 1400.0, 2000.0, -5000.0], np.float32)
    mine = m.inputs(img, tof, imu, veh, 5)
    ref = normalize_inputs(img, tof, imu, veh, use_command=False)
    for k, r in zip(('image', 'tof', 'imu', 'state'), ref):
        assert np.allclose(mine[k][0], r, atol=1e-6), k
    assert mine['mode'][0].tolist() == [0, 0, 0, 0, 0, 1]
    assert emu.apply_gains((1500.0, 1400.0), {'steer_center_us': 1441.0, 'throttle_neutral_us': 1496.0},
                           steer_gain=2.0, throttle_gain=1.5) == (1559.0, 1352.0)
