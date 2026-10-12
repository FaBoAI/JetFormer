"""方策(モデルの構造)の登録簿・共通の処理(jetformer/policies、tools/train.py・train_data.py・build_engine.py・training.py の
方策まわり)のテスト。GPU・実データ・ネットなしで通る(CPU だけ。実データのモデル B があれば一部を比べる)

    cd ~/jetformer && CUDA_VISIBLE_DEVICES= python -m pytest -q tests/test_policies.py
"""

import json
import os
import subprocess
import sys
import types

import cv2
import numpy as np
import pytest
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))
from jetformer import config, policies, training  # noqa: E402
from jetformer import model as jm  # noqa: E402
from jetformer.policies import runtime  # noqa: E402
from jetformer.train_data import GRAVITY, JetFormerDataset  # noqa: E402
import build_engine as be  # noqa: E402
import train  # noqa: E402

MODEL_B = os.path.join(ROOT, 'data', 'runs', 'clean_driving_20261011-nocmd-s3-20261011-024626')
SMALL_V0 = {'dim': 32, 'layers': 1, 'heads': 4}           # 速く書き出すための小さな v0(構造は同じ)


def _x(batch=2, num_modes=6, seed=7):
    g = torch.Generator().manual_seed(seed)
    return dict(image=torch.randn(batch, 3, 224, 224, generator=g), tof=torch.rand(batch, 2, 42, 54, generator=g),
                imu=torch.randn(batch, 10, 6, generator=g) * 0.1, state=torch.randn(batch, 3, generator=g) * 0.1,
                mode=torch.nn.functional.one_hot(torch.tensor([4, 5][:batch]), num_modes).float())


# ---------------------------------------------------------------- (1) torch を読まない
def test_registry_and_drive_do_not_import_torch():
    code = ('import sys; import jetformer.policies, jetformer.policies.post, jetformer.policies.ensemble, jetformer.drive, '
            'jetformer.training, jetformer.model_export; '
            'jetformer.policies.list_specs(); print("torch" in sys.modules)')
    out = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True, timeout=60,
                         env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == 'False'


# ---------------------------------------------------------------- (2)(3) 登録簿・設定
def test_old_config_is_v0_with_frozen_defaults():
    spec, cfg = policies.from_config({'num_modes': 6})
    assert spec.name == 'jetformer_v0' and cfg == {'dim': 256, 'layers': 4, 'heads': 8}
    assert policies.drivable({'num_modes': 6}) == (True, '')
    assert policies.title_of('no_such_policy') == 'no_such_policy'
    assert policies.title_of(None) == policies.get('jetformer_v0').title
    assert policies.inputs_of({}) == list(policies.BASE_INPUTS) and policies.outputs_of({}) == ['actions']
    assert policies.drivable({'inputs': ['image', 'history']})[0] is False
    assert policies.drivable({'outputs': ['waypoints']})[0] is False
    assert policies.drivable({'contract': 99})[0] is False
    assert [s['name'] for s in policies.list_specs()] == ['jetformer_v0', 'act', 'tcp', 'msfusion', 'dinov2']


def test_registry_entries_follow_spec():
    tcp = policies.get('tcp')
    assert tcp.outputs == ('actions', 'waypoints', 'speed_logits') and tcp.postprocess == 'tcp_controller'
    assert tcp.genre_keys['heads'] == {'heads': 2, 'heads_wp': 2, 'heads_v': 10}
    assert policies.get('act').genre_keys['heads'] == {'heads': 2}
    assert policies.get('dinov2').requires_pretrained and policies.get('dinov2').requires
    for name in policies.names():
        s = policies.get(name)
        assert s.genre_keys['embed'] == 'mode_embed.weight'
        assert {'train_rate', 'val_rate', 'min_free_mb', 'amp', 'engine_precision', 'infer_ms', 'params_m'} <= set(s.hints)
    assert policies.get('jetformer_v0').slug == 'v0' and policies.get('msfusion').slug == 'msf'


def test_resolve_rules():
    act = policies.get('act')
    full = act.resolve(None)
    assert full == act.defaults and full is not act.defaults
    assert act.resolve({}) == act.defaults
    r = act.resolve({'dec_layers': '3', 'cvae': 'true', 'kl_weight': 1})        # 型を既定値に合わせる
    assert r['dec_layers'] == 3 and r['cvae'] is True and r['kl_weight'] == 1.0 and set(r) == set(act.defaults)
    with pytest.raises(ValueError, match='知らない設定'):
        act.resolve({'nope': 1})
    with pytest.raises(ValueError):
        act.resolve({'dec_layers': 5})                       # options の範囲外
    with pytest.raises(ValueError):
        act.resolve({'kl_weight': 3.0})                      # choices の外
    with pytest.raises(ValueError):
        act.resolve({'dec_layers': 2.5})                     # 整数に直せない
    with pytest.raises(ValueError, match='管理画面'):
        act.resolve({'ffn': 1024}, ui=True)                  # options に無いキーは UI から変えられない
    assert act.resolve({'ffn': 1024})['ffn'] == 1024        # CLI は変えられる
    assert act.resolve({'ffn': 512}, ui=True)['ffn'] == 512  # 既定値のままなら UI からでも良い
    msf = policies.get('msfusion')
    assert msf.resolve({'scales': [2, 3, 4]}, ui=True)['scales'] == [2, 3, 4]
    with pytest.raises(ValueError):
        msf.resolve({'scales': [1, 4]}, ui=True)
    assert act.changed(act.resolve({'dec_layers': 3})) == {'dec_layers': 3}


def test_stub_modules_are_not_ready_and_v0_is():
    assert policies.get('jetformer_v0').ready() == (True, '')
    for name in ('act', 'tcp', 'msfusion', 'dinov2'):
        ok, why = policies.get(name).ready()
        mod_file = os.path.join(ROOT, 'jetformer', 'policies', name + '.py')
        with open(mod_file, encoding='utf-8') as f:
            stub = policies.STUB_MARK in f.read()
        if stub:
            assert ok is False and 'まだ実装されていません' in why
    missing = policies.PolicySpec('x', 'X', 'jetformer.policies.no_such_module', 'x', '')
    assert missing.ready()[0] is False
    need = policies.PolicySpec('y', 'Y', 'jetformer.policies.jetformer_v0', 'y', '', requires=('data/no/such/file.pth',))
    ok, why = need.ready()
    assert ok is False and 'data/no/such/file.pth' in why


# ---------------------------------------------------------------- (4) v0 は JetFormer をそのまま
def test_v0_builds_plain_jetformer_with_same_keys():
    m = policies.build_policy('jetformer_v0', 6)
    assert type(m) is jm.JetFormer
    assert list(m.state_dict()) == list(jm.JetFormer(num_modes=6).state_dict())


def test_v0_outputs_match_reference_before_the_change():
    """変更の前のコードで作った出力(tests/fixtures/v0_reference.npz)と、登録簿から作った v0 の出力が同じ(同じ乱数の種)"""
    ref = np.load(os.path.join(ROOT, 'tests', 'fixtures', 'v0_reference.npz'))
    if str(ref['torch_version']) != torch.__version__:
        pytest.skip('参照を作った torch の版と違う(%s)' % ref['torch_version'])
    torch.manual_seed(1234)
    m = policies.build_policy('jetformer_v0', 6).eval()
    sd = m.state_dict()
    assert sorted(sd) == list(ref['keys'])
    assert np.allclose([float(sd[k].double().sum()) for k in sorted(sd)], ref['param_sums'], rtol=0, atol=1e-9)
    out = runtime.predict(m, _x())['actions']
    assert np.allclose(out, ref['actions'], rtol=0, atol=1e-6)
    with torch.no_grad():
        direct = m(**_x()).numpy()
    assert np.array_equal(out, direct)


# ---------------------------------------------------------------- (5)(6)(7) 書き出しと入出力の確認
def test_v0_export_has_the_contract_io(tmp_path):
    m = policies.build_policy('jetformer_v0', 6, SMALL_V0)
    io = runtime.export_onnx(m, tmp_path / 'jetformer.onnx', 6)
    assert io['inputs'] == ['image', 'tof', 'imu', 'state', 'mode']
    assert io['input_shapes'] == {'image': [1, 3, 224, 224], 'tof': [1, 2, 42, 54], 'imu': [1, 10, 6], 'state': [1, 3],
                                  'mode': [1, 6]}
    assert io['outputs'] == ['actions'] and io['output_shapes'] == {'actions': [1, 10, 2]}
    assert not (tmp_path / 'jetformer.onnx.tmp').exists()
    b = os.path.join(MODEL_B, 'jetformer.onnx')
    if os.path.exists(b):
        ib = runtime.read_onnx_io(b)
        assert (ib['inputs'], ib['input_shapes'], ib['outputs'], ib['output_shapes']) == \
            (io['inputs'], io['input_shapes'], io['outputs'], io['output_shapes'])


class _StateOnly(torch.nn.Module):
    """state と mode しか使わない小さな方策(入力がグラフから消えることを確かめる)。追加の出力も持つ"""
    EXPORT_OUTPUTS = ('actions', 'extra')

    def __init__(self, num_modes):
        super().__init__()
        self.lin = torch.nn.Linear(3 + num_modes, 20)

    def forward(self, image, tof, imu, state, mode):
        a = self.lin(torch.cat([state, mode], dim=1)).view(-1, 10, 2)
        return {'actions': a, 'extra': a.sum(dim=2)}


def test_unused_inputs_disappear_and_still_pass(monkeypatch, tmp_path):
    mod = types.ModuleType('jetformer.policies._state_only')
    mod.build = lambda num_modes, pretrained=False, **cfg: _StateOnly(num_modes)
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    spec = policies.PolicySpec('_state_only', '試験', mod.__name__, 'so', '', defaults={'x': 1}, outputs=('actions', 'extra'))
    monkeypatch.setitem(policies.REGISTRY, '_state_only', spec)
    m = policies.build_policy('_state_only', 6)
    runtime.check_forward(m, 6)
    io = runtime.export_onnx(m, tmp_path / 'm.onnx', 6)
    assert io['inputs'] == ['state', 'mode'] and io['outputs'] == ['actions', 'extra']
    assert io['output_shapes']['extra'] == [1, 10]
    assert policies.check_io(io['input_shapes'], io['output_shapes'], 6) == (True, '')
    assert policies.drivable({'inputs': io['inputs'], 'outputs': io['outputs']}) == (True, '')
    out = runtime.predict(m, {k: v.numpy() for k, v in _x(1).items()})
    assert set(out) == {'actions', 'extra'} and out['actions'].shape == (1, 10, 2)


def test_check_io_failures():
    ok_in = {'image': [1, 3, 224, 224], 'tof': [1, 2, 42, 54], 'imu': [1, 10, 6], 'state': [1, 3], 'mode': [1, 6]}
    ok_out = {'actions': [1, 10, 2]}
    assert policies.check_io(ok_in, ok_out, 6) == (True, '')
    assert policies.check_io(dict(ok_in, history=[1, 3, 3, 224, 224]), ok_out, 6)[0] is False      # 知らない入力
    assert policies.check_io(dict(ok_in, image=[1, 3, 160, 160]), ok_out, 6)[0] is False           # 形が違う
    assert policies.check_io(dict(ok_in, mode=[1, 8]), ok_out, 6)[0] is False                      # ジャンル数が違う
    assert policies.check_io(ok_in, {'waypoints': [1, 10, 2]}, 6)[0] is False                      # actions が無い
    assert policies.check_io(ok_in, {'actions': [1, 5, 2]}, 6)[0] is False
    assert policies.check_io(dict(ok_in, image=[-1, 3, 224, 224]), ok_out, 6)[0] is False         # 固定でない
    assert policies.check_io(ok_in, {'actions': [1, 10, 2], 'speed': [1, None]}, 6)[0] is False
    assert policies.check_io({'mode': (1, 6)}, {'actions': (1, 10, 2)}, None) == (True, '')         # タプル・ジャンル数を見ない


def test_check_forward_rejects_bad_policies():
    class Bad(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(1))

        def forward(self, image, tof, imu, state, mode):
            return state[:, :1, None].expand(-1, 5, 2) + self.p
    with pytest.raises(RuntimeError, match='形'):
        runtime.check_forward(Bad(), 6)
    m = _StateOnly(6)
    m.EXPORT_OUTPUTS = ('actions', 'missing')
    with pytest.raises(RuntimeError, match='missing'):
        runtime.check_forward(m, 6)


# ---------------------------------------------------------------- (8) run_epoch
class _Fixed(torch.nn.Module):
    """state[:,0] をそのまま全出力にする(L1 を決めて試す)。policy_loss = 2·L1 + 1(内訳 kl = 0.5)"""

    def __init__(self, with_loss=False):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))
        if with_loss:
            self.policy_loss = self._loss
        self.seen_extras = []

    def forward(self, image, tof, imu, state, mode):
        return {'actions': state[:, :1, None].expand(-1, 10, 2) + 0 * self.p, 'aux': state[:, :2]}

    def _loss(self, out, target, extras):
        self.seen_extras.append(extras)
        self.loss_parts = {'kl': torch.tensor(0.5)}
        return 2 * runtime.l1_per_sample(out['actions'], target) + 1 + 0 * self.p


def _batch(errs, weights=None, extras=False):
    b = len(errs)
    out = [torch.zeros(b, 3, 2, 2), torch.zeros(b, 2, 2, 2), torch.zeros(b, 10, 6),
           torch.tensor([[e, 0, 0] for e in errs], dtype=torch.float32), jm.mode_onehot([5] * b, 6), torch.zeros(b, 10, 2)]
    if weights is not None:
        out.append(torch.tensor(weights, dtype=torch.float32))
    if extras:
        out.append({'y_v_tgt': torch.arange(b, dtype=torch.float32)})
    return out


@pytest.mark.parametrize('weights,extras', [(False, False), (True, False), (True, True), (False, True)])
def test_run_epoch_batch_layouts(weights, extras):
    errs = [0.1, 0.2, 0.3, 0.4]
    w = [1.0, 3.0, 0.0, 4.0] if weights else None
    batches = [_batch(errs[:2], w[:2] if w else None, extras), _batch(errs[2:], w[2:] if w else None, extras)]
    st = {}
    mean, per = train.run_epoch(_Fixed(), batches, torch.device('cpu'), amp=False, stats=st)
    expect = sum(e * x for e, x in zip(errs, w)) / sum(w) if w else 0.25
    assert mean == pytest.approx(expect) and st['l1'] == pytest.approx(0.25) and per[5] == pytest.approx(0.25)
    m = _Fixed(with_loss=True)                     # 学習は方策の損失、報告は L1
    opt = torch.optim.SGD(m.parameters(), lr=0.0)
    st = {}
    mean, _ = train.run_epoch(m, batches, torch.device('cpu'), optimizer=opt, amp=False, stats=st)
    assert mean == pytest.approx(expect) and st['l1'] == pytest.approx(0.25)
    if not w:
        assert st['loss'] == pytest.approx(2 * 0.25 + 1)
    assert st['parts'] == {'kl': pytest.approx(0.5)}
    assert all((e is not None) == extras for e in m.seen_extras)
    if extras:
        assert torch.equal(m.seen_extras[0]['y_v_tgt'], torch.arange(2, dtype=torch.float32))


def test_run_epoch_v0_numbers_unchanged():
    """v0 型(Tensor を返す)の学習の数字が今までの式と同じ: mean(L1) と重みつきの Σw·L1/Σw"""
    class V0Like(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(3, 20)

        def forward(self, image, tof, imu, state, mode):
            return self.lin(state).view(-1, 10, 2)
    torch.manual_seed(0)
    m = V0Like()
    g = torch.Generator().manual_seed(3)
    batches = []
    for _ in range(3):
        b = _batch([0.0] * 4)
        b[3] = torch.randn(4, 3, generator=g)
        b[5] = torch.randn(4, 10, 2, generator=g) * 0.1
        batches.append(b)
    with torch.no_grad():
        ref = torch.cat([(m(*b[:5]) - b[5]).abs().mean(dim=(1, 2)) for b in batches])
    st = {}
    mean, _ = train.run_epoch(m, batches, torch.device('cpu'), amp=False, stats=st)
    assert mean == pytest.approx(float(ref.mean()), rel=1e-6)
    assert 'loss' not in st and 'parts' not in st


# ---------------------------------------------------------------- (9)(10) 再開の引数・--init の一致
def test_policy_args_recorded_only_when_not_default():
    args = types.SimpleNamespace(policy='jetformer_v0', policy_cfg=None)
    assert train.policy_args_used(args) == {}
    args.policy_cfg = '{}'
    assert train.policy_args_used(args) == {}
    args.policy, args.policy_cfg = 'act', '{"dec_layers": 3, "cvae": false}'
    assert train.policy_args_used(args) == {'policy': 'act', 'policy_cfg': '{"cvae": false, "dec_layers": 3}'}
    old = {'data': 'd', 'epochs': 30}                 # 前からの last.pth(方策の記録なし)は v0 と同じ
    assert train.args_diff(old, dict(old, **train.policy_args_used(types.SimpleNamespace(policy='jetformer_v0',
                                                                                         policy_cfg=None)))) == {}
    assert 'policy' in train.args_diff(old, dict(old, policy='act'))


def test_check_init_compat():
    act = policies.get('act')
    cfg = act.resolve({'dec_layers': 2})
    train.check_init_compat(None, 'act', cfg, act)                                    # 設定が無ければ確かめない
    train.check_init_compat({'policy': 'act', 'policy_cfg': dict(cfg, kl_weight=1.0)}, 'act', cfg, act)   # 学習だけの設定
    with pytest.raises(SystemExit):
        train.check_init_compat({'policy': 'act', 'policy_cfg': dict(cfg, dec_layers=3)}, 'act', cfg, act)
    with pytest.raises(SystemExit):
        train.check_init_compat({'num_modes': 6}, 'act', cfg, act)                    # 古い学習結果 = v0
    v0 = policies.get('jetformer_v0')
    train.check_init_compat({'num_modes': 6}, 'jetformer_v0', v0.resolve(None), v0)


# ---------------------------------------------------------------- (11) ジャンルの拡張
def test_expand_modes_generic_and_v0_equivalence():
    def state(n, rows_a=2, rows_b=10, dim=4, seed=0):
        g = torch.Generator().manual_seed(seed)
        return {'mode_embed.weight': torch.randn(dim, n, generator=g),
                'heads.weight': torch.randn(n * rows_a, dim, generator=g), 'heads.bias': torch.randn(n * rows_a, generator=g),
                'heads_v.weight': torch.randn(n * rows_b, dim, generator=g), 'heads_v.bias': torch.randn(n * rows_b, generator=g),
                'other': torch.randn(3, generator=g)}
    keys = {'embed': 'mode_embed.weight', 'heads': {'heads': 2, 'heads_v': 10}}
    old, new = state(6), state(8, seed=1)
    out = runtime.expand_modes(old, new, keys, {6: 5, 7: 4})
    assert torch.equal(out['heads_v.weight'].view(8, 10, -1)[:6], old['heads_v.weight'].view(6, 10, -1))
    assert torch.equal(out['heads_v.weight'].view(8, 10, -1)[6], old['heads_v.weight'].view(6, 10, -1)[5])
    assert torch.equal(out['heads.bias'].view(8, 2)[7], old['heads.bias'].view(6, 2)[4])
    assert torch.equal(out['mode_embed.weight'][:, 6], old['mode_embed.weight'][:, 5])
    assert torch.equal(out['other'], old['other'])
    assert runtime.expand_modes(old, old, keys) is old
    with pytest.raises(ValueError):
        runtime.expand_modes(new, old, keys)
    torch.manual_seed(0)
    a, b = jm.JetFormer(num_modes=6, dim=32, layers=1, heads=4), jm.JetFormer(num_modes=8, dim=32, layers=1, heads=4)
    x = train.expand_mode_state(a.state_dict(), b.state_dict(), {6: 5, 7: 4})
    y = runtime.expand_modes(a.state_dict(), b.state_dict(), policies.get('jetformer_v0').genre_keys, {6: 5, 7: 4})
    assert all(torch.equal(x[k], y[k]) for k in x)


# ---------------------------------------------------------------- (12) 最適化のパラメータ
def test_param_groups_skip_frozen_and_v0_keeps_order():
    m = jm.JetFormer(num_modes=6, dim=32, layers=1, heads=4)
    assert runtime.param_groups(m, 1e-3) == list(m.parameters())
    for p in m.image_backbone.parameters():
        p.requires_grad = False
    got = runtime.param_groups(m, 1e-3)
    assert len(got) == len([p for p in m.parameters() if p.requires_grad]) and all(p.requires_grad for p in got)

    class WithGroups(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.b = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)

        def param_groups(self, lr):
            return [{'params': self.a.parameters()}, {'params': self.b.parameters(), 'lr': lr * 0.1}]
    assert runtime.param_groups(WithGroups(), 1.0)[1]['lr'] == pytest.approx(0.1)


# ---------------------------------------------------------------- (13) 学習結果の読み込み(古い設定 = v0)
def test_load_state_and_load_run_old_config(tmp_path):
    torch.manual_seed(5)
    m = jm.JetFormer(num_modes=6).eval()
    torch.save(m.state_dict(), tmp_path / 'best.pth')
    torch.save({'model': m.state_dict(), 'optimizer': {'state': {}, 'param_groups': []}, 'epoch': 3}, tmp_path / 'last.pth')
    for ck in ('best', 'last'):
        sd = runtime.load_state(tmp_path / ('%s.pth' % ck))
        assert all(torch.equal(sd[k], v) for k, v in m.state_dict().items())
    (tmp_path / 'model_config.json').write_text(json.dumps({'num_modes': 6, 'trained_modes': [4, 5]}))   # 古い最小の形
    loaded, cfg, spec, pcfg = runtime.load_run(tmp_path, 'best')
    assert type(loaded) is jm.JetFormer and spec.name == 'jetformer_v0' and pcfg == policies.V0_FROZEN_CFG
    assert not loaded.training
    with torch.no_grad():
        ref = m(**_x()).numpy()
    assert np.array_equal(runtime.predict(loaded, _x())['actions'], ref)
    loaded_last, _, _, _ = runtime.load_run(tmp_path, 'last')
    assert np.array_equal(runtime.predict(loaded_last, _x())['actions'], ref)


# ---------------------------------------------------------------- (14) 追加の正解
MODES = [{'id': i, 'name': n, 'description': ''} for i, n in enumerate(['未設定', '駐車(1)', '駐車(2)', '駐車(3)', '高速走行', '低速走行'])]


def _make_dataset(root, n=40, scenes=('s1', 's2', 's3')):
    (root / 'aligned').mkdir(parents=True)
    cv2.imwrite(str(root / 'img.jpg'), np.full((16, 16, 3), 128, np.uint8))
    (root / 'dataset_card.json').write_text(json.dumps({'modes': MODES}, ensure_ascii=False), encoding='utf-8')
    (root / 'splits.json').write_text(json.dumps({'train': list(scenes[:2]), 'val': list(scenes[2:])}))
    rng = np.random.default_rng(0)
    for k, s in enumerate(scenes):
        imu = np.zeros((n, 10, 6), np.float32)
        imu[:, :, 2] = GRAVITY
        fv = np.ones(n, bool)
        fv[-10:] = False
        np.savez(root / 'aligned' / (s + '.npz'), timestamps_us=(np.arange(n) * 1e6 / 30).astype(np.int64),
                 cam_files=np.array(['img.jpg'] * n), tof=np.zeros((n, 42, 54), np.uint16), imu=imu,
                 vehicle=np.tile(np.array([1500, 1500, 1000, 0], np.float32), (n, 1)),
                 future_actions=(1500 + rng.normal(0, 50, (n, 10, 2))).astype(np.float32), future_valid=fv,
                 vehicle_valid=np.ones(n, bool), mode=np.full(n, 4 + k % 2, np.int16))
    return str(root)


def test_dataset_extra_targets(tmp_path):
    root = _make_dataset(tmp_path / 'ds')
    plain = JetFormerDataset(root, 'train')
    assert len(plain[0]) == 6
    extra = {s: {'v_tgt': np.arange(40, dtype=np.float32) + k, 'waypoints': np.zeros((40, 10, 2), np.float32)}
             for k, s in enumerate(['s1', 's2', 's3'])}
    ds = JetFormerDataset(root, 'train', extra_targets=extra)
    item = ds[3]
    assert len(item) == 7 and isinstance(item[-1], dict) and set(item[-1]) == {'y_v_tgt', 'y_waypoints'}
    k, i = ds.index[3]
    assert float(item[-1]['y_v_tgt']) == float(i + k) and item[-1]['y_waypoints'].shape == (10, 2)
    batch = next(iter(torch.utils.data.DataLoader(ds, 4)))
    assert batch[-1]['y_waypoints'].shape == (4, 10, 2)
    with pytest.raises(KeyError, match='追加の正解'):
        JetFormerDataset(root, 'val', extra_targets={'s1': extra['s1']})
    with pytest.raises(ValueError):
        JetFormerDataset(root, 'train', extra_targets={s: {'v_tgt': np.zeros(5)} for s in extra})


# ---------------------------------------------------------------- (15) 管理画面の学習のコマンド
class _FakePopen(object):
    calls = []

    def __init__(self, cmd, **kw):
        _FakePopen.calls.append(cmd)
        self.cmd, self.pid, self.stdout = cmd, 12345, iter([])

    def poll(self):
        return 0

    def wait(self):
        return 0


@pytest.fixture
def trainer(monkeypatch, tmp_path):
    for ds in ('dsx', 'dsy'):                        # 同じ秒に2回始めても名前が重ならないように2つ
        root = tmp_path / 'datasets' / ds
        root.mkdir(parents=True)
        (root / 'dataset_card.json').write_text('{}')
    monkeypatch.setattr(config, 'DATASET_DIR', tmp_path / 'datasets')
    monkeypatch.setattr(training, 'RUNS_DIR', tmp_path / 'runs')
    monkeypatch.setattr(training, 'mem_available_mb', lambda: 8000)
    monkeypatch.setattr(training.subprocess, 'Popen', _FakePopen)
    _FakePopen.calls = []
    return training.TrainingManager()


def _wait_idle(t):
    for _ in range(100):
        if not t.running:
            return
        import time
        time.sleep(0.02)


def test_training_command_unchanged_for_v0(trainer, tmp_path):
    name = trainer.start('dsx')
    _wait_idle(trainer)
    out = tmp_path / 'runs' / name
    assert name.startswith('dsx-s3-')
    assert _FakePopen.calls[-1] == [sys.executable, str(training.TRAIN_SCRIPT), '--data', str(config.DATASET_DIR / 'dsx'),
                                    '--out', str(out), '--stride', '3', '--epochs', '30', '--batch', '64', '--lr', '0.0003',
                                    '--workers', '2', '--progress-json']
    run = json.loads((out / 'run.json').read_text(encoding='utf-8'))
    assert run['policy'] == 'jetformer_v0' and run['policy_cfg'] == policies.V0_FROZEN_CFG
    assert run['impact_g'] is None and run['seed'] is None
    runs = trainer.list_runs()
    assert runs[0]['params']['policy'] == 'jetformer_v0' and runs[0]['policy_title'] == policies.title_of('jetformer_v0')


def test_training_command_for_other_policies(trainer, monkeypatch, tmp_path):
    monkeypatch.setattr(policies.get('act'), 'ready', lambda: (True, ''))
    name = trainer.start('dsx', policy='act', policy_cfg={'dec_layers': 3}, impact_g=2.5, seed=0)
    _wait_idle(trainer)
    cmd = _FakePopen.calls[-1]
    assert '-act-s3-' in name
    assert cmd[cmd.index('--policy') + 1] == 'act'
    assert json.loads(cmd[cmd.index('--policy-cfg') + 1]) == {'dec_layers': 3}
    assert cmd[cmd.index('--impact-g') + 1] == '2.5' and cmd[cmd.index('--seed') + 1] == '0'
    run = json.loads((tmp_path / 'runs' / name / 'run.json').read_text(encoding='utf-8'))
    assert run['policy_cfg']['dec_layers'] == 3 and run['policy_cfg']['enc_layers'] == 4
    trainer.start('dsy', policy='act')                          # 既定の形なら --policy-cfg は付けない
    _wait_idle(trainer)
    assert '--policy-cfg' not in _FakePopen.calls[-1] and '--policy' in _FakePopen.calls[-1]


def test_training_rejects_bad_policy_requests(trainer, monkeypatch):
    with pytest.raises(ValueError):
        trainer.start('dsx', policy='no_such_policy')
    monkeypatch.setattr(policies.get('act'), 'ready', lambda: (False, '準備ができていません'))
    with pytest.raises(ValueError, match='準備'):
        trainer.start('dsx', policy='act')
    monkeypatch.setattr(policies.get('act'), 'ready', lambda: (True, ''))
    with pytest.raises(ValueError, match='管理画面'):
        trainer.start('dsx', policy='act', policy_cfg={'ffn': 1024})
    monkeypatch.setattr(policies.get('dinov2'), 'ready', lambda: (True, ''))
    with pytest.raises(ValueError, match='事前学習'):
        trainer.start('dsx', policy='dinov2', pretrained=False)
    with pytest.raises(ValueError):
        trainer.start('dsx', impact_g=100)
    with pytest.raises(ValueError):
        trainer.start('dsx', seed=-1)
    assert _FakePopen.calls == []


def test_training_needs_more_memory_for_heavy_policies(trainer, monkeypatch):
    monkeypatch.setattr(training, 'mem_available_mb', lambda: 3600)
    spec = policies.get('jetformer_v0')
    monkeypatch.setitem(spec.hints, 'min_free_mb', 4000)
    with pytest.raises(RuntimeError, match='空きメモリ'):
        trainer.start('dsx')


# ---------------------------------------------------------------- (16) エンジンの作成の設定
def test_build_engine_settings():
    assert be.engine_settings(None) == ('fp16', 1.0)
    assert be.engine_settings({'engine': {'precision': 'BF16', 'workspace_gb': 2}}) == ('bf16', 2.0)
    assert be.engine_settings({'engine': {'precision': 'fp32'}}) == ('fp32', 1.0)
    with pytest.raises(ValueError):
        be.engine_settings({'engine': {'precision': 'int8'}})
    assert be.precision_flags('fp16') == {'fp16': True, 'bf16': False}
    assert be.precision_flags('bf16') == {'fp16': False, 'bf16': True}
    assert be.precision_flags('fp32') == {'fp16': False, 'bf16': False}
    good = ({'image': [1, 3, 224, 224], 'mode': [1, 6]}, {'actions': [1, 10, 2], 'waypoints': [1, 10, 2]})
    assert be.check_engine_io(good, {'num_modes': 6}) == (True, '')
    assert be.check_engine_io(good, None) == (True, '')                      # ジャンル数はエンジンの形から
    assert be.check_engine_io(good, {'num_modes': 8})[0] is False
    assert be.check_engine_io(({'history': [1, 3]}, {'actions': [1, 10, 2]}), None)[0] is False


def test_build_engine_mixed_precision_option(monkeypatch, tmp_path):
    """engine.precision = fp16_norm_fp32(FP16 + 正規化・Softmax だけ FP32)の読み方。ほかの精度の引数は今までと同じ
    (fp32_layers を渡さない)。登録簿の既定は dinov2 だけがこの精度、ほかは fp16 のまま(偽の TensorRT)"""
    assert be.engine_settings({'engine': {'precision': 'FP16_norm_fp32'}}) == ('fp16_norm_fp32', 1.0)
    assert be.precision_flags('fp16_norm_fp32') == {'fp16': True, 'bf16': False, 'fp32_layers': ('norm', 'softmax')}
    for p in ('fp16', 'bf16', 'fp32'):
        assert 'fp32_layers' not in be.precision_flags(p)
    assert 'fp16_norm_fp32' in policies.ENGINE_PRECISIONS
    assert policies.get('dinov2').hints['engine_precision'] == 'fp16_norm_fp32'
    for name in ('jetformer_v0', 'act', 'tcp', 'msfusion'):
        assert policies.get(name).hints['engine_precision'] == 'fp16'
    calls = {}
    fake = types.ModuleType('jetformer.trt_runtime')

    def build(onnx_path, engine_path, fp16=True, workspace_gb=1.0, bf16=False, fp32_layers=()):
        calls.update(fp16=fp16, bf16=bf16, fp32_layers=tuple(fp32_layers))
        with open(engine_path, 'wb') as f:
            f.write(b'engine')
    fake.build_engine = build
    fake.engine_io = lambda path: ({'image': [1, 3, 224, 224], 'mode': [1, 6]}, {'actions': [1, 10, 2]})
    monkeypatch.setitem(sys.modules, 'jetformer.trt_runtime', fake)
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'jetformer.onnx').write_bytes(b'onnx')
    (run / 'model_config.json').write_text(json.dumps({'num_modes': 6, 'engine': {'precision': 'fp16_norm_fp32'}}))
    engine = str(run / 'jetformer_fp16.engine')
    assert be.main(['build_engine.py', str(run / 'jetformer.onnx'), engine]) == 0
    assert calls == {'fp16': True, 'bf16': False, 'fp32_layers': ('norm', 'softmax')}
    (run / 'model_config.json').write_text(json.dumps({'num_modes': 6, 'engine': {'precision': 'fp16'}}))
    calls.clear()
    assert be.main(['build_engine.py', str(run / 'jetformer.onnx'), engine]) == 0
    assert calls == {'fp16': True, 'bf16': False, 'fp32_layers': ()}       # 既定の FP16 は今までと同じ呼び方


# ---------------------------------------------------------------- 学習の通し(CPU・合成データ・v0 の小さな形)
def test_train_main_end_to_end_v0(tmp_path, monkeypatch):
    root = _make_dataset(tmp_path / 'ds')
    out = tmp_path / 'run'
    argv = ['train.py', '--data', root, '--out', str(out), '--epochs', '1', '--batch', '4', '--workers', '0',
            '--max-steps', '1', '--stride', '5', '--no-pretrained', '--no-command-input', '--impact-g', '2.5',
            '--policy-cfg', json.dumps(SMALL_V0), '--seed', '0']
    monkeypatch.setattr(sys, 'argv', argv)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    train.main()
    cfg = json.loads((out / 'model_config.json').read_text(encoding='utf-8'))
    res = json.loads((out / 'result.json').read_text(encoding='utf-8'))
    assert cfg['policy'] == 'jetformer_v0' and cfg['policy_cfg'] == SMALL_V0 and cfg['contract'] == 1
    assert cfg['inputs'] == list(policies.BASE_INPUTS) and cfg['outputs'] == ['actions']
    assert cfg['output_shapes'] == {'actions': [1, 10, 2]} and cfg['engine'] == {'precision': 'fp16', 'workspace_gb': 1.0}
    for k in ('num_modes', 'modes', 'future_steps', 'tof_channels', 'pwm_center', 'pwm_scale', 'tof_max_mm', 'erpm_scale',
              'dataset', 'stride', 'trained_modes', 'train_mode_counts', 'steer_center_us', 'throttle_neutral_us',
              'command_input', 'mask_impacts'):
        assert k in cfg                                     # 今までの欄はそのまま
    assert res['policy'] == 'jetformer_v0' and res['seed'] == 0 and res['peak_mem_mb'] is None
    assert 'train_loss' not in res['history'][0]           # v0 の history は今までと同じ
    ck = torch.load(out / 'last.pth', weights_only=False)
    assert ck['args']['policy_cfg'] == json.dumps(SMALL_V0, sort_keys=True) and 'policy' not in ck['args']
    model, _, spec, pcfg = runtime.load_run(out)
    assert spec.name == 'jetformer_v0' and pcfg == SMALL_V0 and runtime.count_params(model)[0] > 0


def test_build_engine_main_replaces_only_checked_engines(monkeypatch, tmp_path):
    """エンジンは tmp に作り、入出力が合えば置き換え、合わなければ tmp を消して終了コード 2(偽の TensorRT)"""
    calls = {}
    io = {'value': ({'image': [1, 3, 224, 224], 'mode': [1, 6]}, {'actions': [1, 10, 2]})}
    fake = types.ModuleType('jetformer.trt_runtime')

    def build(onnx_path, engine_path, fp16=True, workspace_gb=1.0, bf16=False):
        calls.update(onnx=onnx_path, engine=engine_path, fp16=fp16, bf16=bf16, ws=workspace_gb)
        with open(engine_path, 'wb') as f:
            f.write(b'engine')
    fake.build_engine = build
    fake.engine_io = lambda path: io['value']
    monkeypatch.setitem(sys.modules, 'jetformer.trt_runtime', fake)
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'jetformer.onnx').write_bytes(b'onnx')
    (run / 'model_config.json').write_text(json.dumps({'num_modes': 6, 'engine': {'precision': 'bf16', 'workspace_gb': 2}}))
    engine = str(run / 'jetformer_fp16.engine')
    assert be.main(['build_engine.py', str(run / 'jetformer.onnx'), engine]) == 0
    assert calls['engine'] == engine + '.tmp' and calls['bf16'] is True and calls['fp16'] is False and calls['ws'] == 2.0
    assert open(engine, 'rb').read() == b'engine' and not os.path.exists(engine + '.tmp')
    os.remove(engine)
    io['value'] = ({'history': [1, 3]}, {'actions': [1, 10, 2]})
    assert be.main(['build_engine.py', str(run / 'jetformer.onnx'), engine]) == be.EXIT_BAD_IO
    assert not os.path.exists(engine) and not os.path.exists(engine + '.tmp')
    (run / 'model_config.json').unlink()                     # 設定が無ければ今まで通り FP16・1GB
    io['value'] = ({'mode': [1, 6]}, {'actions': [1, 10, 2]})
    assert be.main(['build_engine.py', str(run / 'jetformer.onnx'), engine]) == 0
    assert calls['fp16'] is True and calls['bf16'] is False and calls['ws'] == 1.0


# ---------------------------------------------------------------- 共有の部品(act・tcp が使う)
def test_blocks_sensor_tokens_match_v0_names_and_values():
    from jetformer.policies import blocks

    class Tok(torch.nn.Module):
        def __init__(self):
            super().__init__()
            blocks.add_sensor_tokens(self, 32, 6, pretrained=False)

        def forward(self, image, tof, imu, state, mode):
            return blocks.sensor_tokens(self, image, tof, imu, state, mode)
    torch.manual_seed(0)
    v0 = jm.JetFormer(num_modes=6, dim=32, layers=1, heads=4).eval()
    tok = Tok().eval()
    v0_keys = {k for k in v0.state_dict() if k.split('.')[0] in
               ('image_backbone', 'image_proj', 'tof_encoder', 'imu_mlp', 'state_mlp', 'mode_embed')}
    assert set(tok.state_dict()) == v0_keys                                       # v0 と同じ名前の層
    tok.load_state_dict({k: v for k, v in v0.state_dict().items() if k in v0_keys})
    x = _x()
    with torch.no_grad():
        t = tok(**x)
        grab = {}
        h = v0.fusion.register_forward_pre_hook(lambda m, args: grab.__setitem__('in', args[0]))
        v0(**x)
        h.remove()
    assert t.shape == (2, blocks.N_SENSOR_TOKENS, 32)
    want = grab['in'][:, :blocks.N_SENSOR_TOKENS] - (v0.pos + v0.modality[v0.modality_index])[:, :blocks.N_SENSOR_TOKENS]
    assert torch.allclose(t, want, atol=1e-5)                                    # v0 の forward と同じトークン
    all_modes = torch.arange(2 * 6 * 3, dtype=torch.float32).view(2, 6, 3)
    sel = blocks.select_mode(all_modes, x['mode'])
    assert torch.equal(sel, torch.stack([all_modes[0, 4], all_modes[1, 5]]))
    enc, dec = blocks.encoder(32, 4, 64, 1), blocks.decoder(32, 4, 64, 1)
    with torch.no_grad():
        mem = enc(t)
        assert dec(torch.zeros(2, 10, 32), mem).shape == (2, 10, 32)
