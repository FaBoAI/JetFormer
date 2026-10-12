"""ACT 型の方策(jetformer/policies/act.py、spec.md §3.1・§7)のテスト。
CPU だけ・GPU・実データ・ネットなしで通る(30 秒未満)

    cd ~/jetformer && CUDA_VISIBLE_DEVICES= nice python -m pytest -q tests/test_policy_act.py
"""

import collections
import json
import os
import sys

import cv2
import numpy as np
import onnx
import pytest
import torch
import torch.nn as nn

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))
from jetformer import policies  # noqa: E402
from jetformer.model import JetFormer  # noqa: E402
from jetformer.policies import act, post, runtime  # noqa: E402
from jetformer.train_data import GRAVITY  # noqa: E402
import train  # noqa: E402

SMALL = {'dim': 32, 'enc_layers': 1, 'dec_layers': 1, 'heads': 4, 'ffn': 64}         # 速く確かめるための小さな形
SMALL_CVAE = dict(SMALL, cvae=True, latent_dim=4, cvae_layers=1)
SENSOR_LAYERS = ('image_backbone', 'image_proj', 'tof_encoder', 'imu_mlp', 'state_mlp', 'mode_embed')   # v0 と同じ名前
ACT_LAYERS = SENSOR_LAYERS + ('pos', 'encoder', 'enc_norm', 'queries', 'decoder', 'dec_norm', 'heads')
CVAE_LAYERS = ('cvae_cls', 'cvae_action', 'cvae_pos', 'cvae_encoder', 'cvae_out', 'latent_proj')
# 書き出しで使ってよい演算 = モデル B の jetformer.onnx の演算(TensorRT FP16 で走っている)+ Identity(Dropout の名残り。
# bench の act_dec2.onnx も同じ組で TensorRT 10.16 のエンジンになった)
ALLOWED_OPS = {'Add', 'Cast', 'Concat', 'Constant', 'ConstantOfShape', 'Conv', 'Div', 'Equal', 'Expand', 'Flatten', 'Gather',
               'Gemm', 'Greater', 'Identity', 'LayerNormalization', 'MatMul', 'MaxPool', 'Mod', 'Mul', 'ReduceSum', 'Relu',
               'Reshape', 'Shape', 'Slice', 'Softmax', 'Sqrt', 'Squeeze', 'Transpose', 'Unsqueeze', 'Where'}
# spec.md §8: ループ・分岐・RNN の層・ATen の代用・antialias の縮小(Resize)・乱数(CVAE の z が書き出しに入っていない)
FORBIDDEN_OPS = {'Loop', 'If', 'Scan', 'GRU', 'LSTM', 'RNN', 'ATen', 'Resize', 'GlobalAveragePool', 'NonZero',
                 'RandomNormal', 'RandomNormalLike', 'RandomUniform', 'RandomUniformLike', 'Multinomial'}
STEER_CENTER_US, THROTTLE_NEUTRAL_US = 1443.0, 1500.0       # 車の約束: ステア > 1443 = 左、スロットル < 1500 = 前進


def _x(batch=2, num_modes=6, genres=(4, 5), seed=7):
    g = torch.Generator().manual_seed(seed)
    ids = torch.tensor([genres[i % len(genres)] for i in range(batch)])
    return dict(image=torch.randn(batch, 3, 224, 224, generator=g), tof=torch.rand(batch, 2, 42, 54, generator=g),
                imu=torch.randn(batch, 10, 6, generator=g) * 0.1, state=torch.randn(batch, 3, generator=g) * 0.1,
                mode=torch.nn.functional.one_hot(ids, num_modes).float())


def _onehot(batch, genre, num_modes):
    return torch.nn.functional.one_hot(torch.full((batch,), genre), num_modes).float()


def _small(num_modes=6, seed=0, **over):
    torch.manual_seed(seed)
    return policies.build_policy('act', num_modes, dict(SMALL, **over))


def _tiny_backbone(model):
    """学習のテストを CPU で速くする: 画像の ResNet-18 と ToF の CNN を、同じ形のトークン([B,512,7,7]・[B,D,6,7])を出す
    小さな層に替えて凍結する(トークンより後ろ = Encoder・行動クエリの Decoder・ヘッド・IMU/操作値の MLP はそのまま学習する)"""
    model.image_backbone = nn.Sequential(nn.AvgPool2d(32), nn.Conv2d(3, 512, 1))         # 224 → 7×7
    model.tof_encoder = nn.Sequential(nn.AvgPool2d(7), nn.Conv2d(2, model.dim, 1))       # 42×54 → 6×7
    for name in ('image_backbone', 'tof_encoder'):
        getattr(model, name).requires_grad_(False)
    return model


def _us(norm):
    return norm * 500.0 + 1500.0


# ---------------------------------------------------------------- 登録簿・大きさ・層の名前
def test_registry_ready_and_default_size():
    assert policies.get('act').ready() == (True, '')
    m = policies.build_policy('act', 6)
    assert type(m) is act.ACTPolicy
    total, trainable = runtime.count_params(m)
    assert abs(total / 1e6 - 15.6) <= 0.3 and trainable == total               # spec.md §3.1: 15.6M ± 0.3M
    assert getattr(m, 'USES_TARGET', False) is False
    assert getattr(m, 'policy_loss', None) is None       # CVAE なしは L1 だけ(学習の損失・history は v0 と同じ扱い)
    assert m.EXPORT_OUTPUTS == ('actions',) and policies.get('act').outputs == ('actions',)
    names = [n for n, _ in m.named_children()] + [n for n, _ in m.named_parameters(recurse=False)]
    assert set(names) == set(ACT_LAYERS)
    assert tuple(m.heads.weight.shape) == (6 * 2, 256) and tuple(m.pos.shape) == (1, 94, 256)
    assert tuple(m.queries.shape) == (1, 10, 256)
    assert len(m.encoder.layers) == 4 and len(m.decoder.layers) == 2
    # センサーのトークン化の層は v0(JetFormer)と同じキー・同じ形(将来の部分的な初期値のため)
    v0 = JetFormer(num_modes=6).state_dict()
    sd = m.state_dict()
    for key, t in v0.items():
        if key.split('.')[0] in SENSOR_LAYERS:
            assert key in sd and sd[key].shape == t.shape, key
    for dec, expect in ((1, 14.8), (4, 17.1)):                               # bench の実測(act・act_dec4)
        assert abs(runtime.count_params(policies.build_policy('act', 6, {'dec_layers': dec}))[0] / 1e6 - expect) <= 0.3


def test_cvae_size_and_hooks():
    m = policies.build_policy('act', 6, {'cvae': True})
    assert type(m) is act.ACTCVAEPolicy and m.USES_TARGET is True and callable(m.policy_loss)
    base = runtime.count_params(policies.build_policy('act', 6))[0]
    extra = (runtime.count_params(m)[0] - base) / 1e6
    assert 0.9 <= extra <= 1.3                            # spec.md §3.1: CVAE で +約1.1M(学習のときだけ使う)
    assert tuple(m.pos.shape) == (1, 95, 256) and len(m.cvae_encoder.layers) == 2
    assert {n for n, _ in m.named_children()} | {n for n, _ in m.named_parameters(recurse=False)} == \
        set(ACT_LAYERS) | set(CVAE_LAYERS)


def test_bad_settings_are_rejected():
    with pytest.raises(ValueError):
        act.build(6, dim=30, heads=8)                     # dim が heads で割り切れない
    with pytest.raises(ValueError):
        act.build(6, **dict(SMALL, enc_layers=0))
    with pytest.raises(TypeError):
        act.build(6, no_such_key=1)
    spec = policies.get('act')
    with pytest.raises(ValueError):
        spec.resolve({'dec_layers': 5}, ui=True)          # 管理画面は 1〜4
    with pytest.raises(ValueError):
        spec.resolve({'kl_weight': 3.0}, ui=True)         # 1.0 / 10.0 だけ
    with pytest.raises(ValueError):
        spec.resolve({'enc_layers': 6}, ui=True)          # 管理画面からは変えられない(CLI はよい)
    assert spec.resolve({'enc_layers': 6})['enc_layers'] == 6


# ---------------------------------------------------------------- 出力の形・ジャンル・並び
@pytest.mark.parametrize('cfg', [SMALL, SMALL_CVAE])
def test_forward_shapes_train_and_eval(cfg):
    torch.manual_seed(0)
    m = policies.build_policy('act', 6, cfg)
    x = _x(batch=2)
    target = torch.zeros(2, 10, 2)
    m.train()
    out = runtime.forward(m, x, target=target, train=True)
    assert set(out) == {'actions'} and tuple(out['actions'].shape) == (2, 10, 2)
    m.eval()
    with torch.no_grad():
        y = runtime.forward(m, x)['actions']
        assert tuple(y.shape) == (2, 10, 2) and torch.isfinite(y).all()
        assert tuple(m.embed(**x).shape) == (2, cfg['dim'])                # offline_rl features 用
    runtime.check_forward(m, 6)                                          # 学習の前の確認が通る


def test_head_layout_is_genre_times_two_plus_steer_throttle():
    """heads の行 = ジャンル×2 + [ステア, スロットル]。10 ステップとも同じヘッド(重み 0 ならバイアスがそのまま出る)"""
    m = _tiny_backbone(_small()).eval()
    with torch.no_grad():
        m.heads.weight.zero_()
        bias = torch.tensor([[0.1 * g + 0.01, -(0.1 * g + 0.02)] for g in range(6)]).flatten()
        m.heads.bias.copy_(bias)
        x = _x(batch=2)
        for g in range(6):
            x['mode'] = _onehot(2, g, 6)
            y = m(**x)
            assert torch.allclose(y[..., 0], torch.full((2, 10), float(bias[2 * g])), atol=1e-6)      # ステア
            assert torch.allclose(y[..., 1], torch.full((2, 10), float(bias[2 * g + 1])), atol=1e-6)  # スロットル


def test_only_the_selected_genre_rows_matter():
    """one-hot で選んだジャンルの heads の行だけが出力に効く(ほかのジャンルの行を変えても同じ)"""
    m = _tiny_backbone(_small()).eval()
    x = _x(batch=2)
    x['mode'] = _onehot(2, 5, 6)
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():
        ref = m(**x).clone()
        m.heads.weight[6:8] += torch.randn(2, 32, generator=g)      # ジャンル 3 の行(3×2, 3×2+1)
        m.heads.bias[6:8] -= 0.5
        assert torch.equal(m(**x), ref)
        x3 = dict(x, mode=_onehot(2, 3, 6))
        y3 = m(**x3)
        m.heads.weight[10:12] += torch.randn(2, 32, generator=g)    # ジャンル 5 の行
        assert (m(**x) - ref).abs().max() > 1e-3
        assert torch.equal(m(**x3), y3)


def test_expand_modes_rows2_keeps_outputs():
    """runtime.expand_modes(genre_keys = {'heads': 2})で増やしたジャンルは、写した元のジャンルと同じ出力"""
    spec = policies.get('act')
    assert spec.genre_keys == {'embed': 'mode_embed.weight', 'heads': {'heads': 2}}
    small = _tiny_backbone(_small(num_modes=6, seed=1)).eval()
    big = _tiny_backbone(_small(num_modes=8, seed=2)).eval()     # 小さな CNN の重みも expand_modes で small から写る
    state = runtime.expand_modes(small.state_dict(), big.state_dict(), spec.genre_keys, {6: 5, 7: 4})
    big.load_state_dict(state, strict=True)
    x = _x(batch=2)
    with torch.no_grad():
        for g_big, g_src in [(0, 0), (4, 4), (5, 5), (6, 5), (7, 4)]:
            y_big = big(**dict(x, mode=_onehot(2, g_big, 8)))
            y_src = small(**dict(x, mode=_onehot(2, g_src, 6)))
            assert torch.allclose(y_big, y_src, atol=1e-5), (g_big, g_src)


# ---------------------------------------------------------------- CVAE
def test_cvae_kl_loss_and_eval_ignores_target():
    torch.manual_seed(0)
    m = _tiny_backbone(policies.build_policy('act', 6, SMALL_CVAE))
    plain = _small()
    assert m.USES_TARGET is True and getattr(plain, 'USES_TARGET', False) is False
    x = _x(batch=3, genres=(4, 5, 4))
    target = torch.randn(3, 10, 2) * 0.3
    # 学習中・target あり: z を引き、KL > 0、損失 = L1 + kl_weight·KL、内訳 kl
    m.train()
    out = runtime.forward(m, x, target=target, train=True)
    kl = m._kl.detach().clone()
    assert tuple(kl.shape) == (3,) and (kl > 0).all()
    per = runtime.sample_loss(m, out, target, None)
    l1 = runtime.l1_per_sample(out['actions'], target)
    assert torch.allclose(per, l1 + SMALL_CVAE.get('kl_weight', 10.0) * kl, atol=1e-5)
    parts = runtime.loss_parts(m)
    assert set(parts) == {'kl'} and parts['kl'] == pytest.approx(float(kl.mean()), rel=1e-5)
    per.mean().backward()
    for name in ('cvae_encoder', 'cvae_out', 'cvae_action', 'latent_proj', 'queries', 'heads', 'image_proj'):
        grads = [p.grad for n, p in m.named_parameters() if n.startswith(name)]
        assert grads and any(g is not None and g.abs().sum() > 0 for g in grads), name
    # target なしの学習(検証と同じ z = 0)は L1 だけ・内訳なし
    out0 = runtime.forward(m, x, target=target, train=False)
    assert m._kl is None
    assert torch.allclose(runtime.sample_loss(m, out0, target, None), runtime.l1_per_sample(out0['actions'], target))
    assert runtime.loss_parts(m) == {}
    # 推論(eval): target の有無で同じ・CVAE の符号化器を変えても同じ(z = 0、潜在トークン = latent_proj のバイアス)
    m.eval()
    with torch.no_grad():
        ref = m(**x)
        assert torch.equal(m(**x, target=target), ref)
        for name in ('cvae_encoder', 'cvae_out', 'cvae_action'):
            for p in getattr(m, name).parameters():
                p.add_(0.5)
        m.cvae_cls.add_(1.0)
        assert torch.equal(m(**x), ref)
        lat = m.latent_proj(torch.zeros(3, m.latent_dim)).unsqueeze(1)
        assert torch.equal(lat, m.latent_token(3, x['mode'], None))       # latent_proj(0) = バイアス


# ---------------------------------------------------------------- 学習で損失が下がる・符号の決まり(合成データ)
@pytest.mark.parametrize('cfg', [dict(SMALL, dropout=0.0), dict(SMALL_CVAE, dropout=0.0)])
def test_tiny_batch_overfits_and_keeps_sign_conventions(cfg):
    """左旋回・右旋回(どちらも前進)・だんだん左へ切る列の 3 つを覚えられ、出力の [ステア, スロットル] の向きと
    ステップの並びが正解のまま(ステア > 1443 µs = 左、スロットル < 1500 µs = 前進)"""
    torch.manual_seed(0)
    m = _tiny_backbone(policies.build_policy('act', 6, cfg))
    g = torch.Generator().manual_seed(3)
    off = torch.tensor([1.0, -1.0, 0.0]).view(3, 1, 1, 1)                     # 3 つの場面を見分けられる入力
    x = dict(image=torch.randn(3, 3, 224, 224, generator=g) * 0.1 + off,
             tof=torch.rand(3, 2, 42, 54, generator=g) * 0.2 + (off + 1.0) * 0.3,
             imu=torch.randn(3, 10, 6, generator=g), state=torch.randn(3, 3, generator=g),
             mode=torch.nn.functional.one_hot(torch.tensor([5, 5, 4]), 6).float())
    steer = torch.stack([torch.full((10,), 1600.0), torch.full((10,), 1300.0), torch.linspace(1450, 1650, 10)])
    throttle = torch.stack([torch.full((10,), 1380.0), torch.full((10,), 1400.0), torch.linspace(1420, 1350, 10)])
    target = (torch.stack([steer, throttle], dim=-1) - 1500.0) / 500.0           # データセットと同じ正規化 [3,10,2]
    steps = 100
    opt = torch.optim.AdamW(runtime.param_groups(m, 3e-3), lr=3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)        # 学習と同じ cosine(最後は安定する)
    first = None
    threads = torch.get_num_threads()
    torch.set_num_threads(1)              # 小さな演算ばかりなので 1 スレッドが速い(CPU の取り合いも減る)
    try:
        for _ in range(steps):
            m.train()
            out = runtime.forward(m, x, target=target, train=True)
            loss = runtime.sample_loss(m, out, target, None).mean()
            first = float(runtime.l1_per_sample(out['actions'].detach(), target).mean()) if first is None else first
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    finally:
        torch.set_num_threads(threads)
    m.eval()
    with torch.no_grad():
        pred = _us(m(**x))                                                       # µs [3,10,2]
    err = float((pred - _us(target)).abs().mean())
    assert err < 0.25 * first * 500.0 and err < 30.0, (first * 500.0, err)       # 推論は z = 0(CVAE でも)
    assert (pred[0, :, 0] > STEER_CENTER_US).all() and (pred[1, :, 0] < STEER_CENTER_US).all()      # 左・右
    assert (pred[:, :, 1] < THROTTLE_NEUTRAL_US).all()                                              # どれも前進
    assert pred[2, -1, 0] - pred[2, 0, 0] > 100.0          # 1 フレーム先 → 10 フレーム先へ左に切っていく(並びが正解どおり)
    assert pred[2, -1, 1] < pred[2, 0, 1]                  # 前進が強くなっていく


# ---------------------------------------------------------------- 書き出し(ONNX)
@pytest.mark.parametrize('cfg', [SMALL, SMALL_CVAE])
def test_onnx_export_contract_and_ops(cfg, tmp_path):
    torch.manual_seed(0)
    m = policies.build_policy('act', 6, cfg)
    path = tmp_path / 'jetformer.onnx'
    io = runtime.export_onnx(m, path, 6)
    assert io['inputs'] == list(policies.BASE_INPUTS)
    assert io['input_shapes'] == dict(policies.BASE_SHAPES, mode=[1, 6])
    assert io['outputs'] == ['actions'] and io['output_shapes'] == {'actions': [1, 10, 2]}
    assert policies.check_io(io['input_shapes'], io['output_shapes'], 6) == (True, '')
    assert not os.path.exists(str(path) + '.tmp')
    graph = onnx.load(str(path), load_external_data=False).graph
    ops = collections.Counter(n.op_type for n in graph.node)
    assert not set(ops) & FORBIDDEN_OPS, set(ops) & FORBIDDEN_OPS
    assert set(ops) <= ALLOWED_OPS, set(ops) - ALLOWED_OPS
    # LayerNorm の数 = Encoder 2/層 + Decoder 3/層 + enc_norm + dec_norm(CVAE の符号化器はグラフに入らない)
    assert ops['LayerNormalization'] == 2 * cfg['enc_layers'] + 3 * cfg['dec_layers'] + 2
    for d in list(graph.input) + list(graph.output):
        assert all(dim.HasField('dim_value') and dim.dim_value > 0 for dim in d.type.tensor_type.shape.dim)
    # 書き出しの後もモデルは使える(eval・CPU・FP32)
    assert tuple(runtime.predict(m, runtime.example_inputs(6))['actions'].shape) == (1, 10, 2)


# ---------------------------------------------------------------- 走行側の予測の重ね合わせ(全方策で共有。既定オフ)
def test_chunks_through_drive_stage_and_temporal_ensembling():
    """ACT の 10 ステップの予測は、走行・評価と同じ post.replay(後処理なし)を通る。ensemble_j = 0(既定)なら
    予測そのもの、ensemble_j > 0 なら前のフレームの予測の重なる行を新しいほど重く混ぜる(jetformer/policies/ensemble.py)"""
    m = _tiny_backbone(_small()).eval()
    x = _x(batch=4, genres=(5,), seed=11)                     # 4 フレーム続けて
    with torch.no_grad():
        outputs = {'actions': m(**x).numpy()}                 # [4,10,2] 正規化のまま(runtime.predict と同じ形)
    cfg = {'policy': 'act', 'num_modes': 6}
    t_ns = (np.arange(4) * 1e9 / 30).astype(np.int64)
    erpm = np.full(4, -2000.0)
    raw = outputs['actions'] * 500.0 + 1500.0
    off, model_pred, infos = post.replay(cfg, (1443.0, 1500.0), outputs, erpm, t_ns, {'ensemble_j': 0, 'ensemble_m': 0.5})
    assert np.array_equal(off, raw.astype(np.float32)) and np.array_equal(model_pred, off) and infos == [None] * 4
    on, _, _ = post.replay(cfg, (1443.0, 1500.0), outputs, erpm, t_ns, {'ensemble_j': 1, 'ensemble_m': 0.5})
    assert np.allclose(on[0], raw[0], atol=1e-3)               # 最初のフレームは重ねるものが無い
    w = np.exp(-0.5)
    assert np.allclose(on[1, :9], (raw[1, :9] + w * raw[0, 1:]) / (1 + w), atol=1e-3)   # 1 フレーム前の予測を 1 行ずらして
    assert np.allclose(on[1, 9], raw[1, 9], atol=1e-3)


# ---------------------------------------------------------------- 学習の通し(tools/train.py、CPU・合成データ・小さな形)
MODES = [{'id': i, 'name': n, 'description': ''} for i, n in enumerate(['未設定', '駐車(1)', '駐車(2)', '駐車(3)', '高速走行', '低速走行'])]


def _make_dataset(root, n=30, scenes=('s1', 's2', 's3')):
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


def test_train_main_end_to_end_cvae(tmp_path, monkeypatch):
    """tools/train.py --policy act(CVAE つきの小さな形)で、学習 → 書き出し → model_config.json・result.json →
    load_run まで通る。走行からは v0 と同じ扱い(後処理なし・drivable)"""
    root = _make_dataset(tmp_path / 'ds')
    out = tmp_path / 'run'
    argv = ['train.py', '--data', root, '--out', str(out), '--epochs', '1', '--batch', '2', '--workers', '0',
            '--max-steps', '1', '--stride', '5', '--no-pretrained', '--no-command-input', '--impact-g', '2.5',
            '--policy', 'act', '--policy-cfg', json.dumps(SMALL_CVAE), '--seed', '0']
    monkeypatch.setattr(sys, 'argv', argv)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    train.main()
    cfg = json.loads((out / 'model_config.json').read_text(encoding='utf-8'))
    res = json.loads((out / 'result.json').read_text(encoding='utf-8'))
    full = policies.get('act').resolve(SMALL_CVAE)
    assert cfg['policy'] == 'act' and cfg['policy_cfg'] == full and cfg['contract'] == 1
    assert cfg['inputs'] == list(policies.BASE_INPUTS) and cfg['outputs'] == ['actions']
    assert cfg['output_shapes'] == {'actions': [1, 10, 2]} and 'postprocess' not in cfg
    assert policies.drivable(cfg) == (True, '') and post.make(cfg, (1443.0, 1500.0)) == (None, '')
    assert res['policy'] == 'act' and res['policy_cfg'] == full and res['seed'] == 0
    assert 'train_loss' in res['history'][0] and res['history'][0]['train_parts']['kl'] >= 0.0   # CVAE の内訳が残る
    assert os.path.exists(out / 'jetformer.onnx')
    ck = torch.load(out / 'last.pth', weights_only=False)
    assert ck['args']['policy'] == 'act'
    model, _, spec, pcfg = runtime.load_run(out)
    assert spec.name == 'act' and pcfg == full and type(model) is act.ACTCVAEPolicy and not model.training
    # --init の一致の確認: kl_weight は学習だけの設定(通る)、cvae は構造(止まる)
    train.check_init_compat(cfg, 'act', dict(full, kl_weight=1.0), spec)
    with pytest.raises(SystemExit):
        train.check_init_compat(cfg, 'act', dict(full, cvae=False), spec)
    with pytest.raises(SystemExit):
        train.check_init_compat(cfg, 'jetformer_v0', policies.V0_FROZEN_CFG, policies.get('jetformer_v0'))
