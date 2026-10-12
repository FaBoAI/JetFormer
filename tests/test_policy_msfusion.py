"""多段融合(TransFuser 型)の方策 jetformer/policies/msfusion.py のテスト(spec.md §3.3・§7)。
CPU だけ・GPU・実データ・ネットなしで通る(ResNet-18 の ImageNet の重みも読まない)

    cd ~/jetformer && CUDA_VISIBLE_DEVICES= nice python -m pytest -q tests/test_policy_msfusion.py
"""

import collections
import itertools
import json
import os
import sys

import numpy as np
import onnx
import pytest
import torch
import torch.nn as nn
import torchvision

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
from jetformer import policies  # noqa: E402
from jetformer.policies import msfusion as msf  # noqa: E402
from jetformer.policies import runtime  # noqa: E402

NUM_MODES = 6
# 速く確かめるための小さな設定(段の数・格子・ToF の枝・ResNet-18 は既定と同じ。交換と後の融合を細く・浅く)
SMALL = {'dim': 64, 'ex_layers': [1, 1, 1, 1], 'ex_ffn': [1, 1, 1, 1], 'late_layers': 1, 'late_heads': 4,
         'dropout': 0.0}
LAYER_NAMES = {'img_stem', 'img_stages', 'tof', 'fuse', 'img_proj', 'tof_proj', 'imu_mlp', 'state_mlp', 'mode_embed',
               'query', 'pos', 'fusion', 'norm', 'trunk', 'heads'}            # spec.md §3.3 の層の名前(head='tokens')
STEER_CENTER, NEUTRAL = 1443.0, 1500.0          # この車: ステア > 中心 = 左、スロットル < 中立 = 前進


def _build(seed=0, num_modes=NUM_MODES, **over):
    torch.manual_seed(seed)
    return policies.build_policy('msfusion', num_modes, dict(SMALL, **over))


def _x(batch=2, num_modes=NUM_MODES, seed=7, genres=(4, 5)):
    g = torch.Generator().manual_seed(seed)
    genres = [genres[i % len(genres)] for i in range(batch)]
    return dict(image=torch.randn(batch, 3, 224, 224, generator=g), tof=torch.rand(batch, 2, 42, 54, generator=g),
                imu=torch.randn(batch, 10, 6, generator=g) * 0.1, state=torch.randn(batch, 3, generator=g) * 0.1,
                mode=torch.nn.functional.one_hot(torch.tensor(genres), num_modes).float())


@pytest.fixture(scope='module')
def default_model():
    torch.manual_seed(0)
    return policies.build_policy('msfusion', NUM_MODES).eval()


# ---------------------------------------------------------------- 登録簿・既定の形
def test_registry_entry_is_ready():
    spec = policies.get('msfusion')
    assert spec.ready() == (True, '')
    assert not hasattr(msf, 'POLICY_STUB')
    assert spec.slug == 'msf' and spec.outputs == ('actions',) and spec.postprocess is None
    assert spec.genre_keys == {'embed': 'mode_embed.weight', 'heads': {'heads': 20}}
    assert spec.resolve() == {'dim': 256, 'scales': [1, 2, 3, 4], 'ex_layers': [2, 2, 2, 1], 'ex_heads': 4,
                              'ex_ffn': [4, 4, 4, 2], 'late_layers': 2, 'late_heads': 8, 'head': 'tokens',
                              'dropout': 0.1}


def test_default_size_layers_and_forward(default_model):
    m = default_model
    total, trainable = runtime.count_params(m)
    assert abs(total / 1e6 - 18.9) <= 0.5 and trainable == total               # 凍結なし(ResNet-18 も学習)
    assert abs(total / 1e6 - policies.get('msfusion').hints['params_m']) <= 0.5
    assert {k.split('.')[0] for k in m.state_dict()} == LAYER_NAMES
    # 交換: 段1〜3 は 2層・FFN×4、段4 は 1層・FFN×2、どれも 4 ヘッド。後の融合は 2層・8 ヘッド・FFN 512
    assert sorted(m.fuse) == ['1', '2', '3', '4']
    for s, (ch, layers, ffn) in {1: (64, 2, 4), 2: (128, 2, 4), 3: (256, 2, 4), 4: (512, 1, 2)}.items():
        f = m.fuse[str(s)]
        assert f.pos.shape == (1, 91, ch) and len(f.tf.layers) == layers
        assert f.tf.layers[0].linear1.out_features == ch * ffn and f.tf.layers[0].self_attn.num_heads == 4
    assert m.pos.shape == (1, 95, 256) and len(m.fusion.layers) == 2
    assert m.fusion.layers[0].self_attn.num_heads == 8 and m.fusion.layers[0].linear1.out_features == 512
    assert m.heads.out_features == NUM_MODES * 20
    out = runtime.check_forward(m, NUM_MODES)                                   # tools/train.py が学習の前に呼ぶもの
    assert tuple(out['actions'].shape) == (2, 10, 2)
    assert getattr(m, 'EXPORT_OUTPUTS', ('actions',)) == ('actions',) and not getattr(m, 'USES_TARGET', False)


@torch.no_grad()
def test_stage_grids_divide_after_padding(default_model):
    m = default_model
    x = _x(batch=1)
    fi, ft = m.img_stem(x['image']), m.tof.prepare(x['tof'])
    assert tuple(ft.shape) == (1, msf.TOF_STEM_CH) + msf.TOF_PADDED_HW == (1, 32, 48, 56)
    windows = {1: ((8, 8), (4, 4)), 2: ((4, 4), (2, 2)), 3: ((2, 2), (1, 1)), 4: ((1, 1), (1, 1))}
    for s in msf.STAGES:
        fi, ft = m.img_stages[s - 1](fi), m.tof.stages[s - 1](ft)
        ch = msf.STAGE_CH[s - 1]
        assert tuple(fi.shape) == (1, ch) + msf.IMG_STAGE_HW[s - 1]
        assert tuple(ft.shape) == (1, ch) + msf.TOF_STAGE_HW[s - 1]
        f = m.fuse[str(s)]
        assert (f.ki, f.kt) == windows[s]
        assert (fi.shape[2] // f.ki[0], fi.shape[3] // f.ki[1]) == msf.IMG_GRID == (7, 7)
        assert (ft.shape[2] // f.kt[0], ft.shape[3] // f.kt[1]) == msf.TOF_GRID == (6, 7)
        fi2, ft2 = f(fi, ft)
        assert fi2.shape == fi.shape and ft2.shape == ft.shape                   # 足し戻しは元の大きさ
        fi, ft = fi2, ft2
    with pytest.raises(ValueError):
        msf.window((28, 28), (8, 8))         # TransFuser の 8×8 は 224 の入力で割り切れない(書き出せない)ので使わない
    with pytest.raises(ValueError):
        msf.window((42, 54), msf.TOF_GRID)   # 詰める前の ToF も割り切れない


def test_tof_padding_keeps_orientation():
    """詰め物は上下 3 行・左右 1 列の 0(無効)だけ。ToF の左右・上下は入れ替わらない(元の [r, c] → [r+3, c+1])"""
    g = torch.Generator().manual_seed(3)
    tof = torch.rand(2, 2, 42, 54, generator=g)
    tof[:, 0, 10:20, :5] = 0.0                                    # 左端の無効ゾーン
    p = msf.ToFStages.padded_input(tof)
    assert tuple(p.shape) == (2, 3, 48, 56)
    assert torch.equal(p[:, :2, 3:45, 1:55], tof)
    assert torch.equal(p[:, 2, 3:45, 1:55], (tof[:, 0] > 0).float())
    assert p[:, 2, 13:23, 1:6].sum() == 0 and p[:, 2, 13:23, 6:].sum() > 0     # 左の無効は左のまま
    border = torch.ones(48, 56, dtype=torch.bool)
    border[3:45, 1:55] = False
    assert p[:, :, border].abs().sum() == 0                                  # 詰めた所は距離・信頼度・有効の印が全部 0


def test_stage_fusion_puts_tokens_back_in_place():
    """交換の出力は元の格子の同じ場所に戻る(左右・上下の反転や画像と ToF の取り違えが無い)。
    Transformer と LayerNorm を素通しにし、位置の埋め込みを 0 にして、平均 → 拡大の足し戻しだけを見る"""
    f = msf.StageFusion(64, (56, 56), (24, 28), layers=1, heads=4, ffn_mult=1, dropout=0.0)
    f.tf, f.norm = nn.Identity(), nn.Identity()
    with torch.no_grad():
        f.pos.zero_()
    fi = torch.zeros(1, 64, 56, 56)
    ft = torch.zeros(1, 64, 24, 28)
    fi[:, 0, :8, :8] = 1.0               # 画像の左上の窓
    ft[:, 1, :4, -4:] = 1.0              # ToF の右上の窓
    with torch.no_grad():
        oi, ot = f(fi, ft)
    di, dt = oi - fi, ot - ft
    assert di[0, 0, :4, :4].min() > 0.99 and di[0, 0, :, 28:].abs().max() == 0 and di[0, 0, 28:, :].abs().max() == 0
    assert dt[0, 1, :2, -2:].min() > 0.99 and dt[0, 1, :, :14].abs().max() == 0 and dt[0, 1, 12:, :].abs().max() == 0
    assert di[0, 1].abs().max() == 0 and dt[0, 0].abs().max() == 0          # 画像と ToF のトークンが混ざらない
    with torch.no_grad():
        oi, ot = f(torch.ones(1, 64, 56, 56), torch.ones(1, 64, 24, 28))
    assert torch.allclose(oi, torch.full_like(oi, 2.0)) and torch.allclose(ot, torch.full_like(ot, 2.0))


# ---------------------------------------------------------------- 設定(scales・head・不正な値)
def test_fewer_scales_mean_fewer_exchange_layers():
    spec = policies.get('msfusion')
    counts = []
    for scales in ([1, 2, 3, 4], [2, 3, 4], [3, 4], [1, 2, 3]):   # 登録簿の選択肢(管理画面・CLI とも)
        m = policies.build_policy('msfusion', NUM_MODES, {'scales': scales})       # 交換の層は既定の深さ
        assert sorted(m.fuse) == [str(s) for s in scales]
        n_layers = sum(len(m.fuse[k].tf.layers) for k in m.fuse)
        assert n_layers == sum([2, 2, 2, 1][s - 1] for s in scales)
        counts.append((n_layers, runtime.count_params(m)[0]))
    no4 = counts.pop()                                            # [1, 2, 3](4段目の交換なし)は既定より小さい
    assert no4[0] == counts[0][0] - 1 and no4[1] < counts[0][1]
    m = msf.build(NUM_MODES, **dict(SMALL, scales=[]))            # 交換なし(モジュールを直接。遅い融合だけ)
    counts.append((len(m.fuse), runtime.count_params(m)[0]))
    assert counts == sorted(counts, reverse=True) and len(set(counts)) == 4 and counts[-1][0] == 0
    with torch.no_grad():
        assert tuple(m.eval()(**_x(batch=1)).shape) == (1, 10, 2)      # 交換なしでも動く(段3・4 だけは学習のテストで)
    assert spec.resolve({'scales': [2, 3, 4]}, ui=True)['scales'] == [2, 3, 4]
    for ui in (True, False):                                   # 選択肢の外は CLI の --policy-cfg でも断る
        with pytest.raises(ValueError):
            spec.resolve({'scales': [1, 3]}, ui=ui)


@pytest.mark.parametrize('bad', [{'scales': [0]}, {'scales': [5]}, {'scales': [2, 2]}, {'ex_layers': [1, 1, 1]},
                                 {'ex_layers': [0, 1, 1, 1]}, {'ex_ffn': [1, 1, 1, 0]}, {'ex_heads': 3},
                                 {'head': 'mlp'}, {'dim': 66}, {'dropout': 1.0}])
def test_bad_config_raises_before_building(bad, monkeypatch):
    def no_resnet(*a, **k):
        raise AssertionError('設定の確認の前に ResNet を作っています')
    monkeypatch.setattr(torchvision.models, 'resnet18', no_resnet)
    with pytest.raises(ValueError):
        msf.build(NUM_MODES, **dict(SMALL, **bad))


def test_pooled_head_and_its_embed():
    x = _x()
    m = _build(head='pooled').eval()
    for gone in ('fusion', 'query', 'pos', 'img_proj', 'tof_proj', 'norm'):
        assert not hasattr(m, gone)
    assert m.global_proj.in_features == 512 and m.trunk[0].in_features == 4 * SMALL['dim']
    assert policies.get('msfusion').resolve({'head': 'pooled'}, ui=True)['head'] == 'pooled'
    with torch.no_grad():
        y = m(**x)
        e = m.embed(**x)
        again = (m.heads(e).view(2, NUM_MODES, 10, 2) * x['mode'].view(2, NUM_MODES, 1, 1)).sum(1)
    assert tuple(y.shape) == (2, 10, 2) and tuple(e.shape) == (2, SMALL['dim'])
    assert torch.allclose(again, y, atol=1e-6)


# ---------------------------------------------------------------- ジャンル
@torch.no_grad()
def test_genre_heads_select_rows_expand_modes_and_embed():
    spec = policies.get('msfusion')
    m = _build().eval()
    x = _x(genres=(4, 5))
    y = m(**x)
    e = m.embed(**x)                                          # tokens: 融合の後の出力トークン(trunk の前)= 特徴 [B,dim]
    again = (m.heads(m.trunk(e)).view(2, NUM_MODES, 10, 2) * x['mode'].view(2, NUM_MODES, 1, 1)).sum(1)
    assert tuple(e.shape) == (2, SMALL['dim']) and torch.allclose(again, y, atol=1e-6)
    m.heads.weight[5 * 20:6 * 20] += 0.5                       # ジャンル 5 の行だけ変える
    y2 = m(**x)
    assert torch.equal(y2[0], y[0]) and not torch.allclose(y2[1], y[1])
    m.heads.weight[5 * 20:6 * 20] -= 0.5
    # 6 → 8 ジャンルに広げ、新しいジャンル 6 をジャンル 5 から写す(--init-mode-copy 6:5)
    big = _build(seed=1, num_modes=8).eval()
    state = runtime.expand_modes(m.state_dict(), big.state_dict(), spec.genre_keys, {6: 5})
    big.load_state_dict(state, strict=True)
    xb = dict(x, mode=torch.nn.functional.one_hot(torch.tensor([4, 5]), 8).float())
    assert torch.allclose(big(**xb), m(**x), atol=1e-5)
    x6 = dict(xb, mode=torch.nn.functional.one_hot(torch.tensor([6, 6]), 8).float())
    x5 = dict(xb, mode=torch.nn.functional.one_hot(torch.tensor([5, 5]), 8).float())
    assert torch.allclose(big(**x6), big(**x5), atol=1e-6)


# ---------------------------------------------------------------- 学習(勾配・損失が下がる・符号の決まり・AMP)
def test_gradients_reach_every_layer():
    """学習の1ステップで、勾配がすべての層(4段の交換・ToF の枝・画像の枝・後の融合・ジャンル)に届く"""
    m = _build().train()
    out = runtime.forward(m, _x(batch=2), train=True)
    runtime.l1_per_sample(out['actions'], torch.full((2, 10, 2), 0.2)).mean().backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    assert not missing, missing
    assert all(m.fuse[k].pos.grad.abs().sum() > 0 for k in m.fuse)
    for p in (m.img_stem[0].weight, m.tof.stem[0].weight, m.mode_embed.weight, m.query, m.heads.weight):
        assert p.grad.abs().sum() > 0


def test_tiny_batch_overfits_and_keeps_sign_conventions():
    """この方策は正解の作り方を変えない(v0 と同じ actions の L1。ラベルの計算は持たない)。正解の符号の決まり
    (ステア > 中心 = 左、スロットル < 中立 = 前進)を、合成の2サンプル(左へ前進・右へ前進)で覚えるか(runtime の学習の経路で)。
    CPU で速く回すため、早い段の画像(stem・layer1・layer2)を凍結し、交換は段3・4 だけにする"""
    m = _build(scales=[3, 4]).train()
    for p in itertools.chain(m.img_stem.parameters(), m.img_stages[0].parameters(), m.img_stages[1].parameters()):
        p.requires_grad_(False)
    x = _x(batch=2, genres=(4, 5))
    us = torch.tensor([[STEER_CENTER + 200.0, NEUTRAL - 150.0],       # 左へ・前進
                       [STEER_CENTER - 200.0, NEUTRAL - 150.0]])      # 右へ・前進
    target = ((us - 1500.0) / 500.0).view(2, 1, 2).expand(2, 10, 2).contiguous()
    opt = torch.optim.AdamW(runtime.param_groups(m, 3e-3), lr=3e-3, weight_decay=1e-4)
    losses = []
    for _ in range(8):
        out = runtime.forward(m, x, target=target, train=True)
        loss = runtime.sample_loss(m, out, target, None).mean()     # policy_loss が無いので L1(v0 と同じ)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    with torch.no_grad():
        pred = runtime.forward(m, x)['actions'] * 500.0 + 1500.0    # [2,10,2] µs
    final = float((pred - (target * 500.0 + 1500.0)).abs().mean() / 500.0)
    assert final < 0.5 * losses[0], (losses, final)
    mean = pred.mean(dim=1)                                          # 10 ステップの平均 [2, (ステア, スロットル)]
    assert mean[0, 0] > STEER_CENTER and mean[1, 0] < STEER_CENTER, mean      # 左・右
    assert (mean[:, 1] < NEUTRAL).all(), mean                                 # 前進(中立より下)


def test_stage_fusion_under_mixed_precision():
    """学習は GPU の AMP(FP16)。交換の足し戻し(Conv の低い精度 + LayerNorm の FP32)・連結で型が混ざっても通るか
    (CPU の bfloat16 の autocast で。モデル全体の確認は CPU では遅いので段1つ)"""
    f = msf.StageFusion(128, (28, 28), (12, 14), layers=1, heads=4, ffn_mult=1, dropout=0.0)
    fi = torch.randn(2, 128, 28, 28, requires_grad=True)
    ft = torch.randn(2, 128, 12, 14, requires_grad=True)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        oi, ot = f(fi.bfloat16(), ft)
        loss = oi.float().abs().mean() + ot.float().abs().mean()
    loss.backward()
    assert oi.shape == fi.shape and ot.shape == ft.shape and torch.isfinite(loss)
    assert f.pos.grad is not None and torch.isfinite(f.pos.grad).all()


# ---------------------------------------------------------------- 書き出し・読み込み
def _ops(path):
    g = onnx.load(str(path), load_external_data=False)
    return g, collections.Counter(n.op_type for n in g.graph.node)


def _attrs(node):
    return {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}


def test_onnx_export_small_config(tmp_path):
    m = _build()
    io = runtime.export_onnx(m, tmp_path / 'jetformer.onnx', NUM_MODES)
    assert io['inputs'] == list(policies.BASE_INPUTS) and io['outputs'] == ['actions']
    assert io['output_shapes'] == {'actions': [1, 10, 2]} and io['input_shapes']['mode'] == [1, NUM_MODES]
    assert policies.check_io(io['input_shapes'], io['output_shapes'], NUM_MODES) == (True, '')
    assert not (tmp_path / 'jetformer.onnx.tmp').exists()
    onnx.checker.check_model(str(tmp_path / 'jetformer.onnx'))
    g, ops = _ops(tmp_path / 'jetformer.onnx')
    assert [o.version for o in g.opset_import if o.domain in ('', 'ai.onnx')] == [17]
    assert not [t for t in ops if 'Adaptive' in t or t in ('ATen', 'GlobalAveragePool', 'Loop', 'If', 'GRU')]
    pools = [_attrs(n) for n in g.graph.node if n.op_type == 'AveragePool']
    # 平均の窓: 画像 8・4・2、ToF 4・2(段3 の ToF と段4 は 1×1 なので平均しない)
    assert sorted(tuple(a['kernel_shape']) for a in pools) == [(2, 2), (2, 2), (4, 4), (4, 4), (8, 8)]
    assert all(a['strides'] == a['kernel_shape'] and not any(a.get('pads', [0])) for a in pools)
    resizes = [_attrs(n) for n in g.graph.node if n.op_type == 'Resize']
    assert len(resizes) == 5
    for a in resizes:                     # antialias なしの双線形(align_corners=False)
        assert a['mode'] == b'linear' and not a.get('antialias', 0)
        assert a['coordinate_transformation_mode'] in (b'half_pixel', b'pytorch_half_pixel')


def test_onnx_export_pooled_head(tmp_path):
    m = _build(head='pooled', scales=[3, 4])
    io = runtime.export_onnx(m, tmp_path / 'p.onnx', NUM_MODES)
    assert io['inputs'] == list(policies.BASE_INPUTS) and io['output_shapes'] == {'actions': [1, 10, 2]}
    _, ops = _ops(tmp_path / 'p.onnx')
    assert ops['ReduceMean'] >= 2 and ops['Resize'] == 1 and ops['AveragePool'] == 1     # 段3: 画像 2×2 だけ(ToF は 1×1)
    assert not [t for t in ops if 'Adaptive' in t or t in ('ATen', 'GlobalAveragePool')]


def test_load_run_roundtrip_and_pretrained_flag(tmp_path, monkeypatch):
    spec = policies.get('msfusion')
    m = _build(seed=5).eval()
    cfg = {'num_modes': NUM_MODES, 'trained_modes': [4, 5], 'policy': 'msfusion', 'policy_cfg': spec.resolve(SMALL),
           'contract': policies.CONTRACT}
    (tmp_path / 'model_config.json').write_text(json.dumps(cfg), encoding='utf-8')
    torch.save(m.state_dict(), tmp_path / 'best.pth')

    import torchvision.models._api as tv_api

    def no_download(*a, **k):
        raise AssertionError('pretrained=False で ImageNet の重みを読もうとしました')
    monkeypatch.setattr(tv_api, 'load_state_dict_from_url', no_download)          # load_run は pretrained=False
    x = {k: v.numpy() for k, v in _x(batch=1).items()}
    ref = runtime.predict(m, x)['actions']
    model, _, s, pcfg = runtime.load_run(tmp_path, ckpt='best')        # strict=True で全部のキーが合う
    assert s.name == 'msfusion' and pcfg == spec.resolve(SMALL) and isinstance(model, msf.MSFusion)
    np.testing.assert_allclose(runtime.predict(model, x)['actions'], ref, atol=1e-6)
    assert policies.drivable(cfg) == (True, '')

    torch.manual_seed(9)
    fake = torchvision.models.resnet18().state_dict()          # 偽の「ImageNet の重み」(ファイル・ネットに触れない)
    monkeypatch.setattr(tv_api, 'load_state_dict_from_url', lambda *a, **k: fake)
    p = msf.build(NUM_MODES, pretrained=True, **SMALL)
    assert torch.equal(p.img_stem[0].weight, fake['conv1.weight'])
    assert torch.equal(p.img_stages[3][1].conv2.weight, fake['layer4.1.conv2.weight'])
