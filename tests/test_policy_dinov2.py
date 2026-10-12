"""DINOv2 の画像特徴の方策(jetformer/policies/dinov2.py・vit.py)のテスト。CPU だけ・ネットなしで通る。
公式の重みの形は合成したもので確かめる(実際の重み data/pretrained/dinov2_vits14_pretrain.pth があれば、それも読む。無ければ skip)。

    cd ~/jetformer && CUDA_VISIBLE_DEVICES= nice python -m pytest -q tests/test_policy_dinov2.py
"""

import json
import os
import sys

import pytest
import torch
import torch.nn.functional as F

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))
from jetformer import policies  # noqa: E402
from jetformer.policies import dinov2, runtime, vit  # noqa: E402

REAL_WEIGHTS = os.path.join(ROOT, dinov2.DEFAULT_WEIGHTS)
TINY = {'vit_depth': 1, 'dim': 32, 'layers': 1, 'heads': 4}      # 速く試すための小さな形(ViT は 1 ブロック)
# 公式のソース(github.com/facebookresearch/dinov2 7764ea0、hub の dinov2_vits14 と同じ設定)で、実際の重み・
# torch.randn(1,3,224,224, Generator().manual_seed(0)) から出した値(forward_features の x_norm_*)。vit.py はこれとビット単位で一致した
OFFICIAL_FINGERPRINT = {'cls8': [0.53731, -0.07742, 2.41219, -1.80327, -3.55929, 3.21084, 3.35636, -0.99181],
                        'p0_4': [0.97058, -1.78708, -0.5451, -1.182], 'p255_4': [1.27953, 1.48248, 0.48912, -2.10582],
                        'patch_mean': 0.11247, 'patch_std': 2.499526, 'cls_norm': 48.0154}


def _x(batch=2, num_modes=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    return dict(image=torch.randn(batch, 3, 224, 224, generator=g), tof=torch.rand(batch, 2, 42, 54, generator=g),
                imu=torch.randn(batch, 10, 6, generator=g) * 0.1, state=torch.randn(batch, 3, generator=g) * 0.1,
                mode=F.one_hot(torch.tensor([4, 5] * batch)[:batch], num_modes).float())


def _tiny(num_modes=6, **over):
    return policies.build_policy('dinov2', num_modes, dict(TINY, **over))


def _official_like(depth, seed=0):
    """公式の dinov2_vits14_pretrain.pth と同じキー・同じ形の合成の重み(mask_token [1,384]・pos_embed [1,1370,384] を含む)"""
    torch.manual_seed(seed)
    sd = vit.DinoViTS14(img_size=vit.OFFICIAL_IMG_SIZE, depth=depth).state_dict()
    sd = {k: v.clone() for k, v in sd.items()}
    for k in sd:
        if k.endswith('gamma'):
            sd[k] = torch.rand_like(sd[k])
    sd['mask_token'] = torch.zeros(1, 384)
    return sd


# ---------------------------------------------------------------- 登録簿
def test_registry_entry_is_ready_and_not_a_stub():
    spec = policies.get('dinov2')
    with open(os.path.join(ROOT, 'jetformer', 'policies', 'dinov2.py'), encoding='utf-8') as f:
        assert policies.STUB_MARK not in f.read()
    ok, why = spec.ready()
    if os.path.exists(REAL_WEIGHTS):
        assert (ok, why) == (True, '')
    else:
        assert ok is False and 'dinov2_vits14_pretrain.pth' in why
    assert spec.requires_pretrained and spec.genre_keys == {'embed': 'mode_embed.weight', 'heads': {'heads': 20}}
    assert set(spec.train_only_keys) == {'finetune_last', 'lr_mult'}
    assert spec.defaults['weights'] == dinov2.DEFAULT_WEIGHTS


def test_default_size_and_tokens():
    m = policies.build_policy('dinov2', 6)                       # 既定の形(ViT 12 ブロック・pool 2)、重みは読まない
    total, trainable = runtime.count_params(m)
    assert abs(total / 1e6 - 24.5) <= 0.2 and abs(trainable / 1e6 - 2.9) <= 0.2
    assert m.n_tokens == 111 and list(m.pos.shape) == [1, 111, 256] and m.vit.pos_embed.shape == (1, 257, 384)
    assert list(m.state_dict())[:2] == ['query', 'pos'] and 'heads.weight' in m.state_dict()


# ---------------------------------------------------------------- 公式の重みの読み込み(vit.py)
def test_load_official_synthetic_strict(tmp_path):
    src = _official_like(depth=2)
    path = tmp_path / 'official.pth'
    torch.save(src, path)
    m = vit.DinoViTS14(img_size=224, depth=2).load_official(path)
    assert m.pos_embed.shape == (1, 257, 384)
    assert torch.equal(m.pos_embed[:, :1], src['pos_embed'][:, :1])          # CLS の位置はそのまま
    assert torch.equal(m.blocks[1].mlp.fc2.weight, src['blocks.1.mlp.fc2.weight'])
    assert torch.equal(m.blocks[0].ls1.gamma, src['blocks.0.ls1.gamma'])
    assert 'mask_token' not in m.state_dict()
    # 足りない・余るキーは strict=True で失敗する。レジスタ版は分かる例外
    bad = dict(src)
    bad.pop('blocks.1.attn.qkv.bias')
    torch.save(bad, tmp_path / 'missing.pth')
    with pytest.raises(RuntimeError):
        vit.DinoViTS14(img_size=224, depth=2).load_official(tmp_path / 'missing.pth')
    torch.save(dict(src, register_tokens=torch.zeros(1, 4, 384)), tmp_path / 'reg.pth')
    with pytest.raises(ValueError, match='レジスタ'):
        vit.DinoViTS14(img_size=224, depth=2).load_official(tmp_path / 'reg.pth')


def test_pos_embed_interpolation_is_the_official_formula():
    g = torch.Generator().manual_seed(0)
    pos = torch.randn(1, 1 + 37 * 37, 8, generator=g)
    got = vit.DinoViTS14.interpolate_pos(pos, 16)
    grid = pos[:, 1:].reshape(1, 37, 37, 8).permute(0, 3, 1, 2)
    ref = F.interpolate(grid, scale_factor=(16.1 / 37, 16.1 / 37), mode='bicubic', antialias=False)   # 公式: (16 + 0.1) / 37
    assert torch.allclose(got[:, 1:], ref.permute(0, 2, 3, 1).reshape(1, 256, 8), atol=1e-6)
    by_size = F.interpolate(grid, size=(16, 16), mode='bicubic', antialias=False)                   # size だけ渡すのとは違う
    assert not torch.allclose(got[:, 1:], by_size.permute(0, 2, 3, 1).reshape(1, 256, 8), atol=1e-4)
    const = torch.full((1, 1 + 37 * 37, 4), 0.7)
    assert torch.allclose(vit.DinoViTS14.interpolate_pos(const, 16), torch.full((1, 257, 4), 0.7), atol=1e-6)
    assert vit.DinoViTS14.interpolate_pos(pos[:, :257], 16).shape == (1, 257, 8)                    # 同じ格子はそのまま


@pytest.mark.skipif(not os.path.exists(REAL_WEIGHTS), reason='DINOv2 の重みが無い')
def test_real_weights_match_official_fingerprint():
    info = dinov2.prepare(None, policies.get('dinov2').resolve(None))
    assert info['ok'] and info['sha256'] == dinov2.WEIGHTS_SHA256
    m = vit.DinoViTS14(img_size=224).load_official(REAL_WEIGHTS).eval()
    x = torch.randn(1, 3, 224, 224, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        h = m(x)[0]
    fp = OFFICIAL_FINGERPRINT
    assert torch.allclose(h[0, :8], torch.tensor(fp['cls8']), atol=2e-3)
    assert torch.allclose(h[1, :4], torch.tensor(fp['p0_4']), atol=2e-3)
    assert torch.allclose(h[256, :4], torch.tensor(fp['p255_4']), atol=2e-3)
    assert abs(float(h[1:].mean()) - fp['patch_mean']) < 1e-3 and abs(float(h[1:].std()) - fp['patch_std']) < 1e-3
    assert abs(float(h[0].norm()) - fp['cls_norm']) < 1e-2


# ---------------------------------------------------------------- 重みのファイルと build
def test_build_without_pretrained_never_touches_weights(monkeypatch):
    def boom(*a, **k):
        raise AssertionError('pretrained=False で重みのファイルに触れた')
    monkeypatch.setattr(torch, 'load', boom)
    monkeypatch.setattr(vit.DinoViTS14, 'load_official', boom)
    monkeypatch.setattr(dinov2, 'weights_path', boom)
    monkeypatch.setattr(dinov2, 'file_sha256', boom)
    m = policies.build_policy('dinov2', 6, dict(TINY, weights='/no/such/dir/dinov2.pth'))
    assert isinstance(m, dinov2.DinoV2Policy)


def test_build_pretrained_errors_and_synthetic_load(tmp_path):
    missing = tmp_path / 'none.pth'
    with pytest.raises(FileNotFoundError, match='DINOv2 の重みがありません') as e:
        dinov2.build(6, pretrained=True, weights=str(missing))
    assert str(missing) in str(e.value)
    with pytest.raises(ValueError, match='vit_depth'):
        dinov2.build(6, pretrained=True, vit_depth=1, weights=str(missing))
    with pytest.raises(ValueError, match='variant'):
        dinov2.build(6, variant='vits14_reg4')
    for bad in ({'pool': 3}, {'finetune_last': 2, 'vit_depth': 1}, {'lr_mult': 0.0}):
        with pytest.raises(ValueError):
            _tiny(**bad)
    # 公式と同じ形の 12 ブロックの合成の重み → pretrained=True で読む(絶対パスの weights)
    src = _official_like(depth=12, seed=3)
    path = tmp_path / 'synthetic_vits14.pth'
    torch.save(src, path)
    m = dinov2.build(6, pretrained=True, weights=str(path), dim=32, layers=1, heads=4)
    assert torch.equal(m.vit.blocks[11].attn.proj.weight, src['blocks.11.attn.proj.weight'])
    assert m.vit.pos_embed.shape == (1, 257, 384) and not any(p.requires_grad for p in m.vit.parameters())


def test_prepare_checks_weights(tmp_path, monkeypatch, capsys):
    spec = policies.get('dinov2')
    with pytest.raises(SystemExit, match='DINOv2 の重みがありません'):
        dinov2.prepare('unused', spec.resolve({'weights': str(tmp_path / 'none.pth')}))
    with pytest.raises(SystemExit, match='vit_depth'):
        dinov2.prepare('unused', spec.resolve({'vit_depth': 1}))
    with pytest.raises(SystemExit, match='variant'):
        dinov2.prepare('unused', spec.resolve({'variant': 'vitb14'}))
    fake = tmp_path / 'fake.pth'
    fake.write_bytes(b'not the official weights')
    calls = []
    real = dinov2.file_sha256
    monkeypatch.setattr(dinov2, 'file_sha256', lambda p: calls.append(p) or real(p))
    info = dinov2.prepare('unused', spec.resolve({'weights': str(fake)}))          # sha256 が違えば警告して続ける
    assert info['ok'] is False and '警告' in capsys.readouterr().out
    assert dinov2.prepare('unused', spec.resolve({'weights': str(fake)}))['sha256'] == info['sha256']
    assert len(calls) == 1                                                           # 2回目は計算し直さない


# ---------------------------------------------------------------- 凍結・学習率の群・train()
def test_frozen_vit_is_not_trained():
    m = _tiny(vit_depth=2)
    assert not any(p.requires_grad for p in m.vit.parameters())
    groups = runtime.param_groups(m, 1e-3)
    assert len(groups) == 1 and 'lr' not in groups[0]
    vit_ids = {id(p) for p in m.vit.parameters()}
    assert not any(id(p) in vit_ids for p in groups[0]['params'])
    assert len(groups[0]['params']) == len([p for p in m.parameters() if p.requires_grad])
    m.train()
    out = runtime.forward(m, _x(), train=True)['actions']
    out.abs().mean().backward()
    assert all(p.grad is None for p in m.vit.parameters())
    assert m.image_proj[1].weight.grad is not None and m.heads.weight.grad is not None
    assert m.image_tokens(_x()['image']).requires_grad is False                    # 凍結の特徴は計算グラフを持たない


def test_finetune_last_two_groups_with_lower_lr():
    m = _tiny(vit_depth=3, finetune_last=2)
    groups = m.param_groups(2e-3)
    assert len(groups) == 2 and groups[1]['lr'] == pytest.approx(2e-4)
    tuned = {n for n, p in m.vit.named_parameters() if p.requires_grad}
    assert tuned == {n for n, _ in m.vit.named_parameters() if n.startswith(('blocks.1.', 'blocks.2.', 'norm.'))}
    assert {id(p) for p in groups[1]['params']} == {id(p) for n, p in m.vit.named_parameters() if n in tuned}
    m2 = _tiny(vit_depth=3, finetune_last=2, lr_mult=0.5)
    assert m2.param_groups(1.0)[1]['lr'] == pytest.approx(0.5)
    m.train()
    runtime.forward(m, _x(), train=True)['actions'].abs().mean().backward()
    assert m.vit.blocks[0].attn.qkv.weight.grad is None and m.vit.patch_embed.proj.weight.grad is None
    assert m.vit.blocks[2].mlp.fc1.weight.grad is not None and m.vit.norm.weight.grad is not None
    opt = torch.optim.AdamW(runtime.param_groups(m, 3e-4), lr=3e-4, weight_decay=1e-4)   # train.py と同じ作り方
    assert [g['lr'] for g in opt.param_groups] == [pytest.approx(3e-4), pytest.approx(3e-5)]


def test_train_keeps_vit_in_eval():
    m = _tiny(vit_depth=2)
    m.train()
    assert m.training and m.fusion.training and m.tof_encoder.training
    assert not m.vit.training and not any(sub.training for sub in m.vit.modules())
    m.eval()
    assert not any(sub.training for sub in m.modules())
    m.train(True)
    assert not m.vit.training


def test_init_compat_allows_lp_ft_but_not_structure_changes():
    import train
    spec = policies.get('dinov2')
    frozen, ft = spec.resolve(None), spec.resolve({'finetune_last': 2, 'lr_mult': 0.2})
    train.check_init_compat({'policy': 'dinov2', 'policy_cfg': frozen}, 'dinov2', ft, spec)     # LP-FT の2段目
    with pytest.raises(SystemExit):
        train.check_init_compat({'policy': 'dinov2', 'policy_cfg': frozen}, 'dinov2', spec.resolve({'pool': 1}), spec)


# ---------------------------------------------------------------- トークン・形・並び
@pytest.mark.parametrize('cfg, n_img, n_tokens', [({'pool': 2}, 65, 111), ({'pool': 1}, 257, 303),
                                                  ({'pool': 2, 'use_cls': False}, 64, 110)])
def test_token_counts_and_forward_shape(cfg, n_img, n_tokens):
    m = _tiny(**cfg).eval()
    assert m.n_tokens == n_tokens and list(m.pos.shape) == [1, n_tokens, 32]
    x = _x()
    with torch.no_grad():
        assert m.image_tokens(x['image']).shape == (2, n_img, 384)
        y = runtime.forward(m, x)['actions']
        assert y.shape == (2, 10, 2) and torch.isfinite(y).all()
        assert m.embed(**x).shape == (2, 32)


def test_image_token_layout_keeps_left_and_right():
    """画像の左上・右上・左下を変えると、平均したトークンの格子(行の順、各行は左から)の対応する1つだけが変わる(左右の反転なし)。
    ViT のブロック 0 個 = 注意で混ざらないので、各トークンは自分の 28×28 px の窓だけで決まる"""
    m = _tiny(vit_depth=0).eval()
    base = torch.zeros(1, 3, 224, 224)
    with torch.no_grad():
        ref = m.image_tokens(base)
        for (r, c), idx in (((0, 0), 1), ((0, 196), 1 + 7), ((196, 0), 1 + 56), ((112, 140), 1 + 4 * 8 + 5)):
            img = base.clone()
            img[:, :, r:r + 28, c:c + 28] = 1.0
            diff = (m.image_tokens(img) - ref).abs().amax(dim=-1)[0]
            assert diff[idx] > 1e-3 and torch.count_nonzero(diff > 1e-6) == 1, (r, c, idx)
    p = torch.arange(16 * 16, dtype=torch.float32).view(1, 256, 1)                 # パッチの番号 = 行 × 16 + 列
    pooled = dinov2.pool_patches(p, 16, 2)[0, :, 0]
    assert pooled[0] == (0 + 1 + 16 + 17) / 4 and pooled[7] == (14 + 15 + 30 + 31) / 4 and pooled[8] == (32 + 33 + 48 + 49) / 4


def test_genre_heads_are_20_rows_per_genre():
    m = _tiny().eval()
    x = _x()                                                     # 1つ目はジャンル 4、2つ目は 5
    with torch.no_grad():
        before = m(**x)
        m.heads.weight[5 * 20:6 * 20] = 0.0
        m.heads.bias[5 * 20:6 * 20] = 0.0
        after = m(**x)
    assert torch.equal(after[0], before[0]) and torch.count_nonzero(after[1]) == 0


# ---------------------------------------------------------------- 学習(小さなバッチを覚えられる)
def test_loss_decreases_on_tiny_batch():
    torch.manual_seed(0)
    m = _tiny(vit_depth=0)                                      # ViT は凍結なので、学習するのは頭だけ
    x = _x(batch=2, seed=1)
    target = torch.rand(2, 10, 2, generator=torch.Generator().manual_seed(2)) - 0.5
    vit_before = {k: v.clone() for k, v in m.vit.state_dict().items()}
    opt = torch.optim.AdamW(runtime.param_groups(m, 5e-3), lr=5e-3, weight_decay=1e-4)
    losses = []
    for _ in range(20):
        m.train()
        out = runtime.forward(m, x, target=target, train=True)
        loss = runtime.sample_loss(m, out, target, None).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < 0.5 * losses[0], losses
    assert all(torch.equal(v, m.vit.state_dict()[k]) for k, v in vit_before.items())      # 凍結した ViT は変わらない


# ---------------------------------------------------------------- 書き出し・読み込み
def test_onnx_export_contract(tmp_path):
    import onnx
    m = _tiny()
    io = runtime.export_onnx(m, tmp_path / 'jetformer.onnx', 6)
    assert io['inputs'] == list(policies.BASE_INPUTS) and io['outputs'] == ['actions']
    assert io['output_shapes'] == {'actions': [1, 10, 2]} and io['input_shapes']['mode'] == [1, 6]
    assert policies.check_io(io['input_shapes'], io['output_shapes'], 6) == (True, '')
    ops = {n.op_type for n in onnx.load(str(tmp_path / 'jetformer.onnx'), load_external_data=False).graph.node}
    assert 'AveragePool' in ops and 'Erf' in ops and 'Softmax' in ops
    forbidden = {'Resize', 'GlobalAveragePool', 'Loop', 'If', 'GRU', 'LSTM', 'Scan'}       # 補間・ループはグラフに無い
    assert not ops & forbidden, ops & forbidden
    assert not (tmp_path / 'jetformer.onnx.tmp').exists()


def test_load_run_roundtrip(tmp_path):
    torch.manual_seed(5)
    m = _tiny(vit_depth=2, finetune_last=2)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.01)
    torch.save(m.state_dict(), tmp_path / 'best.pth')
    pcfg = policies.get('dinov2').resolve(dict(TINY, vit_depth=2, finetune_last=2))
    cfg = {'num_modes': 6, 'policy': 'dinov2', 'policy_cfg': pcfg, 'inputs': list(policies.BASE_INPUTS), 'outputs': ['actions']}
    (tmp_path / 'model_config.json').write_text(json.dumps(cfg), encoding='utf-8')
    got, cfg2, spec, pcfg2 = runtime.load_run(tmp_path)
    assert isinstance(got, dinov2.DinoV2Policy) and spec.name == 'dinov2' and pcfg2 == pcfg
    x = _x()
    a, b = runtime.predict(m, x), runtime.predict(got, x)
    assert list(a) == ['actions'] and (abs(a['actions'] - b['actions']).max() == 0)
    assert policies.drivable(cfg2) == (True, '')
