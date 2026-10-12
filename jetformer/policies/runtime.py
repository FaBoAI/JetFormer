"""方策の torch 側の共通の処理(学習・書き出し・評価の道具が使う。走行は TensorRT のエンジンだけを使うので読まない)

    model, cfg, spec, pcfg = runtime.load_run('data/runs/<学習結果>', ckpt='best', device='cuda')
    out = runtime.predict(model, {'image': ..., 'tof': ..., 'imu': ..., 'state': ..., 'mode': ...})   # FP32 → {名前: np}
    pred_us = out['actions'] * 500 + 1500

方策は forward(image, tof, imu, state, mode) をキーワードで呼ばれ、Tensor [B,10,2] か {'actions': [B,10,2], ...} を返す
(spec.md §2.2)。任意のフック: EXPORT_OUTPUTS・USES_TARGET・policy_loss・loss_parts・param_groups・embed。
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import BASE_INPUTS, FUTURE_STEPS, check_io, from_config

EXPORT_GENRE = 1                  # 書き出しの例の入力のジャンル(今までの train.export_onnx と同じ)


# ---------------------------------------------------------------- 呼び方
def as_outputs(y):
    """方策の戻り値 → {'actions': ..., ...}"""
    if isinstance(y, torch.Tensor):
        return {'actions': y}
    if isinstance(y, dict):
        if 'actions' not in y:
            raise ValueError('方策の出力に actions がありません(出力: %s)' % ', '.join(y))
        return dict(y)
    raise TypeError('方策の出力は Tensor か {名前: Tensor} にしてください(%s)' % type(y).__name__)


def forward(model, inputs, target=None, train=False):
    """inputs: {名前: Tensor}(基本の5つ)。キーワードで呼ぶ → dict。学習中で USES_TARGET の方策だけ target= を渡す"""
    kw = {k: inputs[k] for k in BASE_INPUTS}
    if train and target is not None and getattr(model, 'USES_TARGET', False):
        kw['target'] = target
    return as_outputs(model(**kw))


def l1_per_sample(actions, target):
    """サンプルごとの L1 [B](報告する誤差。どの方策でも同じ)"""
    return (actions.float() - target).abs().mean(dim=(1, 2))


def sample_loss(model, out, target, extras):
    """学習の損失 [B]。方策の policy_loss があればそれ、無ければ L1"""
    fn = getattr(model, 'policy_loss', None)
    if fn is None:
        return l1_per_sample(out['actions'], target)
    per = fn(out, target, extras)
    if per.dim() != 1 or per.shape[0] != target.shape[0]:
        raise RuntimeError('policy_loss はサンプルごとの損失 [B] を返してください(%s)' % list(per.shape))
    return per.float()


def loss_parts(model):
    """方策が直前の policy_loss で置いた内訳 {名前: float}(無ければ {})"""
    parts = getattr(model, 'loss_parts', None) or {}
    return {k: float(v.detach().float().mean()) if isinstance(v, torch.Tensor) else float(v) for k, v in parts.items()}


def param_groups(model, lr):
    """AdamW に渡すパラメータ(群)。方策の param_groups(lr) があればそれ、無ければ学習するパラメータの1群
    (v0 は model.parameters() と同じ並び = 今までの AdamW と同じ)"""
    fn = getattr(model, 'param_groups', None)
    if callable(fn):
        return fn(lr)
    return [p for p in model.parameters() if p.requires_grad]


# ---------------------------------------------------------------- 確かめ・書き出し
def example_inputs(num_modes, batch=1, seed=0, device='cpu', genre=None):
    """基本の5つの例(書き出し・確かめ用。大域の乱数は使わない)。mode は one-hot(ジャンル 5、無ければ最後)"""
    g = torch.Generator().manual_seed(seed)
    genre = min(5, num_modes - 1) if genre is None else genre
    mode = torch.zeros(batch, num_modes)
    mode[:, genre] = 1.0
    x = {'image': torch.randn(batch, 3, 224, 224, generator=g), 'tof': torch.rand(batch, 2, 42, 54, generator=g),
         'imu': torch.randn(batch, 10, 6, generator=g) * 0.1, 'state': torch.randn(batch, 3, generator=g) * 0.1,
         'mode': mode}
    return {k: v.to(device) for k, v in x.items()}


def export_names(model):
    names = tuple(getattr(model, 'EXPORT_OUTPUTS', ('actions',)))
    if not names or names[0] != 'actions':
        raise ValueError('EXPORT_OUTPUTS の先頭は actions にしてください: %s' % (names,))
    return names


@torch.no_grad()
def check_forward(model, num_modes, device='cpu', batch=2):
    """学習の前に1回(30 分の学習の後で書き出しに失敗しないように)。eval で呼び(BatchNorm の統計を変えない)、
    actions [batch,10,2]・有限・EXPORT_OUTPUTS が全部あるか。違えば RuntimeError"""
    was = model.training
    model.eval()
    try:
        out = forward(model, example_inputs(num_modes, batch=batch, device=device))
    finally:
        model.train(was)
    a = out['actions']
    if list(a.shape) != [batch, FUTURE_STEPS, 2]:
        raise RuntimeError('方策の actions の形 %s が [%d, %d, 2] ではありません' % (list(a.shape), batch, FUTURE_STEPS))
    missing = [n for n in export_names(model) if n not in out]
    if missing:
        raise RuntimeError('EXPORT_OUTPUTS の %s が出力にありません' % missing)
    for n in export_names(model):
        if not torch.isfinite(out[n].float()).all():
            raise RuntimeError('方策の出力 %s に数値でない値があります' % n)
    return out


class ExportWrapper(nn.Module):
    """forward(image, tof, imu, state, mode)(位置の引数)→ EXPORT_OUTPUTS の順のタプル"""

    def __init__(self, model, names):
        super().__init__()
        self.model = model
        self.names = tuple(names)

    def forward(self, image, tof, imu, state, mode):
        out = as_outputs(self.model(image=image, tof=tof, imu=imu, state=state, mode=mode))
        return tuple(out[n] for n in self.names)


def _is_v0(model):
    from ..model import JetFormer
    return type(model) is JetFormer


def read_onnx_io(path):
    """ONNX の実際の入出力 {'inputs': [...], 'input_shapes': {名前: 形}, 'outputs': [...], 'output_shapes': {名前: 形}}。
    固定でない次元は -1"""
    import onnx
    m = onnx.load(str(path), load_external_data=False)
    init = {i.name for i in m.graph.initializer}

    def shape(vi):
        return [int(d.dim_value) if d.HasField('dim_value') else -1 for d in vi.type.tensor_type.shape.dim]
    ins = [vi for vi in m.graph.input if vi.name not in init]
    return {'inputs': [vi.name for vi in ins], 'input_shapes': {vi.name: shape(vi) for vi in ins},
            'outputs': [vi.name for vi in m.graph.output], 'output_shapes': {vi.name: shape(vi) for vi in m.graph.output}}


def export_onnx(model, path, num_modes):
    """ONNX に書き出す(opset 17・TorchScript の書き出し dynamo=False・バッチ1・固定の形・MHA の高速経路なし)。
    入力の名前は基本の5つ、出力は EXPORT_OUTPUTS。path + '.tmp' に書いてから置き換える。
    書いた ONNX を読み直して実際の入出力を返す(使っていない入力はグラフから消える。消えるのは良い)。
    走行の約束(check_io)に合わなければ RuntimeError。v0 は今までと同じグラフにするため包まずに書き出す"""
    path = str(path)
    model = model.float().eval().cpu()
    torch.backends.mha.set_fastpath_enabled(False)          # ONNX に無い融合カーネルを使わせない
    names = export_names(model)
    g = torch.Generator().manual_seed(0)
    mode = torch.zeros(1, num_modes)
    mode[0, min(EXPORT_GENRE, num_modes - 1)] = 1.0
    x = (torch.randn(1, 3, 224, 224, generator=g), torch.rand(1, 2, 42, 54, generator=g), torch.zeros(1, 10, 6),
         torch.zeros(1, 3), mode)
    target = model if _is_v0(model) and names == ('actions',) else ExportWrapper(model, names)
    tmp = path + '.tmp'
    try:
        with torch.no_grad():
            torch.onnx.export(target, x, tmp, input_names=list(BASE_INPUTS), output_names=list(names),
                              opset_version=17, dynamo=False)
        io = read_onnx_io(tmp)
        ok, why = check_io(io['input_shapes'], io['output_shapes'], num_modes)
        if not ok:
            raise RuntimeError('書き出した ONNX が走行の約束に合いません: %s' % why)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return io


# ---------------------------------------------------------------- 学習結果の読み込み
def load_state(path):
    """best.pth(state_dict)・last.pth({'model': ..., 'optimizer': ...})のどちらからもモデルの重み"""
    sd = torch.load(str(path), map_location='cpu', weights_only=True)
    if isinstance(sd, dict) and 'model' in sd and 'optimizer' in sd:
        sd = sd['model']
    return sd


def read_config(run_dir):
    with open(Path(run_dir) / 'model_config.json', encoding='utf-8') as f:
        return json.load(f)


def load_run(run_dir, ckpt='best', device='cpu'):
    """model_config.json → 登録簿(無ければ jetformer_v0)→ build(pretrained=False)→ <ckpt>.pth を strict=True で読む → eval。
    戻り値 (model, cfg, spec, resolved_cfg)"""
    run_dir = Path(run_dir)
    cfg = read_config(run_dir)
    spec, pcfg = from_config(cfg)
    model = spec.build(cfg['num_modes'], pcfg, pretrained=False)
    model.load_state_dict(load_state(run_dir / ('%s.pth' % ckpt)), strict=True)
    return model.to(device).eval(), cfg, spec, pcfg


def expand_modes(state, model_state, genre_keys, copy_from=None):
    """学習済みの重み state を、ジャンル数が多いモデル(model_state)に合わせて広げる(train.expand_mode_state の一般形)。

    genre_keys = {'embed': 'mode_embed.weight', 'heads': {層の名前: 1ジャンルの行の数}}。ジャンルに依存するのはこれらだけ。
    元からあるジャンルはそのまま写し、増えたジャンルは copy_from {新ID: 元ID} の元のジャンルの値で初期化する
    (copy_from に無いものはモデルの初期値のまま)。ジャンル数が同じならそのまま返す"""
    copy_from = dict(copy_from or {})
    ek = genre_keys['embed']
    old_n = state[ek].shape[1]
    new_n = model_state[ek].shape[1]
    if old_n == new_n and not copy_from:
        return state
    if new_n < old_n:
        raise ValueError('ジャンル数が減っています(%d → %d)' % (old_n, new_n))
    for dst, src in copy_from.items():
        if not (old_n <= dst < new_n) or not (0 <= src < old_n):
            raise ValueError('--init-mode-copy %d:%d: 新しいジャンルは %d〜%d、元は 0〜%d' % (dst, src, old_n, new_n - 1, old_n - 1))
    out = dict(state)
    emb = model_state[ek].clone()
    emb[:, :old_n] = state[ek]
    for dst, src in copy_from.items():
        emb[:, dst] = emb[:, src]
    out[ek] = emb
    for layer, rows in genre_keys['heads'].items():
        wk, bk = layer + '.weight', layer + '.bias'
        hw = model_state[wk].clone().view(new_n, rows, -1)
        hw[:old_n] = state[wk].view(old_n, rows, -1)
        for dst, src in copy_from.items():
            hw[dst] = hw[src]
        out[wk] = hw.view(new_n * rows, -1)
        if bk in model_state:
            hb = model_state[bk].clone().view(new_n, rows)
            hb[:old_n] = state[bk].view(old_n, rows)
            for dst, src in copy_from.items():
                hb[dst] = hb[src]
            out[bk] = hb.view(new_n * rows)
    return out


# ---------------------------------------------------------------- 評価
def _to_tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device, torch.float32)
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(device)


@torch.no_grad()
def predict(model, inputs):
    """FP32(autocast なし)・eval → {名前: np.ndarray float32}(EXPORT_OUTPUTS だけ。actions は正規化のまま。
    バッチの次元つき = エンジンの run_numpy と同じ形)。inputs は numpy か Tensor の基本の5つ"""
    model.eval()
    p = next(model.parameters(), None)
    device = p.device if p is not None else torch.device('cpu')
    out = forward(model, {k: _to_tensor(inputs[k], device) for k in BASE_INPUTS})
    return {k: out[k].float().cpu().numpy() for k in export_names(model)}      # エンジンと同じ出力だけ


def count_params(model):
    """(全体, 学習するもの)"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def params_m(model):
    total, trainable = count_params(model)
    return {'total': round(total / 1e6, 2), 'trainable': round(trainable / 1e6, 2)}
