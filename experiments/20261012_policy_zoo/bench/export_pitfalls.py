# -*- coding: utf-8 -*-
"""ONNX 書き出し(torch 2.10・TorchScript の書き出し dynamo=False・opset 17)の落とし穴を、小さなモジュールで確かめる(CPU だけ)

    CUDA_VISIBLE_DEVICES= python export_pitfalls.py      → results/export_pitfalls.json
"""

import collections
import io
import os
import warnings

import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import RESULTS, save_json


def try_export(module, args, input_names=None, output_names=None):
    buf = io.BytesIO()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            with torch.no_grad():
                torch.onnx.export(module.eval(), args, buf, input_names=input_names, output_names=output_names,
                                  opset_version=17, dynamo=False)
        except Exception as e:      # noqa: BLE001
            return {'ok': False, 'error': ('%s: %s' % (type(e).__name__, str(e).split('\n')[0]))[:300]}
    m = onnx.load_from_string(buf.getvalue())
    ops = collections.Counter(n.op_type for n in m.graph.node)
    return {'ok': True, 'ops': dict(ops.most_common(12)), 'inputs': [v.name for v in m.graph.input],
            'warnings': sorted(set(str(w.message).split('\n')[0][:160] for w in caught
                                   if 'legacy TorchScript' not in str(w.message) and 'will be removed' not in str(w.message)))}


class Fn(nn.Module):
    """関数を包む。sub に渡したモジュールは子として登録する(登録しないと重みが定数扱いになり書き出しが失敗する)"""

    def __init__(self, fn, sub=None):
        super().__init__()
        self.fn = fn
        self.sub = sub

    def forward(self, *x):
        return self.fn(self.sub, *x) if self.sub is not None else self.fn(*x)


def main():
    out = {}
    x28 = torch.randn(1, 64, 28, 28)
    # 1. adaptive_avg_pool2d: 割り切れない大きさ(TransFuser の 8x8 の格子を 224 入力の 28x28・7x7 に当てる)
    out['adaptive_pool_28_to_8'] = try_export(Fn(lambda x: F.adaptive_avg_pool2d(x, (8, 8))), (x28,))
    out['adaptive_pool_28_to_7'] = try_export(Fn(lambda x: F.adaptive_avg_pool2d(x, (7, 7))), (x28,))
    out['adaptive_pool_7_to_8'] = try_export(Fn(lambda x: F.adaptive_avg_pool2d(x, (8, 8))), (torch.randn(1, 8, 7, 7),))
    # 2. 縮小の antialias(opset 18 から)
    img = torch.randn(1, 3, 224, 224)
    out['interpolate_antialias'] = try_export(
        Fn(lambda x: F.interpolate(x, size=(168, 168), mode='bilinear', antialias=True)), (img,))
    out['interpolate_bilinear'] = try_export(
        Fn(lambda x: F.interpolate(x, size=(168, 168), mode='bilinear', align_corners=False)), (img,))
    # 3. nn.TransformerEncoderLayer の高速経路(fastpath)を切らずに書き出す
    layer = nn.TransformerEncoderLayer(64, 4, 128, batch_first=True, norm_first=True).eval()
    tok = torch.randn(1, 10, 64)
    mha = nn.MultiheadAttention(64, 4, batch_first=True).eval()
    torch.backends.mha.set_fastpath_enabled(True)
    out['encoder_layer_fastpath_on'] = try_export(Fn(lambda m, x: m(x), layer), (tok,))
    out['mha_fastpath_on'] = try_export(Fn(lambda m, q: m(q, q, q, need_weights=False)[0], mha), (tok,))
    torch.backends.mha.set_fastpath_enabled(False)
    out['encoder_layer_fastpath_off'] = try_export(Fn(lambda m, x: m(x), layer), (tok,))
    out['mha_fastpath_off'] = try_export(Fn(lambda m, q: m(q, q, q, need_weights=False)[0], mha), (tok,))
    # 4. GELU(正確な erf 版と tanh 近似)・SDPA・LayerNorm
    out['gelu_erf'] = try_export(nn.GELU(), (tok,))
    out['gelu_tanh'] = try_export(nn.GELU(approximate='tanh'), (tok,))
    q = torch.randn(1, 4, 10, 16)
    out['sdpa'] = try_export(Fn(lambda a, b, c: F.scaled_dot_product_attention(a, b, c)), (q, q, q))
    out['layernorm'] = try_export(nn.LayerNorm(64), (tok,))
    # 5. GRUCell(TCP の経路の GRU)と GRU(層)
    cell = nn.GRUCell(8, 16)
    out['gru_cell'] = try_export(Fn(lambda m, a, h: m(a, h), cell), (torch.randn(1, 8), torch.randn(1, 16)))
    gru = nn.GRU(8, 16, batch_first=True)
    out['gru_layer'] = try_export(Fn(lambda m, a: m(a)[0], gru), (torch.randn(1, 10, 8),))
    # 6. 使っていない入力は消える(interfaces.md §4 の再確認)
    out['unused_inputs'] = try_export(Fn(lambda a, b: a * 2), (torch.randn(1, 3), torch.randn(1, 4)),
                                      input_names=['used', 'unused'])
    # 7. 辞書の出力(そのまま渡すと名前の順が辞書の順になる。ラッパーでタプルにするのが安全)
    out['dict_output'] = try_export(Fn(lambda a: {'actions': a * 2, 'speed': a.sum(1, keepdim=True)}),
                                    (torch.randn(1, 3),), output_names=['actions', 'speed'])
    os.makedirs(RESULTS, exist_ok=True)
    save_json(os.path.join(RESULTS, 'export_pitfalls.json'), out)
    for k, v in out.items():
        print(k, 'OK' if v['ok'] else 'NG', v.get('error', '') or list(v['ops'])[:8], v.get('inputs', ''),
              (v.get('warnings') or [''])[0][:120])


if __name__ == '__main__':
    main()
