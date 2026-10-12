#!/usr/bin/env python3
"""クラウドで学習したモデル(ONNX)をJetson上でTensorRTエンジンに変換する

    python tools/build_engine.py runs/indoor_v1/jetformer.onnx data/models/indoor_v1_fp16.engine

精度・作業メモリは、ONNX と同じフォルダーの model_config.json の engine 欄(precision: fp16(既定)・bf16・fp32・
fp16_norm_fp32、workspace_gb)から決める(無ければ FP16・1GB = 今までと同じ)。fp16_norm_fp32 は FP16 のうち正規化
(LayerNorm)と Softmax の層だけを FP32 で計算する(OBEY_PRECISION_CONSTRAINTS。dinov2 の学習結果で FP16 の精度の確認
2.5 µs に落ちたときの選択肢。速さはほぼ FP16 と同じ)。<engine>.tmp に作ってから、TensorRT で読み直して入出力の
名前と形が走行の約束(jetformer.policies.check_io)に合うか確かめ、合えば置き換える。合わなければ tmp を消して理由を出し、
終了コード 2 で終わる(管理画面は「エンジンの作成に失敗」と表示し、走行の一覧に壊れたモデルが出ない。電源断でも半端な
エンジンが残らない)。
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from jetformer import policies  # noqa: E402

EXIT_BAD_IO = 2
# 精度 → FP32 のまま計算する層の種類(jetformer.trt_runtime.build_engine の fp32_layers)。ほかの精度は空 = 今までと同じ
MIXED_FP32_LAYERS = {'fp16_norm_fp32': ('norm', 'softmax')}


def engine_settings(model_config):
    """model_config(dict か None)→ (precision, workspace_gb)。知らない精度は ValueError"""
    eng = (model_config or {}).get('engine') or {}
    precision = str(eng.get('precision', 'fp16')).lower()
    if precision not in policies.ENGINE_PRECISIONS:
        raise ValueError('engine.precision は %s のどれか: %s' % ('・'.join(policies.ENGINE_PRECISIONS), precision))
    workspace = float(eng.get('workspace_gb', 1.0))
    if not (0.1 <= workspace <= 4.0):
        raise ValueError('engine.workspace_gb は 0.1〜4.0: %s' % workspace)
    return precision, workspace


def precision_flags(precision):
    """'fp16' → {'fp16': True, 'bf16': False} など(trt_runtime.build_engine の引数。fp32 はどちらも False)。
    'fp16_norm_fp32' → {'fp16': True, 'bf16': False, 'fp32_layers': ('norm', 'softmax')}(fp32_layers はこのときだけ入れる)"""
    if precision in MIXED_FP32_LAYERS:
        return {'fp16': True, 'bf16': False, 'fp32_layers': MIXED_FP32_LAYERS[precision]}
    return {'fp16': precision == 'fp16', 'bf16': precision == 'bf16'}


def read_model_config(onnx_path):
    path = os.path.join(os.path.dirname(os.path.abspath(onnx_path)), 'model_config.json')
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def check_engine_io(engine_io, model_config):
    """engine_io = (inputs {名前: 形}, outputs {名前: 形})。num_modes は model_config、無ければエンジンの mode の形"""
    inputs, outputs = engine_io
    num_modes = (model_config or {}).get('num_modes')
    if num_modes is None and 'mode' in inputs:
        num_modes = list(inputs['mode'])[-1]
    return policies.check_io(inputs, outputs, num_modes)


def main(argv):
    if len(argv) != 3:
        raise SystemExit(__doc__)
    from jetformer.trt_runtime import build_engine, engine_io
    onnx_path, engine_path = argv[1], argv[2]
    os.makedirs(os.path.dirname(engine_path) or '.', exist_ok=True)
    cfg = read_model_config(onnx_path)
    precision, workspace = engine_settings(cfg)
    tmp = engine_path + '.tmp'
    t = time.time()
    try:
        build_engine(onnx_path, tmp, workspace_gb=workspace, **precision_flags(precision))
        ok, why = check_engine_io(engine_io(tmp), cfg)
        if not ok:
            print('エンジンの入出力が走行の約束に合いません: %s' % why, flush=True)
            return EXIT_BAD_IO
        os.replace(tmp, engine_path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    print('作成しました: %s (%s・%.0f秒)' % (engine_path, precision.upper(), time.time() - t))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
