#!/usr/bin/env python3
"""JetFormerモデルの推論速度をOrin Nanoで計測する(PyTorch FP16 / TensorRT FP16)

    python tools/benchmark_model.py [--weights model.pth] [--out jetformer_fp16.engine]
    python tools/benchmark_model.py --policy act --num-modes 6 [--policy-cfg '{"dec_layers": 3}'] [--json out.json]

モデルの種類(方策)は jetformer/policies の登録簿から作る(既定 jetformer_v0・ジャンル数 8 = 今までと同じ)。
ONNX の書き出しは学習と同じ runtime.export_onnx(入出力の名前・形を確かめる)。GPU を使うので、ほかの学習・走行と同時に動かさない。
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from jetformer import policies  # noqa: E402
from jetformer.model import IMU_STEPS, TOF_CHANNELS, TOF_SHAPE, mode_onehot  # noqa: E402
from jetformer.policies import runtime  # noqa: E402
from jetformer.trt_runtime import TRTRunner, build_engine  # noqa: E402

NUM_MODES = 8
INPUT_NAMES = policies.BASE_INPUTS


def dummy_inputs(device, dtype, num_modes=NUM_MODES):
    torch.manual_seed(0)
    return (torch.randn(1, 3, 224, 224, device=device, dtype=dtype),
            torch.rand(1, TOF_CHANNELS, *TOF_SHAPE, device=device, dtype=dtype),
            torch.randn(1, IMU_STEPS, 6, device=device, dtype=dtype) * 0.1,
            torch.randn(1, 3, device=device, dtype=dtype) * 0.3,
            mode_onehot([1], num_modes).to(device, dtype))


def bench(fn, n=300, warmup=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t) * 1000)
    return np.array(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='')
    ap.add_argument('--out', default='data/models/jetformer_fp16.engine')
    ap.add_argument('--policy', default=policies.DEFAULT, choices=policies.names(), help='モデルの種類(既定 jetformer_v0)')
    ap.add_argument('--policy-cfg', default=None, help='方策の設定の JSON(既定値から変えるものだけ)')
    ap.add_argument('--num-modes', type=int, default=NUM_MODES, help='ジャンル数(今の学習結果は 6。既定 %d)' % NUM_MODES)
    ap.add_argument('--json', default=None, help='結果を JSON で書く')
    args = ap.parse_args()
    nm = args.num_modes
    spec = policies.get(args.policy)
    pcfg = spec.resolve(json.loads(args.policy_cfg) if args.policy_cfg else None)

    model = spec.build(nm, pcfg, pretrained=False).eval()
    if args.weights:
        model.load_state_dict(runtime.load_state(args.weights))
    n_total, n_train = runtime.count_params(model)
    print('モデルの種類: %s パラメータ数: %.1fM(学習 %.1fM)' % (args.policy, n_total / 1e6, n_train / 1e6))
    result = {'policy': args.policy, 'policy_cfg': pcfg, 'num_modes': nm,
              'params_m': {'total': round(n_total / 1e6, 2), 'trainable': round(n_train / 1e6, 2)}}

    # ---- PyTorch FP16 ----
    m16 = model.cuda().half()
    x16 = dict(zip(INPUT_NAMES, dummy_inputs('cuda', torch.float16, nm)))
    with torch.no_grad():
        t = bench(lambda: runtime.forward(m16, x16))
    print('PyTorch FP16 : 平均 %.2fms  p99 %.2fms  (%.0f fps)' % (t.mean(), np.percentile(t, 99), 1000 / t.mean()))
    result['torch_fp16_ms'] = {'mean': round(float(t.mean()), 3), 'p99': round(float(np.percentile(t, 99)), 3)}

    # ---- ONNX → TensorRT FP16 ----
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    onnx_path = os.path.splitext(args.out)[0] + '.onnx'
    m32 = model.float()
    io = runtime.export_onnx(m32, onnx_path, nm)          # 学習と同じ書き出し(モデルは CPU に移る)
    print('ONNX書き出し:', onnx_path, '%.1fMB' % (os.path.getsize(onnx_path) / 1e6), '入力', io['inputs'], '出力', io['outputs'])
    m32 = m32.cuda()
    x32 = dict(zip(INPUT_NAMES, dummy_inputs('cuda', torch.float32, nm)))
    # 走行プログラムと同じPython版TensorRTでエンジンを作る(trtexecは版が違い読み込めないことがある)
    t0 = time.time()
    build_engine(onnx_path, args.out)
    print('TensorRTエンジン作成: %.0f秒 → %s' % (time.time() - t0, args.out))
    result['engine_build_s'] = round(time.time() - t0, 1)

    # ---- TensorRTの出力がPyTorchと一致するか ----
    runner = TRTRunner(args.out)
    out = runner(**x32)['actions']
    ref32 = runtime.predict(m32, x32)['actions']
    err = np.abs(out.cpu().numpy() - ref32).max()
    print('TensorRT と PyTorch(FP32)の出力差: 最大 %.4f (出力の範囲 %.3f〜%.3f)' % (err, ref32.min(), ref32.max()))
    result['trt_vs_fp32_max'] = float(err)

    t = bench(lambda: runner(**x32))
    print('TensorRT FP16: 平均 %.2fms  p99 %.2fms  (%.0f fps)' % (t.mean(), np.percentile(t, 99), 1000 / t.mean()))
    result['trt_fp16_ms'] = {'mean': round(float(t.mean()), 3), 'p99': round(float(np.percentile(t, 99)), 3)}
    if args.json:
        tmp = args.json + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=1)
        os.replace(tmp, args.json)


if __name__ == '__main__':
    main()
