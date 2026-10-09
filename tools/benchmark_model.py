#!/usr/bin/env python3
"""JetFormerモデルの推論速度をOrin Nanoで計測する(PyTorch FP16 / TensorRT FP16)

    python tools/benchmark_model.py [--weights model.pth] [--out jetformer_fp16.engine]
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from jetformer.model import IMU_STEPS, TOF_SHAPE, JetFormer, mode_onehot  # noqa: E402
from jetformer.trt_runtime import TRTRunner, build_engine  # noqa: E402

NUM_MODES = 8


def dummy_inputs(device, dtype):
    torch.manual_seed(0)
    return (torch.randn(1, 3, 224, 224, device=device, dtype=dtype),
            torch.rand(1, 1, *TOF_SHAPE, device=device, dtype=dtype),
            torch.randn(1, IMU_STEPS, 6, device=device, dtype=dtype) * 0.1,
            torch.randn(1, 3, device=device, dtype=dtype) * 0.3,
            mode_onehot([1], NUM_MODES).to(device, dtype))


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
    args = ap.parse_args()

    model = JetFormer(num_modes=NUM_MODES).eval()
    if args.weights:
        model.load_state_dict(torch.load(args.weights, map_location='cpu', weights_only=True))
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print('パラメータ数: %.1fM' % n_params)

    # ---- PyTorch FP16 ----
    m16 = model.cuda().half()
    x16 = dummy_inputs('cuda', torch.float16)
    with torch.no_grad():
        ref = m16(*x16).float().cpu().numpy()
        t = bench(lambda: m16(*x16))
    print('PyTorch FP16 : 平均 %.2fms  p99 %.2fms  (%.0f fps)' % (t.mean(), np.percentile(t, 99), 1000 / t.mean()))

    # ---- ONNX → TensorRT FP16 ----
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    onnx_path = os.path.splitext(args.out)[0] + '.onnx'
    m32 = model.float()
    x32 = dummy_inputs('cuda', torch.float32)
    torch.backends.mha.set_fastpath_enabled(False)      # ONNXに無い融合カーネルを使わせない
    with torch.no_grad():
        torch.onnx.export(m32, x32, onnx_path, input_names=['image', 'tof', 'imu', 'state', 'mode'],
                          output_names=['actions'], opset_version=17, dynamo=False)
    print('ONNX書き出し:', onnx_path, '%.1fMB' % (os.path.getsize(onnx_path) / 1e6))
    # 走行プログラムと同じPython版TensorRTでエンジンを作る(trtexecは版が違い読み込めないことがある)
    t0 = time.time()
    build_engine(onnx_path, args.out)
    print('TensorRTエンジン作成: %.0f秒 → %s' % (time.time() - t0, args.out))

    # ---- TensorRTの出力がPyTorchと一致するか ----
    runner = TRTRunner(args.out)
    feed = dict(zip(['image', 'tof', 'imu', 'state', 'mode'], x32))
    out = runner(**feed)['actions']
    with torch.no_grad():
        ref32 = m32(*x32).cpu().numpy()
    err = np.abs(out.cpu().numpy() - ref32).max()
    print('TensorRT と PyTorch(FP32)の出力差: 最大 %.4f (出力の範囲 %.3f〜%.3f)' % (err, ref32.min(), ref32.max()))

    t = bench(lambda: runner(**feed))
    print('TensorRT FP16: 平均 %.2fms  p99 %.2fms  (%.0f fps)' % (t.mean(), np.percentile(t, 99), 1000 / t.mean()))


if __name__ == '__main__':
    main()
