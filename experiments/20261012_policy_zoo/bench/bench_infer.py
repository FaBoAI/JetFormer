# -*- coding: utf-8 -*-
"""試作の方策を ONNX(opset 17・バッチ1・固定の形)→ TensorRT FP16 にし、精度・速さ・メモリを測る

    python bench_infer.py <名前> [--export-only]    # protos.VARIANTS の名前、または modelB_engine / dinov2_vit
結果: results/<名前>_infer.json(段ごとに書く。済んだ段は飛ばす = 電源が落ちても続きから)
ONNX とエンジン: work/<名前>.onnx・work/<名前>_fp16.engine
"""

import collections
import os
import sys
import time
import warnings

import numpy as np

from common import RESULTS, WORK, gpu_freq_mhz, load_json, real_inputs, save_json, wait_gpu_free

INPUTS = ['image', 'tof', 'imu', 'state', 'mode']
MODEL_B = '/home/jetson/jetformer/data/runs/clean_driving_20261011-nocmd-s3-20261011-024626'


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def stats(t):
    t = np.asarray(t)
    return {'median_ms': round(float(np.median(t)), 3), 'p90_ms': round(float(np.percentile(t, 90)), 3),
            'p99_ms': round(float(np.percentile(t, 99)), 3), 'mean_ms': round(float(t.mean()), 3),
            'max_ms': round(float(t.max()), 3), 'n': int(len(t))}


def peak_rss_mb():
    """このプロセスの常駐メモリの最大(VmHWM)。Orin は CPU と GPU が同じメモリなので、エンジン作成の重さの目安"""
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmHWM:'):
                    return round(int(line.split()[1]) / 1024, 0)
    except OSError:
        pass
    return None


def build_torch(name, weights_path=None):
    """計測する PyTorch のモデル。weights_path があればその重みを読む(ONNX に書き出したのと同じ重みで比べるため)"""
    import torch
    import torch.nn as nn
    import protos
    if name == 'dinov2_vit':
        class ViTOnly(nn.Module):
            """DINOv2 の ViT だけ(精度の確認用: 公式の重みで FP16 の特徴がどれだけずれるか)"""

            def __init__(self):
                super().__init__()
                self.vit = protos.DinoViTS14(224).load_official()

            def forward(self, image, tof, imu, state, mode):
                return {'tokens': self.vit(image)}
        return ViTOnly().eval()
    if name == 'modelB_engine':
        from jetformer.model import JetFormer
        m = JetFormer(num_modes=6)
        m.load_state_dict(torch.load(os.path.join(MODEL_B, 'best.pth'), map_location='cpu', weights_only=True))
        return m.eval()
    if weights_path:
        m = protos.build(name)
        m.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))
        return m.eval()
    # 初めて作るとき: 学習済みに近い値の範囲にする(FP16 の精度を見るため)。ResNet-18 は ImageNet の重み、
    # BatchNorm の統計は実データ 32 フレームで測り直す(乱数の初期値のままだと活性が層ごとに大きくなり、FP16 の比較にならない)
    torch.manual_seed(0)
    m = protos.build(name, pretrained=True)
    calibrate_bn(m)
    return m.eval()


def calibrate_bn(model, n=32):
    import torch
    bns = [mod for mod in model.modules() if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None            # 累積の平均
    xs = real_inputs(n)
    model.train()
    with torch.no_grad():
        for i in range(0, n, 8):
            model(*[torch.from_numpy(a[i:i + 8]) for a in xs])
    model.eval()


def phase_export(name, res, onnx_path, weights_path):
    import onnx
    import torch
    import torch.nn as nn
    import protos
    torch.backends.mha.set_fastpath_enabled(False)          # ONNX に無い融合カーネル(_native_multi_head_attention)を避ける
    model = build_torch(name).float().cpu()
    if weights_path:
        torch.save(model.state_dict(), weights_path)
    x = protos.example_inputs()
    with torch.no_grad():
        y = protos.outputs_of(model(*x))
    out_names = list(y)

    class Wrap(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, image, tof, imu, state, mode):
            o = protos.outputs_of(self.m(image, tof, imu, state, mode))
            return tuple(o[k] for k in out_names)

    t0 = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with torch.no_grad():
            torch.onnx.export(Wrap(model).eval(), x, onnx_path, input_names=INPUTS, output_names=out_names,
                              opset_version=17, dynamo=False)
    m = onnx.load(onnx_path, load_external_data=False)
    ops = collections.Counter(n.op_type for n in m.graph.node)

    def dims(v):
        return [d.dim_value if d.HasField('dim_value') else d.dim_param for d in v.type.tensor_type.shape.dim]
    msgs = sorted(set('%s: %s' % (w.category.__name__, str(w.message).split('\n')[0][:200]) for w in caught))
    res['export'] = {
        'seconds': round(time.time() - t0, 1), 'onnx_mb': round(os.path.getsize(onnx_path) / 1e6, 1),
        'nodes': len(m.graph.node), 'ops': dict(ops.most_common()),
        'inputs': {v.name: dims(v) for v in m.graph.input}, 'outputs': {v.name: dims(v) for v in m.graph.output},
        'warnings': msgs,
        'params_m': round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
        'trainable_params_m': round(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, 2)}
    log('ONNX', res['export']['nodes'], 'nodes', res['export']['inputs'], res['export']['outputs'])


def phase_build(name, res, onnx_path, engine_path):
    """jetformer/trt_runtime.build_engine と同じ設定(FP16・作業メモリ 1GB)。警告を集めるため Logger だけ自前"""
    import tensorrt as trt

    class Logger(trt.ILogger):
        def __init__(self):
            trt.ILogger.__init__(self)
            self.msgs = []

        def log(self, severity, msg):
            if severity <= trt.ILogger.Severity.WARNING:
                self.msgs.append('%s: %s' % (severity.name, msg[:300]))
    lg = Logger()
    t0 = time.time()
    builder = trt.Builder(lg)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, lg)
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            res['build'] = {'ok': False, 'errors': errs}
            log('ONNX の読み込みに失敗', errs)
            return False
    cfg = builder.create_builder_config()
    cfg.set_flag(trt.BuilderFlag.FP16)
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    ser = builder.build_serialized_network(network, cfg)
    if ser is None:
        res['build'] = {'ok': False, 'errors': lg.msgs[-20:]}
        return False
    with open(engine_path, 'wb') as f:
        f.write(ser)
    res['build'] = {'ok': True, 'seconds': round(time.time() - t0, 1), 'engine_mb': round(os.path.getsize(engine_path) / 1e6, 1),
                    'trt_version': trt.__version__, 'layers_in_network': network.num_layers,
                    'warnings': sorted(set(lg.msgs))[:40], 'process_peak_rss_mb': peak_rss_mb()}
    log('エンジン %.0f 秒' % res['build']['seconds'], res['build']['engine_mb'], 'MB')
    return True


def phase_measure(name, res, engine_path, weights_path):
    import torch
    from jetformer.trt_runtime import TRTRunner
    torch.backends.mha.set_fastpath_enabled(False)
    free0 = torch.cuda.mem_get_info()[0]
    runner = TRTRunner(engine_path)
    torch.cuda.synchronize()
    free1 = torch.cuda.mem_get_info()[0]
    eng = runner.engine
    shapes = {n: list(runner._dev[n].shape) for n in runner.inputs}
    res['engine_io'] = {'inputs': shapes, 'outputs': {k: list(v.shape) for k, v in runner.outputs.items()}}
    try:
        act_mb = eng.device_memory_size_v2 / 1e6
    except AttributeError:
        act_mb = eng.device_memory_size / 1e6
    res['memory'] = {'engine_activation_mb': round(act_mb, 1), 'runner_load_delta_mb': round((free0 - free1) / 1e6, 1)}

    # ---- 精度: 実データ16フレームで TensorRT FP16 と PyTorch FP32(同じ重み)を比べる
    num_modes = shapes.get('mode', [1, 6])[1]
    xs = real_inputs(16, num_modes=num_modes)
    model = build_torch(name, weights_path).float().cuda().eval()
    diffs = collections.defaultdict(list)
    refs = collections.defaultdict(list)
    trts = collections.defaultdict(list)
    d16 = collections.defaultdict(list)
    import protos
    with torch.no_grad():
        for i in range(len(xs[0])):
            feed = {k: xs[j][i:i + 1] for j, k in enumerate(INPUTS)}
            out = runner.run_numpy(**feed)
            tx = [torch.from_numpy(feed[k]).cuda() for k in INPUTS]
            ref = {k: v.float().cpu().numpy() for k, v in protos.outputs_of(model(*tx)).items()}
            with torch.autocast('cuda', dtype=torch.float16):
                r16 = {k: v.float().cpu().numpy() for k, v in protos.outputs_of(model(*tx)).items()}
            for k in ref:
                diffs[k].append(np.abs(out[k] - ref[k]))
                refs[k].append(ref[k])
                trts[k].append(out[k])
                d16[k].append(np.abs(r16[k] - ref[k]))
    acc = {}
    for k in diffs:
        d = np.concatenate([a.ravel() for a in diffs[k]])
        r = np.concatenate([a.ravel() for a in refs[k]])
        a16 = np.concatenate([a.ravel() for a in d16[k]])
        acc[k] = {'trt_max_abs': float(d.max()), 'trt_mean_abs': float(d.mean()), 'ref_std': float(r.std()),
                  'ref_max_abs': float(np.abs(r).max()), 'trt_rel_max': float(d.max() / (np.abs(r).max() + 1e-12)),
                  'torch_autocast_max_abs': float(a16.max())}
        if k == 'actions':
            acc[k]['trt_max_us'] = float(d.max() * 500)
            acc[k]['trt_mean_us'] = float(d.mean() * 500)
        if k == 'tokens':           # ViT の特徴: トークンごとのコサイン類似度(先頭が CLS)
            ref_t = np.concatenate(refs[k], 0)
            out_t = np.concatenate(trts[k], 0)
            cos = (out_t * ref_t).sum(-1) / (np.linalg.norm(out_t, axis=-1) * np.linalg.norm(ref_t, axis=-1) + 1e-12)
            acc[k]['cos_min'] = float(cos.min())
            acc[k]['cos_mean'] = float(cos.mean())
            acc[k]['cls_cos_min'] = float(cos[:, 0].min())
    res['accuracy'] = acc
    log('精度', {k: (round(v['trt_max_abs'], 5), round(v['trt_rel_max'], 5)) for k, v in acc.items()})

    # ---- 速さ
    lat = {'gpu_mhz_before': gpu_freq_mhz()}
    feed_np = {k: xs[j][:1] for j, k in enumerate(INPUTS)}
    feed_cuda = {k: torch.from_numpy(v).cuda() for k, v in feed_np.items()}

    def bench(fn, n=600, warmup=100):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            t.append((time.perf_counter() - t0) * 1000)
        return stats(t)

    # (a) エンジンだけ: 入力をあらかじめ GPU に置き、execute_async_v3 + 同期
    for k in runner.inputs:
        runner._dev[k].copy_(feed_cuda[k])
        runner.context.set_tensor_address(k, runner._dev[k].data_ptr())

    def exec_only():
        runner.context.execute_async_v3(runner.stream.cuda_stream)
        runner.stream.synchronize()
    lat['trt_exec'] = bench(exec_only)
    # (b) benchmark_model.py と同じ __call__(GPU のテンソルを渡す。今の「1.9ms」と同じ測り方)
    lat['trt_call'] = bench(lambda: runner(**feed_cuda))
    # (c) 走行ループと同じ run_numpy(ピン留めメモリへの写し・転送・推論・回収)
    lat['trt_run_numpy'] = bench(lambda: runner.run_numpy(**feed_np))
    lat['gpu_mhz_after'] = gpu_freq_mhz()
    # (d) PyTorch FP16(参考)
    try:
        m16 = model.half()
        x16 = [feed_cuda[k].half() for k in INPUTS]

        def torch_fp16():
            with torch.no_grad():
                m16(*x16)
            torch.cuda.synchronize()
        lat['torch_fp16'] = bench(torch_fp16, n=200, warmup=30)
    except Exception as e:          # noqa: BLE001
        lat['torch_fp16'] = {'error': repr(e)[:200]}
    res['latency'] = lat
    res['memory']['torch_peak_after_mb'] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
    log('速さ exec %.2f / call %.2f / run_numpy %.2f ms(中央値)' % (
        lat['trt_exec']['median_ms'], lat['trt_call']['median_ms'], lat['trt_run_numpy']['median_ms']))


def main():
    name = sys.argv[1]
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(WORK, exist_ok=True)
    out_path = os.path.join(RESULTS, name + '_infer.json')
    res = load_json(out_path)
    res['name'] = name
    if name == 'modelB_engine':
        onnx_path = os.path.join(MODEL_B, 'jetformer.onnx')
        engine_path = os.path.join(MODEL_B, 'jetformer_fp16.engine')      # 読むだけ
        weights_path = None
    else:
        onnx_path = os.path.join(WORK, name + '.onnx')
        engine_path = os.path.join(WORK, name + '_fp16.engine')
        weights_path = None if name == 'dinov2_vit' else os.path.join(WORK, name + '.pth')
        if 'export' not in res or not os.path.exists(onnx_path) or (weights_path and not os.path.exists(weights_path)):
            res = {'name': name}
            phase_export(name, res, onnx_path, weights_path)
            save_json(out_path, res)
    if '--export-only' in sys.argv:
        return
    if 'measure_done' in res:
        log('済み', name)
        return
    if not wait_gpu_free(log=log):
        raise SystemExit('GPU が空かないので止めます')
    if name != 'modelB_engine' and (not res.get('build', {}).get('ok') or not os.path.exists(engine_path)):
        ok = phase_build(name, res, onnx_path, engine_path)
        save_json(out_path, res)
        if not ok:
            raise SystemExit('エンジンを作れませんでした')
    phase_measure(name, res, engine_path, weights_path)
    res['measure_done'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    save_json(out_path, res)


if __name__ == '__main__':
    main()
