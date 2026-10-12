"""TensorRT(Python版)でのエンジン作成と推論

エンジンは必ず「走行プログラムが使うのと同じPython版TensorRT」で作ること
(trtexec はJetPack同梱の別バージョンで、Pythonから読み込めないことがある)。
"""

import tensorrt as trt
import torch

_LOGGER = trt.Logger(trt.Logger.WARNING)


FP32_LAYER_TYPES = {'norm': 'NORMALIZATION', 'softmax': 'SOFTMAX'}   # fp32_layers に書ける名前 → trt.LayerType の名前


def build_engine(onnx_path, engine_path, fp16=True, workspace_gb=1.0, bf16=False, fp32_layers=()):
    """ONNX → エンジン。fp16・bf16 のどちらも False なら FP32(model_config.json の engine.precision から決める)。

    fp32_layers: FP16 の中で FP32 のまま計算する層の種類('norm' = LayerNorm などの正規化、'softmax')。空(既定)なら今までと
    同じ。指定したときは、その種類の層の精度と出力の型を FP32 にし、OBEY_PRECISION_CONSTRAINTS で守らせる
    (engine.precision = fp16_norm_fp32。dinov2 の FP16 の精度の確認のため。2026-10-12)"""
    builder = trt.Builder(_LOGGER)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, _LOGGER)
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError('ONNXの読み込みに失敗: ' + '; '.join(errors))
    cfg = builder.create_builder_config()
    if fp16:
        cfg.set_flag(trt.BuilderFlag.FP16)
    if bf16:
        cfg.set_flag(trt.BuilderFlag.BF16)
    if fp32_layers:
        unknown = [k for k in fp32_layers if k not in FP32_LAYER_TYPES]
        if unknown:
            raise ValueError('fp32_layers は %s のどれか: %s' % ('・'.join(FP32_LAYER_TYPES), unknown))
        want = {getattr(trt.LayerType, FP32_LAYER_TYPES[k]) for k in fp32_layers}
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            if layer.type in want:
                layer.precision = trt.float32
                for j in range(layer.num_outputs):
                    layer.set_output_type(j, trt.float32)
        cfg.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    serialized = builder.build_serialized_network(network, cfg)
    if serialized is None:
        raise RuntimeError('TensorRTエンジンの作成に失敗しました')
    with open(engine_path, 'wb') as f:
        f.write(serialized)
    return engine_path


def engine_io(engine_path):
    """エンジンを TensorRT だけで読み、入出力の形 → (inputs {名前: [形]}, outputs {名前: [形]})(GPU のメモリを確保しない)"""
    with open(engine_path, 'rb') as f:
        engine = trt.Runtime(_LOGGER).deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError('エンジンを読み込めません(TensorRTのバージョン違い?): %s' % engine_path)
    inputs, outputs = {}, {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = [int(d) for d in engine.get_tensor_shape(name)]
        (inputs if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else outputs)[name] = shape
    del engine
    return inputs, outputs


class TRTRunner(object):
    """入出力をtorchのCUDAテンソルで受け渡す。runner(image=..., tof=..., imu=..., state=...) → dict。
    shapes = {名前: 形}(入力と出力。走行の空回しの入力・入出力の確認に使う)"""

    def __init__(self, engine_path):
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(_LOGGER).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError('エンジンを読み込めません(TensorRTのバージョン違い?): %s' % engine_path)
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        self.inputs, self.outputs = [], {}
        self.shapes = {}
        self._dev, self._host, self._out_host = {}, {}, {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            self.shapes[name] = tuple(int(d) for d in shape)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(name)
                self._dev[name] = torch.empty(shape, dtype=torch.float32, device='cuda')
                self._host[name] = torch.empty(shape, dtype=torch.float32).pin_memory()
            else:
                self.outputs[name] = torch.empty(shape, dtype=torch.float32, device='cuda')
                self._out_host[name] = torch.empty(shape, dtype=torch.float32).pin_memory()
                self.context.set_tensor_address(name, self.outputs[name].data_ptr())

    def __call__(self, **inputs):
        keep = []
        for name in self.inputs:
            t = inputs[name].to('cuda', torch.float32).contiguous()
            keep.append(t)
            self.context.set_tensor_address(name, t.data_ptr())
        self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return {k: v.clone() for k, v in self.outputs.items()}

    def run_numpy(self, **arrays):
        """走行ループ用: numpyの入力(バッチ次元込み)→ numpyの出力。固定のピン留めメモリと1本のストリームで
        転送・推論・回収をまとめて行う(__call__ より約1.3ms速い)"""
        for name in self.inputs:
            self._host[name].numpy()[...] = arrays[name]
        with torch.cuda.stream(self.stream):
            for name in self.inputs:
                self._dev[name].copy_(self._host[name], non_blocking=True)
                self.context.set_tensor_address(name, self._dev[name].data_ptr())
            self.context.execute_async_v3(self.stream.cuda_stream)
            for name, t in self.outputs.items():
                self._out_host[name].copy_(t, non_blocking=True)
        self.stream.synchronize()
        return {k: v.numpy().copy() for k, v in self._out_host.items()}
