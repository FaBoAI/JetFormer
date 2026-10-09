"""TensorRT(Python版)でのエンジン作成と推論

エンジンは必ず「走行プログラムが使うのと同じPython版TensorRT」で作ること
(trtexec はJetPack同梱の別バージョンで、Pythonから読み込めないことがある)。
"""

import tensorrt as trt
import torch

_LOGGER = trt.Logger(trt.Logger.WARNING)


def build_engine(onnx_path, engine_path, fp16=True, workspace_gb=1.0):
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
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    serialized = builder.build_serialized_network(network, cfg)
    if serialized is None:
        raise RuntimeError('TensorRTエンジンの作成に失敗しました')
    with open(engine_path, 'wb') as f:
        f.write(serialized)
    return engine_path


class TRTRunner(object):
    """入出力をtorchのCUDAテンソルで受け渡す。runner(image=..., tof=..., imu=..., state=...) → dict"""

    def __init__(self, engine_path):
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(_LOGGER).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError('エンジンを読み込めません(TensorRTのバージョン違い?): %s' % engine_path)
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        self.inputs, self.outputs = [], {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(name)
            else:
                self.outputs[name] = torch.empty(shape, dtype=torch.float32, device='cuda')
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
