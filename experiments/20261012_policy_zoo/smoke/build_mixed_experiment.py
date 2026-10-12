"""実験: FP16 + 指定した種類の層だけ FP32(OBEY_PRECISION_CONSTRAINTS)"""
import sys, time
import tensorrt as trt
onnx_path, engine_path, kinds = sys.argv[1], sys.argv[2], sys.argv[3].split(',')
L = trt.Logger(trt.Logger.WARNING)
b = trt.Builder(L); net = b.create_network(0); p = trt.OnnxParser(net, L)
assert p.parse(open(onnx_path, 'rb').read())
types = {'norm': trt.LayerType.NORMALIZATION, 'softmax': trt.LayerType.SOFTMAX, 'reduce': trt.LayerType.REDUCE,
         'unary': trt.LayerType.UNARY}
want = {types[k] for k in kinds if k in types}
n = 0
counts = {}
for i in range(net.num_layers):
    l = net.get_layer(i)
    counts[str(l.type)] = counts.get(str(l.type), 0) + 1
    if l.type in want or ('heads' in kinds and ('heads' in l.name or 'trunk' in l.name)):
        l.precision = trt.float32
        for j in range(l.num_outputs):
            l.set_output_type(j, trt.float32)
        n += 1
print('layers', net.num_layers, 'fp32 constrained', n, sorted(counts.items(), key=lambda x: -x[1])[:12])
cfg = b.create_builder_config()
cfg.set_flag(trt.BuilderFlag.FP16)
cfg.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
t = time.time()
s = b.build_serialized_network(net, cfg)
assert s is not None
open(engine_path, 'wb').write(s)
print('built %.0fs' % (time.time() - t))
