#!/usr/bin/env python3
"""V-models: どのエンジンで TensorRT の「Using an engine plan file across different models of devices」の警告が出るか。
エンジンを 1 本ずつ読み、前後に印を出す(推論はしない)。jf-heavy --lock big で動かす。"""
import json, sys
from pathlib import Path
REPO = Path('/home/jetson/jetformer'); sys.path.insert(0, str(REPO))
import torch
torch.zeros(1, device='cuda'); torch.cuda.synchronize()
from jetformer.trt_runtime import TRTRunner
C = json.load(open(REPO / 'data/bench_candidates.json'))
for c in C['candidates']:
    print('BEGIN', c['id'], file=sys.stderr, flush=True)
    r = TRTRunner(str(REPO / 'data/runs' / c['run'] / 'jetformer_fp16.engine'))
    del r
    print('END', c['id'], file=sys.stderr, flush=True)
