#!/usr/bin/env python3
"""クラウドで学習したモデル(ONNX)をJetson上でTensorRTエンジンに変換する

    python tools/build_engine.py runs/indoor_v1/jetformer.onnx data/models/indoor_v1_fp16.engine
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from jetformer.trt_runtime import build_engine  # noqa: E402

if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    onnx_path, engine_path = sys.argv[1], sys.argv[2]
    os.makedirs(os.path.dirname(engine_path) or '.', exist_ok=True)
    t = time.time()
    build_engine(onnx_path, engine_path)
    print('作成しました: %s (%.0f秒)' % (engine_path, time.time() - t))
