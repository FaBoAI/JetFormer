# -*- coding: utf-8 -*-
"""作ったエンジンを1つのプロセスに全部読み込み、順番を入れ替えながら3回ずつ測る(測るたびの揺れと順位の確認)

    python latency_repeat.py        → results/latency_repeat.json
"""

import os
import time

import numpy as np

from common import RESULTS, WORK, gpu_freq_mhz, real_inputs, save_json, wait_gpu_free

MODEL_B = '/home/jetson/jetformer/data/runs/clean_driving_20261011-nocmd-s3-20261011-024626/jetformer_fp16.engine'
NAMES = ['modelB_engine', 'v0', 'act', 'act_dec2', 'act_dec4', 'act_d512', 'tcp', 'tcp_dec', 'msfusion', 'msfusion_l2',
         'msfusion_l4', 'msfusion_tf', 'msfusion_rd',
         'dinov2_vit', 'dinov2_p2', 'dinov2_p1', 'dinov2_r168_p2']


def main():
    if not wait_gpu_free():
        raise SystemExit('GPU が空きません')
    import torch
    from jetformer.trt_runtime import TRTRunner
    xs = real_inputs(1)
    feed = dict(zip(['image', 'tof', 'imu', 'state', 'mode'], [a[:1] for a in xs]))
    runners = {}
    for n in NAMES:
        p = MODEL_B if n == 'modelB_engine' else os.path.join(WORK, n + '_fp16.engine')
        if os.path.exists(p):
            runners[n] = TRTRunner(p)
    res = {n: {'exec': [], 'run_numpy': []} for n in runners}
    rng = np.random.default_rng(0)
    for rep in range(3):
        order = list(runners)
        rng.shuffle(order)
        for n in order:
            r = runners[n]
            for k in r.inputs:
                r._host[k].numpy()[...] = feed[k]
                r._dev[k].copy_(r._host[k])
                r.context.set_tensor_address(k, r._dev[k].data_ptr())
            for _ in range(100):
                r.context.execute_async_v3(r.stream.cuda_stream)
            r.stream.synchronize()
            t = []
            for _ in range(500):
                t0 = time.perf_counter()
                r.context.execute_async_v3(r.stream.cuda_stream)
                r.stream.synchronize()
                t.append((time.perf_counter() - t0) * 1000)
            u = []
            for _ in range(500):
                t0 = time.perf_counter()
                r.run_numpy(**feed)
                u.append((time.perf_counter() - t0) * 1000)
            res[n]['exec'].append(round(float(np.median(t)), 3))
            res[n]['run_numpy'].append(round(float(np.median(u)), 3))
            res[n].setdefault('exec_p99', []).append(round(float(np.percentile(t, 99)), 3))
    out = {'gpu_mhz': gpu_freq_mhz(), 'repeats': res}
    torch.cuda.synchronize()
    save_json(os.path.join(RESULTS, 'latency_repeat.json'), out)
    for n, v in res.items():
        print('%-15s exec %s  run_numpy %s' % (n, v['exec'], v['run_numpy']))


if __name__ == '__main__':
    main()
