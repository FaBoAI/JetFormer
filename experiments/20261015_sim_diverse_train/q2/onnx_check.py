#!/usr/bin/env python3
"""C-DIV の推論の確かめ(CPU だけ。FP16 のエンジンは作らない = protocol の no_engine)

学習結果の jetformer.onnx(実車ではこれから TensorRT のエンジンを作る)が PyTorch FP32 と同じ出力を出すかを、
保留した実機の人のきれいな走行から等間隔に 32 フレームで確かめる(10/13 のゲートの accuracy の項目 actions_max_us ≤ 2.5 µs と同じ目安。
ただしエンジン(FP16)ではなく ONNX(FP32)なので、エンジンのゲートの代わりにはならない)。

  1. cd ~/jetformer && CUDA_VISIBLE_DEVICES= python experiments/20261015_sim_diverse_train/q2/onnx_check.py torch     # jetracer の env
  2. cd ~/jetformer && data/sim/venv/bin/python -I experiments/20261015_sim_diverse_train/q2/onnx_check.py onnx      # onnxruntime の venv(読むだけ)
→ /home/jetson/jetformer_reports/sim_diverse_train_20261015/q2/onnx_check/<学習結果>.json
"""

import json
import os
import sys
import time

import numpy as np

REPO = '/home/jetson/jetformer'
OUT = os.environ.get('Q2_ONNX_OUT', '/home/jetson/jetformer_reports/sim_diverse_train_20261015/q2/onnx_check')
RUNS = [r for r in os.environ.get('Q2_ONNX_RUNS', '').split(',') if r] or ['simdiv_20261015-CDIV-s0', 'simdiv_20261015-CDIV-s1']
N = 32
GATE_US = 2.5


def cmd_torch():
    sys.path.insert(0, REPO)
    sys.path.insert(0, REPO + '/tools')
    import torch
    import policy_zoo as pz
    import rl_evaluate as ev
    from jetformer.policies import runtime
    torch.set_num_threads(2)
    held = ev._heldout()
    tab = ev.pred_table()
    cols, _ = ev._val_arrays(tab)
    human = np.isin(tab['scene'], held['human_clean_scenes']) & (tab['part'] == 0)
    clean = np.nonzero(human & cols['fv'] & ~cols['impact'])[0]
    rows = clean[np.linspace(0, len(clean) - 1, N).round().astype(int)]
    os.makedirs(OUT, exist_ok=True)
    for run in RUNS:
        d = os.path.join(REPO, 'data/runs', run)
        res = json.load(open(os.path.join(d, 'result.json'))) if os.path.exists(os.path.join(d, 'result.json')) else {}
        if res.get('state') != 'done':
            print('%s: 学習が済んでいないので飛ばします' % run)
            continue
        out = os.path.join(OUT, run + '_torch.npz')
        if os.path.exists(out):
            print('作成済み: %s' % out)
            continue
        cfg = json.load(open(os.path.join(d, 'model_config.json')))
        fs = pz._Frames(tab, rows, use_command=bool(cfg.get('command_input', True)))
        trained = np.array([int(x) for x in cfg['trained_modes']])
        genres, _ = ev._genres([it[4] for it in fs.items], None, trained)
        items = [fs[i] for i in range(len(fs))]
        frames = {k: np.stack([it[j].numpy() for it in items]).astype(np.float32) for j, k in enumerate(('image', 'tof', 'imu', 'state'))}
        mode = np.zeros((len(rows), cfg['num_modes']), np.float32)
        mode[np.arange(len(rows)), genres] = 1.0
        frames['mode'] = mode
        model, _, _, _ = runtime.load_run(d, 'best', device='cpu')
        acts = []
        for s in range(0, len(rows), 4):
            acts.append(runtime.predict(model, {k: v[s:s + 4] for k, v in frames.items()})['actions'])
        tmp = out + '.tmp.npz'
        np.savez_compressed(tmp, actions=np.concatenate(acts).astype(np.float32), rows=rows, **frames)
        os.replace(tmp, out)
        print('[%s] %s: PyTorch(CPU・best)%d フレーム' % (time.strftime('%H:%M:%S'), run, len(rows)))


def cmd_onnx():
    import hashlib
    import onnxruntime as ort
    for run in RUNS:
        ref_p = os.path.join(OUT, run + '_torch.npz')
        onnx_p = os.path.join(REPO, 'data/runs', run, 'jetformer.onnx')
        if not (os.path.exists(ref_p) and os.path.exists(onnx_p)):
            print('%s: 参照か ONNX が無いので飛ばします' % run)
            continue
        ref = np.load(ref_p)
        blob = open(onnx_p, 'rb').read()
        sess = ort.InferenceSession(blob, providers=['CPUExecutionProvider'])
        names = [i.name for i in sess.get_inputs()]
        t0 = time.time()
        outs = [sess.run(['actions'], {n: ref[n][i:i + 1] for n in names})[0] for i in range(len(ref['rows']))]
        ms = (time.time() - t0) * 1000.0 / len(outs)
        a = np.concatenate(outs).astype(np.float64)
        d = np.abs(a - ref['actions'].astype(np.float64)) * 500.0
        res = {'run': run, 'frames': int(len(a)), 'inputs': names, 'onnxruntime': ort.__version__,
               'onnx_sha256': hashlib.sha256(blob).hexdigest(), 'max_abs_us': round(float(d.max()), 5),
               'mean_abs_us': round(float(d.mean()), 6), 'finite': bool(np.isfinite(a).all()),
               'gate_us': GATE_US, 'pass': bool(np.isfinite(a).all() and d.max() <= GATE_US),
               'onnxruntime_cpu_ms_per_frame': round(ms, 1),
               'note': 'ONNX(FP32・onnxruntime CPU・バッチ 1)と PyTorch FP32(CPU・best.pth・バッチ 4)。FP16 のエンジンは作っていない',
               'checked_at': time.strftime('%Y-%m-%d %H:%M:%S')}
        with open(os.path.join(OUT, run + '.json'), 'w', encoding='utf-8') as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        print(json.dumps(res, ensure_ascii=False))


if __name__ == '__main__':
    {'torch': cmd_torch, 'onnx': cmd_onnx}[sys.argv[1]]()
