#!/usr/bin/env python3
"""学習した1本のエンジンを、走行タブと同じ手順で読めるか確かめる(GPU、1プロセス。PWM は出さない・車は動かさない・
走行のループは回さない)

    python engine_load_check.py <学習結果のフォルダー(data/runs/...)> <結果の json>

1. drive.model_info(走行タブの一覧と同じ関数)で drivable を確かめる
2. 検証の実フレーム 64 個で PyTorch FP32(CPU、best.pth を runtime.load_run)と TensorRT(run_numpy)の出力の差
   (spec.md §5.1 のゲート: actions ≤ 2.5 µs、tcp は waypoints ≤ 0.02 m・速度の期待値 ≤ 0.05 m/s、NaN・Inf なし)
3. drive._run と同じ順: torch の CUDA の初期化 → TRTRunner → policies.check_io → 形から作った 0 で空回し 3 回
4. 速さ: run_numpy(走行と同じ呼び方)とエンジンだけを、100 回の空回しの後 600 回(中央値 ≤ 6.0 ms・p99 ≤ 8.0 ms)

段階3の smoke/engine_check.py の 1〜4 と同じ計算(走行のループの部分は除いた)。

2026-10-14(RL ベストラップ sim_rl_bestlap_20261013): sim_cotrain_20261013/train/engine_load_check.py の写し(計算は同じ)。
実フレームは学習結果の model_config の dataset の val(S-T = 教師の 20 分データ、S-R = RL のベストラップの 20 分データ。
どちらもシミュレーターのジャンル 8・ジャンル数 9)から取る。シミュレーター専用のモデルで、エンジンは実車の走行には使わない。
"""

import collections
import gc
import json
import os
import sys
import time
from pathlib import Path

REPO = Path('/home/jetson/jetformer')
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

DATASETS = REPO / 'data' / 'datasets'
N_FRAMES = 64
WARMUP, TIMED = 100, 600


def pct(x, q):
    return round(float(np.percentile(np.asarray(x), q)), 3)


def write_json(path, obj):
    tmp = str(path) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def real_frames(cfg):
    """検証のシーンから等間隔に 64 個(2.5G の衝突の前後を除く・学習と同じ正規化)→ {名前: [64,...]}"""
    from jetformer.train_data import JetFormerDataset
    ds = JetFormerDataset(str(DATASETS / cfg['dataset']), 'val', stride=1, use_command=cfg.get('command_input', True), mask_impacts=True,
                          impact_g=2.5)
    idx = np.linspace(0, len(ds) - 1, N_FRAMES).round().astype(int)
    cols = [ds[int(i)] for i in idx]
    names = ('image', 'tof', 'imu', 'state', 'mode')
    out = {n: np.stack([c[k].numpy() for c in cols]).astype(np.float32) for k, n in enumerate(names)}
    if out['mode'].shape[1] != cfg['num_modes']:
        raise SystemExit('ジャンル数が違います: データ %d / モデル %d' % (out['mode'].shape[1], cfg['num_modes']))
    with open(DATASETS / cfg['dataset'] / 'splits.json', encoding='utf-8') as f:
        names = json.load(f)['val']
    if len(names) != len(ds.scenes):
        raise SystemExit('val のシーンの数が違います: splits %d / 読めた %d' % (len(names), len(ds.scenes)))
    picks = [[names[ds.index[int(i)][0]], int(ds.index[int(i)][1])] for i in idx]
    return out, {'dataset': cfg['dataset'], 'val_frames': len(ds), 'picks': picks,
                 'modes': sorted({int(m) for m in out['mode'].argmax(1)})}


def torch_reference(run_dir, frames):
    """PyTorch FP32(CPU。TF32 が入らないように)の出力 {名前: [64,...]}"""
    import torch
    from jetformer.policies import runtime
    torch.set_num_threads(4)
    model, cfg, spec, pcfg = runtime.load_run(run_dir, 'best', device='cpu')
    outs = collections.defaultdict(list)
    t = time.time()
    for s in range(0, N_FRAMES, 8):
        o = runtime.predict(model, {k: v[s:s + 8] for k, v in frames.items()})
        for k, v in o.items():
            outs[k].append(v)
    ref = {k: np.concatenate(v) for k, v in outs.items()}
    del model
    gc.collect()
    return ref, round(time.time() - t, 1)


def speed_expectation(logits, bins):
    z = logits - logits.max(axis=-1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=-1, keepdims=True)
    return (p * np.asarray(bins, np.float32)).sum(-1)


def main(run_dir, out_path):
    run_dir = Path(run_dir).resolve()
    res = {'run': run_dir.name, 'checked_at': time.strftime('%Y-%m-%d %H:%M:%S')}
    cfg = json.load(open(run_dir / 'model_config.json', encoding='utf-8'))
    res['policy'] = cfg.get('policy', 'jetformer_v0')

    # ---- 1. 走行タブの一覧と同じ関数
    from jetformer import drive, policies
    from jetformer.policies import post as post_mod
    if run_dir.parent.resolve() != Path(drive.RUNS_DIR).resolve():
        drive.RUNS_DIR = run_dir.parent
    info = drive.model_info(run_dir.name)
    if info is None:
        raise SystemExit('drive.model_info が None(エンジンか model_config.json が無い)')
    res['model_info'] = {k: info.get(k) for k in ('policy', 'policy_title', 'drivable', 'why_not', 'inputs', 'outputs',
                                                  'num_modes', 'trained_modes', 'command_input', 'steer_center_us',
                                                  'throttle_neutral_us', 'best_val_us', 'created_at')}
    res['model_info']['postprocess_kind'] = post_mod.kind_of(info)

    # ---- 2a. 実フレームと PyTorch FP32(CPU。GPU を使う前に済ませる)
    frames, res['frames'] = real_frames(cfg)
    ref_out, ref_s = torch_reference(run_dir, frames)
    res['torch_ref'] = {'device': 'cpu', 'frames': N_FRAMES, 'seconds': ref_s, 'outputs': sorted(ref_out)}

    # ---- 3. drive._run と同じ順でエンジンを読む
    import torch
    from jetformer.trt_runtime import TRTRunner
    torch.zeros(1, device='cuda')
    torch.cuda.synchronize()
    t = time.time()
    runner = TRTRunner(str(Path(drive.RUNS_DIR) / info['name'] / drive.ENGINE_FILE))
    num_modes = info['num_modes']
    ok, why = policies.check_io({n: list(runner.shapes[n]) for n in runner.inputs},
                                {n: list(runner.shapes[n]) for n in runner.outputs}, num_modes)
    res['check_io'] = {'ok': ok, 'why': why, 'inputs': {n: list(runner.shapes[n]) for n in runner.inputs},
                       'outputs': {n: list(runner.shapes[n]) for n in runner.outputs}}
    if not ok:
        write_json(out_path, res)
        raise SystemExit('check_io: ' + why)
    zeros = {n: np.zeros(runner.shapes[n], np.float32) for n in runner.inputs}
    for _ in range(3):
        runner.run_numpy(**zeros)
    res['load_s'] = round(time.time() - t, 2)
    res['engine_mb'] = round((run_dir / drive.ENGINE_FILE).stat().st_size / 2 ** 20, 1)

    # ---- 2b. TensorRT と PyTorch FP32 の差(実フレーム 64 個)
    trt_out = collections.defaultdict(list)
    for i in range(N_FRAMES):
        o = runner.run_numpy(**{n: frames[n][i:i + 1] for n in runner.inputs})
        for k, v in o.items():
            trt_out[k].append(v)
    trt_out = {k: np.concatenate(v) for k, v in trt_out.items()}
    acc = {}
    da = np.abs(trt_out['actions'] - ref_out['actions']) * policies.PWM_SCALE
    acc['actions_max_us'] = round(float(da.max()), 3)
    acc['actions_mean_us'] = round(float(da.mean()), 4)
    acc['actions_p99_us'] = pct(da, 99)
    acc['gate_actions_2p5us'] = bool(da.max() <= 2.5)
    acc['finite'] = bool(all(np.isfinite(v).all() for v in trt_out.values()))
    acc['torch_actions_spread_us'] = round(float(ref_out['actions'].std(axis=0).mean() * policies.PWM_SCALE), 2)
    if 'waypoints' in trt_out:
        dw = np.abs(trt_out['waypoints'] - ref_out['waypoints'])
        acc['waypoints_max_m'] = round(float(dw.max()), 5)
        acc['waypoints_ref_absmax_m'] = round(float(np.abs(ref_out['waypoints']).max()), 3)
        acc['gate_waypoints_0p02m'] = bool(dw.max() <= 0.02)
    if 'speed_logits' in trt_out:
        bins = (cfg.get('postprocess') or {}).get('speed_bins') or list(np.arange(10) * 0.5)
        dv = np.abs(speed_expectation(trt_out['speed_logits'], bins) - speed_expectation(ref_out['speed_logits'], bins))
        acc['speed_exp_max_mps'] = round(float(dv.max()), 5)
        acc['gate_speed_0p05mps'] = bool(dv.max() <= 0.05)
    res['accuracy'] = acc

    # ---- 4. 速さ: run_numpy(走行の呼び方)とエンジンだけ
    feed = {n: frames[n][:1] for n in runner.inputs}
    for _ in range(WARMUP):
        runner.run_numpy(**feed)
    ts = []
    for _ in range(TIMED):
        t0 = time.perf_counter()
        runner.run_numpy(**feed)
        ts.append((time.perf_counter() - t0) * 1000)
    eng = []
    for _ in range(WARMUP):
        runner.context.execute_async_v3(runner.stream.cuda_stream)
        runner.stream.synchronize()
    for _ in range(TIMED):
        t0 = time.perf_counter()
        runner.context.execute_async_v3(runner.stream.cuda_stream)
        runner.stream.synchronize()
        eng.append((time.perf_counter() - t0) * 1000)
    res['latency_ms'] = {'run_numpy_median': pct(ts, 50), 'run_numpy_p99': pct(ts, 99),
                         'engine_median': pct(eng, 50), 'engine_p99': pct(eng, 99),
                         'gate_run_numpy': bool(np.median(ts) <= 6.0 and np.percentile(ts, 99) <= 8.0)}
    gates = [res['model_info']['drivable'], ok, acc['gate_actions_2p5us'], acc['finite'], res['latency_ms']['gate_run_numpy']]
    gates += [acc[k] for k in ('gate_waypoints_0p02m', 'gate_speed_0p05mps') if k in acc]
    res['all_gates'] = bool(all(gates))
    write_json(out_path, res)
    print(json.dumps({'run': res['run'], 'all_gates': res['all_gates'], 'accuracy': acc, 'latency_ms': res['latency_ms']},
                     ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2]))
