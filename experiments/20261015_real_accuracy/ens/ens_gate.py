#!/usr/bin/env python3
"""ENS のエンジンのゲート(protocol.json の arms.ENS_GATE): v0-s2 と msf-s2 の best.pth を 1 つのモジュール(actions の平均)に
まとめ、ONNX → tools/build_engine.py で FP16 のエンジン → 10/12 の engine-check と同じ確かめ。成果物はこのフォルダー(ens/)だけ
(data/runs には置かない = 走行タブの一覧には出ない)。

    cd ~/jetformer && jf-heavy --mem 3G --lock big --need-mb 3200 -- nice -n 10 python -u \\
        experiments/20261015_real_accuracy/ens/ens_gate.py all
  小コマンド(all は 3 つを別のプロセスで順に。出来たものは飛ばす):
    export  CPU: 2 本を読み、平均のモジュールを ONNX(opset 17・バッチ 1・固定の形)に書き出し、実フレーム 64 個の
            PyTorch FP32(CPU)の 2 本の平均を ref.npz に書く
    build   tools/build_engine.py(FP16・作業メモリ 1 GB。このフォルダーの model_config.json の engine 欄)
    check   GPU: TensorRT で 64 フレーム → ref との差(≤ 2.5 µs)・有限、100 回の空回しの後 600 回の run_numpy の
            中央値 ≤ 6 ms・p99 ≤ 8 ms → gate.json
入出力: image [1,3,224,224]・tof [1,2,42,54]・imu [1,10,6]・state [1,3]・mode [1,6] → actions [1,10,2](v0 と同じ)。
車は動かさない・PWM を出さない。
"""

import collections
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = Path('/home/jetson/jetformer')
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'tools'))
RUNS = REPO / 'data' / 'runs'
PAIR = ('realacc_20261015-v0-s2', 'realacc_20261015-msf-s2')
ONNX = HERE / 'ens_p2.onnx'
ENGINE = HERE / 'ens_p2_fp16.engine'
CONFIG = HERE / 'model_config.json'
REF = HERE / 'ref.npz'
GATE = HERE / 'gate.json'
GATES = {'actions_max_us': 2.5, 'run_numpy_median_ms': 6.0, 'run_numpy_p99_ms': 8.0}
N_FRAMES, WARMUP, TIMED = 64, 100, 600
PY = sys.executable


def _write_json(path, obj):
    tmp = str(path) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def make_average_module(models):
    """複数の方策 → forward(image, tof, imu, state, mode) が actions の平均を返す nn.Module(ONNX に書き出す形)"""
    import torch
    from jetformer.policies import runtime

    class Average(torch.nn.Module):
        def __init__(self, ms):
            super().__init__()
            self.ms = torch.nn.ModuleList(ms)

        def forward(self, image, tof, imu, state, mode):
            acts = [runtime.as_outputs(m(image=image, tof=tof, imu=imu, state=state, mode=mode))['actions'] for m in self.ms]
            return torch.stack(acts, 0).mean(0)

    return Average(models).eval()


def cmd_export():
    import torch
    import policy_zoo as pz
    from jetformer import policies
    from jetformer.policies import runtime
    if ONNX.exists() and REF.exists() and CONFIG.exists():
        print('export: 作成済み')
        return
    cfgs = [runtime.read_config(RUNS / n) for n in PAIR]
    if cfgs[0]['num_modes'] != cfgs[1]['num_modes'] or cfgs[0].get('command_input') != cfgs[1].get('command_input'):
        raise SystemExit('2 本のジャンル数・command_input が違います')
    models = [runtime.load_run(RUNS / n, 'best', device='cpu')[0].float().eval() for n in PAIR]
    avg = make_average_module(models)
    torch.backends.mha.set_fastpath_enabled(False)
    nm = cfgs[0]['num_modes']
    g = torch.Generator().manual_seed(0)
    mode = torch.zeros(1, nm)
    mode[0, 5] = 1.0
    x = (torch.randn(1, 3, 224, 224, generator=g), torch.rand(1, 2, 42, 54, generator=g), torch.zeros(1, 10, 6),
         torch.zeros(1, 3), mode)
    tmp = str(ONNX) + '.tmp'
    t0 = time.time()
    with torch.no_grad():
        torch.onnx.export(avg, x, tmp, input_names=list(runtime.BASE_INPUTS), output_names=['actions'], opset_version=17,
                          dynamo=False)
    io = runtime.read_onnx_io(tmp)
    ok, why = policies.check_io(io['input_shapes'], io['output_shapes'], nm)
    if not ok:
        raise SystemExit('書き出した ONNX が走行の約束に合いません: %s' % why)
    os.replace(tmp, ONNX)
    # 実フレーム 64 個(10/12 の engine-check と同じ選び方)の PyTorch FP32(CPU)の 2 本の平均
    frames = pz._engine_frames(cfgs[0])
    torch.set_num_threads(4)
    ref = {n: [] for n in ('avg', 'm0', 'm1')}
    for s in range(0, N_FRAMES, 8):
        sl = {k: v[s:s + 8] for k, v in frames.items()}
        outs = [runtime.predict(m, sl)['actions'] for m in models]
        ref['m0'].append(outs[0])
        ref['m1'].append(outs[1])
        ref['avg'].append((outs[0] + outs[1]) / 2.0)
        with torch.no_grad():
            direct = avg(**{k: torch.from_numpy(np.asarray(v)) for k, v in sl.items()}).numpy()
        if np.abs(direct - ref['avg'][-1]).max() * 500.0 > 1e-3:
            raise SystemExit('平均のモジュールの出力が 2 本の平均と合いません')
    np.savez(REF, **{k: np.concatenate(v) for k, v in ref.items()}, **{'in_' + k: v for k, v in frames.items()})
    cfg = {'policy': 'ens_average', 'members': list(PAIR), 'num_modes': nm, 'trained_modes': cfgs[0]['trained_modes'],
           'command_input': cfgs[0].get('command_input'), 'inputs': io['inputs'], 'input_shapes': io['input_shapes'],
           'outputs': io['outputs'], 'output_shapes': io['output_shapes'], 'engine': {'precision': 'fp16', 'workspace_gb': 1.0},
           'note': '実機の精度の試し(real_accuracy_20261015)の ENS のゲート用。走行タブには出さない(data/runs の外)'}
    _write_json(CONFIG, cfg)
    print('export: %s(%.0f 秒)・ref %d フレーム' % (ONNX.name, time.time() - t0, N_FRAMES), flush=True)


def cmd_build():
    if ENGINE.exists() and ENGINE.stat().st_mtime >= ONNX.stat().st_mtime:
        print('build: 作成済み')
        return
    t0 = time.time()
    r = subprocess.run([PY, '-u', str(REPO / 'tools' / 'build_engine.py'), str(ONNX), str(ENGINE)], cwd=str(REPO))
    if r.returncode != 0 or not ENGINE.exists():
        raise SystemExit('エンジンの作成に失敗しました(終了コード %d)' % r.returncode)
    print('build: %s(%.0f 秒)' % (ENGINE.name, time.time() - t0), flush=True)


def cmd_check():
    import torch
    from jetformer import policies
    from jetformer.trt_runtime import TRTRunner
    z = np.load(REF)
    frames = {k[3:]: z[k] for k in z.files if k.startswith('in_')}
    ref = z['avg']
    torch.zeros(1, device='cuda')
    torch.cuda.synchronize()
    runner = TRTRunner(str(ENGINE))
    nm = json.loads(CONFIG.read_text())['num_modes']
    ok, why = policies.check_io({n: list(runner.shapes[n]) for n in runner.inputs},
                                {n: list(runner.shapes[n]) for n in runner.outputs}, nm)
    res = {'pair': list(PAIR), 'engine': str(ENGINE), 'engine_mb': round(ENGINE.stat().st_size / 2 ** 20, 1),
           'checked_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'check_io': {'ok': ok, 'why': why}}
    if not ok:
        res['all_gates'] = False
        _write_json(GATE, res)
        raise SystemExit('check_io: ' + why)
    zeros = {n: np.zeros(runner.shapes[n], np.float32) for n in runner.inputs}
    for _ in range(3):
        runner.run_numpy(**zeros)
    got = []
    for i in range(N_FRAMES):
        got.append(runner.run_numpy(**{n: frames[n][i:i + 1] for n in runner.inputs})['actions'])
    got = np.concatenate(got)
    da = np.abs(got - ref) * policies.PWM_SCALE
    d0 = np.abs(z['m0'] - ref) * policies.PWM_SCALE
    acc = {'actions_max_us': round(float(da.max()), 3), 'actions_mean_us': round(float(da.mean()), 4),
           'actions_p99_us': round(float(np.percentile(da, 99)), 3), 'finite': bool(np.isfinite(got).all()),
           'gate_actions_2p5us': bool(da.max() <= GATES['actions_max_us']),
           'ref_member_vs_avg_mean_us': round(float(d0.mean()), 2)}
    feed = {n: frames[n][:1] for n in runner.inputs}
    for _ in range(WARMUP):
        runner.run_numpy(**feed)
    ts = []
    for _ in range(TIMED):
        t0 = time.perf_counter()
        runner.run_numpy(**feed)
        ts.append((time.perf_counter() - t0) * 1000)
    en = []
    for _ in range(WARMUP):
        runner.context.execute_async_v3(runner.stream.cuda_stream)
        runner.stream.synchronize()
    for _ in range(TIMED):
        t0 = time.perf_counter()
        runner.context.execute_async_v3(runner.stream.cuda_stream)
        runner.stream.synchronize()
        en.append((time.perf_counter() - t0) * 1000)
    pct = lambda x, q: round(float(np.percentile(np.asarray(x), q)), 3)  # noqa: E731
    lat = {'run_numpy_median': pct(ts, 50), 'run_numpy_p99': pct(ts, 99), 'engine_median': pct(en, 50), 'engine_p99': pct(en, 99)}
    lat['gate_run_numpy'] = bool(lat['run_numpy_median'] <= GATES['run_numpy_median_ms']
                                 and lat['run_numpy_p99'] <= GATES['run_numpy_p99_ms'])
    res.update(accuracy=acc, latency_ms=lat, gates=GATES,
               all_gates=bool(ok and acc['gate_actions_2p5us'] and acc['finite'] and lat['gate_run_numpy']),
               method='実フレーム 64 個(clean_driving の検証から等間隔・10/12 の engine-check と同じ)で TensorRT(run_numpy)と '
                      'PyTorch FP32(CPU・2 本の best.pth の平均)の actions の差。速さは 100 回の空回しの後 600 回')
    _write_json(GATE, res)
    print(json.dumps({'all_gates': res['all_gates'], 'accuracy': acc, 'latency_ms': lat}, ensure_ascii=False), flush=True)
    del runner
    gc.collect()


def cmd_all():
    for sub, env in (('export', {'CUDA_VISIBLE_DEVICES': ''}), ('build', {}), ('check', {})):
        r = subprocess.run([PY, '-u', __file__, sub], cwd=str(REPO), env=dict(os.environ, **env))
        if r.returncode != 0:
            raise SystemExit('%s が失敗しました(終了コード %d)' % (sub, r.returncode))


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ('export', 'build', 'check', 'all'):
        raise SystemExit(__doc__)
    {'export': cmd_export, 'build': cmd_build, 'check': cmd_check, 'all': cmd_all}[sys.argv[1]]()


if __name__ == '__main__':
    main()
