#!/usr/bin/env python3
"""推奨のエンジン(評価の段のベスト)を、走行と同じ手順で読めて回せるかの確認(GPU、1プロセス。PWM は出さない・車は動かさない)

    python drive_path_check.py <学習結果のフォルダー> <結果の json>

smoke/engine_check.py(段階3)の写し。変えたのは、走行のループの場合に A/B/A のカードの固定設定(ステアのゲイン 1.3・
速度のゲイン 1.0・先読み 3・トリム 0)を足したことと、流すフレームをジャンル 5(低速走行)のシーンから選ぶことだけ。
1. 検証の実フレーム 64 個で PyTorch FP32(CPU)と TensorRT(run_numpy)の出力の差
2. 速さ: run_numpy と、エンジンだけを 100 回の空回しの後 600 回
3. drive.model_info → TRTRunner → policies.check_io → 空回し → Driver._loop を本物のエンジンで回し、記録したシーンのフレームを
   30fps で流す。PCA9685 は偽物(I2C に触れない)、記録も偽物、設定ファイルは一時ファイル
"""

import collections
import gc
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO = Path('/home/jetson/jetformer')
sys.path.insert(0, str(REPO))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

DATA = REPO / 'data' / 'datasets' / 'clean_driving_20261011'
N_FRAMES = 64
WARMUP, TIMED = 100, 600
LOOP_FRAMES = 45


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
    ds = JetFormerDataset(str(DATA), 'val', stride=1, use_command=cfg.get('command_input', True), mask_impacts=True,
                          impact_g=2.5)
    idx = np.linspace(0, len(ds) - 1, N_FRAMES).round().astype(int)
    cols = [ds[int(i)] for i in idx]
    names = ('image', 'tof', 'imu', 'state', 'mode')
    out = {n: np.stack([c[k].numpy() for c in cols]).astype(np.float32) for k, n in enumerate(names)}
    if out['mode'].shape[1] != cfg['num_modes']:
        raise SystemExit('ジャンル数が違います: データ %d / モデル %d' % (out['mode'].shape[1], cfg['num_modes']))
    return out, idx


def torch_reference(run_dir, frames):
    """PyTorch FP32(CPU)の出力 {名前: [64,...]}"""
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


class FakePCA(object):
    """PCA9685 の偽物(I2C に触れない。書いた値を残すだけ)"""

    def __init__(self):
        self.pairs = []

    def set_pair_us(self, sc, s, tc, t):
        self.pairs.append((s, t))

    def set_us(self, ch, us):
        pass

    def set_osc_scale(self, scale):
        pass


class FakeRecorder(object):
    recording = False

    def __init__(self):
        self.rows = []

    def add_ai(self, t_ns, data):
        self.rows.append((t_ns, data))


def scene_for_loop(cfg):
    """走行のループに流す、記録したシーンの連続フレーム(検証の最初のシーンの中ほど、学習したジャンル)"""
    splits = json.load(open(DATA / 'splits.json'))
    trained = set(cfg.get('trained_modes') or [])
    if 5 in trained:                                       # A/B/A のカードのジャンル 5(低速走行)
        trained = {5}
    for scene in splits['val']:
        a = np.load(DATA / 'aligned' / (scene + '.npz'))
        ok = a['vehicle_valid'] & np.isin(a['mode'], list(trained))
        start = len(ok) // 3
        for s in range(start, len(ok) - LOOP_FRAMES):
            if ok[s:s + LOOP_FRAMES].all() and (a['vehicle'][s:s + LOOP_FRAMES, 3] < -1000).mean() > 0.5:
                return scene, a, s
    raise SystemExit('ループに流せる連続フレームがありません')


def run_loop(drive, runner, info, ref, a, start, params, num_modes):
    """Driver._loop を本物のエンジンで回す。PCA9685・記録は偽物。戻り値 (driver, fake_pca, フレームの情報, 例外)"""
    from jetformer import clock, config
    from jetformer.sensors.base import Sample
    from jetformer.train_data import normalize_inputs
    d = drive.Driver(hub=None, recorder=FakeRecorder())
    mode_id = int(a['mode'][start])
    d.params = d._clean_params(params)
    d.hb_token = 'smoke'
    d.st = {'phase': 'starting', 'message': '', 'error': None, 'run': info['name'], 'model': info, 'mode_id': mode_id,
            'dry_run': False, 'record': False, 'recording_note': '', 'ref': None, 'started': time.time(),
            'stop_reason': '', 'stats': {}, 'now': None}
    pca = FakePCA()
    out = drive._Output(pca, config.PCA9685, ref, lambda: d.params)
    tof_buf, imu_buf, veh_buf = (collections.deque(maxlen=10), collections.deque(maxlen=60),
                                 collections.deque(maxlen=50))
    cam_slot, cam_evt = [None], threading.Event()
    err = []

    def loop():
        try:
            d._loop(d.st, runner, normalize_inputs, pca, out, ref, num_modes, 0.0, cam_slot, cam_evt,
                    tof_buf, imu_buf, veh_buf, False)
        except Exception as e:          # noqa: BLE001
            err.append(repr(e))
    images = [cv2.imread(str(DATA / str(a['cam_files'][start + i]))) for i in range(LOOP_FRAMES)]   # 先に読む(ループと取り合わない)
    th = threading.Thread(target=loop, daemon=True)
    th.start()
    fed = []
    for i in range(LOOP_FRAMES):
        if not th.is_alive():
            break
        k = start + i
        t = clock.now_ns()
        for j, row in enumerate(a['imu'][k]):              # 撮像までの 10 個の IMU(m/s²・rad/s)
            imu_buf.append((t - (10 - j) * 10_000_000 + 1, row.astype(np.float32)))
        tof_buf.append((t - 5_000_000, a['tof'][k]))
        v = a['vehicle'][k]
        veh_buf.append((t - 3_000_000, {'pwm': [float(v[0]), float(v[1]), float(v[2])], 'valid': [True, True, True],
                                        'select': True, 'moving': abs(float(v[3])) > 0, 'erpm': float(v[3])}))
        image = images[i]
        cam_slot[0] = Sample(t, {'image': image, 'seq': i})
        fed.append({'t_ns': t, 'k': k, 'image': image})
        cam_evt.set()
        d.heartbeat('smoke')
        time.sleep(1 / 30)
    d._stop.set()
    th.join(5)
    return d, pca, fed, (err[0] if err else None)


def same_inputs_as_loop(info, a, k, image, ref):
    """ループと同じ正規化(JPEG の再圧縮・vehicle_state・normalize_inputs)で作った入力(applied=False の最初の形)"""
    from jetformer import config
    from jetformer.train_data import normalize_inputs
    ok, enc = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, config.CAMERA['jpeg_quality']])
    img_d = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    v = a['vehicle'][k]
    vehicle = np.array([ref[0], ref[1], v[2], v[3]], np.float32)    # select=True・前の出力なし → 中心・中立
    img, tof, imu_n, state = normalize_inputs(img_d, a['tof'][k], a['imu'][k].astype(np.float32), vehicle,
                                              info.get('command_input', True))
    return img, tof, imu_n, state


def main(run_dir, out_path):
    run_dir = Path(run_dir).resolve()
    t_all = time.time()
    res = {'run': run_dir.name, 'checked_at': time.strftime('%Y-%m-%d %H:%M:%S')}
    cfg = json.load(open(run_dir / 'model_config.json'))
    res['policy'] = cfg.get('policy', 'jetformer_v0')

    # ---- 1. 実フレームと PyTorch FP32(CPU。GPU を使う前に済ませる)
    frames, idx = real_frames(cfg)
    ref_out, ref_s = torch_reference(run_dir, frames)
    res['torch_ref'] = {'device': 'cpu', 'frames': N_FRAMES, 'seconds': ref_s, 'outputs': sorted(ref_out)}

    # ---- 2. 走行と同じ手順でエンジンを読む(drive._run の 448〜463 行と同じ)
    from jetformer import drive, policies
    from jetformer.policies import post as post_mod
    drive.RUNS_DIR = run_dir.parent
    drive.SETTINGS_PATH = Path(tempfile.mkdtemp(prefix='smoke_drive_')) / 'drive_settings.json'
    info = drive.model_info(run_dir.name)
    if info is None:
        raise SystemExit('drive.model_info が None(エンジンか model_config.json が無い)')
    res['model_info'] = {k: info.get(k) for k in ('policy', 'policy_title', 'drivable', 'why_not', 'inputs', 'outputs',
                                                  'num_modes', 'trained_modes', 'command_input', 'steer_center_us',
                                                  'throttle_neutral_us')}
    res['model_info']['postprocess_kind'] = post_mod.kind_of(info)
    ref = (float(info['steer_center_us']), float(info['throttle_neutral_us']))
    import torch
    from jetformer.trt_runtime import TRTRunner
    torch.zeros(1, device='cuda')
    torch.cuda.synchronize()
    t = time.time()
    runner = TRTRunner(str(drive.RUNS_DIR / info['name'] / drive.ENGINE_FILE))
    num_modes = info['num_modes']
    ok, why = policies.check_io({n: list(runner.shapes[n]) for n in runner.inputs},
                                {n: list(runner.shapes[n]) for n in runner.outputs}, num_modes)
    res['check_io'] = {'ok': ok, 'why': why, 'inputs': {n: list(runner.shapes[n]) for n in runner.inputs},
                       'outputs': {n: list(runner.shapes[n]) for n in runner.outputs}}
    if not ok:
        raise SystemExit('check_io: ' + why)
    zeros = {n: np.zeros(runner.shapes[n], np.float32) for n in runner.inputs}
    for _ in range(3):
        runner.run_numpy(**zeros)
    res['load_s'] = round(time.time() - t, 2)
    res['engine_mb'] = round((run_dir / drive.ENGINE_FILE).stat().st_size / 2 ** 20, 1)

    # ---- 3. TensorRT と PyTorch FP32 の差(実フレーム 64 個)
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
        acc['speed_logits_max_abs'] = round(float(np.abs(trt_out['speed_logits'] - ref_out['speed_logits']).max()), 5)
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

    # ---- 5. 走行のループ(本物のエンジン・偽の PCA9685)
    scene, a, start = scene_for_loop(cfg)
    cases = [('aba_settings', {'steer_gain': 1.3, 'throttle_gain': 1.0, 'lookahead': 3, 'steer_trim_us': 0.0}), ('default', {})]
    if res['model_info']['postprocess_kind'] == 'tcp_controller':
        cases += [('tcp_mode1', {'tcp_mode': 1}), ('tcp_mode2', {'tcp_mode': 2}), ('tcp_mode3', {'tcp_mode': 3}),
                  ('mode0_cap2', {'speed_cap': 2.0}), ('mode1_cap2_scale0.8', {'tcp_mode': 1, 'speed_cap': 2.0,
                                                                               'speed_scale': 0.8})]
    cases.append(('ensemble_j2', {'ensemble_j': 2, 'ensemble_m': 0.5}))
    loops = {}
    for name, extra in cases:
        params = dict(drive.DEFAULT_PARAMS, **extra)
        d, pca, fed, err = run_loop(drive, runner, info, ref, a, start, params, num_modes)
        rows = [r for _, r in d.recorder.rows]
        live = [r for r in rows if not r['hold']]
        holds = [d.st.get('hold_reason')] if len(live) < len(rows) else []
        c = {'error': err, 'frames_fed': len(fed), 'rows': len(rows), 'driven_rows': len(live), 'holds': len(rows) - len(live),
             'hold_reason': holds[0] if holds else None, 'pca_writes': len(pca.pairs),
             'stats': {k: (round(v, 3) if isinstance(v, float) else v) for k, v in (d.st.get('stats') or {}).items()},
             'recording_note': d.st.get('recording_note'), 'post_kind': d.st.get('post_kind')}
        if live:
            mp = np.stack([r['model_pred'] for r in live])
            pr = np.stack([r['pred'] for r in live])
            c['model_pred_finite'] = bool(np.isfinite(mp).all())
            c['pred_equals_model_pred'] = bool(np.array_equal(mp, pr))
            c['pred_minus_model_pred_max_us'] = round(float(np.abs(pr - mp).max()), 2)
            c['infer_ms_median'] = round(float(np.median([r['infer_ms'] for r in live])), 3)
            c['out_us_last'] = [round(float(x), 1) for x in live[-1]['out']]
            c['post_modes'] = sorted({int(r.get('post_mode', -1)) for r in live})
            c['v_hat_median'] = (round(float(np.nanmedian([r['v_hat'] for r in live])), 3)
                                 if any(np.isfinite(r['v_hat']) for r in live) else None)
            c['v_cmd_median'] = (round(float(np.nanmedian([r['v_cmd'] for r in live])), 3)
                                 if any(np.isfinite(r['v_cmd']) for r in live) else None)
            now = d.st.get('now') or {}
            c['now_post'] = now.get('post')
            c['out_steer_left_frac'] = round(float(np.mean([r['out'][0] > ref[0] for r in live])), 3)
            c['out_throttle_forward_frac'] = round(float(np.mean([r['out'][1] < ref[1] for r in live])), 3)
            # ループが記録した model_pred を、同じフレームを同じ正規化で作って run_numpy した値と比べる
            # (prev_pred を入れない今のモデルでは state の操作値は 0、eRPM だけ)
            by_t = {f['t_ns']: f for f in fed}
            diffs = []
            for t_ns, r in d.recorder.rows:
                if r['hold'] or t_ns not in by_t:
                    continue
                f = by_t[t_ns]
                img, tof, imu_n, state = same_inputs_as_loop(info, a, f['k'], f['image'], ref)
                oh = np.zeros((1, num_modes), np.float32)
                oh[0, int(a['mode'][start])] = 1.0
                o = runner.run_numpy(image=img[None], tof=tof[None], imu=imu_n[None], state=state[None], mode=oh)
                diffs.append(float(np.abs(post_mod.Stage.model_pred(o) - r['model_pred']).max()))
                if len(diffs) >= 10:
                    break
            c['loop_vs_direct_max_us'] = round(max(diffs), 4) if diffs else None
            c['loop_vs_direct_frames'] = len(diffs)
        loops[name] = c
    res['drive_loop'] = {'scene': scene, 'start_frame': int(start), 'mode_id': int(a['mode'][start]), 'cases': loops}
    res['ok'] = bool(ok and acc['finite'] and all(c['error'] is None and c['driven_rows'] > 0 and c.get('model_pred_finite')
                                                  for c in loops.values()))
    res['seconds'] = round(time.time() - t_all, 1)
    runner = None
    gc.collect()
    torch.cuda.empty_cache()
    write_json(out_path, res)
    print(json.dumps({'policy': res['policy'], 'ok': res['ok'], 'accuracy': acc, 'latency_ms': res['latency_ms']},
                     ensure_ascii=False))


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
