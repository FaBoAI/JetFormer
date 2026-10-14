#!/usr/bin/env python3
"""ブリッジ用の一式(export_bundle.py が作ったもの)を、シミュレーターの Python とユーザーのブリッジの関数で確かめる(CPU)

    /home/jetson/jetformer/data/sim/venv/bin/python -I -B check_bundle.py <一式のフォルダー>

使う Python は onnxruntime 1.31.0・numpy 2.2.6・opencv 4.13.0 が入ったシミュレーター用の venv(読むだけ。何も入れない)。
ブリッジとアダプター(bridge.py・jetformer_policy.py)は、レポートのフォルダーのシミュレーターの写し(元とバイト単位で同じことを
最初に確かめる)から読み込む。HTTP のサーバーは動かさない(待ち受けの口は作るが、すぐ閉じる。ポートは自分の範囲の 8793)。

1. アプリの「モデル ZIP を読み込む」と同じ順: BridgeServer.save_bundle(ZIP のバイト列)→ activate_model({onnx_token,
   config_token, calibration_token, mode_id 8})。中で validate_io_spec・simulator_profile・Calibration・ONNX のセッションの検査・
   0 の入力での空回しが走る
2. 読み込んだ結果(policy_info)の走行モード 8・学習したジャンル [8]・先読み 1・ゲイン 1.0 / 1.0・中心 1442 / 1498・遅れ 1 を確かめる
3. ブリッジが作る mode の入力(長さ num_modes の one-hot の 8 番目)が、学習のデータの mode の入力と同じことを確かめる
4. ONNX(onnxruntime CPU、バッチ 1)と PyTorch FP32(export_bundle.py が保存した 16 フレームの出力)の actions の差 ≤ 0.5 µs
結果 → <一式のフォルダー>/bundle_check.json(ok が true なら合格)。ブリッジが保存したアップロードの写しは最後に消す。
"""

import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPORT = Path('/home/jetson/jetformer_reports/sim_rl_bestlap_20261013')
SIM_COPY = REPORT / 'sim'
ORIGINAL = Path('/home/jetson/sim/extracted/course-simulator')
PROTOCOL = REPORT / 'protocol.json'
PORT = 8793
SIM_MODE = 8
GATE_ONNX_US = 0.5
PWM_SCALE = 500.0


def sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def sha256(path):
    return sha256_bytes(Path(path).read_bytes())


def write_json(path, obj):
    tmp = str(path) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=str)
    os.replace(tmp, path)


def main(bundle_dir):
    bundle_dir = Path(bundle_dir).resolve()
    name = bundle_dir.name
    res = {'run': name, 'checked_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'python': sys.version.split()[0]}
    # ---- 写しが元と同じか(ユーザーのコードをそのまま使う)
    proto = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    same = {}
    for fn in ('bridge.py', 'jetformer_policy.py', 'calibration.example.json'):
        same[fn] = sha256(SIM_COPY / fn) == sha256(ORIGINAL / fn)
    same['jetformer_policy_vs_protocol'] = sha256(SIM_COPY / 'jetformer_policy.py') == proto['files']['jetformer_policy_adapter']['sha256']
    res['sim_copy_identical'] = same
    if not all(same.values()):
        write_json(bundle_dir / 'bundle_check.json', dict(res, ok=False, why='シミュレーターの写しが元と違います'))
        return 1
    sys.path.insert(0, str(SIM_COPY))
    import numpy as np
    import onnxruntime as ort
    import bridge
    import jetformer_policy
    res['onnxruntime'] = ort.__version__
    res['numpy'] = np.__version__
    exp = json.loads((bundle_dir / 'export.json').read_text(encoding='utf-8'))
    zpath = Path(exp['zip'])
    raw = zpath.read_bytes()
    if sha256_bytes(raw) != exp['zip_sha256']:
        write_json(bundle_dir / 'bundle_check.json', dict(res, ok=False, why='ZIP の sha256 が export.json と違います'))
        return 1
    res['zip'] = str(zpath)
    res['zip_sha256'] = exp['zip_sha256']
    # ---- 1. アプリと同じ順でブリッジに読み込ませる
    runtime_root = REPORT / 'train' / 'bundle_runtime' / name
    if runtime_root.exists():
        shutil.rmtree(runtime_root)
    runtime_root.mkdir(parents=True)
    server = bridge.BridgeServer(PORT, SIM_COPY / 'dist', bridge.PolicyRunner(), runtime_root=runtime_root)
    try:
        t = time.time()
        saved = server.save_bundle(raw)
        res['save_bundle'] = {'defaults': saved['defaults'], 'ignored_files': saved['ignored_files'],
                              'model_sha256': saved['model_sha256'], 'executed_bundle_code': saved['executed_bundle_code']}
        tokens = {'onnx_token': saved['onnx_token'], 'config_token': saved['config_token']}
        if saved.get('calibration_token'):
            tokens['calibration_token'] = saved['calibration_token']
        status = server.activate_model(dict(tokens, mode_id=SIM_MODE))
        res['activate_s'] = round(time.time() - t, 2)
        info = status.get('policy_info') or {}
        got = {'policy_loaded': status.get('policy_loaded'), 'name': status.get('name'),
               'mode_id': info.get('mode_id'), 'mode_name': info.get('mode_name'), 'trained_modes': info.get('trained_modes'),
               'num_modes': (info.get('model_config') or {}).get('num_modes'),
               'command_input': info.get('command_input'), 'steer_center_us': info.get('steer_center_us'),
               'throttle_neutral_us': info.get('throttle_neutral_us'), 'lookahead': info.get('lookahead'),
               'steer_gain': info.get('steer_gain'), 'throttle_gain': info.get('throttle_gain'),
               'control_delay_frames': (info.get('simulator_execution') or {}).get('control_delay_frames'),
               'longitudinal_model': (info.get('simulator_physics') or {}).get('longitudinal_model'),
               'max_steering_rad': info.get('max_steering_rad'), 'max_speed_mps': info.get('max_speed_mps'),
               'bundle_default_calibration': info.get('bundle_default_calibration'),
               'policy_format': info.get('policy_format'), 'onnx_inputs': info.get('onnx_inputs')}
        res['activated'] = got
        want = {'policy_loaded': True, 'mode_id': SIM_MODE, 'trained_modes': [SIM_MODE], 'num_modes': 9,
                'command_input': False, 'steer_center_us': 1442.0, 'throttle_neutral_us': 1498.0, 'lookahead': 1,
                'steer_gain': 1.0, 'throttle_gain': 1.0, 'control_delay_frames': 1, 'bundle_default_calibration': True,
                'policy_format': 'jetformer_v0', 'longitudinal_model': 'real-drive-response-estimate-v1',
                'max_steering_rad': 0.52, 'max_speed_mps': 7.24}
        res['activated_expected'] = {k: (got.get(k) == v or (isinstance(v, list) and list(got.get(k) or []) == v))
                                     for k, v in want.items()}
        res['defaults_expected'] = saved['defaults'] == {'mode_id': SIM_MODE, 'lookahead': 1, 'steer_gain': 1.0,
                                                         'throttle_gain': 1.0}
        res['info_keys'] = sorted(info)
        res['simulator_physics'] = info.get('simulator_physics')
    finally:
        server.server_close()
        shutil.rmtree(runtime_root, ignore_errors=True)
    # ---- 3. ブリッジの mode の入力 = 学習の mode の入力
    cfg = json.loads((bundle_dir / 'model_config.json').read_text(encoding='utf-8'))
    bridge_mode = np.zeros((1, int(cfg['num_modes'])), np.float32)
    bridge_mode[0, SIM_MODE] = 1.0                       # jetformer_policy.Controller.predict と同じ作り方
    ref = np.load(bundle_dir.parent / '_check' / ('%s_torch_ref.npz' % name), allow_pickle=False)
    res['mode_input'] = {'bridge_len': int(bridge_mode.shape[1]), 'train_len': int(ref['mode'].shape[1]),
                         'same_as_training': bool((ref['mode'] == bridge_mode).all())}
    # ---- 4. ONNX(バッチ 1)と PyTorch FP32 の差
    sess = ort.InferenceSession((bundle_dir / 'jetformer.onnx').read_bytes(), providers=['CPUExecutionProvider'])
    names = [i.name for i in sess.get_inputs()]
    outs = []
    for i in range(ref['actions'].shape[0]):
        outs.append(sess.run(['actions'], {n: ref[n][i:i + 1] for n in names})[0])
    onnx_actions = np.concatenate(outs)
    d = np.abs(onnx_actions - ref['actions']) * PWM_SCALE
    res['onnx_vs_torch_max_us'] = round(float(d.max()), 5)
    res['onnx_vs_torch_mean_us'] = round(float(d.mean()), 6)
    res['onnx_inputs'] = names
    res['onnx_l1_vs_label_us'] = round(float(np.abs(onnx_actions - ref['target']).mean() * PWM_SCALE), 2)
    gates = [all(res['activated_expected'].values()), res['defaults_expected'], res['mode_input']['same_as_training'],
             res['onnx_vs_torch_max_us'] <= GATE_ONNX_US, bool(np.isfinite(onnx_actions).all())]
    res['gates'] = {'activated': gates[0], 'defaults': gates[1], 'mode_input': gates[2], 'onnx_vs_torch': gates[3],
                    'finite': gates[4]}
    res['defaults'] = saved['defaults']
    res['ok'] = bool(all(gates))
    write_json(bundle_dir / 'bundle_check.json', res)
    print(json.dumps({'run': name, 'ok': res['ok'], 'gates': res['gates'], 'onnx_vs_torch_max_us': res['onnx_vs_torch_max_us'],
                      'activated': got}, ensure_ascii=False, default=str), flush=True)
    return 0 if res['ok'] else 1


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    sys.exit(main(sys.argv[1]))
