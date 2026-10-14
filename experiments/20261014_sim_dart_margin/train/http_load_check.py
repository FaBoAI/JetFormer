#!/usr/bin/env python3
"""ユーザーのブリッジ(bridge.py の写し)を HTTP で立て、アプリの「モデル ZIP を読み込む」と同じ API で一式を読み込めるか確かめる(CPU)

    /home/jetson/jetformer/data/sim/venv/bin/python -I -B http_load_check.py <一式の ZIP> [<一式の ZIP> ...]

check_bundle.py はブリッジの関数(save_bundle・activate_model)を直接呼ぶ。こちらは実際に待ち受けの口(この作業の範囲の 8815)を開き、
アプリと同じ順で HTTP を送る: POST /api/models/bundle(application/zip)→ POST /api/models/activate({onnx_token, config_token,
calibration_token, mode_id 8})→ GET /api/status。ユーザーのブリッジ・jetformer-web には触らない(別のプロセス・別の口・
一時フォルダーの runtime)。ブリッジが保存したアップロードの写しは最後に消す。
結果 → 標準出力の JSON(ZIP ごとに ok)。全部合格なら終了コード 0。
"""

import hashlib
import json
import shutil
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

SIM_COPY = Path('/home/jetson/jetformer_reports/sim_dart_20261014/sim')
ORIGINAL = Path('/home/jetson/sim/extracted/course-simulator')
PORT = 8815
SIM_MODE = 8


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def request(method, path, body=None, ctype=None):
    req = urllib.request.Request('http://127.0.0.1:%d%s' % (PORT, path), data=body, method=method)
    if ctype:
        req.add_header('Content-Type', ctype)
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return r.status, json.loads(r.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode('utf-8') or '{}')


def check_one(zpath):
    raw = Path(zpath).read_bytes()
    out = {'zip': str(zpath), 'zip_sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
    t = time.time()
    st, saved = request('POST', '/api/models/bundle', raw, 'application/zip')
    out['bundle_status'] = st
    if st != 200:
        out.update(ok=False, why='bundle %s' % saved)
        return out
    out['defaults'] = saved.get('defaults')
    body = {'onnx_token': saved['onnx_token'], 'config_token': saved['config_token'], 'mode_id': SIM_MODE}
    if saved.get('calibration_token'):
        body['calibration_token'] = saved['calibration_token']
    st, status = request('POST', '/api/models/activate', json.dumps(body).encode('utf-8'), 'application/json')
    out['activate_status'] = st
    out['load_s'] = round(time.time() - t, 2)
    if st != 200:
        out.update(ok=False, why='activate %s' % status)
        return out
    st, status = request('GET', '/api/status')
    info = status.get('policy_info') or {}
    got = {'policy_loaded': status.get('policy_loaded'), 'name': status.get('name'), 'mode_id': info.get('mode_id'),
           'trained_modes': info.get('trained_modes'), 'steer_center_us': info.get('steer_center_us'),
           'throttle_neutral_us': info.get('throttle_neutral_us'), 'lookahead': info.get('lookahead'),
           'steer_gain': info.get('steer_gain'), 'throttle_gain': info.get('throttle_gain'),
           'control_delay_frames': (info.get('simulator_execution') or {}).get('control_delay_frames'),
           'bundle_default_calibration': info.get('bundle_default_calibration'), 'model_sha256': info.get('model_sha256')}
    want = {'policy_loaded': True, 'mode_id': SIM_MODE, 'trained_modes': [SIM_MODE], 'steer_center_us': 1442.0,
            'throttle_neutral_us': 1498.0, 'lookahead': 1, 'steer_gain': 1.0, 'throttle_gain': 1.0, 'control_delay_frames': 1,
            'bundle_default_calibration': True}
    out['status'] = got
    out['expected'] = {k: got.get(k) == v for k, v in want.items()}
    out['ok'] = bool(st == 200 and all(out['expected'].values()))
    return out


def main(zips):
    same = {fn: sha256(SIM_COPY / fn) == sha256(ORIGINAL / fn) for fn in ('bridge.py', 'jetformer_policy.py')}
    if not all(same.values()):
        print(json.dumps({'ok': False, 'why': 'シミュレーターの写しが元と違います', 'same': same}, ensure_ascii=False))
        return 1
    sys.path.insert(0, str(SIM_COPY))
    import bridge
    tmp = Path(tempfile.mkdtemp(prefix='jf_http_load_'))
    server = bridge.BridgeServer(PORT, SIM_COPY / 'dist', bridge.PolicyRunner(), runtime_root=tmp)
    th = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': 0.2}, daemon=True)
    th.start()
    results = []
    try:
        for z in zips:
            results.append(check_one(z))
    finally:
        server.shutdown()
        server.server_close()
        shutil.rmtree(tmp, ignore_errors=True)
    res = {'checked_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'port': PORT, 'sim_copy_identical': same,
           'results': results, 'ok': bool(results) and all(r.get('ok') for r in results)}
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0 if res['ok'] else 1


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    sys.exit(main(sys.argv[1:]))
