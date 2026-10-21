"""段 S2 の画面の確認用: 本物の画面(index.html・app.js・style.css)と本物の /api/bench/candidates を、
偽の走行の状態で出す小さなサーバー(センサー・PCA9685・カメラに触れない。jetformer-web とは別のポート)

    python fake_server.py --port 18081 --work /tmp/... [--old]     # --old = 再起動前のサーバー(上限の項目・bench の API なし)

- 学習結果は work/runs の偽物(エンジンは中身の無いファイル)。data/bench_candidates.json は work/bench_candidates.json。
- POST /api/drive/params・/api/drive/start は記録するだけ(start は 409 を返す。走らない)。GET /api/_calls で記録を返す。
"""

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from fastapi import Body, FastAPI, HTTPException  # noqa: E402
from fastapi.responses import Response  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from jetformer import drive  # noqa: E402
from jetformer.web import bench  # noqa: E402

STATIC = ROOT / 'jetformer' / 'web' / 'static'
B = 'clean_driving_20261011-nocmd-s3-20261011-024626'
FIRST = 'first_trial_20261010-s3-20261010-153629'
MSF = 'clean_driving_20261011-msfusion-s0-nocmd-stride3-e30'
SLOW = 'bench_20261021-slow-s0'
TPL = ('bench=real_bench_20261023 day={date} block={block} cand={id} model={run} '
       'battery=#__ V_before=__ V_after=__ check=__ memo=')


def make_runs(work):
    runs = work / 'runs'
    for name, created, policy in ((FIRST, '2026-10-10T15:36:29', None), (B, '2026-10-11T02:46:26', None),
                                  (MSF, '2026-10-11T20:00:00', 'msfusion'), (SLOW, '2026-10-21T12:00:00', None)):
        d = runs / name
        d.mkdir(parents=True, exist_ok=True)
        (d / drive.ENGINE_FILE).write_bytes(b'fake-engine')
        cfg = {'num_modes': 6, 'trained_modes': [1, 2, 3, 4, 5], 'steer_center_us': 1441.0, 'throttle_neutral_us': 1496.0,
               'dataset': 'clean_driving_20261011', 'stride': 3,
               'modes': [{'id': i, 'name': n} for i, n in enumerate(['未設定', '低速', '中速', '高速', '練習', '本番'])]}
        if policy:
            cfg['policy'] = policy
        (d / 'model_config.json').write_text(json.dumps(cfg))
        (d / 'run.json').write_text(json.dumps({'created_at': created}))
    (work / 'drive_recommended.json').write_text(json.dumps({'run': B, 'reason': '方策の比較の評価: ベスト = JetFormer v0'}))
    doc = {'version': 1, 'bench': 'real_bench_20261023', 'protocol_sha256': '1dbb279eba37031a301784868fd6b286eb37acf5f1976b24b5242268f0e0a2bb',
           'created_at': '2026-10-21T13:00:00',
           'baseline': {'mode_id': 5, 'params': dict(drive.DEFAULT_PARAMS, steer_gain=1.0, throttle_gain=1.0, lookahead=3,
                                                     tof_guard=1, ensemble_j=0, speed_cap=0, zone_caps=0)},
           'candidates': [
               {'id': 'B', 'label': 'モデルB(基準)', 'run': B, 'hypothesis': '基準', 'risk': 'low'},
               {'id': 'SLOW', 'label': '遅く入る型', 'run': SLOW, 'hypothesis': 'R1 の手前で自分で緩める', 'risk': 'low'},
               {'id': 'ENS', 'label': '2本の平均', 'run': 'bench_20261021-ens-p2', 'hypothesis': '机上で一番良い → 衝突が減るか', 'risk': 'mid'},
               {'id': 'MSF', 'label': '多段型', 'run': MSF, 'hypothesis': '融合を何段も → ずれに強い', 'risk': 'mid'}],
           'days': [{'date': '2026-10-23', 'blocks': [{'block': 1, 'id': 'B'}, {'block': 2, 'id': 'SLOW'}, {'block': 3, 'id': 'ENS'},
                                                      {'block': 4, 'id': 'B'}, {'block': 5, 'id': 'MSF'}, {'block': 6, 'id': 'B'}]},
                    {'date': '2026-10-24', 'blocks': [], 'note': '10/23 の夜に決める'}],
           'description_template': TPL}
    (work / 'bench_candidates.json').write_text(json.dumps(doc, ensure_ascii=False), encoding='utf-8')


def build(work, old=False):
    drive.RUNS_DIR = work / 'runs'
    drive.RECOMMENDED_PATH = work / 'drive_recommended.json'
    bench.BENCH_PATH = work / 'bench_candidates.json'
    calls = []
    # 今の保存値(data/drive_settings.json と同じ: ステア 1.5・first_trial)
    params = dict(drive.DEFAULT_PARAMS, steer_gain=1.5, throttle_gain=1.0, lookahead=3, tof_guard=1)
    if old:
        params.pop('zone_caps')
    st = {'params': params}
    app = FastAPI()
    if not old:
        app.include_router(bench.router)

    @app.get('/')
    def index():
        html = (STATIC / 'index.html').read_text(encoding='utf-8')
        for name in ('app.js', 'style.css'):
            html = html.replace('/static/%s"' % name, '/static/%s?v=%d"' % (name, int((STATIC / name).stat().st_mtime)))
        return Response(html, media_type='text/html', headers={'Cache-Control': 'no-cache'})

    @app.get('/api/status')
    def status():
        return {'time': time.time(), 'sensors_running': False, 'sensors': {}, 'recorder': {'recording': False, 'saved_scenes': []},
                'session': {}, 'disk': {'free_gb': 41.2, 'total_gb': 250.0}, 'training': False, 'driving': False,
                'config': {'scene_seconds': 20, 'vehicle_protocol': '', 'tof_unit': '', 'camera': ''}}

    @app.get('/api/drive/models')
    def models():
        return drive.list_models()

    @app.get('/api/drive/status')
    def drive_status():
        out = {'phase': 'idle', 'message': '停止中', 'error': None, 'running': False, 'params': dict(st['params']),
               'last': {'run': FIRST, 'mode_id': 5}, 'recording': False, 'elapsed_s': 0, 'stats': {}, 'now': None,
               'history': {'t': []}, 'ref': None,
               'pca9685': {'address': 64, 'freq_hz': 60, 'steering_ch': 0, 'throttle_ch': 1},
               'limits': {k: list(v) for k, v in drive.PARAM_LIMITS.items()}}
        if not old:
            out['zone_caps_spec'] = drive.ZONE_CAPS_SPEC
        return out

    @app.post('/api/drive/params')
    def drive_params(body: dict = Body(...)):
        calls.append({'path': '/api/drive/params', 'body': body})
        for k, v in body.items():
            if k in st['params'] or (k == 'zone_caps' and not old):
                st['params'][k] = v
        return dict(st['params'])

    @app.post('/api/drive/start')
    def drive_start(body: dict = Body(...)):
        calls.append({'path': '/api/drive/start', 'body': body})
        raise HTTPException(409, '確認用のサーバーなので走りません')

    @app.get('/api/_calls')
    def get_calls():
        return calls

    @app.get('/api/{rest:path}')
    def anything(rest: str):
        if rest in ('scenes', 'modes', 'jobs', 'datasets', 'runs', 'policies', 'models'):
            return []
        raise HTTPException(404)

    app.mount('/static', StaticFiles(directory=str(STATIC)), name='static')
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=18081)
    ap.add_argument('--work', required=True)
    ap.add_argument('--old', action='store_true')
    a = ap.parse_args()
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    make_runs(work)
    import uvicorn
    uvicorn.run(build(work, a.old), host='127.0.0.1', port=a.port, log_level='warning')


if __name__ == '__main__':
    main()
