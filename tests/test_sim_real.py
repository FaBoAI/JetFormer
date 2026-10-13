"""導入済みの本物のシミュレーター(data/sim/releases/<版>/ と data/sim/venv)で、ブリッジを起動して中継経由で確かめる。
導入・準備がまだなら skip。ブリッジは CPU だけ・数秒(runtime は一時フォルダー)"""

import subprocess
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from jetformer.sim import bridge_proc, env, install, paths, web

REL = install.active_release()
READY = REL is not None and env.status()['ready']
pytestmark = pytest.mark.skipif(not READY, reason='シミュレーター本体の導入と Python 環境の準備がまだ')


def test_real_bridge_through_proxy(tmp_path):
    bp = bridge_proc.BridgeProcess(python=paths.venv_python(), runner=paths.RUNNER, sim_dir=REL.root, runtime_root=tmp_path / 'rt',
                                   log_path=tmp_path / 'bridge.log', health_timeout_s=20)
    orig = web.bridge
    web.init(bridge=bp)
    try:
        bp.start()
        t0 = time.time()
        while bp.status()['state'] == 'starting' and time.time() - t0 < 30:
            time.sleep(0.2)
        assert bp.status()['state'] == 'running', bp.status()
        app = FastAPI()
        app.include_router(web.router)
        c = TestClient(app)
        r = c.get('/sim/bridge/api/status', headers={'Host': '192.168.5.14:8080', 'Origin': 'http://192.168.5.14:8080'})
        assert r.status_code == 200
        st = r.json()
        assert st['policy_loaded'] is False and st['schema'] == 'course-sim-observation-v1' and st['model_upload']['available']
        # ブリッジ自身は LAN の Host を拒む(中継が書き換えているから通る)
        import http.client
        h = http.client.HTTPConnection('127.0.0.1', bp.port, timeout=5)
        h.request('GET', '/api/status', headers={'Host': '192.168.5.14:8080'})
        assert h.getresponse().status == 403
        # 予約していない API は中継しない
        assert c.get('/sim/bridge/api/models/list').status_code == 404
        r = c.post('/sim/bridge/api/predict', json={'schema': 'x'})
        assert r.status_code in (400, 503) and r.json()['error']['code']
    finally:
        bp.stop()
        web.init(bridge=orig)
    assert bp.status()['state'] == 'stopped'
    assert not list(REL.root.rglob('__pycache__'))          # ユーザーのファイルの写しに何も作らない


def test_real_self_test():
    p = subprocess.run([str(paths.venv_python()), '-I', '-B', str(paths.RUNNER), '--sim-dir', str(REL.root), '--runtime-root', '/nonexistent',
                        '--self-test', '--nice', '0'], capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    assert 'Ran 13 tests' in p.stdout + p.stderr
