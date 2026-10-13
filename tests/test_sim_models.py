"""/api/sim/models: シミュレーターで使えるモデル(jetformer_v0 だけ)のテスト"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from jetformer import model_export
from jetformer.sim import web


def test_only_jetformer_v0_is_sim_compatible(monkeypatch):
    models = [{'name': 'new_tcp', 'policy': 'tcp_v1', 'created_at': '2026-10-13T10:00:00'},
              {'name': 'v0_run', 'policy': 'jetformer_v0', 'created_at': '2026-10-12T10:00:00'},
              {'name': 'old_run', 'created_at': '2026-10-01T10:00:00'}]
    monkeypatch.setattr(model_export, 'list_models', lambda: [dict(m) for m in models])
    app = FastAPI()
    app.include_router(web.router)
    r = TestClient(app).get('/api/sim/models').json()
    assert [m['name'] for m in r] == ['new_tcp', 'v0_run', 'old_run']          # 並び順はそのまま(新しい順)
    assert [m['sim_compatible'] for m in r] == [False, True, True]
    assert 'jetformer_v0' in r[0]['reason'] and r[1]['reason'] is None
