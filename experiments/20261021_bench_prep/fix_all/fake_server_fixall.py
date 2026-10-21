"""Fix-all の画面の確認用: 段 S2 の偽のサーバー(ui_s2/fake_server.py)に、走行の状態を上書きする口を足したもの。
POST /api/_fixall_status {...} → 以後の GET /api/drive/status にその値を重ねる({} で戻す)。jetformer-web(8080)は使わない"""
import argparse
import json
import os
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'ui_s2'))
import fake_server as fs  # noqa: E402
from starlette.responses import JSONResponse  # noqa: E402

OVR = {}


def build(work, old=False):
    app = fs.build(work, old)

    @app.middleware('http')
    async def mw(request, call_next):
        if request.url.path == '/api/_fixall_status' and request.method == 'POST':
            OVR.clear()
            OVR.update(await request.json())
            return JSONResponse({'ok': True})
        resp = await call_next(request)
        if request.url.path == '/api/drive/status' and OVR:
            body = b''.join([c async for c in resp.body_iterator])
            d = json.loads(body)
            d.update(OVR)
            return JSONResponse(d)
        return resp
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=18083)
    ap.add_argument('--work', required=True)
    ap.add_argument('--old', action='store_true')
    a = ap.parse_args()
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    fs.make_runs(work)
    import uvicorn
    uvicorn.run(build(work, a.old), host='127.0.0.1', port=a.port, log_level='warning')


if __name__ == '__main__':
    main()
