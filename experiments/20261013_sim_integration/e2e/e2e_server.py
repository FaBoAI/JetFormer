"""E2E 確認用の JetFormer の試験インスタンス(jetformer-web を再起動せずに、新しいシミュレーターの経路だけを動かす)

    python e2e_server.py <データのフォルダー> <ポート>

- server.py は import しない(センサー・PCA9685 に触れないため)。jetformer.sim.web の Router と、確認に要る
  /api/jobs・/api/models・/api/models/<run>/bundle.zip だけを持つ。127.0.0.1 にだけ bind する。
- データのフォルダーは JETFORMER_DATA にする(sim/ と runs/ は本物へのシンボリックリンク、datasets/ などは試験用)。
"""
import collections
import contextlib
import os
import sys
import threading
import time
import uuid

DATA, PORT = sys.argv[1], int(sys.argv[2])
os.environ['JETFORMER_DATA'] = DATA
sys.path.insert(0, '/home/jetson/jetformer')

import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402

from jetformer import config, model_export  # noqa: E402
from jetformer.sim import web as sim_web  # noqa: E402

assert str(config.DATA_DIR) == DATA, config.DATA_DIR


class Jobs(object):
    """server.Jobs と同じ形(履歴ファイルは書かない)"""

    def __init__(self):
        self.jobs = collections.OrderedDict()
        self.lock = threading.Lock()

    def submit(self, kind, title, fn):
        job_id = uuid.uuid4().hex[:8]
        job = {'id': job_id, 'kind': kind, 'title': title, 'state': 'running', 'progress': 0.0, 'message': '', 'result': None,
               'error': None, 'started': time.time(), 'finished': None}
        with self.lock:
            self.jobs[job_id] = job

        def progress(p, msg=''):
            job['progress'] = round(float(p), 3)
            job['message'] = str(msg)

        def run():
            try:
                job['result'] = fn(progress)
                job['state'] = 'done'
                job['progress'] = 1.0
            except Exception as e:
                job['state'] = 'error'
                job['error'] = '%s: %s' % (type(e).__name__, e)
            job['finished'] = time.time()
        threading.Thread(target=run, daemon=True).start()
        return job

    def list(self):
        with self.lock:
            return list(reversed(self.jobs.values()))


jobs = Jobs()


@contextlib.asynccontextmanager
async def lifespan(_app):
    yield
    sim_web.shutdown()
    print('e2e server: ブリッジを止めました', flush=True)


app = FastAPI(title='JetFormer sim e2e', lifespan=lifespan)
sim_web.init(jobs=jobs, busy=lambda: {'training': False, 'driving': False, 'recording': False}, web_port=8080)
app.include_router(sim_web.router)


@app.get('/api/jobs')
def api_jobs():
    return jobs.list()


@app.get('/api/models')
def api_models():
    return model_export.list_models()


@app.get('/api/models/{name}/bundle.zip')
def api_model_bundle(name: str):
    try:
        body = model_export.iter_bundle_zip(name)
    except KeyError:
        raise HTTPException(404, 'モデルがありません')
    return StreamingResponse(body, media_type='application/zip')


# 管理画面(シミュレーター タブの表示の確認用。ほかのタブの API はこの試験インスタンスにはない)
from fastapi.responses import Response  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

STATIC = config.ROOT / 'jetformer' / 'web' / 'static'
app.mount('/static', StaticFiles(directory=str(STATIC)), name='static')


@app.get('/')
def index():
    return Response((STATIC / 'index.html').read_text(encoding='utf-8'), media_type='text/html', headers={'Cache-Control': 'no-cache'})


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=PORT, log_level='warning', timeout_graceful_shutdown=2)
