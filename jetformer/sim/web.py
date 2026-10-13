"""シミュレーターの経路(server.py が include_router する。server.py は import しない)

    /sim                      → /sim/ → /sim/c/<既定のコース>/
    /sim/c/<cid>/             シミュレーターのページ(dist/index.html に設定と shim を 3 行差し込んだもの)
    /sim/c/<cid>/<path>       scene.js・world.js・assets/ はコースのもの、ほかは dist/ のもの
    /sim/static/<name>        JetFormer のファイル(shim・汎用描画)
    /sim/bridge/api/<path>    ブリッジの API の中継(proxy.py)
    /api/sim/*                管理 API(導入・準備・ブリッジの起動停止・コース・モデル・データセット)

JetFormer の既存の /api/datasets/*・/api/models/* とは重ならない(ブリッジの API は /api 直下につながない)。
"""

import hashlib
import json
import os
import shutil
import threading
import time
import urllib.request
import uuid

import httpx
from fastapi import APIRouter, Body, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response

from .. import config, dataset_import, model_export
from . import bridge_proc, courses, datasets, env, install, paths
from .proxy import Proxy

router = APIRouter()
CID_RE = courses.CID_RE
MIME = {'.js': 'text/javascript; charset=utf-8', '.mjs': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8',
        '.html': 'text/html; charset=utf-8', '.json': 'application/json', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
        '.png': 'image/png', '.txt': 'text/plain; charset=utf-8', '.svg': 'application/octet-stream'}
STATIC_NAMES = ('sim-shim.js', 'sim-shim.css', 'course-scene.js', 'course-world.js')
MAX_COURSE_UPLOAD = courses.MAX_ZIP
SIM_JOB_KINDS = ('sim_install', 'sim_setup', 'import')
SIM_POLICY = 'jetformer_v0'


class _NoJobs(object):
    def submit(self, kind, title, fn):
        raise HTTPException(503, '処理状況(jobs)が使えません')


def _preflight():
    rel = install.active_release()
    if rel is None:
        raise bridge_proc.BridgeError('先にシミュレーター本体を導入してください')
    st = env.status()
    if not st['ready']:
        raise bridge_proc.BridgeError('先に Python 環境を準備してください' + ('(%s)' % st['problem'] if st.get('problem') else ''))
    port = os.environ.get('JETFORMER_SIM_BRIDGE_PORT')
    return {'python': paths.venv_python(), 'runner': paths.RUNNER, 'sim_dir': rel.root, 'runtime_root': paths.runtime_dir(),
            'log_path': paths.logs_dir() / 'bridge.log', 'port': int(port) if port and port.isdigit() else 0}


S = {'jobs': _NoJobs(), 'busy': lambda: {}, 'web_port': None}
bridge = bridge_proc.BridgeProcess(preflight=_preflight)
proxy = Proxy(lambda: bridge.port, note_activity=lambda: bridge.note_activity())
_bstatus = {'t': 0.0, 'value': None, 'port': None}
_importing = set()                 # 取り込みのジョブが動いているセッション ID(同じものを 2 回押したとき・削除との衝突を防ぐ)
_importing_lock = threading.Lock()


def sweep_stale(max_age_s=courses.STALE_S):
    """電源断・強制終了で残った作業ファイルを片付ける(review で追加)。data/sim_courses/ は courses.sweep_stale()、
    data/sim/ はモデルの受け渡し用の .bundle-*.zip と自己診断の selftest-*(max_age_s より古いものだけ)"""
    out = {'courses': None, 'sim': 0}
    try:
        out['courses'] = courses.sweep_stale(max_age_s)
    except Exception:
        pass
    now = time.time()
    try:
        names = os.listdir(paths.sim_dir())
    except OSError:
        names = []
    for name in names:
        p = paths.sim_dir() / name
        try:
            if p.is_symlink() or now - p.stat().st_mtime <= max_age_s:
                continue
            if name.startswith('.bundle-') and name.endswith('.zip') and p.is_file():
                p.unlink(missing_ok=True)
                out['sim'] += 1
            elif name.startswith('selftest-') and p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
                out['sim'] += 1
        except OSError:
            continue
    return out


def init(jobs=None, busy=None, bridge=None, web_port=None):
    """server.py から呼ぶ。jobs = 処理状況、busy = {'training','driving','recording'} を返す関数。bridge はテスト用の差し替え"""
    g = globals()
    if jobs is not None:
        S['jobs'] = jobs
    if busy is not None:
        S['busy'] = busy
    if web_port is not None:
        S['web_port'] = web_port
        proxy.web_port = web_port
    if bridge is not None:
        g['bridge'] = bridge
        proxy.port = lambda: g['bridge'].port
        proxy.note_activity = lambda: g['bridge'].note_activity()
    _bstatus.update(t=0.0, value=None, port=None)
    with _importing_lock:
        _importing.clear()
    sweep_stale()


def shutdown():
    """JetFormer の終了時: ブリッジを止め(3 秒)、中継のクライアントを捨てる"""
    try:
        globals()['bridge'].stop(timeout=3.0)
    except Exception:
        pass
    proxy.close()


def _err(e):
    status = getattr(e, 'status', 400)
    if isinstance(e, courses.CourseError):
        return JSONResponse({'detail': ' / '.join(e.errors), 'errors': e.errors}, status_code=status)
    return HTTPException(status, str(e))


# ---------------------------------------------------------------- ページとファイル
def _no_cache(headers=None):
    h = {'Cache-Control': 'no-cache', 'X-Content-Type-Options': 'nosniff'}
    h.update(headers or {})
    return h


def _etag(*parts):
    return '"%s"' % hashlib.sha1('|'.join(str(p) for p in parts).encode('utf-8')).hexdigest()[:24]


def _file_response(request, p, tag_prefix=''):
    st = p.stat()
    etag = _etag(tag_prefix, p, st.st_size, st.st_mtime_ns)
    if request.headers.get('if-none-match') == etag:
        return Response(status_code=304, headers=_no_cache({'ETag': etag}))
    media = MIME.get(p.suffix.lower(), 'application/octet-stream')
    return FileResponse(p, media_type=media, headers=_no_cache({'ETag': etag}))


def _text_response(request, text, media, tag_prefix=''):
    body = text.encode('utf-8')
    etag = _etag(tag_prefix, hashlib.sha1(body).hexdigest())
    if request.headers.get('if-none-match') == etag:
        return Response(status_code=304, headers=_no_cache({'ETag': etag}))
    if request.method == 'HEAD':
        return Response(status_code=200, media_type=media, headers=_no_cache({'ETag': etag, 'Content-Length': str(len(body))}))
    return Response(body, media_type=media, headers=_no_cache({'ETag': etag}))


def safe_rel(path):
    """URL をデコードした後のパスを確かめて要素に分ける。だめなら None"""
    if not isinstance(path, str) or not path or len(path) > 256:
        return None
    if '\\' in path or '\x00' in path or ':' in path:
        return None
    parts = path.split('/')
    if any(p in ('', '.', '..') for p in parts):
        return None
    return parts


def resolve_under(base, parts):
    """base の中の通常ファイル。途中にシンボリックリンクがあれば None"""
    p = base
    for part in parts:
        p = p / part
        if os.path.islink(p):
            return None
    try:
        rp = p.resolve(strict=True)
        if not rp.is_relative_to(base.resolve()) or not rp.is_file():
            return None
    except (OSError, RuntimeError):
        return None
    return rp


_index_cache = {}


def index_html(rel):
    """dist/index.html に config.js・shim・css の 3 行を差し込む(mtime で覚える)"""
    p = rel.dist / 'index.html'
    shim, css = paths.STATIC / 'sim-shim.js', paths.STATIC / 'sim-shim.css'
    key = (str(p), p.stat().st_mtime_ns, shim.stat().st_mtime_ns, css.stat().st_mtime_ns)
    hit = _index_cache.get(key)
    if hit is not None:
        return hit
    html = p.read_text(encoding='utf-8')
    meta = '<meta charset="utf-8">'
    if html.count(meta) != 1 or html.count('<script type="importmap">') != 1 or html.count('src="./app.js"') != 1:
        raise HTTPException(500, 'シミュレーターの index.html の形が想定と違います(版が変わった可能性があります)')
    inject = ('\n<script src="./_jf/config.js"></script>\n<script src="/sim/static/sim-shim.js?v=%d"></script>\n'
              '<link rel="stylesheet" href="/sim/static/sim-shim.css?v=%d">\n' % (int(shim.stat().st_mtime), int(css.stat().st_mtime)))
    out = html.replace(meta, meta + inject, 1)
    _index_cache.clear()
    _index_cache[key] = out
    return out


def config_js(c, rel):
    cfg = {'course': {'id': c.id, 'name': c.name, 'world_version': c.world_version, 'world_hash': c.world_hash,
                      'renderer': c.renderer, 'kind': c.kind, 'has_shortcut': c.has_shortcut, 'size_m': c.size_m,
                      'features': c.features},
           'bridge': '/sim/bridge', 'jetformer_port': S['web_port'] or config.WEB_PORT, 'courses_api': '/api/sim/courses',
           'status_api': '/api/sim/status', 'home': '/', 'release': rel.id}
    return 'window.JETFORMER_SIM = Object.freeze(%s);\n' % courses.js_json(cfg)


@router.get('/sim', include_in_schema=False)
def sim_root_noslash(request: Request):
    return RedirectResponse('/sim/' + ('?' + request.url.query if request.url.query else ''), status_code=307)


@router.get('/sim/', include_in_schema=False)
def sim_root(request: Request):
    cid = courses.get_default()
    return RedirectResponse('/sim/c/%s/' % cid + ('?' + request.url.query if request.url.query else ''), status_code=307)


@router.get('/sim/c/{cid}', include_in_schema=False)
def sim_course_noslash(cid: str, request: Request):
    if not CID_RE.match(cid):
        raise HTTPException(404, 'コースがありません')
    return RedirectResponse('/sim/c/%s/' % cid + ('?' + request.url.query if request.url.query else ''), status_code=308)


@router.api_route('/sim/c/{cid}/{path:path}', methods=['GET', 'HEAD'], include_in_schema=False)
def sim_course_file(cid: str, path: str, request: Request):
    if not CID_RE.match(cid):
        raise HTTPException(404, 'コースがありません')
    rel = install.active_release()
    if rel is None:
        raise HTTPException(503, 'シミュレーター本体が導入されていません(JetFormer の「シミュレーター」タブで導入してください)')
    c = courses.get_course(cid)
    if c is None:
        raise HTTPException(404, 'コースがありません: %s' % cid)
    if path in ('', 'index.html'):
        return _text_response(request, index_html(rel), 'text/html; charset=utf-8', rel.id)
    parts = safe_rel(path)
    if parts is None:
        raise HTTPException(404)
    rel_path = '/'.join(parts)
    if rel_path == '_jf/config.js':
        return _text_response(request, config_js(c, rel), 'text/javascript; charset=utf-8', rel.id)
    if rel_path == '_jf/course-data.js':
        text = courses.course_data_module(cid)
        if text is None:
            raise HTTPException(404)
        return _text_response(request, text, 'text/javascript; charset=utf-8', rel.id)
    if parts[0] == '_jf':
        raise HTTPException(404)
    if rel_path in ('scene.js', 'world.js'):
        p = c.file_for(rel_path)
        if p is None or not p.is_file() or p.is_symlink():
            raise HTTPException(404)
        return _file_response(request, p, rel.id + cid)
    if parts[0] == 'assets' and len(parts) == 2:
        if c.assets_dir is not None:
            p = resolve_under(c.assets_dir, parts[1:])
            if p is not None:
                return _file_response(request, p, rel.id + cid)
        p = resolve_under(rel.dist / 'assets', parts[1:])
    else:
        p = resolve_under(rel.dist, parts)
    if p is None:
        raise HTTPException(404)
    return _file_response(request, p, rel.id)


@router.api_route('/sim/static/{name}', methods=['GET', 'HEAD'], include_in_schema=False)
def sim_static(name: str, request: Request):
    if name not in STATIC_NAMES:
        raise HTTPException(404)
    return _file_response(request, paths.STATIC / name, 'jf')


# ---------------------------------------------------------------- 中継
@router.api_route('/sim/bridge/api/{path:path}', methods=['GET', 'HEAD', 'POST'], include_in_schema=False)
async def sim_bridge_proxy(path: str, request: Request):
    return await proxy.handle(request, path)


# ---------------------------------------------------------------- 管理 API
def _bridge_status_sync(timeout=1.0):
    port = bridge.port
    if not port:
        return None
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open('http://127.0.0.1:%d/api/status' % port, timeout=timeout) as r:
            return json.loads(r.read().decode('utf-8'))
    except Exception:
        return None


async def _bridge_api_status():
    now = time.monotonic()
    port = bridge.port
    if port and _bstatus['port'] == port and now - _bstatus['t'] < 2.0:
        return _bstatus['value']
    value = await proxy.bridge_status() if port else None
    _bstatus.update(t=now, value=value, port=port)
    return value


def _summarize_bridge_status(b, s):
    s = dict(s)
    if b:
        dc = b.get('dataset_collection') or {}
        s.update(policy_loaded=bool(b.get('policy_loaded')), policy_name=b.get('name'),
                 recording=bool(dc.get('recording')), session_id=dc.get('session_id') if dc.get('recording') else None,
                 frames=dc.get('frames') if dc.get('recording') else None,
                 model_upload=bool((b.get('model_upload') or {}).get('available')),
                 policy_error=b.get('policy_error'))
    else:
        s.update(policy_loaded=False, policy_name=None, recording=False, session_id=None)
    return s


def _status_sync():
    rel = install.active_release()
    rec = rel.record if rel else {}
    dc = rec.get('default_course') or {}
    return {
        'release': {'installed': rel is not None, 'id': rel.id if rel else None, 'created_at': rec.get('created_at'),
                    'installed_at': rec.get('installed_at'), 'default_world_version': dc.get('world_version'),
                    'warnings': rec.get('warnings', []), 'skipped': len(rec.get('skipped', []))},
        'inbox_dir': str(paths.inbox_dir()), 'inbox': install.list_inbox(), 'releases': install.list_releases(),
        'env': env.status(), 'bridge': bridge.status(), 'busy': S['busy']() or {}, 'default_course': courses.get_default(),
        'op_running': paths.OP_LOCK.locked(),
    }


@router.get('/api/sim/status')
async def api_sim_status():
    out = await run_in_threadpool(_status_sync)
    b = await _bridge_api_status() if out['bridge']['state'] == 'running' else None
    out['bridge'] = _summarize_bridge_status(b, out['bridge'])
    return out


@router.post('/api/sim/install')
def api_sim_install(body: dict = Body(default={})):
    name = body.get('file')
    try:
        p = install.inbox_file(name)
    except install.InstallError as e:
        raise _err(e)
    if paths.OP_LOCK.locked():
        raise HTTPException(409, 'ほかの導入・準備が動いています。終わってからもう一度押してください')
    activate = bool(body.get('activate', True))
    return S['jobs'].submit('sim_install', 'シミュレーターの導入: %s' % p.name,
                            lambda progress: install.install_release(p.name, activate=activate, progress=progress,
                                                                     bridge_running=bridge.running))


@router.post('/api/sim/releases/{rid}/activate')
def api_sim_release_activate(rid: str):
    try:
        return install.activate(rid, bridge_running=bridge.running())
    except install.InstallError as e:
        raise _err(e)


@router.delete('/api/sim/releases/{rid}')
def api_sim_release_delete(rid: str):
    try:
        return install.delete_release(rid)
    except install.InstallError as e:
        raise _err(e)


@router.post('/api/sim/setup')
def api_sim_setup(body: dict = Body(default={})):
    if bridge.running():
        raise HTTPException(409, 'ブリッジを止めてから準備してください')
    if paths.OP_LOCK.locked():
        raise HTTPException(409, 'ほかの導入・準備が動いています。終わってからもう一度押してください')
    if install.active_release() is None:
        raise HTTPException(409, '先にシミュレーター本体を導入してください')
    force = bool(body.get('force'))
    return S['jobs'].submit('sim_setup', 'シミュレーターの準備(作り直し)' if force else 'シミュレーターの準備',
                            lambda progress: env.setup(force=force, progress=progress))


@router.post('/api/sim/bridge/start')
def api_sim_bridge_start():
    try:
        return bridge.start()
    except bridge_proc.BridgeError as e:
        raise _err(e)


@router.post('/api/sim/bridge/stop')
def api_sim_bridge_stop(body: dict = Body(default={})):
    if not body.get('force'):
        b = _bridge_status_sync()
        if b and (b.get('dataset_collection') or {}).get('recording'):
            raise HTTPException(409, 'シミュレーターで収集中です。止めると収集中のデータは失われます(それでも止めるなら force)')
    _bstatus.update(t=0.0, value=None, port=None)
    return bridge.stop()


@router.get('/api/sim/bridge/log')
def api_sim_bridge_log(lines: int = Query(default=200, ge=0, le=300)):
    return bridge.logs(lines)


@router.post('/api/sim/bridge/load_model')
async def api_sim_bridge_load_model(body: dict = Body(...)):
    """(ブラウザなしの確認用)JetFormer の学習済みモデルをサーバー側でブリッジに読み込む。ページはトークンを持たないので
    シミュレーターの「設定を反映」は使えない"""
    run = str(body.get('run') or '')
    models = {m['name']: m for m in await run_in_threadpool(model_export.list_models)}
    if run not in models:
        raise HTTPException(404, 'モデルがありません')
    if models[run].get('policy', SIM_POLICY) != SIM_POLICY:
        raise HTTPException(400, 'シミュレーターは TransFuser軽量型(jetformer_v0)だけに対応しています')
    port = bridge.port
    if not port:
        raise HTTPException(409, 'ブリッジが動いていません')
    tmp = paths.sim_dir() / ('.bundle-%s.zip' % uuid.uuid4().hex[:8])

    def write():
        with open(tmp, 'wb') as f:
            for chunk in model_export.iter_bundle_zip(run):
                f.write(chunk)
        return tmp.stat().st_size
    try:
        size = await run_in_threadpool(write)
        client = proxy.client()
        host = {'Host': '127.0.0.1:%d' % port}

        async def gen():
            with open(tmp, 'rb') as f:
                while True:
                    chunk = await run_in_threadpool(f.read, 1 << 20)
                    if not chunk:
                        break
                    yield chunk
        r = await client.post('http://127.0.0.1:%d/api/models/bundle' % port, params={'name': run + '.zip'}, content=gen(),
                              headers=dict(host, **{'Content-Type': 'application/zip', 'Content-Length': str(size)}),
                              timeout=httpx.Timeout(300.0, connect=3.0))
        b = r.json()
        if r.status_code != 200:
            raise HTTPException(r.status_code, 'ブリッジがモデル ZIP を受け付けませんでした: %s' % ((b.get('error') or {}).get('message')))
        req = {'onnx_token': b['onnx_token'], 'config_token': b['config_token']}
        if b.get('calibration_token'):
            req['calibration_token'] = b['calibration_token']
        mode = body.get('mode_id', (b.get('defaults') or {}).get('mode_id'))
        if mode is not None:
            try:
                req['mode_id'] = int(mode)
            except (TypeError, ValueError):
                raise HTTPException(400, 'mode_id は整数にしてください')
        r = await client.post('http://127.0.0.1:%d/api/models/activate' % port, json=req, headers=host,
                              timeout=httpx.Timeout(180.0, connect=3.0))
        a = r.json()
        if r.status_code != 200:
            raise HTTPException(r.status_code, 'モデルを有効にできませんでした: %s' % ((a.get('error') or {}).get('message')))
        _bstatus.update(t=0.0, value=None, port=None)
        return {'ok': True, 'run': run, 'policy_loaded': a.get('policy_loaded'), 'name': a.get('name'),
                'mode_id': (a.get('policy_info') or {}).get('mode_id')}
    finally:
        tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------- コース
@router.get('/api/sim/courses')
def api_sim_courses():
    return courses.list_courses()


@router.put('/api/sim/courses/import')
async def api_sim_course_import(request: Request, filename: str = Query(default='course.zip'), replace: bool = False):
    if 'transfer-encoding' in request.headers and 'content-length' not in request.headers:
        raise HTTPException(411, 'Content-Length が必要です')
    try:
        length = int(request.headers.get('content-length') or 0)
    except ValueError:
        raise HTTPException(400, 'Content-Length が不正です')
    if length > MAX_COURSE_UPLOAD:
        raise HTTPException(413, 'コースの ZIP が大きすぎます(64 MiB まで)')
    base = paths.courses_dir()
    base.mkdir(parents=True, exist_ok=True)
    tmp = base / ('.upload-%s.zip' % uuid.uuid4().hex[:12])
    size = 0
    try:
        with open(tmp, 'wb') as f:
            async for chunk in request.stream():
                size += len(chunk)
                if size > MAX_COURSE_UPLOAD:
                    raise HTTPException(413, 'コースの ZIP が大きすぎます(64 MiB まで)')
                f.write(chunk)
        if size == 0:
            raise HTTPException(400, '空のファイルです')
        try:
            return await run_in_threadpool(courses.import_package, tmp, filename, replace)
        except courses.CourseError as e:
            return _err(e)
    finally:
        tmp.unlink(missing_ok=True)


@router.get('/api/sim/courses/{cid}')
def api_sim_course(cid: str):
    c = courses.get_course(cid)
    if c is None:
        raise HTTPException(404, 'コースがありません')
    return c.to_dict(courses.get_default())


@router.delete('/api/sim/courses/{cid}')
def api_sim_course_delete(cid: str):
    try:
        return courses.delete_course(cid)
    except courses.CourseError as e:
        return _err(e)


@router.post('/api/sim/courses/{cid}/default')
def api_sim_course_default(cid: str):
    try:
        return courses.set_default(cid)
    except courses.CourseError as e:
        return _err(e)


@router.get('/api/sim/courses/{cid}/package.zip')
def api_sim_course_package(cid: str):
    try:
        data = courses.export_package(cid)
    except courses.CourseError as e:
        return _err(e)
    return Response(data, media_type='application/zip',
                    headers={'Content-Disposition': 'attachment; filename="%s-course.zip"' % cid, 'Cache-Control': 'no-store'})


# ---------------------------------------------------------------- モデル・データセット
@router.get('/api/sim/models')
def api_sim_models():
    out = []
    for m in model_export.list_models():
        ok = m.get('policy', SIM_POLICY) == SIM_POLICY
        out.append(dict(m, sim_compatible=ok, reason=None if ok else 'シミュレーターは TransFuser軽量型(jetformer_v0)だけに対応しています'))
    return out


@router.get('/api/sim/datasets')
def api_sim_datasets():
    b = _bridge_status_sync() if bridge.port else None
    dc = (b or {}).get('dataset_collection') or {}
    out = datasets.list_sim_datasets(dc.get('session_id') if dc.get('recording') else None)
    out['in_flight'] = proxy.busy()
    return out


@router.post('/api/sim/datasets/{sid}/import')
def api_sim_dataset_import(sid: str):
    try:
        p, info = datasets.archive_path(sid)
    except datasets.DatasetError as e:
        raise _err(e)
    free = shutil.disk_usage(str(config.DATA_DIR if config.DATA_DIR.exists() else paths.sim_dir())).free
    if free < 2.2 * info['bytes']:
        raise HTTPException(507, '空き容量が足りません(ZIP %.1f GB の 2.2 倍が必要、空き %.1f GB)' % (info['bytes'] / 1e9, free / 1e9))
    fn = info['filename']
    with _importing_lock:
        if sid in _importing:
            raise HTTPException(409, 'このデータセットはいま取り込んでいます(処理状況を見てください)')
        _importing.add(sid)

    def run(progress):
        try:
            return dataset_import.import_archive(p, simulator=True, source_name=fn, progress=progress)
        finally:
            with _importing_lock:
                _importing.discard(sid)
    try:
        return S['jobs'].submit('import', 'シミュレーターから取り込み: %s(%.1f MB)' % (fn, info['bytes'] / 1e6), run)
    except BaseException:
        with _importing_lock:
            _importing.discard(sid)
        raise


@router.delete('/api/sim/datasets/{sid}')
def api_sim_dataset_delete(sid: str):
    with _importing_lock:
        if sid in _importing:
            raise HTTPException(409, 'このデータセットはいま JetFormer に取り込んでいます。終わってから消してください')
    b = _bridge_status_sync() if bridge.port else None
    dc = (b or {}).get('dataset_collection') or {}
    try:
        return datasets.delete(sid, in_flight=proxy.busy(), recording_sid=dc.get('session_id') if dc.get('recording') else None)
    except datasets.DatasetError as e:
        raise _err(e)
