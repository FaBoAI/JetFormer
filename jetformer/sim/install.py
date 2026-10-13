"""シミュレーター本体(ユーザーの course-simulator の ZIP)の導入

- 導入元は Jetson のフォルダー inbox_dir()(既定 ~/sim)に置いた ZIP だけ。ブラウザからのアップロードでは本体(Python・JS)を入れない。
- ZIP の中身の全ファイルを COMPLETE-MANIFEST.json(schema course-simulator-complete-v1)の sha256 と照合しながら、
  data/sim/releases/<release_id>/ に展開する。dataset-downloads/(取り込み済みの大きなデータ)と runtime/ は入れない。
- release_id = <manifest の created_at を YYYYMMDD-HHMMSS>-<COMPLETE-MANIFEST.json の sha256 先頭 8 桁>
- 使う版は data/sim/active.json で決める。導入した版のファイルは変えない(ブリッジも -B で __pycache__ を作らない)。
"""

import datetime
import hashlib
import json
import os
import re
import shutil
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from . import paths, ziputil

SCHEMA = 'course-simulator-complete-v1'
MANIFEST_NAME = 'COMPLETE-MANIFEST.json'
MAX_ZIP_BYTES = 4 << 30
MAX_ENTRIES = 5000
MAX_TOTAL = 2 << 30
MAX_RATIO = 200
MAX_MANIFEST = 4 << 20
EXCLUDED_PREFIXES = ('dataset-downloads/', 'runtime/')
REQUIRED = ('bridge.py', 'dataset_recorder.py', 'jetformer_policy.py', 'jetformer_upload.py', 'calibration.example.json',
            'requirements-model.txt', 'dist/index.html', 'dist/app.js', 'dist/scene.js', 'dist/world.js', 'dist/sim-core.js',
            'qa/combine_scene_datasets.py', 'training/speed_physics.py')
RELEASE_RE = re.compile(r'^\d{8}-\d{6}-[0-9a-f]{8}$')
SHA_RE = re.compile(r'^[0-9a-f]{64}$')
CHUNK = 1 << 20


class InstallError(ValueError):
    """導入できない(日本語の理由)。status は HTTP の目安"""

    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


@dataclass
class Release:
    id: str
    root: Path
    manifest: dict
    record: dict = field(default_factory=dict)
    files: dict = field(default_factory=dict)            # パス → sha256(manifest のもの)
    trusted_sha256: set = field(default_factory=set)      # このリリースのファイルの sha256(参考。コースの JS の信頼には trusted_js を使う)
    trusted_js: set = field(default_factory=set)          # 同じフォルダーの (scene.js の sha256, world.js の sha256) の組(dist・同梱スナップショット)

    @property
    def dist(self):
        return self.root / 'dist'


def js_pairs(files):
    """{パス: sha256} から、scene.js と world.js が同じフォルダーにそろっているところの (scene の sha256, world の sha256) の集合。
    コースとして実行してよい JS はこの組だけ(リリースのほかの JS(app.js・テストなど)を scene.js として動かさせないため。review で追加)"""
    dirs = {}
    for path, sha in files.items():
        d, _, name = str(path).rpartition('/')
        if name in ('scene.js', 'world.js'):
            dirs.setdefault(d, {})[name] = sha
    return {(v['scene.js'], v['world.js']) for v in dirs.values() if len(v) == 2}


# ---------------------------------------------------------------- ZIP を読む
def _read_manifest(zf, info):
    raw = ziputil.read_limited(zf, info, MAX_MANIFEST, MANIFEST_NAME)
    try:
        m = ziputil.strict_json(raw, MANIFEST_NAME, max_depth=8)
    except ValueError as e:
        raise InstallError(str(e))
    return raw, m


def check_manifest(m):
    """COMPLETE-MANIFEST.json の形を確かめて {パス: {bytes, sha256}} を返す"""
    if not isinstance(m, dict) or m.get('schema') != SCHEMA:
        raise InstallError('COMPLETE-MANIFEST.json の schema が %s ではありません' % SCHEMA)
    files = m.get('files')
    if not isinstance(files, list) or not files:
        raise InstallError('COMPLETE-MANIFEST.json の files が配列ではありません')
    out, seen = {}, set()
    for f in files:
        if not isinstance(f, dict):
            raise InstallError('COMPLETE-MANIFEST.json の files の形が違います')
        p, n, h = f.get('path'), f.get('bytes'), f.get('sha256')
        try:
            ziputil.check_path(p)
        except ziputil.ZipCheckError as e:
            raise InstallError('COMPLETE-MANIFEST.json: %s' % e)
        if not isinstance(n, int) or isinstance(n, bool) or n < 0 or not isinstance(h, str) or not SHA_RE.match(h):
            raise InstallError('COMPLETE-MANIFEST.json: %s の bytes・sha256 が不正です' % p)
        if p.casefold() in seen:
            raise InstallError('COMPLETE-MANIFEST.json: 同じパスが重複しています: %s' % p)
        seen.add(p.casefold())
        out[p] = {'bytes': n, 'sha256': h}
    return out


def release_id_of(manifest, manifest_raw):
    try:
        created = datetime.datetime.fromisoformat(str(manifest.get('created_at')))
    except ValueError:
        raise InstallError('COMPLETE-MANIFEST.json の created_at が日時ではありません')
    return '%s-%s' % (created.strftime('%Y%m%d-%H%M%S'), hashlib.sha256(manifest_raw).hexdigest()[:8])


def _excluded(rel):
    return rel.startswith(EXCLUDED_PREFIXES)


def inspect_zip(path):
    """ZIP の中央ディレクトリと COMPLETE-MANIFEST.json だけを読んで判定する(展開しない)。
    {kind: 'simulator'|'other', release_id, top, problem, entries}"""
    path = Path(path)
    out = {'kind': 'other', 'release_id': None, 'top': None, 'problem': None}
    try:
        if path.stat().st_size > MAX_ZIP_BYTES:
            raise InstallError('ZIP が大きすぎます(4 GiB まで)')
        with zipfile.ZipFile(path) as zf:
            plan = _plan(zf)
            out.update(kind='simulator', release_id=plan['release_id'], top=plan['top'],
                       created_at=plan['manifest'].get('created_at'), files=len(plan['include']), bytes=plan['bytes'])
    except (InstallError, ziputil.ZipCheckError) as e:
        out['problem'] = str(e)
    except (OSError, zipfile.BadZipFile) as e:
        out['problem'] = 'ZIP として読めません: %s' % e
    return out


def _plan(zf):
    """導入の計画: どのエントリーをどこに入れるか。おかしなところがあれば InstallError"""
    try:
        entries = ziputil.scan(zf, MAX_ENTRIES, max_ratio=MAX_RATIO)
    except ziputil.ZipCheckError as e:
        raise InstallError(str(e))
    tops = {parts[0] for _, parts, _ in entries}
    if len(tops) != 1:
        raise InstallError('ZIP の中身が 1 つのフォルダーにまとまっていません(シミュレーター本体の ZIP ではありません)')
    top = tops.pop()
    files = {}
    for info, parts, kind in entries:
        if kind == 'file':
            if len(parts) < 2:
                raise InstallError('ZIP の最上位にファイルがあります: %s' % info.filename)
            files['/'.join(parts[1:])] = info
    if MANIFEST_NAME not in files:
        raise InstallError('%s/%s がありません(シミュレーター本体の ZIP ではありません)' % (top, MANIFEST_NAME))
    raw, manifest = _read_manifest(zf, files[MANIFEST_NAME])
    mfiles = check_manifest(manifest)
    rid = release_id_of(manifest, raw)
    include, skipped, warnings = {}, [], []
    for rel, info in files.items():
        if rel == MANIFEST_NAME:
            include[rel] = (info, None)
        elif _excluded(rel):
            skipped.append(rel)
        elif rel in mfiles:
            if info.file_size != mfiles[rel]['bytes']:
                raise InstallError('%s の大きさが COMPLETE-MANIFEST.json と違います' % rel)
            include[rel] = (info, mfiles[rel])
        elif '/' not in rel and rel.lower().endswith(('.txt', '.md')):
            include[rel] = (info, None)
            warnings.append('COMPLETE-MANIFEST.json にない説明ファイルを入れました: %s' % rel)
        else:
            skipped.append(rel)
            warnings.append('COMPLETE-MANIFEST.json にないファイルは入れません: %s' % rel)
    missing = [p for p in mfiles if not _excluded(p) and p not in files]
    if missing:
        raise InstallError('COMPLETE-MANIFEST.json にあるファイルが ZIP にありません: %s%s' % (
            ', '.join(missing[:5]), ' ほか' if len(missing) > 5 else ''))
    lacking = [p for p in REQUIRED if p not in include]
    if lacking:
        raise InstallError('シミュレーター本体に必要なファイルがありません: %s' % ', '.join(lacking))
    total = sum(info.file_size for info, _ in include.values())
    if total > MAX_TOTAL:
        raise InstallError('展開後の大きさが 2 GiB を超えます')
    return {'top': top, 'manifest': manifest, 'manifest_raw': raw, 'manifest_files': mfiles, 'release_id': rid,
            'include': include, 'skipped': sorted(skipped), 'warnings': warnings, 'bytes': total}


# ---------------------------------------------------------------- inbox
_inbox_cache = {}


def list_inbox():
    """inbox_dir() 直下の *.zip(通常ファイルだけ)と判定。結果は (パス, 大きさ, mtime) で覚えておく"""
    d = paths.inbox_dir()
    out = []
    try:
        names = sorted(os.listdir(d))
    except OSError:
        return out
    installed = {r['id'] for r in list_releases()}
    for name in names:
        p = d / name
        if not name.lower().endswith('.zip') or p.is_symlink() or not p.is_file():
            continue
        st = p.stat()
        key = (str(p), st.st_size, st.st_mtime)
        info = _inbox_cache.get(key)
        if info is None:
            info = inspect_zip(p)
            _inbox_cache[key] = info
        out.append({'file': name, 'bytes': st.st_size, 'mtime': st.st_mtime, 'kind': info['kind'],
                    'release_id': info.get('release_id'), 'installed': info.get('release_id') in installed,
                    'problem': info.get('problem'), 'created_at': info.get('created_at')})
    return out


def inbox_file(name):
    """inbox の ZIP を名前で選ぶ(パスは受け付けない)"""
    if not isinstance(name, str) or not name or '/' in name or '\\' in name or name in ('.', '..') or '\x00' in name:
        raise InstallError('ファイル名だけを指定してください', 400)
    p = paths.inbox_dir() / name
    if not name.lower().endswith('.zip') or p.is_symlink() or not p.is_file():
        raise InstallError('%s に %s がありません' % (paths.inbox_dir(), name), 404)
    return p


# ---------------------------------------------------------------- 導入済みの版
def _record_path(rid):
    return paths.releases_dir() / ('%s.json' % rid)


def _read_json(p):
    try:
        with open(p, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def list_releases():
    d = paths.releases_dir()
    active = active_id()
    out = []
    try:
        names = sorted(os.listdir(d), reverse=True)
    except OSError:
        return out
    for name in names:
        if not RELEASE_RE.match(name) or not (d / name).is_dir():
            continue
        rec = _read_json(_record_path(name))
        if not rec:
            continue
        out.append({'id': name, 'active': name == active, 'installed_at': rec.get('installed_at'),
                    'created_at': rec.get('created_at'), 'bytes': rec.get('bytes'), 'files': rec.get('files'),
                    'default_course': rec.get('default_course'), 'warnings': rec.get('warnings', [])})
    return out


def active_id():
    a = _read_json(paths.active_file())
    rid = a.get('release') if isinstance(a, dict) else None
    if isinstance(rid, str) and RELEASE_RE.match(rid) and (paths.releases_dir() / rid).is_dir() and _record_path(rid).is_file():
        return rid
    return None


_release_cache = {}


def get_release(rid):
    """導入済みの版(Release)。なければ None。manifest は (記録の mtime) で覚えておく"""
    if not isinstance(rid, str) or not RELEASE_RE.match(rid):
        return None
    root = paths.releases_dir() / rid
    rec_path = _record_path(rid)
    try:
        key = (str(root), rec_path.stat().st_mtime)
    except OSError:
        return None
    if not root.is_dir():
        return None
    rel = _release_cache.get(key)
    if rel is None:
        rec = _read_json(rec_path) or {}
        manifest = _read_json(root / MANIFEST_NAME) or {}
        files = {f['path']: f['sha256'] for f in manifest.get('files', []) if isinstance(f, dict) and 'path' in f}
        rel = Release(id=rid, root=root, manifest=manifest, record=rec, files=files, trusted_sha256=set(files.values()),
                      trusted_js=js_pairs(files))
        _release_cache[key] = rel
    return rel


def active_release():
    rid = active_id()
    return get_release(rid) if rid else None


def all_releases():
    return [r for r in (get_release(x['id']) for x in list_releases()) if r]


def _write_active(rid):
    paths.sim_dir().mkdir(parents=True, exist_ok=True)
    paths.write_durable(paths.active_file(), json.dumps({'release': rid}))


def activate(rid, bridge_running=False):
    if bridge_running:
        raise InstallError('ブリッジを止めてから版を切り替えてください', 409)
    if not get_release(rid):
        raise InstallError('導入済みの版ではありません: %s' % rid, 404)
    _write_active(rid)
    return {'ok': True, 'active': rid}


def delete_release(rid):
    if not get_release(rid):
        raise InstallError('導入済みの版ではありません: %s' % rid, 404)
    if rid == active_id():
        raise InstallError('使っている版は削除できません(別の版を使うようにしてから削除してください)', 409)
    root = (paths.releases_dir() / rid)
    if root.resolve().parent != paths.releases_dir().resolve():
        raise InstallError('削除できない場所です', 400)
    shutil.rmtree(root)
    _record_path(rid).unlink(missing_ok=True)
    return {'ok': True}


# ---------------------------------------------------------------- 導入
def _default_course_info(root):
    out = {}
    for name, key in (('scene.js', 'scene_sha256'), ('world.js', 'world_sha256')):
        p = root / 'dist' / name
        if p.is_file():
            out[key] = hashlib.sha256(p.read_bytes()).hexdigest()
    try:
        m = re.search(r"export const WORLD_VERSION\s*=\s*'([^']{1,80})'", (root / 'dist' / 'world.js').read_text(encoding='utf-8'))
        out['world_version'] = m.group(1) if m else None
    except OSError:
        out['world_version'] = None
    return out


def install_release(name, activate=True, progress=None, bridge_running=lambda: False):
    """inbox の ZIP(名前で指定)を導入する。戻り値 {release_id, already_installed, activated, files, bytes, skipped, warnings}"""
    progress = progress or (lambda p, m='': None)
    zpath = inbox_file(name)
    if not paths.OP_LOCK.acquire(blocking=False):
        raise InstallError('ほかの導入・準備が動いています。終わってからもう一度押してください', 409)
    try:
        return _install(zpath, activate, progress, bridge_running)
    finally:
        paths.OP_LOCK.release()


def _install(zpath, do_activate, progress, bridge_running):
    if zpath.stat().st_size > MAX_ZIP_BYTES:
        raise InstallError('ZIP が大きすぎます(4 GiB まで)')
    rdir = paths.releases_dir()
    rdir.mkdir(parents=True, exist_ok=True)
    for old in rdir.glob('.tmp-*'):             # 前回の途中のもの(電源断など)
        shutil.rmtree(old, ignore_errors=True)
    progress(0.02, 'ZIP を確かめています')
    with zipfile.ZipFile(zpath) as zf:
        plan = _plan(zf)
        rid = plan['release_id']
        dest = rdir / rid
        result = {'release_id': rid, 'source_file': zpath.name, 'files': len(plan['include']), 'bytes': plan['bytes'],
                  'skipped': plan['skipped'], 'warnings': plan['warnings']}
        if dest.is_dir() and _record_path(rid).is_file():
            result['already_installed'] = True
        else:
            free = shutil.disk_usage(str(rdir)).free
            if free < plan['bytes'] + (1 << 30):
                raise InstallError('空き容量が足りません(必要 %.1f GB、空き %.1f GB)' % (
                    (plan['bytes'] + (1 << 30)) / 1e9, free / 1e9), 507)
            if dest.exists():
                shutil.rmtree(dest)
            _record_path(rid).unlink(missing_ok=True)
            stage = rdir / ('.tmp-%s' % uuid.uuid4().hex[:12])
            try:
                _extract(zf, plan, stage, progress)
                os.chmod(stage, 0o755)
                # 電源断に備えて、展開した中身をディスクに書き切ってから置き、記録はそのあとに書く
                # (「記録だけあって中身が 0 バイトの版」を作らない。851 ファイルを 1 つずつ fsync すると SD カードで遅いので 1 回で。review で追加)
                progress(0.96, 'ディスクに書き込んでいます')
                os.sync()
                os.replace(stage, dest)
                paths.fsync_dir(rdir)
            finally:
                if stage.exists():
                    shutil.rmtree(stage, ignore_errors=True)
            rec = {'release_id': rid, 'source_file': zpath.name, 'source_bytes': zpath.stat().st_size,
                   'manifest_sha256': hashlib.sha256(plan['manifest_raw']).hexdigest(),
                   'created_at': plan['manifest'].get('created_at'),
                   'installed_at': datetime.datetime.now().isoformat(timespec='seconds'),
                   'files': len(plan['include']), 'bytes': plan['bytes'], 'skipped': plan['skipped'],
                   'warnings': plan['warnings'], 'default_course': _default_course_info(dest)}
            paths.write_durable(_record_path(rid), json.dumps(rec, ensure_ascii=False, indent=1))
            result['already_installed'] = False
    result['activated'] = False
    if do_activate:
        if active_id() is None or not bridge_running():
            _write_active(rid)
            result['activated'] = True
        else:
            result['message'] = 'ブリッジが動いているので、使う版は切り替えていません。ブリッジを止めてから「この版を使う」を押してください'
    progress(1.0, '導入済み: %s' % rid if result['already_installed'] else '導入しました: %s' % rid)
    return result


def _extract(zf, plan, stage, progress):
    stage.mkdir(parents=True)
    total = max(plan['bytes'], 1)
    done = 0
    last = time.monotonic()
    for rel, (info, expect) in sorted(plan['include'].items()):
        out = stage.joinpath(*rel.split('/'))
        if stage.resolve() not in out.resolve().parents:
            raise InstallError('展開先の外に出るパスがあります: %s' % rel)
        out.parent.mkdir(parents=True, exist_ok=True)
        h = hashlib.sha256()
        n = 0
        limit = info.file_size
        with zf.open(info) as src, open(out, 'wb') as dst:
            while True:
                chunk = src.read(CHUNK)
                if not chunk:
                    break
                n += len(chunk)
                if n > limit:
                    raise InstallError('%s が宣言より大きく展開されました' % rel)
                h.update(chunk)
                dst.write(chunk)
                done += len(chunk)
                if time.monotonic() - last > 0.5:
                    last = time.monotonic()
                    progress(0.05 + 0.9 * done / total, '展開と照合 %d / %d MB' % (done >> 20, total >> 20))
        os.chmod(out, 0o644)
        if expect is not None and (n != expect['bytes'] or h.hexdigest() != expect['sha256']):
            raise InstallError('%s の sha256 が COMPLETE-MANIFEST.json と一致しません(ZIP が壊れているか、書き換えられています)' % rel)
    for d in sorted({p for p in stage.rglob('*') if p.is_dir()}):
        os.chmod(d, 0o755)
