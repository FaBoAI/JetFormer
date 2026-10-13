"""シミュレーターのコースの登録簿

コースの種類(kind):
- release     Default。使っている版(リリース)の dist/scene.js・world.js(今は v3)。削除できない
- snapshot    リリースに同梱のスナップショット(qa/course-snapshots/<id>/。旧 v2)。スナップショットの manifest と
              COMPLETE-MANIFEST.json の両方と sha256 が一致するものだけ。削除できない
- trusted-js  アップロードしたコースのうち、scene.js・world.js の sha256 が導入済みリリースのファイルと一致するもの
- data        それ以外のアップロード。physics-world.json(データ)を JetFormer の汎用描画(static/course-scene.js)で描く

アップロードされた JS は実行しない(シミュレーターのページは JetFormer と同じオリジンで動くので、JS は JetFormer の
API を呼べてしまい、実車を動かせるため)。アップロードの形式は 2 つ:
  (a) シミュレーター作者のスナップショット形式: manifest.json + physics-world.json [+ scene.js, world.js] [+ assets/]
  (b) JetFormer 形式: course.json(format jetformer-sim-course/v1)+ physics-world.json [+ assets/]
"""

import datetime
import hashlib
import io
import json
import math
import os
import re
import shutil
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from . import install, paths, worldhash, ziputil

CID_RE = re.compile(r'^[a-z0-9][a-z0-9-]{0,47}$')
UPLOAD_CID_RE = re.compile(r'^[a-z0-9][a-z0-9-]{1,47}$')
WV_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$')
SURFACE_RE = re.compile(r'^[a-z0-9][a-z0-9-]{0,31}$')
TEXTURE_RE = re.compile(r'^[a-z0-9][a-z0-9_-]{0,47}\.(jpg|png)$')
COLOR_RE = re.compile(r'^#[0-9a-fA-F]{6}$')
RESERVED = {'default', 'static', 'bridge', 'api', 'c', '_jf'}
SEGMENT_KINDS = ('rail', 'curtain', 'gate-foot')
DISK_KINDS = ('post-foot', 'post', 'gate-post')
ALL_TEACHERS = ['slow', 'speed', 'speed-fast', 'speed-rc', 'speed-optimized']
GENERIC_TEACHERS = ['slow', 'speed']
DIST_TEXTURES = ('carpet.jpg', 'carpet-normal.jpg', 'turf.jpg', 'turf-normal.jpg', 'blue-pad.jpg', 'black-mat.jpg')
COLOR_KEYS = ('rail', 'rail_alt', 'curtain', 'post', 'gate', 'floor', 'background')

MAX_ZIP = 64 << 20
MAX_UNPACKED = 128 << 20
MAX_ENTRIES = 256
MAX_PHYSICS = 4 << 20
MAX_JSON = 64 << 10
MAX_JS = 1 << 20
MAX_TEXTURE = 8 << 20
MAX_TEXTURE_PX = 4096
MAX_TEXTURES = 16
COORD_LIMIT = 50.0
SPAWN_ERROR_M = 0.10
SPAWN_WARN_M = 0.23            # 車体 0.42 × 0.19 の対角の半分
STALE_S = 600                  # これより古い作業ファイル(.upload-*・.tmp-*)は、電源断などで残ったものとして消す

# 登録・置き換え・削除・既定の変更は同時に 1 つだけ(同じ id を 2 つ同時に送ると os.replace が 500 になるため。review で追加)
_LOCK = threading.RLock()


class CourseError(ValueError):
    """コースを登録・利用できない。errors は日本語の理由の一覧、status は HTTP の目安"""

    def __init__(self, errors, status=400):
        self.errors = [errors] if isinstance(errors, str) else list(errors)
        self.status = status
        super().__init__(' / '.join(self.errors))


# ---------------------------------------------------------------- 既知のコース
def _load_known():
    try:
        with open(paths.PKG_DATA / 'known_courses.json', encoding='utf-8') as f:
            return json.load(f).get('courses', [])
    except (OSError, ValueError):
        return []


KNOWN = _load_known()


def known_by_js(scene_sha, world_sha):
    for k in KNOWN:
        if k.get('scene_sha256') == scene_sha and k.get('world_sha256') == world_sha:
            return k
    return None


def _sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _features(cid):
    if cid == 'default':
        return {'teachers': list(ALL_TEACHERS), 'collection_plan': True}
    return {'teachers': list(GENERIC_TEACHERS), 'collection_plan': False}


@dataclass
class Course:
    id: str
    kind: str
    name: str
    world_version: str = None
    world_hash: str = None
    canonical_sha256: str = None
    description: str = ''
    stats: dict = field(default_factory=dict)
    size_m: list = None
    has_shortcut: bool = False
    release: object = None              # install.Release
    js_dir: Path = None                 # scene.js・world.js のあるフォルダー(release・snapshot・trusted-js)
    assets_dir: Path = None             # コース固有のテクスチャ(なければ dist/assets)
    physics_path: Path = None
    created_at: str = None
    warnings: list = field(default_factory=list)
    visual: dict = field(default_factory=dict)
    source: dict = field(default_factory=dict)

    @property
    def renderer(self):
        return 'generic' if self.kind == 'data' else 'simulator'

    @property
    def deletable(self):
        return self.kind in ('data', 'trusted-js')

    @property
    def features(self):
        return _features(self.id)

    def file_for(self, rel):
        """scene.js・world.js をどこから返すか"""
        if rel not in ('scene.js', 'world.js'):
            return None
        if self.kind == 'data':
            return paths.STATIC / ('course-scene.js' if rel == 'scene.js' else 'course-world.js')
        return self.js_dir / rel if self.js_dir else None

    def asset_for(self, name):
        """assets/<name>: コースのものがあればそれ、なければ dist/assets"""
        if self.assets_dir is not None:
            p = self.assets_dir / name
            if p.is_file() and not p.is_symlink():
                return p
        if self.release is not None:
            return self.release.dist / 'assets' / name
        return None

    def to_dict(self, default_id=None):
        return {'id': self.id, 'name': self.name, 'description': self.description, 'kind': self.kind,
                'renderer': self.renderer, 'world_version': self.world_version, 'world_hash': self.world_hash,
                'canonical_sha256': self.canonical_sha256, 'stats': self.stats, 'size_m': self.size_m,
                'has_shortcut': self.has_shortcut, 'features': self.features, 'deletable': self.deletable,
                'is_default': self.id == default_id, 'warnings': list(self.warnings), 'created_at': self.created_at,
                'open_url': '/sim/c/%s/' % self.id, 'source': self.source}


# ---------------------------------------------------------------- リリースのコース(Default・スナップショット)
_release_courses_cache = {}


def _release_courses(rel):
    key = (rel.id, rel.record.get('installed_at'), str(rel.root))
    hit = _release_courses_cache.get(key)
    if hit is not None:
        return hit
    out = [_default_course(rel)] + _snapshot_courses(rel)
    _release_courses_cache.clear()
    _release_courses_cache[key] = out
    return out


def _default_course(rel):
    scene_sha = _sha256_file(rel.dist / 'scene.js') if (rel.dist / 'scene.js').is_file() else None
    world_sha = _sha256_file(rel.dist / 'world.js') if (rel.dist / 'world.js').is_file() else None
    k = known_by_js(scene_sha, world_sha)
    c = Course(id='default', kind='release', name='Default(標準コース)', release=rel, js_dir=rel.dist, assets_dir=None)
    if k:
        c.name = k['label']
        c.description = k.get('description', '')
        c.world_version, c.world_hash, c.canonical_sha256 = k['world_version'], k['world_hash'], k.get('canonical_sha256')
        c.stats, c.size_m, c.has_shortcut = dict(k['stats']), list(k['size_m']), bool(k.get('has_shortcut'))
        if k.get('physics_file'):
            c.physics_path = paths.PKG_DATA / k['physics_file']
    else:
        try:
            m = re.search(r"export const WORLD_VERSION\s*=\s*'([^']{1,80})'", (rel.dist / 'world.js').read_text(encoding='utf-8'))
            c.world_version = m.group(1) if m else None
        except OSError:
            c.world_version = None
        c.name = 'Default(この版の標準コース)'
        c.description = '導入した版の dist のコース。world hash は未計算です'
        c.has_shortcut = True
        c.warnings.append('この版の Default コースは JetFormer が知らない形です(world hash は未計算)')
    if rel.files.get('dist/scene.js') != scene_sha or rel.files.get('dist/world.js') != world_sha:
        c.warnings.append('dist の scene.js・world.js が COMPLETE-MANIFEST.json と一致しません')
    return c


def _snapshot_courses(rel):
    out = []
    base = rel.root / 'qa' / 'course-snapshots'
    try:
        names = sorted(os.listdir(base))
    except OSError:
        return out
    for name in names:
        d = base / name
        if d.is_symlink() or not d.is_dir() or not WV_RE.match(name):
            continue
        try:
            man = json.loads((d / 'manifest.json').read_text(encoding='utf-8'))
            files = man.get('files') or {}
            ok = True
            for fn in ('scene.js', 'world.js', 'physics-world.json'):
                p = d / fn
                sha = _sha256_file(p) if p.is_file() and not p.is_symlink() else None
                rel_path = 'qa/course-snapshots/%s/%s' % (name, fn)
                if sha is None or files.get(fn) != sha or rel.files.get(rel_path) != sha:
                    ok = False
            if not ok:
                continue
            world = json.loads((d / 'physics-world.json').read_text(encoding='utf-8'))
            desc = worldhash.describe(world)
            if man.get('world_hash') and man['world_hash'] != desc['world_hash']:
                continue
            wv = str(man.get('world_version') or name)
            cid = _id_from(wv)
            if not cid or cid in RESERVED:
                continue
            k = known_by_js(files.get('scene.js'), files.get('world.js'))
            b = world.get('bounds') or {}
            out.append(Course(
                id=cid, kind='snapshot', name=k['label'] if k else '同梱コース %s' % wv,
                description=k.get('description', '') if k else str(man.get('reason') or '')[:200],
                world_version=wv, world_hash=desc['world_hash'], canonical_sha256=desc['canonical_sha256'],
                stats=desc['stats'], size_m=list(k['size_m']) if k else _size_of(b), release=rel, js_dir=d,
                has_shortcut=bool(world.get('requiredCrossings')), physics_path=d / 'physics-world.json'))
        except (OSError, ValueError, TypeError, AttributeError):
            continue
    return out


def _size_of(bounds):
    try:
        return [round(float(bounds['maxX']) - float(bounds['minX']), 2), round(float(bounds['maxZ']) - float(bounds['minZ']), 2)]
    except (KeyError, TypeError, ValueError):
        return None


def _id_from(text):
    s = re.sub(r'[^a-z0-9-]+', '-', str(text).lower()).strip('-')
    s = re.sub(r'-{2,}', '-', s)[:48].strip('-')
    return s or None


# ---------------------------------------------------------------- アップロードしたコース
def _uploaded_course(d, trusted):
    try:
        meta = json.loads((d / 'course.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    if not isinstance(meta, dict) or meta.get('format') != 'jetformer-sim-course-meta/v1' or meta.get('id') != d.name:
        return None
    kind = meta.get('kind')
    warnings = list(meta.get('warnings') or [])
    js_dir = None
    if kind == 'trusted-js':
        shas = tuple((_sha256_file(d / f) if (d / f).is_file() and not (d / f).is_symlink() else None) for f in ('scene.js', 'world.js'))
        if shas in trusted:
            js_dir = d
        else:
            kind = 'data'
            warnings.append('JS の元になったリリースが見つからないので、簡易描画で表示します')
    elif kind != 'data':
        return None
    assets = d / 'assets'
    return Course(id=d.name, kind=kind, name=str(meta.get('name') or d.name), description=str(meta.get('description') or ''),
                  world_version=meta.get('world_version'), world_hash=meta.get('world_hash'),
                  canonical_sha256=meta.get('canonical_sha256'), stats=meta.get('stats') or {}, size_m=meta.get('size_m'),
                  has_shortcut=bool(meta.get('has_shortcut')), js_dir=js_dir,
                  assets_dir=assets if assets.is_dir() and not assets.is_symlink() else None,
                  physics_path=d / 'physics-world.json', created_at=meta.get('created_at'), warnings=warnings,
                  visual=meta.get('visual') or {}, source=meta.get('source') or {})


def _trusted_js():
    """導入済みのどれかの版で、同じフォルダーにそろっている (scene.js, world.js) の sha256 の組(install.js_pairs)"""
    out = set()
    for r in install.all_releases():
        out |= r.trusted_js
    return out


def list_course_objects():
    """[Course](Default、同梱、アップロード(新しい順))"""
    rel = install.active_release()
    out = list(_release_courses(rel)) if rel else []
    trusted = _trusted_js()
    uploaded = []
    try:
        names = os.listdir(paths.courses_dir())
    except OSError:
        names = []
    taken = {c.id for c in out}
    for name in names:
        d = paths.courses_dir() / name
        if not UPLOAD_CID_RE.match(name) or name in RESERVED or name in taken or d.is_symlink() or not d.is_dir():
            continue
        c = _uploaded_course(d, trusted)
        if c:
            c.release = rel
            uploaded.append(c)
    uploaded.sort(key=lambda c: (c.created_at or '', c.id), reverse=True)
    return out + uploaded


def get_course(cid):
    """Course。なければ None(シミュレーター未導入なら release・snapshot はない)"""
    if not isinstance(cid, str) or not CID_RE.match(cid):
        return None
    for c in list_course_objects():
        if c.id == cid:
            return c
    return None


def _settings_path():
    return paths.courses_dir() / 'settings.json'


def get_default():
    try:
        with open(_settings_path(), encoding='utf-8') as f:
            cid = json.load(f).get('default')
    except (OSError, ValueError, AttributeError):
        cid = None
    if isinstance(cid, str) and cid != 'default' and get_course(cid) is not None:
        return cid
    return 'default'


def set_default(cid):
    with _LOCK:
        if get_course(cid) is None:
            raise CourseError('コースがありません: %s' % cid, 404)
        paths.courses_dir().mkdir(parents=True, exist_ok=True)
        paths.write_durable(_settings_path(), json.dumps({'default': cid}))
        return {'ok': True, 'default': cid}


def list_courses():
    default = get_default()
    return {'default': default, 'courses': [c.to_dict(default) for c in list_course_objects()]}


def delete_course(cid):
    with _LOCK:
        c = get_course(cid)
        if c is None:
            raise CourseError('コースがありません: %s' % cid, 404)
        if not c.deletable:
            raise CourseError('Default と同梱のコースは削除できません', 409)
        d = paths.courses_dir() / cid
        base = paths.courses_dir().resolve()
        if d.is_symlink() or d.resolve().parent != base:
            raise CourseError('削除できない場所です', 400)
        was_default = get_default() == cid
        # 先に作業用の名前に変えてから消す(消している途中で電源が切れても、半分消えたコースが一覧に出ない)
        trash = paths.courses_dir() / ('.del-%s' % uuid.uuid4().hex[:12])
        os.replace(d, trash)
        paths.fsync_dir(paths.courses_dir())
        shutil.rmtree(trash, ignore_errors=True)
        if was_default:
            set_default('default')
        return {'ok': True}


def _meta_id(d):
    try:
        meta = json.loads((d / 'course.json').read_text(encoding='utf-8'))
        cid = meta.get('id') if isinstance(meta, dict) and meta.get('format') == 'jetformer-sim-course-meta/v1' else None
    except (OSError, ValueError):
        return None
    return cid if isinstance(cid, str) and UPLOAD_CID_RE.match(cid) and cid not in RESERVED else None


def sweep_stale(max_age_s=STALE_S):
    """電源断・強制終了で data/sim_courses/ に残った作業ファイルを片付ける(review で追加)。
    - .upload-*.zip(受け取り途中の ZIP)・.tmp-*(登録途中)・.del-*(削除途中): max_age_s より古ければ消す
    - .old-*(置き換えの途中で止まったときの前のコース): 元の場所が空いていれば戻し、そうでなければ消す
    戻り値 {'removed': n, 'restored': [cid]}"""
    base = paths.courses_dir()
    out = {'removed': 0, 'restored': []}
    with _LOCK:
        try:
            names = os.listdir(base)
        except OSError:
            return out
        now = time.time()
        for name in names:
            p = base / name
            try:
                if p.is_symlink():
                    continue
                if name.startswith('.old-') and p.is_dir():
                    cid = _meta_id(p)
                    dest = base / cid if cid else None
                    if dest is not None and _meta_id(dest) != cid:
                        if dest.exists() and not dest.is_symlink():
                            shutil.rmtree(dest, ignore_errors=True)
                        if not dest.exists():
                            os.replace(p, dest)
                            out['restored'].append(cid)
                            continue
                    shutil.rmtree(p, ignore_errors=True)
                    out['removed'] += 1
                elif name.startswith(('.upload-', '.tmp-', '.del-', 'settings.json.tmp')) and now - p.stat().st_mtime > max_age_s:
                    if p.is_dir():
                        shutil.rmtree(p, ignore_errors=True)
                    else:
                        p.unlink(missing_ok=True)
                    out['removed'] += 1
            except OSError:
                continue
        if out['removed'] or out['restored']:
            paths.fsync_dir(base)
    return out


# ---------------------------------------------------------------- 検証
def _num(v, what, errors, lo=None, hi=None, lo_open=False):
    if not (isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)):
        errors.append('%s が有限の数ではありません' % what)
        return None
    if lo is not None and (v <= lo if lo_open else v < lo):
        errors.append('%s は %s %s にしてください(今は %r)' % (what, lo, 'より大きく' if lo_open else '以上', v))
        return None
    if hi is not None and v > hi:
        errors.append('%s は %s 以下にしてください(今は %r)' % (what, hi, v))
        return None
    return v


def _coord(v, what, errors):
    return _num(v, what, errors, -COORD_LIMIT, COORD_LIMIT)


def _xz(v, what, errors):
    if not (isinstance(v, list) and len(v) == 2):
        errors.append('%s は [x, z] の 2 つの数にしてください' % what)
        return None
    x, z = _coord(v[0], what + '[0]', errors), _coord(v[1], what + '[1]', errors)
    return None if x is None or z is None else [x, z]


def _seg_dist(p, a, b):
    ax, az, bx, bz = a[0], a[1], b[0], b[1]
    dx, dz = bx - ax, bz - az
    L = dx * dx + dz * dz
    t = 0.0 if L == 0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - az) * dz) / L))
    return math.hypot(p[0] - (ax + t * dx), p[1] - (az + t * dz))


def validate_world(world):
    """physics-world.json を確かめて (きれいにした world, 警告) を返す。だめなら CourseError(理由の一覧)。
    値は丸めない(sim-core の normalizedWorld は黙って既定値にするので、その前にエラーにする)"""
    errors, warnings = [], []
    if not isinstance(world, dict):
        raise CourseError('physics-world.json はオブジェクトにしてください')
    allowed = {'segments', 'disks', 'surfaces', 'bounds', 'route', 'requiredCrossings', 'spawn'}
    extra = sorted(set(world) - allowed)
    if extra:
        warnings.append('physics-world.json の未知の項目は使いません: %s' % ', '.join(extra[:8]))
    out = {}
    segs = world.get('segments', [])
    if not isinstance(segs, list) or len(segs) > 2000:
        errors.append('segments は 0〜2000 件の配列にしてください')
        segs = []
    out['segments'] = []
    for i, s in enumerate(segs):
        w = 'segments[%d]' % i
        if not isinstance(s, dict):
            errors.append('%s がオブジェクトではありません' % w)
            continue
        a, b = _xz(s.get('a'), w + '.a', errors), _xz(s.get('b'), w + '.b', errors)
        r = _num(s.get('radius'), w + '.radius', errors, 0, 0.5, lo_open=True)
        h = _num(s.get('height'), w + '.height', errors, 0, 5, lo_open=True)
        if s.get('kind') not in SEGMENT_KINDS:
            errors.append('%s.kind は %s のどれかにしてください' % (w, '・'.join(SEGMENT_KINDS)))
        if None not in (a, b, r, h) and s.get('kind') in SEGMENT_KINDS:
            out['segments'].append({'a': a, 'b': b, 'radius': r, 'height': h, 'kind': s['kind']})
    disks = world.get('disks', [])
    if not isinstance(disks, list) or len(disks) > 4000:
        errors.append('disks は 0〜4000 件の配列にしてください')
        disks = []
    out['disks'] = []
    for i, s in enumerate(disks):
        w = 'disks[%d]' % i
        if not isinstance(s, dict):
            errors.append('%s がオブジェクトではありません' % w)
            continue
        x, z = _coord(s.get('x'), w + '.x', errors), _coord(s.get('z'), w + '.z', errors)
        r = _num(s.get('radius'), w + '.radius', errors, 0, 0.5, lo_open=True)
        h = _num(s.get('height'), w + '.height', errors, 0, 5, lo_open=True)
        if s.get('kind') not in DISK_KINDS:
            errors.append('%s.kind は %s のどれかにしてください' % (w, '・'.join(DISK_KINDS)))
        if None not in (x, z, r, h) and s.get('kind') in DISK_KINDS:
            out['disks'].append({'x': x, 'z': z, 'radius': r, 'height': h, 'kind': s['kind']})
    surfs = world.get('surfaces')
    if not isinstance(surfs, list) or not 1 <= len(surfs) <= 64:
        errors.append('surfaces は 1〜64 件の配列にしてください')
        surfs = []
    out['surfaces'] = []
    ids = set()
    for i, s in enumerate(surfs):
        w = 'surfaces[%d]' % i
        if not isinstance(s, dict):
            errors.append('%s がオブジェクトではありません' % w)
            continue
        sid = s.get('id')
        if not isinstance(sid, str) or not SURFACE_RE.match(sid):
            errors.append('%s.id は英小文字・数字・- の 32 文字以内にしてください' % w)
        elif sid in ids:
            errors.append('%s.id が重複しています: %s' % (w, sid))
        x, z = _coord(s.get('x'), w + '.x', errors), _coord(s.get('z'), w + '.z', errors)
        ww = _num(s.get('w'), w + '.w', errors, 0, 50, lo_open=True)
        dd = _num(s.get('d'), w + '.d', errors, 0, 50, lo_open=True)
        mu = _num(s.get('mu'), w + '.mu', errors, 0.01, 3)
        if isinstance(sid, str) and SURFACE_RE.match(sid) and sid not in ids and None not in (x, z, ww, dd, mu):
            ids.add(sid)
            out['surfaces'].append({'id': sid, 'x': x, 'z': z, 'w': ww, 'd': dd, 'mu': mu})
    b = world.get('bounds')
    bounds = None
    if not isinstance(b, dict):
        errors.append('bounds({minX, maxX, minZ, maxZ})がありません')
    else:
        vals = {k: _coord(b.get(k), 'bounds.' + k, errors) for k in ('minX', 'maxX', 'minZ', 'maxZ')}
        if None not in vals.values():
            if not (vals['minX'] < vals['maxX'] and vals['minZ'] < vals['maxZ']):
                errors.append('bounds は minX < maxX、minZ < maxZ にしてください')
            elif vals['maxX'] - vals['minX'] > 60 or vals['maxZ'] - vals['minZ'] > 60:
                errors.append('bounds の幅と奥行きは 60 m 以下にしてください')
            else:
                bounds = vals
    out['bounds'] = bounds
    route = world.get('route')
    out['route'] = []
    if not isinstance(route, list) or not 3 <= len(route) <= 400:
        errors.append('route(ゲートの列)は 3〜400 点にしてください')
        route = []
    for i, p in enumerate(route):
        q = _xz(p, 'route[%d]' % i, errors)
        if q is None:
            continue
        if out['route'] and math.hypot(q[0] - out['route'][-1][0], q[1] - out['route'][-1][1]) <= 0.05:
            errors.append('route[%d] が前の点に近すぎます(0.05 m より離してください)' % i)
        if bounds and not (bounds['minX'] + 0.2 <= q[0] <= bounds['maxX'] - 0.2 and bounds['minZ'] + 0.2 <= q[1] <= bounds['maxZ'] - 0.2):
            errors.append('route[%d] が bounds の内側(0.2 m の余裕)にありません' % i)
        out['route'].append(q)
    if len(out['route']) >= 3 and math.hypot(out['route'][0][0] - out['route'][-1][0], out['route'][0][1] - out['route'][-1][1]) <= 0.05:
        errors.append('route の最後の点が最初の点と重なっています(最後のゲートは自動で最初の点に戻ります。最後の点を消してください)')
    rc = world.get('requiredCrossings', [])
    out['requiredCrossings'] = []
    if not isinstance(rc, list) or len(rc) > 8:
        errors.append('requiredCrossings は 0〜8 件の配列にしてください')
        rc = []
    surf_by_id = {s['id']: s for s in out['surfaces']}
    for i, g in enumerate(rc):
        w = 'requiredCrossings[%d]' % i
        if not isinstance(g, dict):
            errors.append('%s がオブジェクトではありません' % w)
            continue
        gid = g.get('id', 'crossing-%d' % i)
        if not isinstance(gid, str) or not SURFACE_RE.match(gid):
            errors.append('%s.id は英小文字・数字・- にしてください' % w)
        s = surf_by_id.get(g.get('surface_id'))
        if s is None:
            errors.append('%s.surface_id が surfaces にありません: %r' % (w, g.get('surface_id')))
        lo, hi, z = _coord(g.get('minX'), w + '.minX', errors), _coord(g.get('maxX'), w + '.maxX', errors), _coord(g.get('z'), w + '.z', errors)
        d = g.get('direction')
        if d not in (1, -1) or isinstance(d, bool):
            errors.append('%s.direction は 1 か -1 にしてください' % w)
        if None in (lo, hi, z) or s is None:
            continue
        if not lo < hi:
            errors.append('%s は minX < maxX にしてください' % w)
            continue
        eps = 1e-6
        if not (s['x'] - s['w'] / 2 - eps <= lo and hi <= s['x'] + s['w'] / 2 + eps and s['z'] - s['d'] / 2 - eps <= z <= s['z'] + s['d'] / 2 + eps):
            errors.append('%s の開口が路面 %s の中にありません' % (w, s['id']))
            continue
        if isinstance(gid, str) and SURFACE_RE.match(gid) and d in (1, -1):
            out['requiredCrossings'].append({'id': gid, 'surface_id': s['id'], 'minX': lo, 'maxX': hi, 'z': z, 'direction': d})
    sp = world.get('spawn')
    out['spawn'] = None
    if not isinstance(sp, dict):
        errors.append('spawn({x, z, yaw})がありません')
    else:
        x, z = _coord(sp.get('x'), 'spawn.x', errors), _coord(sp.get('z'), 'spawn.z', errors)
        yaw = _num(sp.get('yaw'), 'spawn.yaw', errors, -math.pi, math.pi)
        if None not in (x, z, yaw):
            out['spawn'] = {'x': x, 'z': z, 'yaw': yaw}
            if bounds and not (bounds['minX'] < x < bounds['maxX'] and bounds['minZ'] < z < bounds['maxZ']):
                errors.append('spawn が bounds の内側にありません')
            near = None
            for s in out['segments']:
                gap = _seg_dist((x, z), s['a'], s['b']) - s['radius']
                near = gap if near is None else min(near, gap)
            for s in out['disks']:
                gap = math.hypot(x - s['x'], z - s['z']) - s['radius']
                near = gap if near is None else min(near, gap)
            if near is not None and near < SPAWN_ERROR_M:
                errors.append('spawn が壁や柱に近すぎます(%.2f m。0.10 m 以上離してください)' % near)
            elif near is not None and near < SPAWN_WARN_M:
                warnings.append('spawn が壁や柱に近いです(%.2f m)。車体がぶつかる恐れがあります' % near)
    if errors:
        raise CourseError(errors[:40] + (['ほか %d 件' % (len(errors) - 40)] if len(errors) > 40 else []))
    return out, warnings


def _image_size(data):
    """PNG・JPEG の大きさ (w, h) をヘッダーから読む(展開しない)。種類が違えば None"""
    if data[:8] == b'\x89PNG\r\n\x1a\n' and len(data) >= 24 and data[12:16] == b'IHDR':
        return 'png', int.from_bytes(data[16:20], 'big'), int.from_bytes(data[20:24], 'big')
    if data[:3] == b'\xff\xd8\xff':
        i = 2
        while i + 9 < len(data):
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7 or marker == 0xFF:
                i += 1 if marker == 0xFF else 2
                continue
            length = int.from_bytes(data[i + 2:i + 4], 'big')
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                return 'jpg', int.from_bytes(data[i + 7:i + 9], 'big'), int.from_bytes(data[i + 5:i + 7], 'big')
            i += 2 + length
        return 'jpg', 0, 0
    return None


def clean_texture(name, data):
    """テクスチャを確かめて再エンコードしたバイト列を返す(メタデータを落とす)。だめなら CourseError"""
    import cv2
    import numpy as np
    if not TEXTURE_RE.match(name):
        raise CourseError('テクスチャの名前は英小文字・数字・_・- で .jpg か .png にしてください: %s' % name)
    if len(data) > MAX_TEXTURE:
        raise CourseError('テクスチャが大きすぎます(8 MiB まで): %s' % name)
    info = _image_size(data)
    ext = name.rsplit('.', 1)[1]
    if info is None or info[0] != ext:
        raise CourseError('%s の中身が拡張子と違います(JPEG・PNG だけ受け付けます)' % name)
    if not (0 < info[1] <= MAX_TEXTURE_PX and 0 < info[2] <= MAX_TEXTURE_PX):
        raise CourseError('%s の大きさが 1〜%d px ではありません' % (name, MAX_TEXTURE_PX))
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED if ext == 'png' else cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise CourseError('%s を画像として読めません' % name)
    if img.shape[0] > MAX_TEXTURE_PX or img.shape[1] > MAX_TEXTURE_PX:
        raise CourseError('%s が大きすぎます' % name)
    ok, buf = cv2.imencode('.jpg' if ext == 'jpg' else '.png', img, [cv2.IMWRITE_JPEG_QUALITY, 90] if ext == 'jpg' else [])
    if not ok:
        raise CourseError('%s を保存し直せません' % name)
    return buf.tobytes()


def _check_visual(v, asset_names, errors):
    if v is None:
        return {}
    if not isinstance(v, dict):
        errors.append('course.json の visual はオブジェクトにしてください')
        return {}
    textures = set(asset_names) | set(DIST_TEXTURES)
    out = {}
    if 'floor_texture' in v:
        if v['floor_texture'] not in textures:
            errors.append('visual.floor_texture が assets にも標準のテクスチャにもありません: %r' % v['floor_texture'])
        else:
            out['floor_texture'] = v['floor_texture']
    st = v.get('surface_textures')
    if st is not None:
        if not isinstance(st, dict) or len(st) > 64:
            errors.append('visual.surface_textures は {路面の id: テクスチャ名} にしてください')
        else:
            out['surface_textures'] = {}
            for k, t in st.items():
                if not SURFACE_RE.match(str(k)) or t not in textures:
                    errors.append('visual.surface_textures の %r → %r が使えません' % (k, t))
                else:
                    out['surface_textures'][k] = t
    cs = v.get('colors')
    if cs is not None:
        if not isinstance(cs, dict):
            errors.append('visual.colors はオブジェクトにしてください')
        else:
            out['colors'] = {}
            for k, c in cs.items():
                if k not in COLOR_KEYS or not isinstance(c, str) or not COLOR_RE.match(c):
                    errors.append('visual.colors の %r は %s のどれかで "#rrggbb" にしてください' % (k, '・'.join(COLOR_KEYS)))
                else:
                    out['colors'][k] = c.lower()
    unknown = set(v) - {'floor_texture', 'surface_textures', 'colors'}
    if unknown:
        errors.append('visual の未知の項目: %s' % ', '.join(sorted(map(str, unknown))[:5]))
    return out


def _name_ok(name):
    return isinstance(name, str) and 1 <= len(name) <= 40 and not any(ord(c) < 0x20 or 0x7f <= ord(c) < 0xa0 for c in name)


def validate_package(path):
    """コースの ZIP を確かめる。戻り値 dict(meta, world, assets{名前: bytes}, js{名前: bytes}, warnings, format)。
    だめなら CourseError(理由の一覧)"""
    path = Path(path)
    if path.stat().st_size > MAX_ZIP:
        raise CourseError('ZIP が大きすぎます(64 MiB まで)', 413)
    try:
        zf = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError):
        raise CourseError('ZIP として読めません')
    with zf:
        try:
            entries = ziputil.scan(zf, MAX_ENTRIES, max_ratio=200, ratio_min_bytes=1 << 16)
        except ziputil.ZipCheckError as e:
            raise CourseError(str(e))
        files = [(info, parts) for info, parts, kind in entries if kind == 'file']
        if sum(info.file_size for info, _ in files) > MAX_UNPACKED:
            raise CourseError('展開後の大きさが 128 MiB を超えます', 413)
        roots = {parts[0] for _, parts in files}
        strip = 0
        if len(roots) == 1 and all(len(p) > 1 for _, p in files):
            strip = 1                                # 1 つの最上位フォルダーの下にまとまっている
        byname = {}
        warnings = []
        for info, parts in files:
            rel = '/'.join(parts[strip:])
            byname[rel] = info
        if 'COMPLETE-MANIFEST.json' in byname or 'bridge.py' in byname:
            raise CourseError('シミュレーター本体の ZIP です。Jetson の ~/sim に置いて「導入」してください(コースのアップロードには使えません)')
        accepted = {}
        for rel, info in byname.items():
            if rel in ('manifest.json', 'course.json', 'physics-world.json', 'scene.js', 'world.js'):
                accepted[rel] = info
            elif rel.startswith('assets/') and rel.count('/') == 1 and re.search(r'\.(jpg|jpeg|png)$', rel, re.I):
                accepted[rel] = info
            elif '/' not in rel and re.match(r'^README.*\.(txt|md)$', rel, re.I):
                continue
            else:
                warnings.append('使わないファイルを無視しました: %s' % rel)
        if 'physics-world.json' not in accepted:
            raise CourseError('physics-world.json がありません(コースの形はデータで受け付けます)')
        if 'manifest.json' not in accepted and 'course.json' not in accepted:
            raise CourseError('manifest.json(シミュレーターのスナップショット形式)か course.json(JetFormer 形式)が必要です')

        def read_json(name, limit):
            raw = ziputil.read_limited(zf, accepted[name], limit, name)
            try:
                return raw, ziputil.strict_json(raw, name, max_depth=8)
            except ValueError as e:
                raise CourseError(str(e))
        try:
            phys_raw, world = read_json('physics-world.json', MAX_PHYSICS)
            manifest = read_json('manifest.json', MAX_JSON)[1] if 'manifest.json' in accepted else None
            course = read_json('course.json', MAX_JSON)[1] if 'course.json' in accepted else None
        except ziputil.ZipCheckError as e:
            raise CourseError(str(e), 413)
        js = {}
        for name in ('scene.js', 'world.js'):
            if name in accepted:
                try:
                    js[name] = ziputil.read_limited(zf, accepted[name], MAX_JS, name)
                except ziputil.ZipCheckError as e:
                    raise CourseError(str(e), 413)
        assets = {}
        tex = [r for r in accepted if r.startswith('assets/')]
        if len(tex) > MAX_TEXTURES:
            raise CourseError('テクスチャは %d 枚までです' % MAX_TEXTURES)
        for rel in tex:
            name = rel.split('/', 1)[1]
            try:
                raw = ziputil.read_limited(zf, accepted[rel], MAX_TEXTURE, rel)
            except ziputil.ZipCheckError as e:
                raise CourseError(str(e), 413)
            assets[name] = clean_texture(name, raw)

    errors = []
    meta = {}
    fmt = 'jetformer' if course is not None else 'snapshot'
    if manifest is not None:
        if not isinstance(manifest, dict):
            raise CourseError('manifest.json はオブジェクトにしてください')
        mfiles = manifest.get('files') or {}
        if not isinstance(mfiles, dict):
            errors.append('manifest.json の files は {ファイル名: sha256} にしてください')
            mfiles = {}
        if 'physics-world.json' in mfiles and mfiles['physics-world.json'] != hashlib.sha256(phys_raw).hexdigest():
            errors.append('physics-world.json の sha256 が manifest.json と一致しません')
        for name, raw in js.items():
            if name in mfiles and mfiles[name] != hashlib.sha256(raw).hexdigest():
                errors.append('%s の sha256 が manifest.json と一致しません' % name)
        meta.update(world_version=manifest.get('world_version'), world_hash=manifest.get('world_hash'),
                    description=manifest.get('reason'))
    if course is not None:
        if not isinstance(course, dict) or course.get('format') != 'jetformer-sim-course/v1':
            raise CourseError('course.json の format は "jetformer-sim-course/v1" にしてください')
        for k in ('name', 'world_version', 'id', 'description', 'world_hash', 'visual'):
            if k in course:
                meta[k] = course[k]
        unknown = set(course) - {'format', 'name', 'world_version', 'id', 'description', 'world_hash', 'visual'}
        if unknown:
            warnings.append('course.json の未知の項目は使いません: %s' % ', '.join(sorted(unknown)[:5]))
        if manifest is not None and manifest.get('world_version') and course.get('world_version') != manifest.get('world_version'):
            errors.append('course.json と manifest.json の world_version が違います')
    wv = meta.get('world_version')
    if not isinstance(wv, str) or not WV_RE.match(wv):
        errors.append('world_version は英数字・._- の 80 文字以内にしてください(コースごとに一意)')
    name = meta.get('name', wv)
    if not _name_ok(name):
        errors.append('name は 1〜40 文字にしてください(制御文字は使えません)')
    desc = meta.get('description') or ''
    if not isinstance(desc, str):
        desc = ''
    desc = ''.join(c for c in desc if ord(c) >= 0x20)[:200]
    cid = meta.get('id') if meta.get('id') is not None else (_id_from(wv) if isinstance(wv, str) else None)
    if not isinstance(cid, str) or not UPLOAD_CID_RE.match(cid):
        errors.append('id は英小文字・数字・- の 2〜48 文字にしてください: %r' % (cid,))
    elif cid in RESERVED:
        errors.append('id に予約語(%s)は使えません' % '・'.join(sorted(RESERVED)))
    visual = _check_visual(meta.get('visual'), assets.keys(), errors) if fmt == 'jetformer' else {}
    try:
        clean, w2 = validate_world(world)
        warnings += w2
    except CourseError as e:
        errors += e.errors
        clean = None
    if errors:
        raise CourseError(errors)
    desc_hash = worldhash.describe(clean)
    declared = meta.get('world_hash')
    if declared is not None and declared != desc_hash['world_hash']:
        raise CourseError('world_hash が計算した値と違います(宣言 %s、計算 %s)。シミュレーター本体の版の違いが疑われます'
                          % (declared, desc_hash['world_hash']))
    if 'scene.js' in js or 'world.js' in js:
        if set(js) != {'scene.js', 'world.js'}:
            warnings.append('scene.js と world.js は 2 つそろっていないと使いません。簡易描画で表示します')
            js = {}
    return {'meta': {'id': cid, 'name': name, 'description': desc, 'world_version': wv, 'visual': visual},
            'world': clean, 'hash': desc_hash, 'assets': assets, 'js': js, 'warnings': warnings, 'format': fmt}


def import_package(path, filename='course.zip', replace=False):
    """コースの ZIP を確かめて登録する。戻り値 {course, already_registered}。同時に 1 つだけ(_LOCK)"""
    with _LOCK:
        sweep_stale()
        return _import_package(path, filename, replace)


def _import_package(path, filename, replace):
    pkg = validate_package(path)
    meta, h = pkg['meta'], pkg['hash']
    cid, wv = meta['id'], meta['world_version']
    existing = list_course_objects()
    for c in existing:
        if c.world_version == wv:
            if c.canonical_sha256 and c.canonical_sha256 == h['canonical_sha256']:
                return {'course': c.to_dict(get_default()), 'already_registered': True}
            if not (replace and c.id == cid and c.deletable):
                raise CourseError('同じ world_version(%s)で形が違うコースがあります(%s)。world_version を変えてください' % (wv, c.name), 409)
    same_id = next((c for c in existing if c.id == cid), None)
    if same_id is not None and not (replace and same_id.deletable):
        raise CourseError('同じ id のコースがあります: %s%s' % (cid, '' if same_id.deletable else '(置き換えられません)'), 409)
    warnings = list(pkg['warnings'])
    for c in existing:
        if c.canonical_sha256 == h['canonical_sha256'] and c.world_version != wv:
            warnings.append('同じ形のコース「%s」があります' % c.name)
    kind = 'data'
    js = pkg['js']
    if js:
        shas = {n: hashlib.sha256(b).hexdigest() for n, b in js.items()}
        trusted = _trusted_js()
        k = known_by_js(shas['scene.js'], shas['world.js'])
        if (shas['scene.js'], shas['world.js']) in trusted and (k is None or k.get('canonical_sha256') == h['canonical_sha256']):
            kind = 'trusted-js'
            if k is None:
                warnings.append('scene.js・world.js は導入済みの版のものですが、physics-world.json と同じ形かは確かめていません')
        else:
            warnings.append('JS は実行しません。簡易描画で表示します(scene.js・world.js が導入済みのシミュレーターのものと一致しません)')
            js = {}
    world = pkg['world']
    b = world['bounds']
    record = {'format': 'jetformer-sim-course-meta/v1', 'id': cid, 'name': meta['name'], 'description': meta['description'],
              'kind': kind, 'world_version': wv, 'world_hash': h['world_hash'], 'canonical_sha256': h['canonical_sha256'],
              'sim_core_sha256': (install.active_release().files.get('dist/sim-core.js') if install.active_release() else None),
              'stats': h['stats'], 'size_m': _size_of(b), 'has_shortcut': bool(world['requiredCrossings']),
              'source': {'filename': os.path.basename(str(filename))[:120], 'bytes': Path(path).stat().st_size,
                         'sha256': _sha256_file(path), 'format': pkg['format']},
              'visual': meta['visual'], 'assets': sorted(pkg['assets']), 'warnings': warnings,
              'created_at': datetime.datetime.now().isoformat(timespec='seconds')}
    base = paths.courses_dir()
    base.mkdir(parents=True, exist_ok=True)
    stage = base / ('.tmp-%s' % uuid.uuid4().hex[:12])
    try:
        stage.mkdir()
        (stage / 'physics-world.json').write_text(json.dumps(world, ensure_ascii=False), encoding='utf-8')
        if pkg['assets']:
            (stage / 'assets').mkdir()
            for n, data in pkg['assets'].items():
                (stage / 'assets' / n).write_bytes(data)
        for n, data in js.items():
            (stage / n).write_bytes(data)
        (stage / 'course.json').write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding='utf-8')
        for f in stage.rglob('*'):              # 電源断に備えて中身を書き切ってから置く(review で追加)
            if f.is_file():
                paths.fsync_file(f)
        paths.fsync_dir(stage)
        if (stage / 'assets').is_dir():
            paths.fsync_dir(stage / 'assets')
        dest = base / cid
        if dest.exists():
            if dest.is_symlink() or dest.resolve().parent != base.resolve():
                raise CourseError('置き換えられない場所です', 400)
            old = base / ('.old-%s' % uuid.uuid4().hex[:12])
            os.replace(dest, old)                # ここで止まっても sweep_stale() が前のコースを戻す
            os.replace(stage, dest)
            paths.fsync_dir(base)
            shutil.rmtree(old, ignore_errors=True)
        else:
            os.replace(stage, dest)
            paths.fsync_dir(base)
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
    c = get_course(cid)
    return {'course': c.to_dict(get_default()) if c else dict(record), 'already_registered': False}


# ---------------------------------------------------------------- 汎用描画のデータ・書き出し
def course_world(c):
    """コースの physics-world(dict)。なければ None"""
    p = c.physics_path
    if p is None or not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def js_json(value):
    """<script> や JS モジュールに埋め込んでも安全な JSON(ensure_ascii + < > & のエスケープ)"""
    s = json.dumps(value, ensure_ascii=True, separators=(',', ':'))
    return s.replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')


def course_data_module(cid):
    """/sim/c/<cid>/_jf/course-data.js の中身(data コースだけ)。JS のコードとして文字列を組み立てない"""
    c = get_course(cid)
    if c is None or c.kind != 'data':
        return None
    w = course_world(c)
    if w is None:
        return None
    data = {'format': 'jetformer-sim-course-data/v1', 'id': c.id, 'world_version': c.world_version, 'name': c.name,
            'physics': {k: w.get(k) for k in ('segments', 'disks', 'surfaces', 'bounds')},
            'world': {'route': w.get('route'), 'requiredCrossings': w.get('requiredCrossings', []), 'spawn': w.get('spawn')},
            'visual': c.visual or {},
            'assets': sorted(os.listdir(c.assets_dir)) if c.assets_dir else []}
    return 'export default %s;\n' % js_json(data)


def export_package(cid):
    """JetFormer 形式(course.json + physics-world.json [+ assets/])の ZIP を bytes で返す(雛形)"""
    c = get_course(cid)
    if c is None:
        raise CourseError('コースがありません: %s' % cid, 404)
    w = course_world(c)
    if w is None:
        raise CourseError('このコースは書き出せません(形のデータがありません。Default が JetFormer の知らない版の場合など)', 409)
    course = {'format': 'jetformer-sim-course/v1', 'name': ('%s のコピー' % c.name)[:40] if c.kind != 'data' else c.name,
              'world_version': c.world_version, 'world_hash': c.world_hash,
              'description': '書き出し元: %s(%s)。編集したら world_version と id を変えてアップロードしてください' % (c.name, c.world_version)}
    if c.kind == 'data':
        course['id'] = c.id
        if c.visual:
            course['visual'] = c.visual
    buf = io.BytesIO()
    top = c.id
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('%s/course.json' % top, json.dumps(course, ensure_ascii=False, indent=1))
        z.writestr('%s/physics-world.json' % top, json.dumps(w, ensure_ascii=False, indent=1))
        if c.kind == 'data' and c.assets_dir:
            for n in sorted(os.listdir(c.assets_dir)):
                p = c.assets_dir / n
                if TEXTURE_RE.match(n) and p.is_file() and not p.is_symlink():
                    z.write(p, '%s/assets/%s' % (top, n))
        z.writestr('%s/README.txt' % top, (
            'JetFormer のシミュレーター用コース(jetformer-sim-course/v1)\n'
            '- physics-world.json: 衝突の形(segments・disks)・路面(surfaces)・範囲(bounds)・ゲート列(route)・必須横断(requiredCrossings)・開始位置(spawn)\n'
            '- course.json: 名前・world_version(コースごとに一意)・見た目(visual)\n'
            '- assets/: テクスチャ(任意。jpg・png)\n'
            '編集したら course.json の world_version と id を変え、world_hash の行を消してから ZIP にしてアップロードしてください。\n'
            'JS(scene.js・world.js)は実行しません。JetFormer の汎用描画で表示します。\n'))
    return buf.getvalue()
