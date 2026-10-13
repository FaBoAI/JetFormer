"""シミュレーター(ブリッジ)で集めたデータセットの一覧・取り込み・削除

ブリッジは data/sim/runtime/datasets/<セッション ID 32 桁>/ に ZIP と archive-info.json を作る(archive-info.json は
ZIP を .part から置き換えた後に書かれるので、あれば ZIP は完成している)。ここではブリッジの archive_path() と同じ
確かめ方で完成した ZIP を見つけ、JetFormer の dataset_import.import_archive() でそのまま(写しを作らずに)取り込む。
"""

import datetime
import json
import os
import re
import shutil
import zipfile
from pathlib import Path

from .. import config
from . import paths

SID_RE = re.compile(r'^[0-9a-f]{32}$')
SCHEMA = 'course-sim-jetformer-dataset-v1'
MAX_CARD = 8 << 20


class DatasetError(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def root():
    return paths.runtime_dir() / 'datasets'


def archive_path(sid):
    """完成した ZIP の (パス, archive-info)。なければ DatasetError(404)"""
    if not isinstance(sid, str) or not SID_RE.match(sid):
        raise DatasetError('セッション ID の形が違います', 404)
    base = root()
    d = base / sid
    marker = d / 'archive-info.json'
    if d.is_symlink() or not d.is_dir() or marker.is_symlink() or not marker.is_file():
        raise DatasetError('完成したデータセットがありません: %s' % sid, 404)
    try:
        info = json.loads(marker.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        raise DatasetError('archive-info.json を読めません', 404)
    fn = info.get('filename', '') if isinstance(info, dict) else ''
    if (not isinstance(info, dict) or info.get('schema') != SCHEMA or info.get('session_id') != sid
            or not isinstance(fn, str) or Path(fn).name != fn or not fn.endswith('.zip')):
        raise DatasetError('archive-info.json の中身がおかしいです', 404)
    p = d / fn
    try:
        ok = (not p.is_symlink() and p.is_file() and p.resolve().is_relative_to(base.resolve())
              and p.stat().st_size == info.get('bytes'))
    except OSError:
        ok = False
    if not ok:
        raise DatasetError('データセットの ZIP がありません(大きさが合わない・消えたなど)', 404)
    return p, info


_summary_cache = {}


def zip_summary(p):
    """ZIP の中の dataset_card.json(<名前>/dataset_card.json)から {dataset_name, frames, scenes}。中央ディレクトリとカードだけを読む"""
    st = p.stat()
    key = (str(p), st.st_size, st.st_mtime)
    hit = _summary_cache.get(key)
    if hit is not None:
        return hit
    out = {'dataset_name': None, 'frames': None, 'scenes': None, 'training_frames': None}
    try:
        with zipfile.ZipFile(p) as zf:
            cards = [i for i in zf.infolist() if i.filename.count('/') == 1 and i.filename.endswith('/dataset_card.json')]
            if len(cards) == 1 and cards[0].file_size <= MAX_CARD:
                out['dataset_name'] = cards[0].filename.split('/')[0]
                card = json.loads(zf.read(cards[0]).decode('utf-8'))
                if isinstance(card, dict):
                    for k in ('frames', 'scenes', 'training_frames'):
                        v = card.get(k)
                        if isinstance(v, list):
                            v = len(v)
                        if isinstance(v, int) and not isinstance(v, bool):
                            out[k] = v
    except (OSError, ValueError, zipfile.BadZipFile, KeyError, UnicodeDecodeError):
        pass
    if len(_summary_cache) > 256:
        _summary_cache.clear()
    _summary_cache[key] = out
    return out


def _imported(name, filename, size):
    if not name:
        return False
    try:
        with open(Path(config.DATASET_DIR) / name / 'import_source.json', encoding='utf-8') as f:
            src = json.load(f)
        return src.get('file') == filename and src.get('bytes') == size
    except (OSError, ValueError, AttributeError):
        return False


def _dir_bytes(d):
    total = 0
    for dirpath, _dirs, files in os.walk(d):
        for f in files:
            try:
                total += os.lstat(os.path.join(dirpath, f)).st_size
            except OSError:
                pass
    return total


def list_sim_datasets(active_session=None):
    """{datasets: [...](新しい順), incomplete: [{session_id, bytes, recording}], free_gb}"""
    base = root()
    out, incomplete = [], []
    try:
        names = os.listdir(base)
    except OSError:
        names = []
    for sid in names:
        d = base / sid
        if not SID_RE.match(sid) or d.is_symlink() or not d.is_dir():
            continue
        try:
            p, info = archive_path(sid)
        except DatasetError:
            incomplete.append({'session_id': sid, 'bytes': _dir_bytes(d), 'recording': sid == active_session})
            continue
        summ = zip_summary(p)
        coll = d / 'collection-summary.json'
        item = {'session_id': sid, 'filename': info['filename'], 'bytes': info['bytes'], 'sha256': info.get('sha256'),
                'kind': 'collection' if coll.is_file() and not coll.is_symlink() else 'scene',
                'created_at': datetime.datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec='seconds'),
                'dataset_name': summ['dataset_name'], 'frames': summ['frames'], 'scenes': summ['scenes'],
                'training_frames': summ['training_frames'], 'duration_s': None, 'work_bytes': None}
        if item['kind'] == 'collection':
            try:
                cs = json.loads(coll.read_text(encoding='utf-8'))
                for k in ('scenes', 'frames', 'duration_s', 'training_frames'):
                    if isinstance(cs.get(k), (int, float)) and not isinstance(cs.get(k), bool):
                        item[k] = cs[k]
            except (OSError, ValueError, AttributeError):
                pass
        elif item['scenes'] is None:
            item['scenes'] = 1
        if item['duration_s'] is None and isinstance(item['frames'], int):
            item['duration_s'] = round(item['frames'] / 30.0, 1)
        item['imported'] = _imported(summ['dataset_name'], info['filename'], info['bytes'])
        item['import_name'] = summ['dataset_name']
        out.append(item)
    out.sort(key=lambda x: x['created_at'], reverse=True)
    try:
        free = shutil.disk_usage(str(paths.sim_dir() if paths.sim_dir().exists() else config.DATA_DIR)).free
    except OSError:
        free = None
    return {'datasets': out, 'incomplete': incomplete, 'free_gb': round(free / 1e9, 1) if free is not None else None}


def delete(sid, in_flight=0, recording_sid=None):
    """runtime/datasets/<sid>/ ごと消す(ZIP・展開形・_frames)。収集中や中継中は DatasetError(409)"""
    if not isinstance(sid, str) or not SID_RE.match(sid):
        raise DatasetError('セッション ID の形が違います', 404)
    if recording_sid and sid == recording_sid:
        raise DatasetError('このセッションは収集中です。収集を終えてから消してください', 409)
    if in_flight:
        raise DatasetError('データセットの保存・まとめ・ダウンロードが動いています。終わってから消してください', 409)
    base = root()
    d = base / sid
    if d.is_symlink() or not d.is_dir():
        raise DatasetError('ありません: %s' % sid, 404)
    if d.resolve().parent != base.resolve():
        raise DatasetError('消せない場所です', 400)
    shutil.rmtree(d)
    return {'ok': True}
