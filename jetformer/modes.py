"""走行ジャンル(駐車・高速走行など)の表

IDはモデルの出力ヘッドの番号に直結するため、一度付けたIDは変えない・再利用しない。
名前や説明は後から変えてよい。ID 0 は「未設定」で、学習には使わない。
"""

import json
import os
import threading

from . import config

MODES_FILE = config.DATA_DIR / 'modes.json'
UNSET = 0

DEFAULT_MODES = [
    {'id': 0, 'name': '未設定', 'description': 'ジャンル未設定(学習には使わない)'},
    {'id': 1, 'name': '駐車(1)', 'description': ''},
    {'id': 2, 'name': '駐車(2)', 'description': ''},
    {'id': 3, 'name': '駐車(3)', 'description': ''},
    {'id': 4, 'name': '高速走行', 'description': ''},
    {'id': 5, 'name': '低速走行', 'description': ''},
]

_lock = threading.RLock()


def load():
    if not MODES_FILE.exists():
        # 初回に既定の表をファイルへ書き出し、以後のIDを固定する
        with _lock:
            if not MODES_FILE.exists():
                _save([dict(m) for m in DEFAULT_MODES])
    with open(MODES_FILE, encoding='utf-8') as f:
        return json.load(f)


def _save(modes):
    MODES_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(MODES_FILE) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(modes, f, ensure_ascii=False, indent=1)
    os.replace(tmp, MODES_FILE)


def get(mode_id):
    for m in load():
        if m['id'] == mode_id:
            return m
    return None


def name_of(mode_id):
    m = get(mode_id)
    return m['name'] if m else 'ID %d' % mode_id


def add(name, description=''):
    name = (name or '').strip()
    if not name:
        raise ValueError('ジャンル名を入力してください')
    with _lock:
        modes = load()
        if any(m['name'] == name for m in modes):
            raise ValueError('同じ名前のジャンルがあります: %s' % name)
        new = {'id': max(m['id'] for m in modes) + 1, 'name': name, 'description': description or ''}
        modes.append(new)
        _save(modes)
        return new


def update(mode_id, name=None, description=None):
    if mode_id == UNSET:
        raise ValueError('「未設定」は変更できません')
    with _lock:
        modes = load()
        for m in modes:
            if m['id'] == mode_id:
                if name is not None:
                    name = name.strip()
                    if not name:
                        raise ValueError('ジャンル名を入力してください')
                    if any(o['name'] == name and o['id'] != mode_id for o in modes):
                        raise ValueError('同じ名前のジャンルがあります: %s' % name)
                    m['name'] = name
                if description is not None:
                    m['description'] = description
                _save(modes)
                return m
    raise KeyError(mode_id)


def validate(mode_id):
    if get(int(mode_id)) is None:
        raise ValueError('存在しないジャンルIDです: %s' % mode_id)
    return int(mode_id)
