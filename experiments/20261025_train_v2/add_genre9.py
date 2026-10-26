#!/usr/bin/env python3
"""ジャンル 9「AI・きれいな速い周」を data/modes.json に足す(2026-10-25。了承済み: main session から)。

jetformer.modes.add(管理画面のジャンルの追加と同じ関数)で足す。既存の id は変えない。
足す前の modes.json を写しておく(experiments/20261025_train_v2/modes.json.before_genre9)。
もう id 9 が同じ名前・説明であれば何もしない。違う中身の id 9 があれば止まる。

    /home/jetson/miniconda/envs/jetracer/bin/python -I /home/jetson/jetformer/experiments/20261025_train_v2/add_genre9.py
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path('/home/jetson/jetformer')
sys.path.insert(0, str(ROOT))
from jetformer import modes  # noqa: E402

HERE = Path(__file__).resolve().parent
PROTOCOL = Path('/home/jetson/jetformer_reports/train_v2_20261025/protocol_v2.json')
BEFORE_SHA = 'f40f64bc07a4e72acfd3a278f9810e711744e44e2818a3cf19ddfb79589a6d14'   # protocol_v2.json の defaults_untouched


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    entry = json.loads(PROTOCOL.read_text(encoding='utf-8'))['tables']['genre9_entry']
    cur = modes.load()
    old = [m for m in cur if m['id'] == entry['id']]
    if old:
        m = old[0]
        if m['name'] == entry['name'] and m['description'] == entry['description']:
            print('ジャンル 9 はもうあります(何もしない):', json.dumps(m, ensure_ascii=False))
            print('modes.json の sha256:', sha(modes.MODES_FILE))
            return
        raise SystemExit('id 9 に別の中身があります(止める): %s' % json.dumps(m, ensure_ascii=False))
    before = sha(modes.MODES_FILE)
    if before != BEFORE_SHA:
        raise SystemExit('modes.json が protocol の時点と違います(止める): %s' % before)
    if max(m['id'] for m in cur) + 1 != entry['id']:
        raise SystemExit('次の id が %d ではありません(止める)' % entry['id'])
    shutil.copy2(modes.MODES_FILE, HERE / 'modes.json.before_genre9')
    new = modes.add(entry['name'], entry['description'])
    after = modes.load()
    assert new['id'] == entry['id'], new
    assert [m for m in after if m['id'] != entry['id']] == cur, '既存のジャンルが変わった'
    print('足した:', json.dumps(new, ensure_ascii=False))
    print('modes.json の sha256: 前 %s → 後 %s' % (before, sha(modes.MODES_FILE)))


if __name__ == '__main__':
    main()
