#!/usr/bin/env python3
"""記録の段: 検証に通った集約 ZIP を JetFormer のデータセットとして取り込む(2026-10-14、design.md 4.6 節)

    source ~/miniconda/etc/profile.d/conda.sh && conda activate jetracer
    jf-heavy --mem 2G --lock none -- python -I ~/jetformer/experiments/20261013_sim_rl_bestlap/record/import_dataset.py <集約ZIP>

  - jetformer.dataset_import.import_archive(zip, simulator=True)(シーンのタブ用の展開 to_scenes=True も行う)
  - そのあと jetformer.dataset.set_dataset_tags(name, ['simulator', 'rl-bestlap'])
  - ジャンルは 8(シミュレーター)のまま。data/modes.json は変えない(design.md 4.7 節: 新しいジャンル 9 は作らない)
  - 取り込んだ名前・大きさ・フレーム数などを /home/jetson/jetformer_reports/sim_rl_bestlap_20261013/rl_dataset.json に書く(S-R の学習はこれを使う)
  jetformer-web は再起動しない。jetformer/web・jetformer/sim*・data/sim* は触らない(取り込みは data/datasets/ と data/raw/ に入る)。
"""
from __future__ import annotations

import datetime
import hashlib
import json
import sys
from pathlib import Path

JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF))     # jetformer のパッケージだけを明示して読む(-I なのでカレントや PYTHONPATH は読まない)
from jetformer import config, dataset  # noqa: E402
from jetformer.dataset_import import import_archive  # noqa: E402

REPORT = Path('/home/jetson/jetformer_reports/sim_rl_bestlap_20261013')
TAGS = ['simulator', 'rl-bestlap']


def file_sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    zpath = Path(sys.argv[1]).resolve()
    modes_before = file_sha(config.DATA_DIR / 'modes.json') if (config.DATA_DIR / 'modes.json').exists() else None
    log = []
    info = import_archive(zpath, simulator=True, source_name=zpath.name,
                          progress=lambda f, msg: (log.append([round(f, 3), msg]), print(f'[取り込み] {f:.0%} {msg}', flush=True)),
                          to_scenes=True)
    card = dataset.set_dataset_tags(info['name'], TAGS)
    modes_after = file_sha(config.DATA_DIR / 'modes.json') if (config.DATA_DIR / 'modes.json').exists() else None
    root = config.DATASET_DIR / info['name']
    size = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    scenes = info.get('scenes') or {}
    out = {'schema': 'rl-bestlap-dataset/v1', 'created_at': datetime.datetime.now().isoformat(timespec='seconds'),
           'name': info['name'], 'path': str(root), 'archive': str(zpath), 'archive_bytes': zpath.stat().st_size,
           'archive_sha256': file_sha(zpath), 'dataset_bytes_on_disk': size, 'tags': card.get('tags'), 'source': card.get('source'),
           'genre_id': 8, 'num_modes': card.get('num_modes'), 'mode_frames': card.get('mode_frames'),
           'scenes': card.get('scenes'), 'frames': card.get('frames'), 'training_frames': card.get('training_frames'),
           'train_scenes': card.get('train_scenes'), 'val_scenes': card.get('val_scenes'), 'test_scenes': card.get('test_scenes'),
           'scene_log': card.get('scene_log'), 'import_summary': info.get('summary'), 'import_warnings': info.get('warnings'),
           'scenes_tab': {k: (len(v) if isinstance(v, list) else v) for k, v in scenes.items()} if isinstance(scenes, dict) else scenes,
           'modes_json_sha256_before': modes_before, 'modes_json_sha256_after': modes_after,
           'modes_json_unchanged': modes_before == modes_after, 'progress_log': log[-5:]}
    (REPORT / 'rl_dataset.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
