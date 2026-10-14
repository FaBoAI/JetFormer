#!/usr/bin/env python3
"""記録 D1・D2・D3 の集約 ZIP を JetFormer のデータセットとして取り込む(2026-10-14、sim_dart_20261014。ベストラップの import_dataset.py の写しを広げたもの)

    source ~/miniconda/etc/profile.d/conda.sh && conda activate jetracer
    jf-heavy --mem 2G --lock none -- python -I import_dataset.py <集約ZIP> --dataset D1 --out <結果.json> [--temp-data-dir <一時の data>]

  - jetformer.dataset_import.import_archive(zip, simulator=True, to_scenes=<protocol の to_scenes>)
      D1・D3(DART)はシーンのタブに展開しない(シーンから作り直すと、正解が記録の乱し入りの操作から計算し直されるため)。D2 は展開する。
  - そのあと jetformer.dataset.set_dataset_tags(name, <protocol の tags>)
  - 取り込みの学習除外(jetformer/sim/label_mask.py)が掛かっていないこと(label_masked_frames が無い・0)を確かめる。
    差し込みモジュールは info.jf を出さないので、掛からないはず(掛かっていたら付け替えと混ざるので失敗にする)
  - --temp-data-dir: JETFORMER_DATA をその一時のフォルダーにして取り込む(試しの取り込み。data/modes.json の写しを置く)。
    本番の data/ には何も書かない
  - ジャンルは 8(シミュレーター)のまま。data/modes.json は変えない。jetformer-web は再起動しない。
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

JF = Path('/home/jetson/jetformer')
PROTOCOL = Path('/home/jetson/jetformer_reports/sim_dart_20261014/protocol.json')


def file_sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('zip')
    ap.add_argument('--dataset', choices=['D1', 'D2', 'D3'], required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--temp-data-dir', default='')
    ap.add_argument('--to-scenes', choices=['protocol', 'yes', 'no'], default='protocol')
    a = ap.parse_args()
    spec = json.loads(PROTOCOL.read_text())['recording']['datasets'][a.dataset]
    real_modes = JF / 'data' / 'modes.json'
    if a.temp_data_dir:
        tmp = Path(a.temp_data_dir).resolve()
        if tmp == (JF / 'data').resolve() or JF / 'data' in tmp.parents:
            sys.exit('一時の data は本番の data/ の外に置いてください')
        (tmp / 'datasets').mkdir(parents=True, exist_ok=True)
        (tmp / 'raw').mkdir(parents=True, exist_ok=True)
        shutil.copy2(real_modes, tmp / 'modes.json')
        os.environ['JETFORMER_DATA'] = str(tmp)
    sys.path.insert(0, str(JF))     # jetformer のパッケージだけを明示して読む(-I なのでカレントや PYTHONPATH は読まない)
    from jetformer import config, dataset  # noqa: E402
    from jetformer.dataset_import import import_archive, validate_dataset  # noqa: E402
    if a.temp_data_dir:
        assert config.DATA_DIR == Path(a.temp_data_dir).resolve(), (config.DATA_DIR, a.temp_data_dir)
    to_scenes = spec['to_scenes'] if a.to_scenes == 'protocol' else a.to_scenes == 'yes'
    zpath = Path(a.zip).resolve()
    real_modes_before = file_sha(real_modes)
    log = []
    info = import_archive(zpath, simulator=True, source_name=zpath.name,
                          progress=lambda f, msg: (log.append([round(f, 3), msg]), print(f'[取り込み] {f:.0%} {msg}', flush=True)),
                          to_scenes=to_scenes)
    card = dataset.set_dataset_tags(info['name'], spec['tags'])
    root = config.DATASET_DIR / info['name']
    errors, warnings, summary = validate_dataset(root)
    masked = (info.get('summary') or {}).get('label_masked_frames')
    card = json.loads((root / 'dataset_card.json').read_text(encoding='utf-8'))
    lm = card.get('label_mask')
    out = {'schema': 'jf-sim-dart-dataset/v1', 'created_at': datetime.datetime.now().isoformat(timespec='seconds'), 'dataset_key': a.dataset,
           'name': info['name'], 'path': str(root), 'data_dir': str(config.DATA_DIR), 'temporary': bool(a.temp_data_dir),
           'archive': str(zpath), 'archive_bytes': zpath.stat().st_size, 'archive_sha256': file_sha(zpath),
           'tags': card.get('tags'), 'source': card.get('source'), 'to_scenes': to_scenes,
           'num_modes': card.get('num_modes'), 'mode_frames': card.get('mode_frames'), 'scenes': card.get('scenes'), 'frames': card.get('frames'),
           'training_frames': card.get('training_frames'), 'train_scenes': card.get('train_scenes'), 'val_scenes': card.get('val_scenes'),
           'teacher_id': None, 'import_summary': info.get('summary'), 'import_warnings': info.get('warnings'),
           'label_masked_frames': masked, 'card_label_mask': lm,
           'label_mask_clear': (not masked) and not (lm and int(lm.get('masked_frames', 0) or 0) > 0),
           'validate_errors': errors, 'validate_warnings': warnings, 'validate_summary': summary,
           'real_modes_json_unchanged': file_sha(real_modes) == real_modes_before, 'progress_log': log[-5:]}
    try:
        first = json.loads((root / 'splits.json').read_text())['train'][0]
        sess = json.loads((root / 'simulation' / 'scenes' / first / 'session.json').read_text())
        out['teacher_id'] = sess['context']['teacher']['id']
    except Exception as e:  # noqa: BLE001
        out['teacher_id'] = f'読めません: {e}'
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False, indent=1))
    if not out['label_mask_clear']:
        sys.exit('取り込みで学習除外(label_mask)が掛かりました。付け替えと混ざるので止めます')
    if errors:
        sys.exit('validate_dataset のエラー: ' + ' / '.join(errors))


if __name__ == '__main__':
    main()
