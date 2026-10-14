#!/usr/bin/env python3
"""外部の参照(作者の shortcut-rc-finetuned)を、アプリの「モデル ZIP を読み込む」で読める ZIP にする(2026-10-14)

    python -I make_ext_bundle.py

  - 元のフォルダー(/home/jetson/sim/extracted/course-simulator/models/shortcut-rc-finetuned)の 4 つのファイル
    (jetformer.onnx・model_config.json・io_spec.json・calibration.json)を、中身を変えずに ZIP の 1 つのフォルダーに入れる。
  - protocol.json の files(external_ref_onnx・external_ref_model_config・external_ref_calibration)の sha256 と一致することを確かめる。
  - 評価する 5 つの一式の sha256 を eval/bundles/bundles.json に書く。元のファイルは変更しない。
"""
from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_common import EVAL, EXT_DIR, EXT_ZIP, MODELS, REPORT  # noqa: E402


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    protocol = json.loads((REPORT / 'protocol.json').read_text(encoding='utf-8'))
    expected = {'jetformer.onnx': protocol['files']['external_ref_onnx']['sha256'],
                'model_config.json': protocol['files']['external_ref_model_config']['sha256'],
                'calibration.json': protocol['files']['external_ref_calibration']['sha256']}
    files = ['jetformer.onnx', 'model_config.json', 'io_spec.json', 'calibration.json']
    src = {f: sha(EXT_DIR / f) for f in files}
    for f, s in expected.items():
        assert src[f] == s, f'{f} の sha256 が protocol と違います'
    EXT_ZIP.parent.mkdir(parents=True, exist_ok=True)
    tmp = EXT_ZIP.with_suffix('.tmp')
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_STORED) as z:
        for f in files:
            z.write(EXT_DIR / f, f'shortcut-rc-finetuned/{f}')
    tmp.replace(EXT_ZIP)
    with zipfile.ZipFile(EXT_ZIP) as z:
        for f in files:
            assert hashlib.sha256(z.read(f'shortcut-rc-finetuned/{f}')).hexdigest() == src[f]
    out = {'schema': 'sim-rl-bestlap-eval-bundles/v1',
           'note': '閉ループ評価に読み込んだ一式。学習した 4 本は学習の段の bundles/、外部の参照は元のフォルダーの 4 ファイルをそのまま ZIP にしたもの',
           'external_source_files_sha256': src, 'models': []}
    for m in MODELS:
        out['models'].append({'label': m['label'], 'arm': m['arm'], 'seed': m['seed'], 'run': m['run'], 'zip': m['zip'],
                              'zip_sha256': sha(m['zip']), 'onnx': m['onnx'], 'onnx_sha256': sha(m['onnx']),
                              'mode': m['mode'], 'throttle_gain': m['throttle_gain']})
    (EVAL / 'bundles' / 'bundles.json').write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
