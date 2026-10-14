#!/usr/bin/env python3
"""合格したシーンを 1 つの集約 ZIP にまとめる(アプリの「1つのZIPにまとめる」と同じ処理を、空き容量の取り置きだけ小さくして行う)
(2026-10-14、sim_dart_20261014)

    source ~/miniconda/etc/profile.d/conda.sh && conda activate jetracer
    python -I build_collection.py --result <browser_collect の result.json> --name rl-dart-fast-10min --out <collection.json> [--reserve-gib 6]

  なぜ: ユーザーの qa/combine_scene_datasets.py(アプリの /api/datasets/collection が呼ぶもの)は、まとめる前に
  「ディスクの空き ≥ 20 GiB + 元の ZIP の 2 倍」を求める。2026-10-14 13:30 の空きは 約 14 GB で、準備の段の試しで
  「Collection exceeds storage budget or 20GiB free-space reserve」で止まった。
  やり方: コピーした sim/qa/combine_scene_datasets.py の本文を読み、取り置きの 1 行
      max_bytes=5*1024**3;reserve=20*1024**3
  だけを reserve=<--reserve-gib> GiB に替えて(1 か所だけ替わったことを確かめる)、同じ combine_collection を呼ぶ。ほかの検査
  (シーンの 600 フレーム・30 Hz・未来ラベル・衝突なし・契約が同じ・開始が分割をまたがない など)と ZIP の中身・名前の付け方はそのまま。
  ユーザーのファイル(/home/jetson/sim/...)とコピーの qa/ のファイルは変えない。ブリッジの runtime/datasets/<新しい ID>/ に ZIP ができる。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import types
from pathlib import Path

SIM = Path('/home/jetson/jetformer_reports/sim_dart_20261014/sim')
SRC = SIM / 'qa' / 'combine_scene_datasets.py'
LINE = 'max_bytes=5*1024**3;reserve=20*1024**3'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--result', required=True, help='browser_collect.py の result.json(accepted・sidecars の session_id を使う)')
    ap.add_argument('--name', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--reserve-gib', type=float, default=6.0)
    ap.add_argument('--sessions-root', default=str(SIM / 'runtime' / 'datasets'))
    a = ap.parse_args()
    res = json.loads(Path(a.result).read_text())
    side = Path(a.result).parent / 'sidecars'
    ids = [json.loads((side / f'{n}.json').read_text())['session_id'] for n in res['accepted']]
    text = SRC.read_text(encoding='utf-8')
    if text.count(LINE) != 1:
        sys.exit('combine_scene_datasets.py の取り置きの行が想定と違います')
    patched = text.replace(LINE, f'max_bytes=5*1024**3;reserve=int({a.reserve_gib}*1024**3)')
    sys.path.insert(0, str(SIM))   # ブリッジ(run_bridge.py)と同じく、コピーの sim を探す場所に足す(combine_scene_datasets が training/ を読む)
    mod = types.ModuleType('combine_scene_datasets_reserve')
    mod.__file__ = str(SRC)   # ROOT(= コピーの sim)を元と同じにする
    exec(compile(patched, str(SRC) + ' (reserve patched)', 'exec'), mod.__dict__)
    manager = types.SimpleNamespace(root=Path(a.sessions_root).resolve())
    free_before = shutil.disk_usage(manager.root).free
    result = mod.combine_collection(manager, {'name': a.name, 'session_ids': ids, 'expected_frames': 600, 'expected_duration_s': 20,
                                              'require_collision_free': True})
    cid = result['download_url'].rsplit('/', 1)[-1]
    zpath = manager.root / cid / result['filename']
    out = {'request': {'name': a.name, 'session_ids': ids}, 'response': result, 'zip': str(zpath), 'reserve_gib': a.reserve_gib,
           'free_bytes_before': free_before, 'free_bytes_after': shutil.disk_usage(manager.root).free,
           'combine_source_sha256': hashlib.sha256(SRC.read_bytes()).hexdigest(), 'patched_line': LINE + ' → reserve=' + str(a.reserve_gib) + ' GiB'}
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
