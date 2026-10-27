#!/usr/bin/env python3
"""10/27 fix4(小さな残り): 学習 v3 の段 1 の本ごとに、エンジン・engine_check.json・走行の一覧(drive.model_info。
add_v1_block.drivable と同じ読み方)を読むだけ。GPU・TensorRT は使わない(ファイルを読むだけ)。何も書かない。

    /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python \
        experiments/20261026_fix/fix4/v3_engines.py
"""
import json
import sys
from pathlib import Path

JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF))
from jetformer import drive  # noqa: E402

GENRE = 5           # add_v1_block.GENRE と同じ(11/3 のジャンル)


def main():
    rows = []
    for d in sorted((JF / 'data' / 'runs').glob('v3_20261026-*')):
        eng = d / drive.ENGINE_FILE
        chk = drive._read_json(d / drive.ENGINE_CHECK_FILE) if (d / drive.ENGINE_CHECK_FILE).exists() else None
        info = drive.model_info(d.name)
        rows.append({
            'run': d.name,
            'engine': eng.exists(),
            'engine_mtime': eng.stat().st_mtime if eng.exists() else None,
            'engine_check_file': chk is not None,
            'check_ok': (chk or {}).get('ok'), 'accuracy_ok': (chk or {}).get('accuracy_ok'),
            'speed_ok': (chk or {}).get('speed_ok'), 'p99_ms': (chk or {}).get('run_numpy_p99_ms'),
            'checked_at': (chk or {}).get('checked_at'),
            'model_info': info is not None,
            'engine_check_current': bool(info and info.get('engine_check')),     # 今のエンジンの確認か(大きさ・時刻)
            'drivable': None if info is None else bool(info.get('drivable')),
            'why_not': None if info is None else info.get('why_not'),
            'trained_modes': None if info is None else info.get('trained_modes'),
            'genre_ok': None if info is None else GENRE in (info.get('trained_modes') or []),
        })
    print(json.dumps(rows, ensure_ascii=False, indent=1))
    n = len(rows)
    print('本 %d・エンジンあり %d・走れる(ジャンル %d)%d・engine_check 合格 %d・速さだけ不合格 %s'
          % (n, sum(r['engine'] for r in rows), GENRE, sum(bool(r['drivable'] and r['genre_ok']) for r in rows),
             sum(r['check_ok'] is True for r in rows),
             [r['run'] for r in rows if r['accuracy_ok'] is True and r['speed_ok'] is False]), file=sys.stderr)


if __name__ == '__main__':
    main()
