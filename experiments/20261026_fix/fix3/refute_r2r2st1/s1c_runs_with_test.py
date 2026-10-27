#!/usr/bin/env python3
"""s1 の続き: 手順書 3 章の順(発進の前進の確かめ → そのあと後ろの見張りを ON)で、朝の確かめが block=1 の記録として残ると、
② runs_1103 は回 1 の「最初の本」を確かめの記録にし、本物の回 1 の本に「再起動で OFF に戻った?」の印を付ける。
作り物の log.json だけ(tmp)。本物の data/raw は読まない"""
import contextlib, importlib.machinery, importlib.util, io, json, sys, tempfile
from pathlib import Path
RUNS = Path('/home/jetson/jetformer/experiments/20261026_fix/plan/runs_1103.py')
BASE = {'steer_gain': 1.0, 'throttle_gain': 0.9, 'lookahead': 3, 'tof_guard': 1, 'zone_caps': 0, 'start_kick': 1, 'speed_auto': 0,
        'speed_ceiling': 0, 'ceil_left_mps': 4.1, 'launch_boost': 0, 'launch_us': 250, 'rec_back_guard': 0, 'rec_hold_keep': 0,
        'launch_no_kick': 0}


def rec(raw, name, ai_no, block, tune, cand, official, memo, params):
    d = raw / name
    (d / ('%s-0001' % name)).mkdir(parents=True)
    desc = ('ai_no=%d bench=real_bench_20261023 day=2026-11-03 block=%d cand=%s model=clean_driving_20261011-nocmd-s3-20261011-024626 '
            'battery=#A V_before=8.30 V_after=__ check=__ tune=%s throttle_gain=0.9 official=%d memo=%s') % (ai_no, block, cand, tune, official, memo)
    (d / 'log.json').write_text(json.dumps({'description': desc, 'drive': {
        'run': 'clean_driving_20261011-nocmd-s3-20261011-024626', 'dry_run': False, 'params': dict(BASE, **params)}},
        ensure_ascii=False), encoding='utf-8')


with tempfile.TemporaryDirectory(dir=str(Path(__file__).parent)) as td:
    raw = Path(td) / 'raw'
    # 朝の確かめ(s1 の結果: 基準の設定のゲイン 1.00・発進の前進 ON・plan_check の違いなし → cand=B official=1・AI No.1)
    rec(raw, '20261103-081500', 1, 1, '1-1・P1確認・天井OFF・4〜6周', 'B', 1, '試し', {'throttle_gain': 1.0, 'launch_boost': 1})
    # 手順書 3 章: 確かめのあと見張りを ON → 回 1
    rec(raw, '20261103-090000', 2, 1, '1-1・P1確認・天井OFF・4〜6周', 'EXP', 0, '', {'throttle_gain': 0.92, 'rec_back_guard': 1})
    rec(raw, '20261103-091000', 3, 2, '1-2・P1・床1本目・止まって3周', 'EXP', 0, '', {'throttle_gain': 0.92, 'rec_back_guard': 1})
    rec(raw, '20261103-092000', 4, 3, '1-3・P1・床2本目・止まって3周', 'EXP', 0, '', {'throttle_gain': 0.92, 'rec_back_guard': 1})
    loader = importlib.machinery.SourceFileLoader('runs_r2s', str(RUNS))
    spec = importlib.util.spec_from_file_location('runs_r2s', str(RUNS), loader=loader)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    m.RAW = raw
    argv, buf = sys.argv, io.StringIO()
    sys.argv = ['runs_1103.py', '--day', '20261103', '--json']
    try:
        with contextlib.redirect_stdout(buf):
            m.main()
    finally:
        sys.argv = argv
    out = buf.getvalue().strip().splitlines()
    print('\n'.join(out[:-1]))
    for r in json.loads(out[-1]):
        print(r.get('log'), r.get('run_id'), 'No.%s' % r.get('ai_no'), r.get('diffs'))
