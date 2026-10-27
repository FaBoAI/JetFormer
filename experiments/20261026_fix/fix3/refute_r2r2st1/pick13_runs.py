#!/usr/bin/env python3
"""反証 r2-r2-state-1 の確かめ(追加): 朝に 13 を選んでから確かめを走ると、② runs_1103 は本物の回 3 の本に「回 3 の形が 3-2(No.1)と違う」を付けるか。
作り物の log.json だけ(tmp)。本物の data/raw は読まない"""
import contextlib, importlib.machinery, importlib.util, io, json, sys, tempfile
from pathlib import Path
RUNS = Path('/home/jetson/jetformer/experiments/20261026_fix/plan/runs_1103.py')
BASE = {'steer_gain': 1.0, 'throttle_gain': 0.95, 'lookahead': 3, 'tof_guard': 1, 'zone_caps': 0, 'start_kick': 1, 'speed_auto': 0,
        'speed_ceiling': 1, 'ceil_left_mps': 4.1, 'launch_boost': 0, 'launch_us': 250, 'rec_back_guard': 0, 'rec_hold_keep': 0,
        'launch_no_kick': 0}
TUNE = {12: '3-1', 13: '3-2', 14: '3-3', 15: '3-4'}


def rec(raw, name, ai_no, block, cand, official, memo, params):
    d = raw / name
    (d / ('%s-0001' % name)).mkdir(parents=True)
    desc = ('ai_no=%d bench=real_bench_20261023 day=2026-11-03 block=%d cand=%s model=clean_driving_20261011-nocmd-s3-20261011-024626 '
            'battery=#A V_before=8.30 V_after=__ check=__ tune=%s throttle_gain=0.9 official=%d memo=%s') % (ai_no, block, cand, TUNE[block], official, memo)
    (d / 'log.json').write_text(json.dumps({'description': desc, 'drive': {
        'run': 'clean_driving_20261011-nocmd-s3-20261011-024626', 'dry_run': False, 'params': dict(BASE, **params)}},
        ensure_ascii=False), encoding='utf-8')


with tempfile.TemporaryDirectory(dir=str(Path(__file__).parent)) as td:
    raw = Path(td) / 'raw'
    # 朝の確かめ: 13 を選んだあと基準の設定(ゲイン 1.00・天井 OFF)+ 発進の前進 ON
    rec(raw, '20261103-081500', 1, 13, 'B', 1, '試し', {'throttle_gain': 1.0, 'speed_ceiling': 0, 'launch_boost': 1})
    # 本物の回 3(回 2 のよい形 = 天井 ON・0.95)
    rec(raw, '20261103-120000', 12, 12, 'EXP', 0, '', {})
    rec(raw, '20261103-121000', 13, 13, 'EXP', 0, '', {'launch_boost': 1})
    rec(raw, '20261103-122000', 14, 14, 'EXP', 0, '', {})
    rec(raw, '20261103-123000', 15, 15, 'EXP', 0, '', {'launch_boost': 1})
    loader = importlib.machinery.SourceFileLoader('runs_st1', str(RUNS))
    spec = importlib.util.spec_from_file_location('runs_st1', str(RUNS), loader=loader)
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
