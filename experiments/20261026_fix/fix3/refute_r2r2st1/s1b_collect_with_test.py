#!/usr/bin/env python3
"""s1 の続き: 朝の車輪を浮かせた確かめ(説明欄 block=N・official=1・ゲイン 1.00・発進の前進 1・AI 側は約 1 秒)が記録に入ったとき、
⑤ collect.py --day 20261103 の出力。作り物の記録(tests/test_standing_3lap.write_standing_log)を tmp に書く。本物の data/raw は読まない"""
import contextlib, importlib.machinery, importlib.util, io, json, sys, tempfile
from pathlib import Path
JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF / 'tests'))
from test_standing_3lap import write_standing_log   # noqa: E402

COLLECT = JF / 'experiments/20261026_fix/diag_launch/collect.py'
PRM = {'throttle_gain': 0.95, 'steer_gain': 1.0, 'speed_ceiling': 1, 'ceil_all_mps': 4.5, 'ceil_left_mps': 4.1, 'speed_auto': 0,
       'auto_target_mps': 4.27, 'launch_boost': 0, 'launch_us': 250.0, 'launch_mps': 2.5, 'launch_max_s': 1.2, 'zone_caps': 0}
ON = dict(PRM, launch_boost=1)
# 朝の確かめ: 基準の設定(ゲイン 1.00・天井 OFF)+ 発進の前進 ON
TEST = dict(PRM, throttle_gain=1.0, speed_ceiling=0, launch_boost=1)
DESC = 'ai_no=%d bench=real_bench_20261023 day=2026-11-03 block=%d cand=%s model=runB official=%d memo=%s'


def main():
    with tempfile.TemporaryDirectory(dir=str(Path(__file__).parent)) as td:
        tmp = Path(td)
        raw = tmp / 'raw'; raw.mkdir()
        bench = tmp / 'bench.json'
        blocks = [{'block': 1, 'run_id': '1-1'}, {'block': 13, 'run_id': '3-2'}] + [{'block': b, 'run_id': '3-%d' % (b - 11)} for b in range(12, 20) if b != 13]
        bench.write_text(json.dumps({'candidates': [{'id': 'B', 'run': 'runB'}], 'days': [{'date': '2026-11-03', 'blocks': blocks}]}), encoding='utf-8')
        n = [0]

        def add(block, params, desc, **kw):
            log = '20261103-%02d%02d00' % (8 + n[0] // 12, (n[0] * 5) % 60)
            n[0] += 1
            kw.setdefault('dur', 30.0)
            if params.get('launch_boost'):
                kw.setdefault('t_ai', 1.3); kw.setdefault('t_move', 1.4); kw.setdefault('launch', (1.3, 0.5, 1))
            write_standing_log(raw, log, run='runB', description=desc, ramp_s=0.5, params=params, **kw)
            return log
        # 朝の確かめ(手順書 3 章 1〜5。block は朝に選んだ番号のまま。memo に「試し」)
        t = add(1, TEST, DESC % (1, 1, 'B', 1, '試し'), human_after=2.4, stop_after=2.3)
        # 回 3(計画どおり。ON は全部 R1 に届く)
        for b in range(12, 20):
            add(b, ON if b % 2 else PRM, DESC % (b, b, 'EXP', 0, 'x'))
        loader = importlib.machinery.SourceFileLoader('collect_r2s', str(COLLECT))
        spec = importlib.util.spec_from_file_location('collect_r2s', str(COLLECT), loader=loader)
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        m.RAW, m.LAB, m.BENCH, m.OUT = raw, tmp / 'nolab', bench, tmp / 'out'
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            m.main(['--day', '20261103', '--out', str(tmp / 'o.json')])
        print('朝の確かめの記録:', t)
        out = buf.getvalue().splitlines()
        i = next(k for k, x in enumerate(out) if x.startswith('発進の前進の比べ'))
        print('\n'.join(out[i:]))


main()
