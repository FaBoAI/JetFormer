"""反証 7-2・docs: 回 3 の途中で起こし直しがあり、左の値(4.2 → 既定 4.1)を入れ直し忘れた日に、本物の ⑤ collect.py --ignore-gain が
出す行を見る(作り物の記録。作業場所 fix7/tmp_review_docs_r2 の下に作って最後に消す。本物の data/raw・labels は読まない・書かない)。
fix3/approvals.md 8(ユーザーが了承する「回 3 の判定の読み方」)の表に、この「満たさない」が載っているかと照らす。"""
import contextlib
import io
import json
import shutil
import sys
from pathlib import Path

JF = Path('/home/jetson/jetformer')
for p in (JF, JF / 'tools', JF / 'tests', JF / 'experiments/20261026_fix/fix7/tools_r1/probes'):
    sys.path.insert(0, str(p))
import gen_course as gc  # noqa: E402
from test_fix3_r1_2 import load_mod  # noqa: E402

TMP = JF / 'experiments/20261026_fix/fix7/tmp_review_docs_r2/form_split'
PRM = {'throttle_gain': 0.9, 'steer_gain': 1.0, 'speed_ceiling': 1, 'ceil_all_mps': 4.5, 'ceil_left_mps': 4.2, 'speed_auto': 0,
       'auto_target_mps': 4.27, 'launch_boost': 0, 'launch_us': 250.0, 'launch_mps': 2.5, 'launch_max_s': 1.2, 'zone_caps': 0,
       'start_kick': 1, 'rec_back_guard': 1, 'rec_hold_keep': 1, 'launch_no_kick': 0}
DESC = 'ai_no=%d bench=real_bench_20261023 day=2026-11-03 block=%d cand=EXP model=runB'


def main():
    if TMP.exists():
        shutil.rmtree(TMP)
    try:
        raw = TMP / 'raw'
        raw.mkdir(parents=True)
        bench = TMP / 'bench.json'
        bench.write_text(json.dumps({'candidates': [{'id': 'B', 'run': 'runB'}],
                                     'days': [{'date': '2026-11-03', 'blocks': [{'block': b, 'run_id': '3-%d' % (b - 11)}
                                                                                for b in range(12, 20)]}]}), encoding='utf-8')
        for i, b in enumerate(range(12, 20)):
            on = b % 2 == 1                                  # 3-2・3-4・3-6・3-8 が ON
            left = 4.2 if b < 16 else 4.1                    # 3-5 の前に起こし直し → 左を入れ直し忘れ(既定 4.1)
            prm = dict(PRM, launch_boost=1 if on else 0, ceil_left_mps=left)
            d_move, ramp = (0.15 + 0.03 * i, 0.3) if on else (0.20 + 0.01 * i, 0.6)
            vk = gc.launch_keys(1.0, d_move, ramp, until=30.0)
            gc.write_log(raw, '20261103-13%02d00' % (i * 5), 30.0, vk, [(1.0, 28.0)], prm, DESC % (i + 12, b),
                         launch=[(1.0, 0.5)] if on else [])
        m = load_mod('collect_review_docs_r2', JF / 'experiments/20261026_fix/diag_launch/collect.py')
        m.RAW, m.LAB, m.BENCH, m.OUT = raw, TMP / 'nolab', bench, TMP / 'out'
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            m.main(['--day', '20261103', '--ignore-gain', '--out', str(TMP / 'ig.json')])
        out = buf.getvalue()
        for x in out.splitlines():
            if any(w in x for w in ('組に分かれた', 'ON − OFF', '満たさない', '使ってよい')) or x.startswith('  on ') or x.startswith('  off '):
                print('⑤:', x)
    finally:
        shutil.rmtree(TMP.parent, ignore_errors=True)
    ap = (Path('/home/jetson/jetformer_reports/fix_20261026/fix3/approvals.md')).read_text(encoding='utf-8')
    sec8 = ap.split('## 8.')[1].split('## 9.')[0]
    print('approvals 8 に「組に分かれ」:', '組に分かれ' in sec8, '/ 「形」:', '形' in sec8, '/ 「走り直し」:', '走り直し' in sec8)


if __name__ == '__main__':
    main()
