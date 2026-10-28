#!/usr/bin/env python3
"""反証 8-1 回目 ui の指摘 runs-opt-mark-on-the-on-runs を打ち消す側で確かめる(票 A)。
本物の plan/runs_1103.py を import して RAW だけ作り物の記録(この台本の場所の tmp_a1/)に替えて流す。読むだけ(data/raw は見ない)。
各本の drive.params は 11/3 のブロックの expect(null でない項目)どおり = 試験の選択肢のほかは「計画どおり」。"""
import contextlib
import importlib.util
import io
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = Path('/home/jetson/jetformer/experiments/20261026_fix/plan/runs_1103.py')
BENCH = json.loads(Path('/home/jetson/jetformer/data/bench_candidates.json').read_text(encoding='utf-8'))
BLOCKS = {b['block']: b for d in BENCH['days'] if d['date'] == '2026-11-03' for b in d['blocks']}
RID = {b['run_id']: n for n, b in BLOCKS.items()}
RUN_B = next(c['run'] for c in BENCH['candidates'] if c['id'] == 'B')
BASE = {'steer_gain': 1.0, 'throttle_gain': 0.9, 'lookahead': 3, 'tof_guard': 1, 'zone_caps': 0, 'start_kick': 1,
        'speed_auto': 0, 'speed_ceiling': 0, 'ceil_left_mps': 4.1, 'ceil_all_mps': 4.5, 'launch_boost': 0, 'launch_us': 250,
        'launch_mps': 2.5, 'launch_max_s': 1.2, 'auto_target_mps': 4.27, 'rev_s': 0.4, 'rev_us': 300, 'steer_trim_us': 0}
ON = {'rec_back_guard': 1, 'rec_hold_keep': 1, 'launch_no_kick': 0}
OFF = {'rec_back_guard': 0, 'rec_hold_keep': 0, 'launch_no_kick': 0}


def rec(raw, name, ai_no, run_id, opts):
    d = raw / name
    (d / ('%s-0001' % name)).mkdir(parents=True)
    p = dict(BASE, **opts)
    if run_id:
        blk = RID[run_id]
        p.update({k: v for k, v in (BLOCKS[blk]['expect'] or {}).items() if v is not None})
        desc = 'ai_no=%d bench=x day=2026-11-03 block=%d cand=B tune=%s battery=#A V_before=8.30 memo=' % (ai_no, blk, run_id)
    else:
        desc = 'ai_no=%d 朝の発進の前進の確かめ memo=' % ai_no          # block の無い本(本「—」)
    (d / 'log.json').write_text(json.dumps({'description': desc, 'drive': {'run': RUN_B, 'dry_run': False, 'params': p}},
                                           ensure_ascii=False), encoding='utf-8')


def run(raw):
    spec = importlib.util.spec_from_file_location('runs_1103_a1', RUNS)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.RAW = raw
    argv, buf = sys.argv, io.StringIO()
    sys.argv = ['runs_1103.py', '--day', '20261103', '--json']
    try:
        with contextlib.redirect_stdout(buf):
            m.main()
    finally:
        sys.argv = argv
    out = buf.getvalue().strip().splitlines()
    return out[:-1], json.loads(out[-1])


SCEN = {
    # E: 回 1 は ON。回 2 の前に起き直して 2-1 を OFF のまま走り、2-2 の前に入れ直した
    'E_round2_first_off': [('20261103-090000', 1, '1-1', ON), ('20261103-090500', 2, '1-2', ON),
                           ('20261103-100000', 3, '2-1', OFF), ('20261103-100500', 4, '2-2', ON), ('20261103-101000', 5, '2-3', ON)],
    # K: 朝の確かめ(本「—」)→ 選択肢を入れる前に 1-1 → 1-2 から ON
    'K_morning_forgot': [('20261103-081500', 1, None, OFF), ('20261103-083000', 2, '1-1', OFF),
                         ('20261103-083500', 3, '1-2', ON), ('20261103-084000', 4, '1-3', ON)],
    # C(比べ): 1-3 の前に起き直して OFF のまま(回の途中で OFF)
    'C_mid_round_off': [('20261103-090000', 1, '1-1', ON), ('20261103-090500', 2, '1-2', ON), ('20261103-091000', 3, '1-3', OFF)],
    # F(比べ): 回 2 は全部 OFF のまま(回の境で起き直し、入れ直さない)
    'F_round2_all_off': [('20261103-090000', 1, '1-1', ON), ('20261103-090500', 2, '1-2', ON),
                         ('20261103-100000', 3, '2-1', OFF), ('20261103-100500', 4, '2-2', OFF)],
    # G(比べ): 朝から全部 ON
    'G_all_on': [('20261103-090000', 1, '1-1', ON), ('20261103-090500', 2, '1-2', ON), ('20261103-100000', 3, '2-1', ON)],
}

if __name__ == '__main__':
    tmp = HERE / 'tmp_a1'
    out = []
    for name, rows in SCEN.items():
        raw = tmp / name / 'raw'
        if raw.exists():
            shutil.rmtree(raw)
        for log, ai, rid, opts in rows:
            rec(raw, log, ai, rid, opts)
        table, js = run(raw)
        out.append('===== %s' % name)
        out.append('走った本: ' + ' / '.join('%s(No.%d) %s' % (rid or '—', ai, 'ON' if opts['rec_back_guard'] else 'OFF')
                                        for log, ai, rid, opts in rows))
        for r in js:
            opt = [x for x in r['diffs'] if '試験の選択肢' in x]
            out.append('  %-4s No.%s 見張り=%s 印=%s その他の違い=%s' % (
                r['run_id'] or '—', r['ai_no'], r['options']['rec_back_guard'], opt or 'なし',
                [x for x in r['diffs'] if '試験の選択肢' not in x] or 'なし'))
        out.extend('  | ' + x for x in table if x.startswith('試験の選択肢') or '本・計画と違う' in x)
    shutil.rmtree(tmp)
    txt = '\n'.join(out)
    (HERE / 'out' / 'a1_runs_first_off_after.txt').write_text(txt + '\n', encoding='utf-8')
    print(txt)
