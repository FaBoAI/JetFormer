"""R2-rec-2 の確かめ(係 rec2・作り物の記録だけ。本物の data は読まない・書かない):
同じ log.json(天井 4.1 + 自動合わせ ON)で、A = 最後まで ON、B = 25 秒で走りながら OFF(画面のスイッチと同じ結果:
AI_CMD の ceil_on・sa_on が False に、gains は設定のまま)を作り、道具の表で A と B が区別されるかを見る。"""
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np

JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF / 'tests'))
sys.path.insert(0, str(JF / 'tools'))
import section_scorecard as ss  # noqa: E402
import quick_eval_by_params as qp  # noqa: E402
import standing_3lap as s3  # noqa: E402
from test_lap_anchor import write_lap_log  # noqa: E402

spec = importlib.util.spec_from_file_location('runs_1103', JF / 'experiments/20261026_fix/plan/runs_1103.py')
r1103 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r1103)

TMP = JF / 'experiments/20261026_fix/fix2/tmp_rec2'
if TMP.exists():
    shutil.rmtree(TMP)
raw = TMP / 'raw'
raw.mkdir(parents=True)
P = {'throttle_gain': 0.95, 'speed_ceiling': 1, 'ceil_all_mps': 4.5, 'ceil_left_mps': 4.1, 'speed_auto': 1,
     'auto_target_mps': 4.27, 'launch_boost': 0}


def make(log, off_at):
    d, anchors, _ = write_lap_log(raw, log, dur=60.0, false_laps=(), params=P,
                                  description='ai_no=8 bench=real_bench_20261023 day=2000-11-03 block=8 cand=EXP tune=2-3')
    for s in sorted(d.glob('*/AI_CMD.npz')):
        z = dict(np.load(s))
        t = (z['timestamps_us'] - z['timestamps_us'][0]) / 1e6 + (int(s.parent.name[-4:]) - 1) * 20.0
        on = t < off_at
        n = len(t)
        g = np.where(on, 0.95 * 1.04, 0.95)
        z['gains'] = np.stack([np.ones(n), g], axis=1).astype(np.float32)
        z.update(ceil_on=on, ceil_mode=np.zeros(n, np.int8), ceil_applied=np.zeros(n, bool), ceil_left=np.zeros(n, bool),
                 sa_on=on, sa_m=np.where(on, 1.04, np.nan).astype(np.float32), sa_gain=g.astype(np.float32),
                 sa_src=np.ones(n, np.int8), sa_event=np.zeros(n, np.int8), launch_on=np.zeros(n, bool),
                 launch_end=np.zeros(n, np.int8))
        np.savez(s, **z)
    return d, anchors


out = {}
for name, log, off_at in (('A 最後まで ON', '20001103-130000', 1e9), ('B 25 秒で OFF', '20001103-140000', 25.0)):
    d, anchors = make(log, off_at)
    res = s3.analyze_log(log, raw, TMP / 'labels', {'x': 'B'})
    key, txt = qp.key_of(res)
    sp = ss.score_log(log, TMP / 'labels', raw)['speed_ctrl']
    ev = (d / 'events.jsonl')
    out[name] = {'key': key, 'text': txt, 'flags_of': qp.flags_of(res), 'sc_flags': sp['flags'],
                 'ai_frames': sp.get('ai_frames'), 'ceil_on_frames': sp['ceil']['on_frames'],
                 'auto_on_frames': sp['auto']['on_frames'], 'ctrl_cells': ss.ctrl_cells({'speed_ctrl': sp, 'vbat': {}})[:2] +
                 (ss.ctrl_cells({'speed_ctrl': sp, 'vbat': {}})[3],),
                 'passes': res['laps']['passes'], 'clean': res['laps']['clean'],
                 'anchors_after_off': sum(1 for a in anchors if a >= off_at),
                 'runs_1103_diffs': r1103.diffs({'speed_ceiling': 1, 'speed_auto': 1, 'ceil_left_mps': 4.1,
                                                 'throttle_gain': 0.95}, json.loads((d / 'log.json').read_text())['drive']['params']),
                 'events_jsonl': ev.read_text() if ev.exists() else '(無い)'}
    out[name]['_res'] = res

for k, v in out.items():
    print('==', k)
    for kk, vv in v.items():
        if kk != '_res':
            print('  %-18s %s' % (kk, vv))
rows = qp.group_rows([out[k]['_res'] for k in out])
print('== group_rows: 組の数 %d' % len(rows))
for r in rows:
    print('  ', r['text'], '本数', r['runs'], r['logs'], '通過', r['passes'], 'きれいな周', r['clean_laps'])
print('== run_rows の印:', [(r['log'], r['flags']) for r in qp.run_rows([out[k]['_res'] for k in out])])
