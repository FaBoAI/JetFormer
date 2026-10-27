"""tools:F7 の確かめ 2: 回 3(止まって 1〜2 周)で比べる「動く → A1 中央値」も記録ごとの状態で混ざるか(読むだけ)"""
import copy, sys
from pathlib import Path
ROOT = Path('/home/jetson/jetformer')
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'tools'))
import standing_3lap as s3
import quick_eval_by_params as q
r = s3.run(['20261024-183123'], None, None, str(ROOT / 'data' / 'bench_candidates.json'))[0]
r = copy.deepcopy(r)
r['settings'].update(launch_boost=1, launch_us=250.0, launch_mps=2.5, launch_max_s=1.2)
st = r['standing']
for s in st:
    print('発進 t_ai', s['t_ai'], 'to_r1_s', s.get('to_r1_s'), 'move_after_s', s.get('move_after_s'), 'valid', s.get('valid'), s.get('why'))
r1 = [i for i, s in enumerate(st) if s.get('to_r1_s') is not None]
for i, s in enumerate(st):
    s['launch'].update(recorded=True, on_frames=0, events=[[s['t_ai'], 'skip', 'ToF が無い・古い・ガード 0']])
st[r1[0]]['launch'].update(on_frames=20, on_s=0.67, end_reason='速さに届いた', events=[[st[r1[0]]['t_ai'], 'start', '']])
g = q.group_rows([r])[0]
print('組', g['text'], '動く→A1 の数', g['launch_n'], '中央値', g['launch_to_r1_median_s'], '印', q.flags_of(r))
print('出た発進だけの 動く→A1', st[r1[0]]['to_r1_s'] - st[r1[0]]['move_after_s'])
