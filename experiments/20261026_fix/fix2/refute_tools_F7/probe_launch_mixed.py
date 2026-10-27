"""tools:F7 の確かめ(読むだけ): launch_state が記録ごとなので、1 本の中で「出ず」の発進と「出た」発進が混ざると
「出ず」の発進の止まってから 3 周が「発進 250」の組に入り、印も出ないか。"""
import copy, json, sys
from pathlib import Path
ROOT = Path('/home/jetson/jetformer')
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'tools'))
import standing_3lap as s3
import quick_eval_by_params as q

day = sys.argv[1] if len(sys.argv) > 1 else '20261024'
logs, skipped = s3.day_logs(day, None, None)
res = s3.run(logs, None, None, str(ROOT / 'data' / 'bench_candidates.json'))
multi = []
for r in res:
    if 'error' in r:
        continue
    st = r.get('standing') or []
    nv = sum(1 for s in st if s.get('valid'))
    print(r['log'], 'standing', len(st), 'valid', nv, 'launch_boost', (r.get('settings') or {}).get('launch_boost'))
    if len(st) >= 2 and nv >= 1:
        multi.append(r)
print('記録', len(res), '・発進 2 回以上で valid 1 以上の記録', len(multi))
if not multi:
    sys.exit(0)
r0 = copy.deepcopy(multi[0])
r0['settings']['launch_boost'] = 1
r0['settings'].update(launch_us=250.0, launch_mps=2.5, launch_max_s=1.2)
st = r0['standing']
vi = [i for i, s in enumerate(st) if s.get('valid')][0]
# valid な発進 = 「前が近い」で出ず。ほかの発進(どれか 1 つ)= 出た
for i, s in enumerate(st):
    lr = s.setdefault('launch', {})
    lr.update(recorded=True, on_frames=0, on_s=None, end_reason=None, events=[[s['t_ai'], 'skip', '前が近い(TTC の中央値 < 3.3 秒)']])
oi = [i for i in range(len(st)) if i != vi][0]
st[oi]['launch'].update(on_frames=20, on_s=0.67, end_reason='速さに届いた', events=[[st[oi]['t_ai'], 'start', '']])
print('valid の発進', vi, 'from_move_s', st[vi]['from_move_s'], 'launch', st[vi]['launch']['on_frames'], '/ 出た発進', oi,
      'valid', st[oi].get('valid'))
print('launch_state(記録) =', q.launch_state(r0))
k, txt = q.key_of(r0)
print('key_of の文 =', txt)
print('flags_of =', q.flags_of(r0))
for g in q.group_rows([r0]):
    print('組', g['text'], 'standing_n', g['standing_n'], 'standing_list_s', g['standing_list_s'])
# collect.py(diag_launch)の発進ごとの決まり(258 行)
per = []
for s in st:
    lr = s['launch']
    started = bool(lr.get('on_frames')) or any(e[1] == 'start' for e in lr.get('events') or [])
    per.append('on' if started else 'on_not_started')
print('collect.py の決まりで発進ごと =', per)
print('standing_table の発進の前進の列 =', [s3.launch_text(s['launch']) for s in st])
