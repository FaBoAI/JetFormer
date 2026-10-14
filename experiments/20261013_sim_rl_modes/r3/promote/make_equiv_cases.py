"""昇格の段(2026-10-15): equiv_runtime.mjs に渡す比べの組を作る。
各モードの試験の組から、組ごとに先頭の数本 + シールド・抜け出し・乱しが働いた回を選び、
  - 候補の runtime(試験に使ったもの)と rt_merged(ページと同じく mode.json の runtime を modeRuntime に渡す)を閉ループで比べる
  - no_wall・line は modeRuntime を渡さない道(Node の試験・シーンの組の作り方)でも一部を比べる
  - fastest・speed・obstacle は今の共有の rl/runtime と比べる(変わっていないこと)
  - 保存した F-ID(snapshot の列)を rt_merged に与えて比べる
    python3 -I r3/promote/make_equiv_cases.py > cases.json
"""
import json
import sys
from pathlib import Path

EXP = Path('/home/jetson/jetformer/experiments/20261013_sim_rl_modes')
R3 = Path('/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3')
REP = Path('/home/jetson/jetformer_reports/sim_rl_modes_20261013')
CAND = EXP / 'r3/candidates'
MERGED = str(EXP / 'r3/promote/rt_merged')
OLD = str(EXP / 'rl/runtime')
REG = Path('/home/jetson/jetformer/data/sim_rl/modes')


def rj(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def mrt(path):
    return rj(path)['runtime']


cases, fids = [], []

# ---- no_wall r4
nw_rt = mrt(CAND / 'no_wall_r4/mode.json')
nw_pol = str(CAND / 'no_wall_r4/policy.json')
rows = [json.loads(l) for l in (R3 / 'no_wall_r4/test/rows_test.jsonl').read_text().splitlines() if l.strip()]
pick, seen = [], {}
for r in rows:
    s = r['set']
    seen[s] = seen.get(s, 0) + 1
    esc = (r.get('labels') or {}).get('fb', {})
    busy = (r.get('shield_frames') or 0) >= 8 or any('escape' in k for k in esc)
    if seen[s] <= 3 or (busy and sum(1 for x in pick if x['set'] == s and x.get('_busy')) < 3):
        pick.append({**r, '_busy': busy})
for k, r in enumerate(pick):
    dr = r.get('dr')
    c = {'id': 'nw-%s-%s' % (r['set'], r['id']), 'set': r['set'], 'mode': 'no_wall', 'a_rt': str(CAND / 'no_wall_r4/runtime'), 'b_rt': MERGED,
         'policy': nw_pol, 'start': r['start'], 'params': {}, 'dr': dr, 'b_mode_runtime': nw_rt if k % 4 != 3 else None}
    cases.append(c)
for f in ('fid_std', 'fid_slow', 'fid_corner', 'fid_escape', 'fid_escape_rev'):
    fids.append({'mode': 'no_wall', 'file': str(CAND / ('no_wall_r4/%s.json' % f)), 'rt': MERGED, 'policy': nw_pol, 'mode_runtime': nw_rt})

# ---- recover r4
rc_rt = mrt(CAND / 'recover_r4/mode.json')
rc_pol = str(CAND / 'recover_r4/policy.json')
ev = rj(R3 / 'rc4/eval_test.json')['tests']
want = {'RC-P100-R4': 16, 'RC-U60-R4': 8, 'RC-REC60': 4, 'RC-STD': 0}
for t, n in want.items():
    rows = ev[t]
    sel = rows[:n] + [r for r in rows if r['id'] in ('P27',)]
    for r in sel:
        if r.get('kick') or r['test'] == 'RC-STD':
            continue
        cases.append({'id': 'rc-%s-%s' % (t, r['id']), 'set': t, 'mode': 'recover', 'a_rt': str(CAND / 'recover_r4/runtime'), 'b_rt': MERGED,
                      'policy': rc_pol, 'start': r['start'], 'params': {k: v for k, v in (r.get('params') or {}).items() if k in ('perturb_class',)},
                      'dr': None, 'overrides': {'perturbation': False} if t in ('RC-U60-R4', 'RC-REC60') else {}, 'b_mode_runtime': rc_rt})
for f in ('fid_FP_P11', 'fid_FP_P24', 'fid_RP_P10', 'fid_RP_P27'):
    fids.append({'mode': 'recover', 'file': str(R3 / ('check_recover_r4/fid/%s.json' % f)), 'rt': MERGED, 'policy': rc_pol, 'mode_runtime': rc_rt})

# ---- line r4b
ln_rt = mrt(CAND / 'line_r4/mode.json')
ln_pol = str(CAND / 'line_r4/policy.json')
ev = rj(R3 / 'line_r4b/eval_test_r4b.json')['tests']
for t, rows in ev.items():
    n = 6 if t == 'LN-FAM6' else 5
    busy = [r for r in rows if (r.get('shield_frames') or 0) >= 2][:3]
    for k, r in enumerate(rows[:n] + [b for b in busy if b not in rows[:n]]):
        params = {'line': r['line'], 'c_v': r['c_v']}
        cases.append({'id': 'ln-%s-%s' % (t, r['id']), 'set': t, 'mode': 'line', 'a_rt': str(CAND / 'line_r4/runtime'), 'b_rt': MERGED,
                      'policy': ln_pol, 'start': r['start'], 'params': params, 'dr': None, 'a_mode_runtime': ln_rt, 'b_mode_runtime': ln_rt})
for f in ('fid_std_left08', 'fid_ln_g30r120_r4b_g8_shield'):
    fids.append({'mode': 'line', 'file': str(CAND / ('line_r4/%s.json' % f)), 'rt': MERGED, 'policy': ln_pol, 'mode_runtime': ln_rt})

# ---- fastest・speed・obstacle(共有の今の rl/runtime と比べる)
starts = [{'gate': 0, 'left_offset_m': 0, 'left_yaw_deg': 0, 'seed': 1}, {'gate': 5, 'left_offset_m': .1, 'left_yaw_deg': -8, 'seed': 77},
          {'gate': 13, 'left_offset_m': -.15, 'left_yaw_deg': 12, 'seed': 4242}, {'gate': 24, 'left_offset_m': .05, 'left_yaw_deg': 3, 'seed': 99},
          {'gate': 30, 'left_offset_m': -.05, 'left_yaw_deg': -14, 'seed': 31337}]
for key, pol, params_list in (('fastest', REG / 'fastest/policy.json', [{}]),
                              ('speed', REG / 'speed/policy.json', [{'c_v': 1.0}, {'c_v': .5}, {'c_v': .75}]),
                              ('obstacle', EXP / 'candidates/obstacle/policy.json', [{}])):
    rt = mrt(REG / key / 'mode.json')
    for i, st in enumerate(starts[:5 if key != 'obstacle' else 2]):
        cases.append({'id': '%s-%d' % (key, i), 'set': 'regression', 'mode': key, 'a_rt': OLD, 'b_rt': MERGED, 'policy': str(pol), 'start': st,
                      'params': params_list[i % len(params_list)], 'dr': None, 'b_mode_runtime': rt})
for key, f in (('fastest', REP / 'eval/fastest/fid_std.json'), ('speed', REP / 'eval/speed/r2/fid_std_speed_r2_cv1.json'),
               ('speed', REP / 'eval/speed/r2/fid_std_speed_r2_cv0.5.json')):
    fids.append({'mode': key, 'file': str(f), 'rt': MERGED, 'policy': str(REG / key / 'policy.json'), 'mode_runtime': mrt(REG / key / 'mode.json')})

json.dump({'cases': cases, 'fids': fids}, sys.stdout, ensure_ascii=False, indent=1)
print('cases', len(cases), {m: sum(1 for c in cases if c['mode'] == m) for m in ('no_wall', 'recover', 'line', 'fastest', 'speed', 'obstacle')}, 'fids', len(fids), file=sys.stderr)
