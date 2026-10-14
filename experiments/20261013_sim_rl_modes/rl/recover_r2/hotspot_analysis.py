"""乱しの始まりの集まり(黒シートの出口・奥の直線の出口)と、黒シートの出口の舵のパルスがなぜ失敗するか(2026-10-14、recover r2)。

確認 recover r1 の走り(check_recover_r1/runs/rows_*.jsonl、ユーザーの sim-core.js・ブラウザの組で走らせたもの)を読むだけ(走らせ直さない)。
離脱復活の方策の乱しありの組(REPLAY-P100 = 試験の組の作り直し、FRESH-P = 新しい開始 200 回)を使う。
    python -I rl/recover_r2/hotspot_analysis.py <runs のフォルダ> <room_table.json> <出力.json>
"""
import collections
import json
import math
import sys
from pathlib import Path

runs, room_path, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
L = 26.119648724030924
room = json.loads(room_path.read_text())


def room_at(s):
    # 使える幅(すき間 8 cm、左 ρ_L・右 ρ_R)と、すき間 5 cm で動ける幅(width05)。格子点の値
    i = int(round((s % L) / room['ds_m'])) % room['n']
    return room['rhoL'][i], room['rhoR'][i], room['width05'][i]


rows = []
for fn in sorted(runs.glob('rows_*.jsonl')):
    for line in open(fn):
        r = json.loads(line)
        if r['set'] in ('REPLAY-P100', 'FRESH-P'):
            rows.append(r)


def zone(sm):
    if 8.0 <= sm < 8.25:
        return 'black_sheet_exit'
    if 19.25 <= sm < 19.55:
        return 'back_straight_exit'
    return 'other'


evs = []
for r in rows:
    pf, T = r['pf'], r['frames']
    coll_f = T if r['collisions'] else None
    for k, e in enumerate(r['events']):
        ps, pe = e['ps'], e['pe'] if e['pe'] is not None else T
        nxt = r['events'][k + 1]['ps'] if k + 1 < len(r['events']) else T
        sm = e['s'] % L
        v0 = pf['v'][ps] if ps < len(pf['v']) else None
        failed_coll = coll_f is not None and coll_f <= nxt
        c_s = (pf['s'][min(T - 1, len(pf['s']) - 1)] % L) if failed_coll else None
        evs.append({'set': r['set'], 'id': r['id'], 'type': e['type'], 'cls': e['cls'], 'sign': e.get('sign'), 'kind': e.get('kind'), 'k': e.get('k'),
                    'frames': e.get('frames'), 's_mod': sm, 'zone': zone(sm), 'v0': v0, 'collision_after': failed_coll,
                    'coll_s_mod': c_s, 'coll_dt_from_ps': ((T - ps) / 30) if failed_coll else None,
                    'collider': (r['collision'] or {}).get('kind') if failed_coll else None, 'ey0': pf['ey'][ps] if ps < len(pf['ey']) else None})

res = {'source': str(runs), 'episodes': len(rows), 'events': len(evs)}
# 始まりの位置の分布(s を 0.25 m のマスに)
hist = collections.Counter(math.floor(e['s_mod'] / .25) * .25 for e in evs)
top = hist.most_common(6)
res['start_hist_top'] = [{'s_from': round(b, 2), 'n': n, 'share': round(n / len(evs), 4)} for b, n in top]
res['zone_share'] = {z: round(sum(e['zone'] == z for e in evs) / len(evs), 4) for z in ('black_sheet_exit', 'back_straight_exit', 'other')}
# 黒シートの出口の舵のパルス
bs = [e for e in evs if e['zone'] == 'black_sheet_exit' and e['type'] == 'steer_pulse']
res['black_sheet_exit_steer_pulse'] = {
    'n': len(bs), 'collided': sum(e['collision_after'] for e in bs),
    'v0_max': max(e['v0'] for e in bs), 'v0_median': sorted(e['v0'] for e in bs)[len(bs) // 2],
    'by_sign': {str(sg): {'n': sum(e['sign'] == sg for e in bs), 'collided': sum(e['sign'] == sg and e['collision_after'] for e in bs)} for sg in (1, -1)},
    'by_class': {c: {'n': sum(e['cls'] == c for e in bs), 'collided': sum(e['cls'] == c and e['collision_after'] for e in bs)} for c in ('弱', '中', '強')},
    'collision_s_mod_hist': dict(collections.Counter(round(math.floor(e['coll_s_mod'] / .5) * .5, 1) for e in bs if e['collision_after'])),
    'collision_dt_from_ps_median_s': sorted(e['coll_dt_from_ps'] for e in bs if e['collision_after'])[sum(e['collision_after'] for e in bs) // 2] if any(e['collision_after'] for e in bs) else None,
    'collider_kinds': dict(collections.Counter(e['collider'] for e in bs if e['collision_after'])),
}
# 使える幅(room_table)の s 8〜11 m
res['room_8_to_11'] = [{'s': round(s, 2), 'rhoL': round(room_at(s)[0], 3), 'rhoR': round(room_at(s)[1], 3), 'width05': round(room_at(s)[2], 3)} for s in [8 + .25 * i for i in range(13)]]
out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
print(json.dumps(res, ensure_ascii=False, indent=1)[:3500])

# 向き × 強さ(黒シートの出口の舵のパルス)と、全部の出来事の失敗のうち黒シートの出口の右のパルスが占める割合
res['black_sheet_exit_steer_pulse']['by_sign_class'] = {f'{"左" if sg > 0 else "右"}・{c}': {'n': sum(e['sign'] == sg and e['cls'] == c for e in bs),
    'collided': sum(e['sign'] == sg and e['cls'] == c and e['collision_after'] for e in bs)} for sg in (1, -1) for c in ('弱', '中', '強')}
coll_all = [e for e in evs if e['collision_after']]
bsr = [e for e in coll_all if e['zone'] == 'black_sheet_exit' and e['type'] == 'steer_pulse' and e['sign'] == -1]
res['collisions_after_event'] = {'n': len(coll_all), 'black_sheet_exit_right_pulse': len(bsr),
    'by_zone_type': dict(collections.Counter(f"{e['zone']}:{e['type']}" for e in coll_all))}
res['events_by_zone_type'] = dict(collections.Counter(f"{e['zone']}:{e['type']}" for e in evs))
# 試験の組だけ(REPLAY-P100)の同じ数え
rp = [e for e in evs if e['set'] == 'REPLAY-P100']
res['replay_p100'] = {'events': len(rp), 'collided': sum(e['collision_after'] for e in rp),
    'black_sheet_exit_right_pulse': {'n': sum(e['zone'] == 'black_sheet_exit' and e['type'] == 'steer_pulse' and e['sign'] == -1 for e in rp),
                                     'collided': sum(e['zone'] == 'black_sheet_exit' and e['type'] == 'steer_pulse' and e['sign'] == -1 and e['collision_after'] for e in rp)}}
out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
print(json.dumps({k: res[k] for k in ('collisions_after_event', 'events_by_zone_type', 'replay_p100')}, ensure_ascii=False, indent=1))
print(json.dumps(res['black_sheet_exit_steer_pulse']['by_sign_class'], ensure_ascii=False))
