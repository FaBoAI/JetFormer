"""no_wall r4 の下調べのまとめ(組ごとの衝突・最小すき間・止まり・有効・抜け出し・作動・1 フレームの時間と、崩れた回の一覧)。
    python3 summ_dev_nw4.py <rows.jsonl> [...]"""
import json
import sys
from collections import defaultdict

for path in sys.argv[1:]:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    by = defaultdict(list)
    for r in rows:
        by[r['set']].append(r)
    print('==', path, len(rows))
    for k, v in by.items():
        v = [r for r in v if not r.get('error')]
        if not v:
            continue
        fr = sum(r['labels']['frames'] for r in v)
        lap1 = sorted(r['laps'][0]['elapsed'] for r in v if r.get('laps'))
        ms = sorted(r['ctrl_ms_per_frame'] for r in v if r.get('ctrl_ms_per_frame'))
        print(f"{k:10s} n {len(v):4d} coll {sum(r['collisions'] > 0 for r in v):3d} min {100 * min(r['min_clearance'] for r in v):6.2f} <8 {sum(r['min_clearance'] < .08 for r in v):3d} "
              f"stall {sum(bool(r['stall']) for r in v):3d} v3 {sum(bool(r['valid_episode_r3']) for r in v):4d} esc {sum((r.get('escapes') or 0) > 0 for r in v):3d} "
              f"back {sum(r.get('back_frames', 0) > 0 for r in v):3d} sh {sum(r['shield_frames'] for r in v) / fr:.4f} shmax {max(r['shield_frames'] / max(1, r['labels']['frames']) for r in v):.3f} "
              f"lap1 {lap1[len(lap1) // 2] if lap1 else -1:.2f} ms {ms[len(ms) // 2] if ms else -1:.1f} p99f {max(r.get('frame_ms_p99') or 0 for r in v):.0f} maxf {max(r.get('frame_ms_max') or 0 for r in v):.0f}")
        bad = [r for r in v if r['collisions'] or r['stall'] or not r['valid_episode_r3'] or r['min_clearance'] < .08]
        for r in bad[:12]:
            st = r['start']
            fb = dict(sorted(r['labels']['fb'].items(), key=lambda x: -x[1])[:4])
            print(f"    {r['id']:12s} g{st.get('gate')} o{st.get('left_offset_m', 0):+.3f} y{st.get('left_yaw_deg', 0):+.2f} c0 {100 * (st.get('initial_clearance_m') or 0):.2f} "
                  f"coll {r['collisions']} min {100 * r['min_clearance']:.2f} stall {r['stall']} vl {r['valid_laps']} esc {r.get('escapes')} dr {(r.get('dr') or {}).get('kind')} {fb}")
