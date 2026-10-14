# r3 の下調べの集計(1 行 1 回の JSONL → 組ごとの衝突・最小すき間・シールドの作動・止まり)
import json, sys, collections
rows = [json.loads(l) for f in sys.argv[1:] for l in open(f) if l.strip()]
by = collections.defaultdict(list)
for r in rows: by[(r.get('set'), r.get('corner') or '')].append(r)
for k, rs in sorted(by.items()):
    n = len(rs); col = sum(1 for r in rs if (r.get('collisions') or 0) > 0); err = sum(1 for r in rs if 'error' in r)
    cl = [r['min_clearance'] for r in rs if (r.get('collisions') or 0) == 0 and r.get('min_clearance') is not None]
    lt5 = sum(1 for c in cl if c < .05); lt8 = sum(1 for c in cl if c < .08)
    fr = sum(r.get('frames', 0) for r in rs); sh = sum(r.get('shield_frames', 0) for r in rs)
    gov = sum(r.get('labels', {}).get('governor', 0) for r in rs)
    lap0 = sum(1 for r in rs if r.get('valid_episode') and r.get('valid_laps', 0) == 0)
    valid1 = sum(1 for r in rs if r.get('valid_episode') and r.get('valid_laps', 0) >= 1)
    wall = sum(r.get('wall_s', 0) for r in rs)
    print(f"{k[0]:8s} {k[1]:30s} n={n:4d} col={col:3d} err={err} min={min(cl) if cl else None!s:.6s} <5cm={lt5} <8cm={lt8} valid+lap={valid1} valid_lap0={lap0} sh={100*sh/max(fr,1):.2f}% gov={100*gov/max(fr,1):.2f}% wall/run={wall/max(n,1):.2f}s")
    cols = [r for r in rs if (r.get('collisions') or 0) > 0]
    for r in cols[:12]:
        c = r.get('collision') or {}
        print('   col', r['id'], 'g', r['start'].get('gate'), 't', c.get('time'), c.get('collider'), 'sh', r.get('shield_frames'), 'fb', r.get('labels', {}).get('fb'))
