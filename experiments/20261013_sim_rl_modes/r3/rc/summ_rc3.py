# r3 の下調べ(離脱復活 + シールド)の集計。判定の数え方の近似(立て直しの線の比べは省き、衝突だけで出来事の成否を数える)
#   python3 summ_rc3.py <rows_dir> [<rows_dir> ...]
import json, sys, glob, collections
for d in sys.argv[1:]:
    rows = [json.loads(l) for f in sorted(glob.glob(d + '/rows_*.jsonl')) for l in open(f) if l.strip()]
    by = collections.defaultdict(list)
    for r in rows: by[r['set']].append(r)
    for k, rs in sorted(by.items()):
        rs = [r for r in rs if not r.get('invalid_spawn')]
        n = len(rs); val = [r for r in rs if r['valid_episode']]
        # 止まり: フレーム 30 以降で 0.2 m/s 未満が 30 フレーム以上続く(ブレーキの乱しの区間 [ps, pe + 30) を除く)
        stall = 0
        for r in rs:
            v = r['pf']['v']; ev = r.get('events') or []
            brake = set()
            for e in ev:
                if e.get('type') == 'speed_pulse' and e.get('kind') == 'brake' and e.get('pe') is not None:
                    brake.update(range(e['ps'], e['pe'] + 30))
            run = 0; st = False
            for i, x in enumerate(v):
                if i >= 30 and abs(x) < .2 and i not in brake: run += 1
                else: run = 0
                if run >= 30: st = True; break
            r['_stall'] = st; stall += st
        evs = 0; fail = 0; bytype = collections.Counter(); failtype = collections.Counter()
        for r in rs:
            ev = [e for e in (r.get('events') or []) if e['ps'] < r['frames']]
            for i, e in enumerate(ev):
                evs += 1; bytype[e['type']] += 1
                nxt = ev[i + 1]['ps'] if i + 1 < len(ev) else 10**9
                pe = e['pe'] if e.get('pe') is not None else r['frames']
                if r['collisions'] and r['frames'] < min(nxt, pe + 90) + 1 and r['frames'] >= e['ps']:
                    fail += 1; failtype[e['type']] += 1
        cl = [r['min_clearance'] for r in val]
        shf = sum(sum(1 for c, l in zip(r['pf']['ch'], r['pf']['l']) if c and l == 1) for r in rs); frs = sum(len(r['pf']['ch']) for r in rs)
        print(f"{d.split('/')[-1]:8s} {k:6s} n={n} valid={len(val)} valid_no_stall={sum(1 for r in val if not r['_stall'])} stall={stall} col={sum(1 for r in rs if r['collisions'])} "
              f"min_valid={min(cl) if cl else None!s:.6s} <3cm={sum(c < .03 for c in cl)} events={evs} ev/run={evs/max(n,1):.2f} ev_fail={fail} succ≈{1-fail/max(evs,1):.3f} "
              f"fail_by_type={dict(failtype)} n_by_type={dict(bytype)} shield={100*shf/max(frs,1):.2f}%")
