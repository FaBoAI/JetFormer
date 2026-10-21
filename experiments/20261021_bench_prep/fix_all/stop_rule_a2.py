"""Fix-all: 止める条件の直し A2 の候補を 10/10 の AI 走行 11 本に当てる(読み取りだけ・標準ライブラリ)。
(d) 同じ場所で、止まって人が取り返した衝突(交代 class after_collision・failure)が 3 周続けて
(e) 同じ、2 周続けて
(f) 同じ場所の「止まる衝突」(outcome stop)が 3 周続けて(取り返しを問わない)
"""
import json, sys
SC = "/home/jetson/jetformer_reports/bench_prep_20261021/build/mock_bench/nightly_20261010/scorecard/scorecard.json"
LB = "/home/jetson/jetformer/data/labels/%s.json"
d = json.load(open(SC))
out = {}
for log, p in sorted(d['per_log'].items()):
    lab = json.load(open(LB % log))
    ev = lab['events']
    ta = [q['ta_s'] for q in p['passes'] if q['ai']]

    def lap_of(t):
        k = None
        for i, x in enumerate(ta):
            if t >= x - 0.5:
                k = i
        return k

    def runs(items, n):
        by = {}
        for loc, t in items:
            k = lap_of(t)
            if k is None:
                continue
            by.setdefault(loc, set()).add(k)
        hits = []
        for loc, ks in by.items():
            for k in sorted(ks):
                if all(k + j in ks for j in range(n)):
                    hits.append((loc, k + n - 1))
        first = min((k for _, k in hits), default=None)
        return len(hits), (ta[first] if first is not None and first < len(ta) else None)
    tk = [(x.get('loc'), x['t_s']) for x in ev['takeovers'] if x.get('class') == 'after_collision' and x.get('failure')]
    stops = [(c.get('loc'), c['t_s']) for c in ev['collisions'] if c.get('ai') and c.get('outcome') == 'stop']
    d3 = runs(tk, 3)
    e2 = runs(tk, 2)
    f3 = runs(stops, 3)
    out[log] = dict(desc=lab['description'], passes=len(ta), takeovers_after_collision=len(tk), stop_collisions=len(stops),
                    d_takeover_3laps=d3[0], d_first_ta=d3[1], e_takeover_2laps=e2[0], e_first_ta=e2[1],
                    f_stop_3laps=f3[0], f_first_ta=f3[1])
for k, v in out.items():
    print(k, v)
summ = {r: sum(1 for v in out.values() if v[r] > 0) for r in ('d_takeover_3laps', 'e_takeover_2laps', 'f_stop_3laps')}
print('fired (logs of 11):', summ)
if len(sys.argv) > 1:
    json.dump({'per_log': out, 'fired_logs': summ}, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1)
