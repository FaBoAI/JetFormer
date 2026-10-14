# r3 の下調べ(離脱復活の oracle の試作)の集計
#   python3 summ_oracle.py <oracle_s*.jsonl ...> [--json 出力.json]
# 出す値:
#   1. 乱しの種類・強さ・向きごとの「立て直せる」割合(全部の位置 / 今の記録の決まりで始めてよい位置だけ)
#   2. 今の記録の決まり(種類 0.45/0.40/0.15・強さ等確率・左右等確率。わざと外すは試作に無いので舵のパルスで代わりにしない)で、
#      始めてよい位置に一様に起きる乱しのうち、立て直せない割合の見積もり(舵のパルスとブレーキ/踏み込みだけ)
#   3. 区間ごと(黒シートの出口 s 8.0〜8.25・奥の直線の出口 19.25〜19.5・ほか)の立て直せる割合
#   4. 「立て直せる組だけ」にしたとき、始めてよい位置(フレーム)のうち少なくとも 1 つの組が使える割合(出来事の数が保てるか)
import json, sys, collections
args = [a for a in sys.argv[1:] if not a.startswith('--')]
out = sys.argv[sys.argv.index('--json') + 1] if '--json' in sys.argv else None
if out in args: args.remove(out)
rows = [json.loads(l) for f in args for l in open(f) if l.strip()]
def key(r): return (r['type'], r['cls'], r['sign'] if r['type'] == 'steer_pulse' else r['kind'])
def zone(s):
    if 8.0 <= s < 8.25: return 'black_sheet_exit'
    if 19.25 <= s < 19.5: return 'back_straight_exit'
    return 'other'
res = {'n_rows': len(rows), 'by_combo': {}, 'by_zone_allowed': {}, 'states': 0}
by = collections.defaultdict(lambda: [0, 0, 0, 0])
for r in rows:
    b = by[key(r)]; b[0] += 1; b[1] += r['recoverable']
    if r['allowed_v1']: b[2] += 1; b[3] += r['recoverable']
print('種類 強さ 向き: 全部の位置 立て直せる / 始めてよい位置で 立て直せる')
for k in sorted(by, key=str):
    b = by[k]
    res['by_combo'][' '.join(map(str, k))] = {'all': b[0], 'rec': b[1], 'allowed': b[2], 'allowed_rec': b[3]}
    print(f"  {str(k):34s} {b[1]:4d}/{b[0]:4d} = {b[1]/max(b[0],1):.2f}   {b[3]:4d}/{b[2]:4d} = {b[3]/max(b[2],1):.2f}")
# 今の記録の決まりでの見積もり(始めてよい位置だけ)。舵のパルス 0.45・ブレーキ/踏み込み 0.15 を、その 2 つの中で 0.75 / 0.25 に読み替え
w = {}
for c in ['弱', '中', '強']:
    for sg in [1, -1]: w[('steer_pulse', c, sg)] = 0.75 / 3 / 2
    for kd in ['brake', 'burst']: w[('speed_pulse', c, kd)] = 0.25 / 3 / 2
num = sum(w[k] * by[k][3] / max(by[k][2], 1) for k in w)
print(f"今の記録の決まり(舵のパルス・ブレーキ/踏み込みだけ、始めてよい位置で一様)で立て直せる割合の見積もり: {num:.3f}")
res['v1_mix_recoverable_estimate'] = num
zb = collections.defaultdict(lambda: [0, 0])
for r in rows:
    if not r['allowed_v1']: continue
    z = zone(r['s']); zb[(z, r['type'])][0] += 1; zb[(z, r['type'])][1] += r['recoverable']
print('区間 × 種類(始めてよい位置だけ):')
for k in sorted(zb):
    print(f"  {str(k):40s} {zb[k][1]:4d}/{zb[k][0]:4d} = {zb[k][1]/max(zb[k][0],1):.2f}")
    res['by_zone_allowed'][' '.join(k)] = {'n': zb[k][0], 'rec': zb[k][1]}
# 状態(開始, f)ごと: 始めてよい位置で、立て直せる組がいくつあるか
st = collections.defaultdict(list)
for r in rows: st[(r['start'], r['f'])].append(r)
allowed = [v for v in st.values() if v[0]['allowed_v1']]
anyrec = sum(1 for v in allowed if any(x['recoverable'] for x in v))
steer_any = sum(1 for v in allowed if any(x['recoverable'] and x['type'] == 'steer_pulse' for x in v))
strong_any = sum(1 for v in allowed if any(x['recoverable'] and x['cls'] == '強' and x['type'] == 'steer_pulse' for x in v))
print(f"状態 {len(st)}、うち今の決まりで始めてよい {len(allowed)}。そのうち立て直せる組が 1 つ以上 {anyrec}、舵のパルスで 1 つ以上 {steer_any}、強い舵のパルスで 1 つ以上 {strong_any}")
res.update({'states': len(st), 'allowed_states': len(allowed), 'allowed_states_any_recoverable': anyrec, 'allowed_states_steer_recoverable': steer_any, 'allowed_states_strong_steer_recoverable': strong_any})
# 立て直せる組だけにしたとき、始めてよい位置(s)の広がり: 0.25 m のマスのうち、どれかの組が立て直せるマス
cells = collections.defaultdict(lambda: [0, 0])
for v in st.values():
    s = v[0]['s']; c = int(s / .25)
    cells[c][0] += 1; cells[c][1] += any(x['recoverable'] for x in v if x['type'] == 'steer_pulse')
print(f"0.25 m のマス {len(cells)}、舵のパルスで立て直せる状態があるマス {sum(1 for c in cells.values() if c[1] > 0)}")
res['cells_025'] = len(cells); res['cells_steer_recoverable'] = sum(1 for c in cells.values() if c[1] > 0)
# 速さと立て直せる割合(舵のパルス・中と強)
vb = collections.defaultdict(lambda: [0, 0])
for r in rows:
    if r['type'] != 'steer_pulse' or r['cls'] == '弱': continue
    b = vb[round(r['v'] * 2) / 2]; b[0] += 1; b[1] += r['recoverable']
print('速さ(0.5 m/s のマス)と 中・強の舵のパルスの立て直せる割合:', {k: f"{v[1]}/{v[0]}" for k, v in sorted(vb.items())})
res['steer_mid_strong_by_speed'] = {str(k): v for k, v in sorted(vb.items())}
if out: json.dump(res, open(out, 'w'), ensure_ascii=False, indent=1)
