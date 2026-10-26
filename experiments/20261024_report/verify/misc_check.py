"""verify/misc_check.py — こまかい主張の確かめ(生の記録だけ)。
1) No.9 のゲインが全部の AI のフレームで 0.90 か(説明欄は 1.1)。
2) No.3(2本の平均)の発進: 補助(assist)のあと、モデルの前進の指令はどうなったか。
3) No.9 の発進: AI 側にした時刻・そのときの eRPM・最初の R1 の起点までの時間。
4) No.17 の 3 周の窓の中身。
"""
import json, os, sys
import numpy as np
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H)
import rawload
R = json.load(open(os.path.join(H, 'recount.json')))
d9 = R['20261024-162033']
print('1) No.9 gains_set (steer, throttle):', d9['gains_set'], '説明欄 throttle_gain =', d9['desc_tg'], 'params', d9['params'].get('throttle_gain'))
r = rawload.load('20261024-135608')
sel = r['a_sel'] & ~r['a_dry']
t = r['a_t']; thr = 1496 - r['a_out'][:, 1]; thr_m = 1496 - r['a_model'][:, 1]
k = np.where(sel)[0]
print('2) No.3 AI frames', len(k), 't %.2f〜%.2f' % (t[k[0]], t[k[-1]]))
for i in k[::6]:
    ve = np.interp(t[i], r['v_t'], r['v_erpm'])
    print('   t=%.2f assist=%d out_fwd=%.0f model_fwd=%.0f erpm=%.0f' % (t[i], r['a_assist'][i], thr[i], thr_m[i], ve))
r9 = rawload.load('20261024-162033')
vs = r9['v_sel']; vt = r9['v_t']
on = vt[1:][np.diff(vs.astype(int)) == 1]
print('3) No.9 AI on at', np.round(on, 2), 'erpm there', [round(float(np.interp(x, vt, r9['v_erpm']))) for x in on])
a = [x for x in d9['anchors'] if not x['false']]
print('   first anchors', [round(x['t'], 2) for x in a[:2]], '→ start→first anchor %.2f s' % (a[0]['t'] - on[0]))
print('   laps', [(l['t0'], l['lap']) for l in d9['laps'] if l['ok']][:3])
d17 = R['20261024-184549']
L = [l for l in d17['laps'] if l['ok']]
for i in range(len(L) - 2):
    print('4) No.17 window', round(sum(l['lap'] for l in L[i:i + 3]), 3), [(l['lap'], l['gmax']) for l in L[i:i + 3]])
