"""verify/cond_check.py — 電池と明るさの「印の向き」を、解析 B と別のやり方で数え直す。生の記録だけを使う。

電池: 前進の指令 cmd = 1496 − out[1](µs)を車の時間に合わせ、1 次遅れ(τ = 0.4 秒)でならす。
      直線(|舵| ≤ 60 µs)・AI 側・速さ 1 m/s 超・衝突の前後 1 秒を除く・ならした指令 120〜260 µs のフレームで、
      前進の eRPM = a + b × cmd を走行ごとに当てはめ、cmd = 180・200 µs のときの eRPM を出す(同じ指令でどれだけ回るか)。
      あわせて、解析 B の (a) に近い「eRPM ÷ 指令」の中央値と、そのときの指令の中央値も出す(指令が小さいと比が大きくなる偏りを見る)。
明るさ: AI 側のフレームの 3 枚に 1 枚の画像の輝度 Y(0.299R + 0.587G + 0.114B)の平均。
"""
import json, os, sys
import numpy as np
from PIL import Image
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
import rawload
R = json.load(open(os.path.join(H, 'recount.json')))

out = {}
for lg, d in R.items():
    if 'error' in d or not lg.startswith('20261024') or d['ai_s'] < 5:
        continue
    r = rawload.load(lg)
    vt, ve = r['v_t'], r['v_erpm']
    at = r['a_t']
    ai_v = r['v_sel'] & ~np.interp(vt, at, r['a_dry'].astype(float)).round().astype(bool)
    k = np.searchsorted(at, vt, side='right') - 1
    k = k.clip(0, len(at) - 1)
    cmd = 1496.0 - r['a_out'][k, 1]
    steer = r['a_out'][k, 0] - 1441.0
    dt = np.diff(vt, prepend=vt[0]).clip(0, 0.05)
    lp = np.zeros_like(cmd); tau = 0.4
    for i in range(1, len(cmd)):
        a = dt[i] / (tau + dt[i]); lp[i] = lp[i - 1] + a * (cmd[i] - lp[i - 1])
    fwd = -ve
    near_col = np.zeros(len(vt), bool)
    for c in d['collisions']:
        near_col |= (vt > c['t_first'] - 1.0) & (vt < c['t_first'] + 1.0)
    m = ai_v & (np.abs(steer) <= 60) & (fwd * 0.5e-3 > 1.0) & ~near_col & (lp >= 120) & (lp <= 260)
    res = dict(no=d['ai_no'], setting='%s %.2f/%.2f' % (d['model'], d['gains'][1], d['gains'][0]), n=int(m.sum()))
    if m.sum() > 100:
        b, a0 = np.polyfit(lp[m], fwd[m], 1)
        res.update(erpm_at180=round(a0 + b * 180), erpm_at200=round(a0 + b * 200), slope=round(b, 2),
                   ratio_med=round(float(np.median(fwd[m] / lp[m])), 2), cmd_med=round(float(np.median(lp[m])), 1))
    w = ai_v & (fwd > 0)
    res['erpm_p90_ai'] = round(float(np.percentile(fwd[ai_v], 90))) if ai_v.any() else None
    # 明るさ
    ct = r['c_t']; files = r['c_files']
    ai_c = np.interp(ct, vt, ai_v.astype(float)) > 0.5
    idx = np.where(ai_c)[0][::3]
    ys = []
    for i in idx:
        im = np.asarray(Image.open(files[i]).convert('RGB'), np.float32)
        ys.append(float((0.299 * im[..., 0] + 0.587 * im[..., 1] + 0.114 * im[..., 2]).mean()))
    res['Y_mean'] = round(float(np.mean(ys)), 2) if ys else None
    res['Y_n'] = len(ys)
    out[lg] = res
    print(lg, res, flush=True)
json.dump(out, open(os.path.join(H, 'cond_check.json'), 'w'), ensure_ascii=False, indent=1)
