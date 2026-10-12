"""(動画用)fig/src/fig_verdict.py の写し。動画の決まり(時刻・先の予定を言わない)に合わせて 2 か所だけ変えた:
- 副題から固定した時刻を外す
- 下の「次の一歩」の箱を、今日できたこと(学習・TensorRT・管理画面)の箱に替える
数字は元の図と同じく eval/zoo_eval.json から読む。出力は video/fig/verdict_video.png。"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'fig', 'src'))
from zoolib import *  # noqa: F401,F403
from zoolib import Canvas, load, PCOL, POLICIES, INK, INK2, MUTED, GRID, GREEN, RED, BLUE, VIOLET, tint

D = load()
P = D['policy']
ev = D['ev']
delta = D['delta']
vd = ev['verdict']
assert vd['best'] == 'jetformer_v0' and vd['rule'] == 4, vd

c = Canvas()
c.title('ベストは TransFuser軽量型(JetFormer v0 = 今のモデル B)のまま',
        '結果を見る前に固定した規則で判定。新しい 4 つはどれも v0 と同じくらいで、v0 を超えたものは無かった')

# ---- 左: 判定の流れ
c.txt(1.2, 72.6, '判定の流れ(規則 1〜4)', size=19, weight='bold', ha='left')
diffs = [P[p]['diff'] for p in POLICIES if p != 'jetformer_v0']
steps = [
    (56.8, 12.8, '① 候補: ゲート合格・崩れなし・悪い所なし', '→ ACT型・TCP型',
     '外れた: 多段型(DE の出口)・DINOv2型(FP16)', MUTED),
    (44.4, 9.0, '② 主指標で v0 より 10 µs 以上良い', 'なし', '差は %+.1f〜%+.1f µs(δ の 1/10 ほど)' % (min(diffs), max(diffs)), MUTED),
    (32.0, 9.0, '③ R1 の進入・DE の出口で「良い」', 'なし', '陰性対照(v0 の種どうし)も確かめた', MUTED),
]
X0, W0 = 1.0, 62.0
for (yb, h, head, res, note, col) in steps:
    c.box(X0, yb, W0, h, 'white', col, lw=1.8)
    c.txt(X0 + 1.4, yb + h - 2.6, head, size=17, weight='bold', ha='left')
    c.txt(X0 + 1.4, yb + 2.4, note, size=16, color=INK2, ha='left')
    if res.startswith('→'):
        c.txt(X0 + 1.4, yb + h / 2, res, size=17, weight='bold', color=INK, ha='left')
    else:
        c.txt(X0 + W0 - 1.4, yb + h - 2.6, res, size=17, weight='bold', color=INK, ha='right')
for y0, y1 in ((56.8, 53.4), (44.4, 41.0)):
    c.arrow(X0 + W0 / 2, y0, X0 + W0 / 2, y1, color=MUTED)
c.arrow(X0 + W0 / 2, 32.0, X0 + W0 / 2, 29.2, color=GREEN)
c.box(X0, 13.0, W0, 16.2, tint(GREEN, 0.16), GREEN, lw=3.0)
c.txt(X0 + 1.4, 25.8, '④ ベスト = TransFuser軽量型', size=24, weight='bold', ha='left', color=INK)
c.txt(X0 + 1.4, 21.2, '走行はモデル B のエンジンのまま(変更なし)', size=17, ha='left')
c.txt(X0 + 1.4, 17.8, '管理画面の走行タブで「★推奨」として最初に選ばれる', size=16, color=INK2, ha='left')
c.txt(X0 + 1.4, 15.0, '作り直した DINOv2型 を候補に入れても結論は同じ', size=16, color=INK2, ha='left')

# ---- 右: どれくらい確か
XR = 67.5
c.txt(XR, 72.6, 'どれくらい確か?', size=19, weight='bold', ha='left')
c.txt(XR, 69.2, '主指標 [µs]: 方策の差は、種を変えただけの差と同じ大きさ', size=16, color=INK2, ha='left')
br = [abs(v) for k, v in ev['primary']['best_minus_last_us'].items() if k != 'B']
sr = [ev['primary']['seed_range_us'][p]['best'] for p in POLICIES]
bars = [('区別の幅 δ', 0, delta, INK2), ('best と last の差', min(br), max(br), MUTED),
        ('種だけ変えた差', min(sr), max(sr), MUTED), ('方策と v0 の差', 0.0, max(diffs), VIOLET)]
ax = c.fig.add_axes([0.66, 0.60, 0.315, 0.20])
for i, (lab, lo, hi, col) in enumerate(bars):
    ax.barh(i, hi - lo, left=lo, height=0.52, color=tint(col, 0.55) if col != INK2 else tint(INK2, 0.18), edgecolor=col, lw=1.4)
    ax.text(hi + 0.25, i, ('%.0f µs' % hi) if lab.startswith('区別') else ('%.2f〜%.2f' % (lo, hi) if lo > 0 else '〜%.1f' % hi),
            va='center', ha='left', fontsize=16, color=INK)
ax.set_yticks(range(len(bars)))
ax.set_yticklabels([b[0] for b in bars], fontsize=16)
ax.set_xlim(0, 13.5)
ax.set_xticks([0, 5, 10])
ax.tick_params(axis='x', labelsize=16)
ax.grid(axis='x', color=GRID, lw=1.0)
ax.set_axisbelow(True)

c.txt(XR, 45.0, 'R1・DE [µs]: 方策の差は、v0 の種どうしの差と同じ大きさ', size=16, color=INK2, ha='left')
r1, de = ev['r1'], ev['de']
pol_vals = [r1['per_policy'][p]['d_mean'] for p in r1['per_policy']] + [de['per_policy'][p]['d_mean'] for p in de['per_policy']]
nc_vals = [r1['nc'][k]['d_mean'] for k in r1['nc']] + [de['nc'][k]['d_mean'] for k in de['nc']]
ax2 = c.fig.add_axes([0.66, 0.37, 0.315, 0.125])
ax2.axvspan(-delta, delta, color=tint(BLUE, 0.08), zorder=0)
for i, (lab, vals, col) in enumerate([('v0 の種どうし', nc_vals, MUTED), ('4 方策 − v0', pol_vals, VIOLET)]):
    ax2.plot([min(vals), max(vals)], [i, i], color=col, lw=5, solid_capstyle='round', zorder=3)
    ax2.scatter(vals, [i] * len(vals), s=55, color=col, edgecolor='white', zorder=4)
    ax2.text(max(vals) + 0.8, i, '%+.1f〜%+.1f' % (min(vals), max(vals)), va='center', ha='left', fontsize=16, color=INK)
ax2.axvline(0, color=INK2, lw=1.0)
ax2.set_yticks([0, 1])
ax2.set_yticklabels(['v0 の種どうし', '4 方策 − v0'], fontsize=16)
ax2.set_ylim(-0.6, 1.6)
ax2.set_xlim(-12, 17)
ax2.set_xticks([-10, 0, 10])
ax2.tick_params(axis='x', labelsize=16)
ax2.set_xlabel('出力の差の平均(帯 = ±10 µs)', fontsize=16, color=INK2, labelpad=1)
ax2.grid(axis='x', color=GRID, lw=1.0)
ax2.set_axisbelow(True)

items = [('言える', 'v0 を 10 µs 以上超える方策は無い', GREEN),
         ('言えない', '4 つの中の順位(差がばらつき程度)', INK2),
         ('まだ分からない', '実車での差(机上だけ・種は 2 本)', RED)]
for i, (k, s, col) in enumerate(items):
    y = 20.6 - i * 3.0
    c.txt(XR, y, k, size=17, weight='bold', color=col, ha='left')
    c.txt(XR + 17.0, y, s, size=17, ha='left')

# ---- 下: 今日できたこと(動画用。元の図では「次の一歩」)
c.box(1.0, 1.4, 133.4, 8.6, tint(INK2, 0.07), GRID, lw=1.6)
c.txt(2.4, 7.4, '4 つとも使える', size=17, weight='bold', ha='left')
c.txt(2.4, 3.9, '5 つの方策 × 2 本 = 10 本を学習し、全部 TensorRT に。管理画面の学習タブ・走行タブから、同じ安全の仕組みで使える',
      size=17, ha='left')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fig')
os.makedirs(OUT, exist_ok=True)
c.save(os.path.abspath(os.path.join(OUT, 'verdict_video.png')))
