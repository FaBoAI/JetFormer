#!/usr/bin/env python3
"""報告・動画のための画面の一覧(sim_diverse_train_20261015、2026-10-15): S-DIV の E2(完走した開始)を追従の視点で撮った窓と車載カメラを 1 枚に並べる
→ fig/eval_screens_sdiv_e2.png。画面は eval/closed_loop_screens の走行(本番の E2 と軌跡の差 0)で撮ったもの。"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

R = Path('/home/jetson/jetformer_reports/sim_diverse_train_20261015')
S = R / 'screens' / 'lap_SDIV_E2'
for name in ('Noto Sans CJK JP', 'Noto Sans JP', 'IPAexGothic'):
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        matplotlib.rcParams['font.family'] = name
        break
stems = sorted({p.name.rsplit('_', 1)[0] for p in S.glob('eval_SDIV-s0-E2-single_f*_window.png')})
fig, axes = plt.subplots(2, len(stems), figsize=(4.2 * len(stems), 6.2), facecolor='#fcfcfb',
                         gridspec_kw={'height_ratios': [1.25, 1]})
for i, st in enumerate(stems):
    f = int(st.rsplit('_f', 1)[1])
    w = plt.imread(S / f'{st}_window.png')
    h, wd = w.shape[:2]
    axes[0][i].imshow(w[int(h * .14):int(h * .70), int(wd * .02):int(wd * .745)])   # 3D の画面だけを切り出す
    axes[0][i].set_title(f'{f / 30:.1f} 秒(フレーム {f})', fontsize=10, color='#0b0b0b')
    axes[1][i].imshow(plt.imread(S / f'{st}_camera.png'))
    for ax in (axes[0][i], axes[1][i]):
        ax.set_xticks([])
        ax.set_yticks([])
axes[0][0].set_ylabel('追従の視点', fontsize=10)
axes[1][0].set_ylabel('車載カメラ(224×224。JetFormer の入力)', fontsize=9)
fig.suptitle('S-DIV が E2(ゲート 0・右 3 cm・右 3°)から 8.36 秒で完走した走り(シミュレーターの閉ループ、遅延 1)', fontsize=11, x=.01, ha='left')
fig.tight_layout()
out = R / 'fig' / 'eval_screens_sdiv_e2.png'
fig.savefig(out, dpi=110, facecolor='#fcfcfb')
print(out)
