#!/usr/bin/env python3
"""S2・S3・G1 の図(2026-10-16)。fidelity.json・fidelity/*・g1_replay.json から描く(読むだけ)

    python figures_s23.py → /home/jetson/jetformer_reports/sim_calibration_20261016/fig/
      fidelity_heldout_sections.png   保留のシーンの 1 秒の窓: 区間ごとの 4 つの指標(P_default・P_cfg・P_cal)
      fidelity_heldout_examples.png   保留のシーンの窓の例: 同じ指令を流した軌跡(実車・較正の前・後)
      g1_r1_entry_lap_loss.png        G1: R1 の進入の速さ(記録 → 層あり、仮想の閉ループ)と、周ごとのラップの損
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import calib_data as cd  # noqa: E402
import calib_model as cm  # noqa: E402
import fidelity as fd  # noqa: E402

REPORT = fd.REPORT
FIG = REPORT / 'fig'
INK, INK2, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#8f8e88', '#e4e3de', '#fcfcfb'
BLUE, ORANGE = '#2a78d6', '#eb6834'
COL = {'P_default': MUTED, 'P_cfg': ORANGE, 'P_cal': BLUE}
LAB = {'P_default': '較正の前(今の物理)', 'P_cfg': '設定欄だけ(P_cfg)', 'P_cal': '較正した物理(P_cal)'}
for f in ('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',):
    try:
        font_manager.fontManager.addfont(f)
    except Exception:
        pass
plt.rcParams.update({
    'font.family': ['Noto Sans CJK JP', 'DejaVu Sans'], 'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK2,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
    'axes.facecolor': SURF, 'figure.facecolor': SURF, 'savefig.facecolor': SURF, 'axes.titlesize': 11, 'axes.titlecolor': INK,
    'legend.frameon': False, 'lines.linewidth': 2.0, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.unicode_minus': False,
})


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / name, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(FIG / name)


def fig_sections():
    F = json.loads((REPORT / 'fidelity.json').read_text())
    t = F['w1']['table']
    groups = ['all', 'human', 'ai', 'R1', 'GM', 'DE', 'LH', 'HS', 'SW']
    gl = ['全体', '人', 'AI', 'R1', 'GM', 'DE', 'LH', 'HS', 'SW']
    mets = [('e_v_rmse', '1 秒後の速さの誤差 [m/s]'), ('e_r_rms', 'ヨーレートの誤差 [rad/s]'),
            ('e_psi_rmse', '1 秒後の向きの誤差 [度]'), ('e_p_rms', '1 秒後の位置の誤差 [m]')]
    fig, axs = plt.subplots(2, 2, figsize=(11, 6.4))
    x = np.arange(len(groups))
    w = 0.27
    for ax, (m, title) in zip(axs.ravel(), mets):
        for k, nm in enumerate(('P_default', 'P_cfg', 'P_cal')):
            vals = [t[g][nm][m] for g in groups]
            ax.bar(x + (k - 1) * w, vals, w - 0.02, color=COL[nm], label=LAB[nm], edgecolor=SURF, linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{a}\n{t[g]["n"]}' for a, g in zip(gl, groups)], fontsize=8.5)
        ax.set_title(title, loc='left')
        ax.grid(axis='x', visible=False)
        ax.axvline(2.5, color=GRID, lw=1.0)
    axs[0, 0].legend(loc='upper left', fontsize=8.5)
    fig.suptitle('保留のシーン(774 窓・41 シーン)で、実車の指令を流したときの誤差(RMS。小さいほど実車に近い)。数字は窓の数',
                 x=0.01, ha='left', fontsize=10.5, color=INK)
    fig.tight_layout()
    save(fig, 'fidelity_heldout_sections.png')


def fig_examples():
    W = fd.load_heldout('w1')
    tru = cm.truth(W)
    js = {nm: fd.load_js(fd.OUTD / 'js_w1', nm, W['n'], cd.N_WIN) for nm in ('P_default', 'P_cfg', 'P_cal')}
    errs = {nm: cm.errors(js[nm], tru) for nm in js}
    picks = []
    rng = np.random.default_rng(7)
    for sec in ('R1', 'GM', 'LH', 'DE'):
        idx = np.nonzero((W['sec'] == sec) & (W['op'] == 0))[0]
        # 典型: P_cal の位置の誤差が、その区間の中央値に近い窓(改善の大きい窓を選んで見栄えを良くしない)
        ep = errs['P_cal']['e_p'][idx]
        j = idx[np.argsort(np.abs(ep - np.median(ep)))[:3]]
        picks.append((sec, int(rng.choice(j))))
    fig, axs = plt.subplots(1, 4, figsize=(12, 3.6))
    for ax, (sec, i) in zip(axs, picks):
        ax.plot(tru['x'][i], tru['y'][i], color=INK, lw=2.4, label='実車(記録)')
        for nm in ('P_default', 'P_cfg', 'P_cal'):
            ax.plot(js[nm]['x'][i], js[nm]['y'][i], color=COL[nm], lw=2.0, label=LAB[nm], ls='--' if nm == 'P_cfg' else '-')
        ax.plot([0], [0], 'o', color=INK, ms=5)
        ax.set_aspect('equal', 'datalim')
        ax.set_title(f'{sec}({W["scene_name"][i][-4:]}・{W["t0"][i]:.1f} 秒)', loc='left', fontsize=10)
        ax.text(0.02, 0.02, '位置の誤差: 前 %.2f → 後 %.2f m' % (errs['P_default']['e_p'][i], errs['P_cal']['e_p'][i]),
                transform=ax.transAxes, fontsize=8, color=INK2)
        ax.set_xlabel('前へ [m]')
    axs[0].set_ylabel('左へ [m]')
    axs[0].legend(loc='upper left', fontsize=7.5)
    fig.suptitle('同じ 1 秒の実車の操作を流したときの軌跡(保留のシーン、人の操作。各区間で P_cal の誤差が中央値に近い窓)',
                 x=0.01, ha='left', fontsize=10.5, color=INK)
    fig.tight_layout()
    save(fig, 'fidelity_heldout_examples.png')


def fig_g1():
    G = json.loads((REPORT / 'g1_replay.json').read_text())
    L = G['v1.1']['logs']
    laps = [l for e in L.values() if 'virtual' in e for l in e['virtual']['R1_2.5']['nominal']]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.0), gridspec_kw={'width_ratios': [1.2, 1]})
    ax = axs[0]
    ent = [(l['r1_entry_rec'], l['r1_entry_capped']) for l in laps if 'r1_entry_rec' in l]
    for a, b in ent:
        ax.plot([0, 1], [a, b], color=MUTED if a <= 2.5 else BLUE, lw=1.6, marker='o', ms=6)
    ax.axhline(2.5, color=INK2, lw=1.0, ls=':')
    ax.text(-0.18, 2.505, 'R1 の上限 2.5 m/s', ha='left', va='bottom', fontsize=8.5, color=INK2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['記録(層なし)', '層あり(仮想の閉ループ)'])
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylabel('R1 の進入の速さ [m/s](起点の手前 0.4 秒の最大)')
    ax.set_title(f'AI の周 {len(ent)} 周: 上限を超えて入った周(青)が上限まで下がる', loc='left', fontsize=10)
    ax.grid(axis='x', visible=False)
    ax = axs[1]
    loss = np.array([l['loss_s'] for l in laps])
    order = np.argsort(loss)
    ax.bar(np.arange(len(loss)), loss[order], 0.7, color=BLUE, edgecolor=SURF)
    ax.axhline(0.3, color=INK2, lw=1.0, ls=':')
    ax.axhline(loss.mean(), color=INK, lw=1.2)
    ax.text(-0.4, 0.302, 'G1 の基準 0.3 秒', ha='left', va='bottom', fontsize=8.5, color=INK2)
    ax.text(-0.4, loss.mean() + 0.002, '平均 %.2f 秒' % loss.mean(), ha='left', va='bottom', fontsize=8.5, color=INK)
    ax.set_xticks([])
    ax.set_ylabel('1 周のタイムの損 [秒]')
    ax.set_title('R1 2.5・GM 2.2・DE 3.0 m/s を全部入れた計算上の損(周ごと)', loc='left', fontsize=10)
    ax.grid(axis='x', visible=False)
    fig.suptitle('安全層 v1.1 の再生(10/10 の AI の記録、車の動きは較正の物理の前後の式で計算)', x=0.01, ha='left', fontsize=10.5, color=INK)
    fig.tight_layout()
    save(fig, 'g1_r1_entry_lap_loss.png')


if __name__ == '__main__':
    fig_sections()
    fig_examples()
    fig_g1()
