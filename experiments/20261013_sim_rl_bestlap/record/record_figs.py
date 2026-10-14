#!/usr/bin/env python3
"""記録の段の図(2026-10-14)

    source ~/miniconda/etc/profile.d/conda.sh && conda activate jetracer
    python -I ~/jetformer/experiments/20261013_sim_rl_bestlap/record/record_figs.py

  fig/record_laps.png        シーンごとの周回時間(1 周目 = 停止から・2 周目 = フライング)。
                             記録(ブラウザ)・記録前の予想(Node)・教師の 20 分データ(同じ 60 の開始)を並べる
  fig/record_trajectories.png 記録した 60 シーンの軌跡(上から)と、教師の標準の開始の軌跡
入力: recording/verify/collection-integrity.json・recording/prediction.json・env/teacher_baseline.json・env/records/
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPORT = Path('/home/jetson/jetformer_reports/sim_rl_bestlap_20261013')
WORLD = Path('/home/jetson/jetformer/experiments/20261013_sim_rl_bestlap/sim/world-v3.json')
SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
BLUE, ORANGE, GRAY = '#2a78d6', '#eb6834', '#8a8984'


def jp_font():
    import matplotlib
    from matplotlib import font_manager
    for name in ('Noto Sans CJK JP', 'Noto Sans JP', 'IPAexGothic', 'IPAGothic', 'TakaoGothic', 'VL Gothic', 'Droid Sans Fallback'):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            matplotlib.rcParams['font.family'] = name
            return name
    return None


def style(ax, title, ylabel='', xlabel=''):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=10.5, loc='left')
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.grid(True, color=GRID, lw=.6)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    jp_font()
    integ = json.loads((REPORT / 'recording/verify/collection-integrity.json').read_text())
    rows = sorted([r for r in integ['rows'] if r.get('pass')], key=lambda r: r['index'])
    teacher = {s['name'].replace('rc20-mode8-', 'rlbest20-mode8-'): s['laps']
               for s in json.loads((REPORT / 'env/teacher_baseline.json').read_text())['REC60_original']['scenes']}
    idx = np.array([r['index'] for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), facecolor=SURFACE, sharex=True)
    for k, (ax, title) in enumerate(zip(axes, ('1 周目(停止から)', '2 周目(フライングラップ)'))):
        rec = np.array([r['lap_times'][k] for r in rows])
        pred = np.array([r['prediction']['predicted_laps'][k] for r in rows])
        tea = np.array([teacher[r['name']][k] for r in rows])
        style(ax, title, '周回時間 [秒]' if k == 0 else '', 'シーンの番号(0〜47 train・48〜59 val)')
        ax.axvspan(47.5, 59.5, color=GRID, alpha=.45, lw=0)
        ax.scatter(idx, tea, s=22, color=GRAY, label=f'教師の 20 分データ(平均 {tea.mean():.3f} 秒)', zorder=2)
        ax.scatter(idx, pred, s=34, facecolors=SURFACE, edgecolors=ORANGE, linewidths=1.4,
                   label=f'記録前の予想・Node(平均 {pred.mean():.3f} 秒)', zorder=3)
        ax.scatter(idx, rec, s=26, color=BLUE, edgecolors=SURFACE, linewidths=.8,
                   label=f'RL の記録・ブラウザ(平均 {rec.mean():.3f} 秒)', zorder=4)
        ax.legend(fontsize=8, frameon=False, labelcolor=INK2, loc='center right')
        ax.set_xlim(-1, 60)
    d1 = np.mean([teacher[r['name']][0] - r['lap_times'][0] for r in rows])
    d2 = np.mean([teacher[r['name']][1] - r['lap_times'][1] for r in rows])
    fig.suptitle(f'記録した {len(rows)} シーンの周回時間: 同じ開始の教師より、平均で 1 周目 {d1:.2f} 秒・2 周目 {d2:.2f} 秒速い',
                 color=INK, fontsize=11, x=.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, .94))
    out = REPORT / 'fig' / 'record_laps.png'
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)

    # 軌跡(上から)。sim の座標: x 右、z は画面の下向き → 図では z を反転せず、上から見た向きは x 右・z 下
    world = json.loads(WORLD.read_text())
    fig, ax = plt.subplots(figsize=(11.5, 7.6), facecolor=SURFACE)
    style(ax, '記録した 60 シーンの軌跡(上から。青 = RL の記録 60 本、灰の破線 = 教師の標準の開始、橙の点 = ゲート、灰の円 = 柱と脚)', 'z [m]', 'x [m]')
    for s in world['surfaces']:
        if s['id'] == 'carpet':
            continue
        col = {'shortcut': '#3b3b3b', 'turf': '#7fbf7f', 'ptfe': '#e8e2c8', 'blue-mat': '#b9d3f0'}.get(s['id'], '#cccccc')
        ax.add_patch(plt.Rectangle((s['x'] - s['w'] / 2, s['z'] - s['d'] / 2), s['w'], s['d'], color=col, alpha=.35, lw=0))
        ax.text(s['x'], s['z'], s['id'], fontsize=7, color=INK2, ha='center', va='center')
    for seg in world['segments']:
        ax.plot([seg['a'][0], seg['b'][0]], [seg['a'][1], seg['b'][1]], color='#9b4a3c', lw=1.2, alpha=.8)
    for dsk in world['disks']:
        ax.add_patch(plt.Circle((dsk['x'], dsk['z']), max(dsk.get('radius', .02), .02), color='#6f6e69', alpha=.55, lw=0))
    for r in rows:
        rep = json.loads((Path(r['dataset']) / 'simulation/evaluation-report.json').read_text())
        x = [t['x'] for t in rep['trace']]; z = [t['z'] for t in rep['trace']]
        ax.plot(x, z, color=BLUE, lw=.6, alpha=.22)
    tr = json.loads((REPORT / 'env/records/teacher_REC60_rc20-mode8-train-00_original.json').read_text())['trace']
    ax.plot([t['x'] for t in tr], [t['z'] for t in tr], color=GRAY, lw=1.3, ls='--', alpha=.9)
    for i, (gx, gz) in enumerate(world['route']):
        ax.plot(gx, gz, marker='o', ms=3, color=ORANGE, alpha=.9)
    b = world['bounds']
    ax.set_xlim(b['minX'], b['maxX']); ax.set_ylim(b['maxZ'], b['minZ'])
    ax.set_aspect('equal')
    fig.tight_layout()
    out2 = REPORT / 'fig' / 'record_trajectories.png'
    fig.savefig(out2, dpi=120, facecolor=SURFACE)
    plt.close(fig)
    print(json.dumps({'laps': str(out), 'trajectories': str(out2)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
